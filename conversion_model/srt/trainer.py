import torch
import numpy as np
import torch.nn as nn
import os
import math
from contextlib import nullcontext
import srt.utils.visualize as vis

from tqdm import tqdm
from srt.utils.common import mse2psnr, concatenate_dict, gather_all, get_rank, get_world_size
from modules.util import ImagePyramide
from modules.perceptual_loss import *
from modules.temporal_align import multiscale_temporal_consistency_loss
from collections import defaultdict
    


class FSRTTrainer:
    def __init__(self, model, optimizer, cfg, device, out_dir, kp_detector, discriminator, optimizer_disc):
        self.model = model
        self.kp_detector = kp_detector
        self.optimizer = optimizer
        self.config = cfg
        self.device = device
        self.out_dir = out_dir
        self.dataset_type = cfg.get('data', {}).get('dataset', 'vox')
        self.is_rppg_video = self.dataset_type == 'rppg_video'
        training_cfg = cfg.get('training', {})
        disc_cfg = cfg.get('discriminator', {})

        self.phase2_start = training_cfg.get('iters_in_phase1', 0)
        self.phase3_start = training_cfg.get('iters_in_phase1', 0) + training_cfg.get('iters_in_phase2', 0)
        self.disc_warmup_iters = training_cfg.get('disc_warmup_iters', 0)
        self.statistical_regularization = training_cfg.get('statistical_regularization', False)
        self.simulate_out_of_frame_motion = cfg['data'].get('simulate_out_of_frame_motion', False)
        self.scales = training_cfg.get('scales', [1.0])
        self.pyramid = ImagePyramide(self.scales, 3)
        self.perceptual_loss_weight = training_cfg.get('perceptual_loss_weight', [0.0] * 5)
        self.vgg19 = None
        if (not self.is_rppg_video) and sum(self.perceptual_loss_weight) > 0.0:
            self.vgg19 = init_perceptual_loss(device=self.device)
        self.lambda_var = training_cfg.get('variance_loss_weight', 0.0)
        self.lambda_cov = training_cfg.get('covariance_loss_weight', 0.0)
        self.lambda_invar = training_cfg.get('invariance_loss_weight', 0.0)
        self.lambda_mse = training_cfg.get('mse_loss_weight', 1.0)
        self.generator_gan_weight = training_cfg.get('generator_gan_loss_weight', 0.0)
        self.discriminator_gan_weight = training_cfg.get('discriminator_gan_loss_weight', 0.0)
        self.generator_gan_feature_matching_weight = training_cfg.get('generator_gan_feature_matching', [0.0] * 4)
        self.disc_scales = disc_cfg.get('scales', [1])
        self.use_disc = disc_cfg.get('use_disc', False)
        self.optimizer_disc = optimizer_disc
        self.discriminator = discriminator

        if self.is_rppg_video:
            # Step-2 objective: sequential rollout with L_rec + L_temp.
            loss_weights = training_cfg.get('loss_weights', {})
            self.lambda_rec = float(loss_weights.get('rec', 1.0))
            self.lambda_temp = float(loss_weights.get('temp', 0.1))
            self.rppg_use_motion_conditioning = bool(
                training_cfg.get('rppg_use_motion_conditioning', True)
            )
            self._rppg_conditioning_warned = False

            self.temporal_scales = tuple(training_cfg.get('temporal_scales', [1.0, 0.5, 0.25]))
            self.temporal_scale_weights = training_cfg.get('temporal_scale_weights', None)

            if self.temporal_scale_weights is not None:
                self.temporal_scale_weights = tuple(float(x) for x in self.temporal_scale_weights)
                if len(self.temporal_scale_weights) != len(self.temporal_scales):
                    raise ValueError('training.temporal_scale_weights must match training.temporal_scales.')

            self.rppg_task_mode = training_cfg.get('task_mode', 'm2s')
            data_num_pixels = int(cfg.get('data', {}).get('num_pixels', 65536))
            self.rppg_decode_chunk = int(training_cfg.get('rppg_decode_chunk', min(data_num_pixels, 8192)))
            if self.rppg_decode_chunk < 1:
                self.rppg_decode_chunk = min(data_num_pixels, 8192)
            self.rppg_framewise_backward = bool(training_cfg.get('rppg_framewise_backward', False))
            self.rppg_recompute_encoder_per_frame = bool(
                training_cfg.get('rppg_recompute_encoder_per_frame', False)
            )
            # Memory-safe option: backward chunk-by-chunk to avoid keeping all chunk graphs.
            self.rppg_streaming_backward = bool(training_cfg.get('rppg_streaming_backward', False))
            # In streaming mode we freeze encoder by default, so chunks can backward independently.
            self.rppg_freeze_encoder = bool(training_cfg.get('rppg_freeze_encoder', self.rppg_streaming_backward))
            # Temporal loss is logged but not backpropagated in streaming mode to keep memory bounded.
            self.rppg_stream_detach_temp = bool(training_cfg.get('rppg_stream_detach_temp', True))
            if self.rppg_streaming_backward and self.rppg_framewise_backward:
                raise ValueError(
                    "training.rppg_streaming_backward and training.rppg_framewise_backward cannot both be true."
                )
            if self.rppg_streaming_backward and not self.rppg_freeze_encoder:
                raise ValueError("training.rppg_streaming_backward=true requires training.rppg_freeze_encoder=true.")
            # Disable legacy branches that depend on Vox-specific augmentations.
            self.statistical_regularization = False
            self.use_disc = False

    def evaluate(self, val_loader):
        ''' Performs an evaluation.
        Args:
            val_loader (dataloader): pytorch dataloader
        '''
        self.model.eval()
        eval_lists = defaultdict(list)

        loader = val_loader if get_rank() > 0 else tqdm(val_loader)
        sceneids = []

        for data in loader:
            sceneids.append(data['sceneid'])
            eval_step_dict = self.eval_step(data)

            for k, v in eval_step_dict.items():
                eval_lists[k].append(v)

        sceneids = torch.cat(sceneids, 0).cuda()
        sceneids = torch.cat(gather_all(sceneids), 0)

        print(f'Evaluated {len(torch.unique(sceneids))} unique scenes.')
        eval_dict = {k: torch.cat([v_.unsqueeze(0) if v_.dim() == 0 else v_ for v_ in v],dim=0) for k, v in eval_lists.items()} 
        eval_dict = concatenate_dict(eval_dict)  # Concatenate across processes
        eval_dict = {k: v.mean().item() for k, v in eval_dict.items()}  # Average across batch_size
        print('Evaluation results:')
        print(eval_dict)
        return eval_dict

    def train_step(self, data, it):
        self.model.train()
        self.optimizer.zero_grad()
        if self.is_rppg_video and getattr(self, 'rppg_streaming_backward', False):
            return self._train_step_rppg_streaming(data, it)
        if self.is_rppg_video and getattr(self, 'rppg_framewise_backward', False):
            return self._train_step_rppg_framewise(data, it)
        loss, loss_terms, disc_items = self.compute_loss(data, it)
        loss = loss.mean(0)
        if self.statistical_regularization:
            loss+=loss_terms['reg_loss'].mean()
        loss_terms = {k: v.mean(0).item() for k, v in loss_terms.items()}
        loss.backward()
        self.optimizer.step()
        
        if it >= self.phase3_start and self.use_disc:
            self.optimizer_disc.zero_grad()
            loss_disc = 0
            
            discriminator_maps_generated = self.discriminator(disc_items['generated'], kp=disc_items['target_kp_disc'], detach=True)
            discriminator_maps_real = self.discriminator(disc_items['real'], kp=disc_items['target_kp_disc'])

            for scale in self.disc_scales:
                key = 'prediction_map_%s' % scale
                value = (1. - discriminator_maps_real[key]) ** 2 + discriminator_maps_generated[key] ** 2
                loss_disc += self.discriminator_gan_weight * value.mean()
            loss_terms['disc_gan'] = loss_disc
            loss_disc.backward()
            self.optimizer_disc.step()
        return loss.item(), loss_terms

    def _train_step_rppg_streaming(self, data, it):
        """Memory-safe rPPG step: backward per chunk to avoid graph accumulation."""
        device = self.device
        clip_frames = data.get('clip_frames')
        if clip_frames is None:
            raise ValueError("rppg_video batch missing 'clip_frames'.")
        clip_frames = clip_frames.to(device)
        if clip_frames.dim() != 5:
            raise ValueError(f"Expected clip_frames [B,T,C,H,W], got {tuple(clip_frames.shape)}")

        driving_frames = data.get('driving_frames')
        if driving_frames is not None:
            driving_frames = driving_frames.to(device)
            if driving_frames.shape != clip_frames.shape:
                raise ValueError(
                    f"driving_frames shape {tuple(driving_frames.shape)} must match clip_frames {tuple(clip_frames.shape)}."
                )
        else:
            driving_frames = clip_frames

        bs, t_steps, c, h, w = clip_frames.shape
        if c != 3:
            raise ValueError(f"rppg_video expects 3-channel RGB frames, got C={c}.")
        if t_steps < 1:
            raise ValueError("rppg_video clip must contain at least one frame.")

        has_paired_target = self._get_has_paired_target(data, batch_size=bs, device=device)
        paired_denom = torch.clamp(has_paired_target.sum(), min=1.0)
        num_kp = self._get_num_kp()

        source_frames = clip_frames[:, :1]
        source_kps, encoder_expression = self._build_rppg_source_conditioning(
            source_frames=source_frames,
            num_kp=num_kp,
            device=device,
            dtype=clip_frames.dtype,
        )
        source_pos = self._build_full_image_pos(
            batch_size=bs,
            height=h,
            width=w,
            num_views=1,
            device=device,
            dtype=clip_frames.dtype,
        )

        enc_ctx = torch.no_grad() if self.rppg_freeze_encoder else nullcontext()
        with enc_ctx:
            z = self.model.encoder(source_frames, source_kps, source_pos, expression_vector=encoder_expression)
        if self.rppg_freeze_encoder:
            z = z.detach()

        target_pos = source_pos[:, 0].reshape(bs, h * w, 2)

        rec_sum = clip_frames.new_zeros(())
        rec_weight_sum = clip_frames.new_zeros(())
        temp_sum = clip_frames.new_zeros(())
        temp_count = 0
        temporal_detail_sum = defaultdict(lambda: clip_frames.new_zeros(()))

        prev_generated = None
        prev_input_frame = None
        prev_state = None
        n_pix = target_pos.shape[1]
        chunk_size = max(1, int(getattr(self, 'rppg_decode_chunk', n_pix)))

        for t in range(t_steps):
            # Keep temporal decoder conditioning consistent with inference path.
            cur_input = driving_frames[:, t]
            cur_target = driving_frames[:, t]
            target_kps, decoder_expression = self._build_rppg_driving_conditioning(
                driving_frame=cur_input,
                n_pix=n_pix,
                num_kp=num_kp,
                device=device,
                dtype=clip_frames.dtype,
            )
            cur_target_flat = cur_target.permute(0, 2, 3, 1).reshape(bs, n_pix, 3)
            frame_rec_per_sample = clip_frames.new_zeros((bs,))
            pred_buffer = torch.zeros((bs, n_pix, 3), device=device, dtype=clip_frames.dtype)
            extras_first = {}

            for start in range(0, n_pix, chunk_size):
                end = min(n_pix, start + chunk_size)
                pos_chunk = target_pos[:, start:end]
                kps_chunk = target_kps[:, start:end]
                tgt_chunk = cur_target_flat[:, start:end]

                temporal_inputs = {
                    'cur_frame': cur_input,
                    'prev_frame': prev_input_frame,
                    'prev_generated': prev_generated,
                    'prev_feat': (prev_state or {}).get('prev_feat'),
                }
                try:
                    pred_chunk, extras_chunk = self.model.decoder(
                        z,
                        pos_chunk,
                        kps_chunk,
                        expression_vector=decoder_expression,
                        temporal_inputs=temporal_inputs,
                        return_temporal_state=True,
                    )
                except TypeError:
                    pred_chunk, extras_chunk = self.model.decoder(
                        z,
                        pos_chunk,
                        kps_chunk,
                        expression_vector=decoder_expression,
                    )
                    if not isinstance(extras_chunk, dict):
                        extras_chunk = {}

                if start == 0:
                    extras_first = extras_chunk if isinstance(extras_chunk, dict) else {}

                chunk_frac = float(end - start) / float(n_pix)
                mse_chunk = ((pred_chunk - tgt_chunk) ** 2).mean(dim=(1, 2))
                frame_rec_per_sample = frame_rec_per_sample + mse_chunk * chunk_frac

                # Backprop chunk contribution immediately to cap peak memory.
                chunk_loss = self.lambda_rec * ((mse_chunk * has_paired_target).sum() / paired_denom)
                chunk_loss = chunk_loss * (chunk_frac / float(t_steps))
                chunk_loss.backward()

                pred_buffer[:, start:end] = pred_chunk.detach()

            rec_sum = rec_sum + (frame_rec_per_sample * has_paired_target).sum().detach()
            rec_weight_sum = rec_weight_sum + has_paired_target.sum().detach()

            pred_frame = pred_buffer.view(bs, h, w, 3).permute(0, 3, 1, 2).contiguous()

            if t > 0 and prev_generated is not None:
                temporal_aux = extras_first.get('temporal_aux', {})
                flow_t = temporal_aux.get('flow')
                if flow_t is not None:
                    temp_cur = pred_frame.detach() if self.rppg_stream_detach_temp else pred_frame
                    temp_t, detail_t = multiscale_temporal_consistency_loss(
                        cur_frame=temp_cur,
                        prev_frame=prev_generated,
                        flow=flow_t,
                        scales=self.temporal_scales,
                        scale_weights=self.temporal_scale_weights,
                    )
                    temp_sum = temp_sum + temp_t.detach()
                    temp_count += 1
                    for key, value in detail_t.items():
                        temporal_detail_sum[key] = temporal_detail_sum[key] + value.detach()

            prev_state = extras_first.get('temporal_state', None)
            prev_generated = pred_frame.detach()
            prev_input_frame = cur_input

        self.optimizer.step()

        rec_denom = torch.clamp(rec_weight_sum, min=1.0)
        rec_loss = rec_sum / rec_denom
        if temp_count > 0:
            temp_loss = temp_sum / float(temp_count)
        else:
            temp_loss = clip_frames.new_zeros(())

        total_loss = self.lambda_rec * rec_loss + self.lambda_temp * temp_loss
        loss_terms = {
            'mse': float(rec_loss.item()),
            'rec': float(rec_loss.item()),
            'temp': float(temp_loss.item()),
            'loss_total': float(total_loss.item()),
            'paired_weight_sum': float(rec_weight_sum.item()),
            'temp_steps': float(temp_count),
        }
        if temp_count > 0:
            for key, value in temporal_detail_sum.items():
                loss_terms[f'temp_{key}'] = float((value / float(temp_count)).item())
        return float(total_loss.item()), loss_terms

    def _train_step_rppg_framewise(self, data, it):
        """Memory-safe non-streaming rPPG step with frame-wise backward."""
        device = self.device
        clip_frames = data.get('clip_frames')
        if clip_frames is None:
            raise ValueError("rppg_video batch missing 'clip_frames'.")
        clip_frames = clip_frames.to(device)
        if clip_frames.dim() != 5:
            raise ValueError(f"Expected clip_frames [B,T,C,H,W], got {tuple(clip_frames.shape)}")

        driving_frames = data.get('driving_frames')
        if driving_frames is not None:
            driving_frames = driving_frames.to(device)
            if driving_frames.shape != clip_frames.shape:
                raise ValueError(
                    f"driving_frames shape {tuple(driving_frames.shape)} must match clip_frames {tuple(clip_frames.shape)}."
                )
        else:
            driving_frames = clip_frames

        bs, t_steps, c, h, w = clip_frames.shape
        if c != 3:
            raise ValueError(f"rppg_video expects 3-channel RGB frames, got C={c}.")
        if t_steps < 1:
            raise ValueError("rppg_video clip must contain at least one frame.")

        has_paired_target = self._get_has_paired_target(data, batch_size=bs, device=device)
        paired_denom = torch.clamp(has_paired_target.sum(), min=1.0)
        num_kp = self._get_num_kp()

        source_frames = clip_frames[:, :1]
        source_kps, encoder_expression = self._build_rppg_source_conditioning(
            source_frames=source_frames,
            num_kp=num_kp,
            device=device,
            dtype=clip_frames.dtype,
        )
        source_pos = self._build_full_image_pos(
            batch_size=bs,
            height=h,
            width=w,
            num_views=1,
            device=device,
            dtype=clip_frames.dtype,
        )

        target_pos = source_pos[:, 0].reshape(bs, h * w, 2)

        if self.rppg_recompute_encoder_per_frame:
            z_base = None
        else:
            z_base = self.model.encoder(
                source_frames, source_kps, source_pos, expression_vector=encoder_expression
            )

        rec_sum = clip_frames.new_zeros(())
        rec_weight_sum = clip_frames.new_zeros(())
        temp_sum = clip_frames.new_zeros(())
        temp_count = 0
        temporal_detail_sum = defaultdict(lambda: clip_frames.new_zeros(()))

        prev_generated = None
        prev_input_frame = None
        prev_state = None
        expected_temp_steps = max(1, t_steps - 1)

        def decode_frame_chunked(z_latent, cur_input, prev_input, prev_gen, prev_st):
            n_pix = target_pos.shape[1]
            step = max(1, int(getattr(self, 'rppg_decode_chunk', n_pix)))
            pred_chunks = []
            first_extras = {}

            for start in range(0, n_pix, step):
                end = min(n_pix, start + step)
                pos_chunk = target_pos[:, start:end]
                kps_chunk = target_kps[:, start:end]
                temporal_inputs = {
                    'cur_frame': cur_input,
                    'prev_frame': prev_input,
                    'prev_generated': prev_gen,
                    'prev_feat': (prev_st or {}).get('prev_feat'),
                }
                try:
                    pred_chunk, extras_chunk = self.model.decoder(
                        z_latent,
                        pos_chunk,
                        kps_chunk,
                        expression_vector=decoder_expression,
                        temporal_inputs=temporal_inputs,
                        return_temporal_state=True,
                    )
                except TypeError:
                    pred_chunk, extras_chunk = self.model.decoder(
                        z_latent,
                        pos_chunk,
                        kps_chunk,
                        expression_vector=decoder_expression,
                    )
                    if not isinstance(extras_chunk, dict):
                        extras_chunk = {}
                pred_chunks.append(pred_chunk)
                if start == 0:
                    first_extras = extras_chunk

            pred_pixels_full = torch.cat(pred_chunks, dim=1)
            if not isinstance(first_extras, dict):
                first_extras = {}
            return pred_pixels_full, first_extras

        for t in range(t_steps):
            if self.rppg_recompute_encoder_per_frame:
                z_t = self.model.encoder(
                    source_frames, source_kps, source_pos, expression_vector=encoder_expression
                )
            else:
                z_t = z_base

            cur_input = driving_frames[:, t]
            cur_target = driving_frames[:, t]
            target_kps, decoder_expression = self._build_rppg_driving_conditioning(
                driving_frame=cur_input,
                n_pix=target_pos.shape[1],
                num_kp=num_kp,
                device=device,
                dtype=clip_frames.dtype,
            )
            pred_pixels, extras = decode_frame_chunked(
                z_latent=z_t,
                cur_input=cur_input,
                prev_input=prev_input_frame,
                prev_gen=prev_generated,
                prev_st=prev_state,
            )

            pred_frame = pred_pixels.view(bs, h, w, 3).permute(0, 3, 1, 2).contiguous()

            rec_per_sample = ((pred_frame - cur_target) ** 2).mean(dim=(1, 2, 3))
            rec_sum = rec_sum + (rec_per_sample * has_paired_target).sum().detach()
            rec_weight_sum = rec_weight_sum + has_paired_target.sum().detach()

            frame_loss = self.lambda_rec * ((rec_per_sample * has_paired_target).sum() / paired_denom)
            frame_loss = frame_loss / float(t_steps)

            if t > 0 and prev_generated is not None:
                temporal_aux = extras.get('temporal_aux', {})
                flow_t = temporal_aux.get('flow')
                if flow_t is not None:
                    temp_t, detail_t = multiscale_temporal_consistency_loss(
                        cur_frame=pred_frame,
                        prev_frame=prev_generated,
                        flow=flow_t,
                        scales=self.temporal_scales,
                        scale_weights=self.temporal_scale_weights,
                    )
                    temp_sum = temp_sum + temp_t.detach()
                    temp_count += 1
                    for key, value in detail_t.items():
                        temporal_detail_sum[key] = temporal_detail_sum[key] + value.detach()
                    frame_loss = frame_loss + (self.lambda_temp * temp_t / float(expected_temp_steps))

            retain_graph = (not self.rppg_recompute_encoder_per_frame) and (t < t_steps - 1)
            frame_loss.backward(retain_graph=retain_graph)

            prev_state = extras.get('temporal_state', None)
            prev_generated = pred_frame.detach()
            prev_input_frame = cur_input

        self.optimizer.step()

        rec_denom = torch.clamp(rec_weight_sum, min=1.0)
        rec_loss = rec_sum / rec_denom
        if temp_count > 0:
            temp_loss = temp_sum / float(temp_count)
        else:
            temp_loss = clip_frames.new_zeros(())

        total_loss = self.lambda_rec * rec_loss + self.lambda_temp * temp_loss
        loss_terms = {
            'mse': float(rec_loss.item()),
            'rec': float(rec_loss.item()),
            'temp': float(temp_loss.item()),
            'loss_total': float(total_loss.item()),
            'paired_weight_sum': float(rec_weight_sum.item()),
            'temp_steps': float(temp_count),
        }
        if temp_count > 0:
            for key, value in temporal_detail_sum.items():
                loss_terms[f'temp_{key}'] = float((value / float(temp_count)).item())
        return float(total_loss.item()), loss_terms


        
    def extract_keypoints_and_expression(self, img_src, img_driv, img_src_augm=None, img_driv_augm=None): 
        '''
        Shapes:
            img_src:       [bs,nsrc,3,h,w]
            img_driv:      [bs,(ndriv),3,h,w]
        '''
        assert self.kp_detector is not None
        if len(img_driv.shape) == 4:
            img_driv = img_driv.unsqueeze(1) 
            if img_driv_augm is not None:
                img_driv_augm = img_driv_augm.unsqueeze(1) 
            
        bs, nsrc, c, h, w = img_src.shape
        nkp = self.kp_detector.num_kp
        ndriv = img_driv.shape[1]    
        img = torch.cat([img_src,img_driv], dim = 1).view(-1,c,h,w)
        if img_src_augm is not None:
            img_augm = torch.cat([img_src_augm,img_driv_augm], dim = 1).view(-1,c,h,w)

        with torch.no_grad():
            kps, latent_dict = self.kp_detector(img)
            kps = kps.view(bs,nsrc+ndriv,nkp,2)
            heatmaps = latent_dict['heatmap'].view(bs,nsrc+ndriv,nkp,latent_dict['heatmap'].shape[-2],latent_dict['heatmap'].shape[-1])
            feature_maps = latent_dict['feature_map'].view(bs,nsrc+ndriv,latent_dict['feature_map'].shape[-3],latent_dict['feature_map'].shape[-2],latent_dict['feature_map'].shape[-1])
            
        if img_src_augm is not None:
            with torch.no_grad():
                _, latent_dict_augm = self.kp_detector(img_augm)
                heatmaps_augm = latent_dict_augm['heatmap'].view(bs,nsrc+ndriv,nkp,latent_dict_augm['heatmap'].shape[-2],latent_dict_augm['heatmap'].shape[-1])
                feature_maps_augm = latent_dict_augm['feature_map'].view(bs,nsrc+ndriv,latent_dict_augm['feature_map'].shape[-3],latent_dict_augm['feature_map'].shape[-2],latent_dict_augm['feature_map'].shape[-1])
            
            
        kps_src, kps_driv = torch.split(kps,[nsrc,ndriv], dim=1)
        _, heatmap_driv = torch.split(heatmaps,[nsrc,ndriv], dim=1)
        _, feature_map_driv = torch.split(feature_maps,[nsrc,ndriv], dim=1)
        
        if kps_driv.shape[1] == 1:
            kps_driv = kps_driv.squeeze(1)
        
        expression_vector_src , expression_vector = torch.split(self.model.expression_encoder(feature_maps.flatten(0,1),heatmaps.flatten(0,1)).view(bs,nsrc+ndriv,-1), [nsrc,ndriv], dim = 1) 
        if expression_vector.shape[1] == 1:
            expression_vector = expression_vector.squeeze(1)
            
        if img_src_augm is not None:
            expression_vector_src_augm , expression_vector_augm = torch.split(self.model.expression_encoder(feature_maps_augm.flatten(0,1),heatmaps_augm.flatten(0,1)).view(bs,nsrc+ndriv,-1), [nsrc,ndriv], dim = 1) 
            if expression_vector_augm.shape[1] == 1:
                expression_vector_augm = expression_vector_augm.squeeze(1)
        else:
            expression_vector_src_augm = expression_vector_augm = None

        return kps_src, kps_driv, expression_vector, expression_vector_src, expression_vector_augm, expression_vector_src_augm
    
    def _get_num_kp(self):
        if self.kp_detector is not None and hasattr(self.kp_detector, 'num_kp'):
            return int(self.kp_detector.num_kp)
        model_cfg = self.config.get('model', {})
        enc_kwargs = model_cfg.get('encoder_kwargs', {})
        return int(enc_kwargs.get('num_kp', 10))

    def _build_full_image_pos(self, batch_size, height, width, num_views, device, dtype):
        ys = torch.arange(height, device=device, dtype=dtype)
        xs = torch.arange(width, device=device, dtype=dtype)
        try:
            grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        except TypeError:
            grid_y, grid_x = torch.meshgrid(ys, xs)

        grid_x = (grid_x + 0.5 - (width / 2.0)) / (width / 2.0)
        grid_y = (grid_y + 0.5 - (height / 2.0)) / (height / 2.0)
        grid = torch.stack([grid_x, grid_y], dim=-1)
        grid = grid.unsqueeze(0).unsqueeze(0).repeat(batch_size, num_views, 1, 1, 1)
        return grid

    def _get_has_paired_target(self, data, batch_size, device):
        has_paired_target = data.get('has_paired_target', None)
        if has_paired_target is None:
            return torch.ones(batch_size, device=device, dtype=torch.float32)
        if isinstance(has_paired_target, list):
            has_paired_target = torch.tensor([1.0 if bool(x) else 0.0 for x in has_paired_target], device=device)
        elif not torch.is_tensor(has_paired_target):
            has_paired_target = torch.tensor(has_paired_target, device=device)
        has_paired_target = has_paired_target.to(device=device)
        if has_paired_target.dim() == 0:
            has_paired_target = has_paired_target.repeat(batch_size)
        return has_paired_target.float()

    def _warn_rppg_conditioning_once(self, message):
        if not getattr(self, '_rppg_conditioning_warned', False):
            print(f'[WARN] {message}')
            self._rppg_conditioning_warned = True

    def _fit_expression_vector(self, expression, target_dim, batch_size, device, dtype, add_view_dim=False):
        if target_dim <= 0:
            return None
        if expression is None:
            return None
        if not torch.is_tensor(expression):
            expression = torch.tensor(expression, device=device)
        expression = expression.to(device=device, dtype=dtype).detach()
        if expression.dim() == 3 and expression.shape[1] == 1:
            expression = expression.squeeze(1)
        elif expression.dim() != 2:
            expression = expression.reshape(batch_size, -1)

        cur_dim = int(expression.shape[-1])
        if cur_dim < target_dim:
            pad = expression.new_zeros((batch_size, target_dim - cur_dim))
            expression = torch.cat([expression, pad], dim=-1)
        elif cur_dim > target_dim:
            expression = expression[:, :target_dim]

        if add_view_dim:
            expression = expression.unsqueeze(1)
        return expression

    def _extract_kp_expression_frame(self, frame):
        if self.kp_detector is None:
            return None, None
        keypoints, latent_dict = self.kp_detector(frame)
        expression = None
        expression_encoder = getattr(self.model, 'expression_encoder', None)
        if expression_encoder is not None:
            feature_map = latent_dict.get('feature_map')
            heatmap = latent_dict.get('heatmap')
            if feature_map is not None and heatmap is not None:
                if heatmap.dim() == 5 and heatmap.shape[2] == 1:
                    heatmap = heatmap.squeeze(2)
                expression = expression_encoder(feature_map, heatmap)
        return keypoints, expression

    def _build_rppg_source_conditioning(self, source_frames, num_kp, device, dtype):
        bs = source_frames.shape[0]
        source_kps = torch.zeros((bs, 1, num_kp, 2), device=device, dtype=dtype)
        source_expression = None
        source_frame = source_frames[:, 0]

        if self.rppg_use_motion_conditioning:
            if self.kp_detector is None:
                self._warn_rppg_conditioning_once(
                    'rPPG motion conditioning enabled but kp_detector is missing; falling back to zero keypoints.'
                )
            else:
                try:
                    source_kp_frame, source_expression = self._extract_kp_expression_frame(source_frame)
                    if source_kp_frame is not None:
                        source_kps = source_kp_frame[:, None].to(device=device, dtype=dtype)
                except Exception as exc:
                    self._warn_rppg_conditioning_once(
                        f'rPPG source conditioning extraction failed ({exc}); falling back to zero keypoints.'
                    )

        encoder_expression = None
        if hasattr(self.model.encoder, 'encode_with_expression') and self.model.encoder.encode_with_expression:
            expr_size = int(getattr(self.model.encoder, 'expression_size', 0))
            if expr_size > 0:
                encoder_expression = self._fit_expression_vector(
                    source_expression,
                    target_dim=expr_size,
                    batch_size=bs,
                    device=device,
                    dtype=dtype,
                    add_view_dim=True,
                )
                if encoder_expression is None:
                    encoder_expression = source_frames.new_zeros((bs, 1, expr_size))

        return source_kps, encoder_expression

    def _build_rppg_driving_conditioning(self, driving_frame, n_pix, num_kp, device, dtype):
        bs = driving_frame.shape[0]
        target_kps = torch.zeros((bs, n_pix, num_kp, 2), device=device, dtype=dtype)
        driving_expression = None

        if self.rppg_use_motion_conditioning:
            if self.kp_detector is None:
                self._warn_rppg_conditioning_once(
                    'rPPG motion conditioning enabled but kp_detector is missing; falling back to zero keypoints.'
                )
            else:
                try:
                    driving_kp_frame, driving_expression = self._extract_kp_expression_frame(driving_frame)
                    if driving_kp_frame is not None:
                        target_kps = driving_kp_frame[:, None].repeat(1, n_pix, 1, 1).to(device=device, dtype=dtype)
                except Exception as exc:
                    self._warn_rppg_conditioning_once(
                        f'rPPG driving conditioning extraction failed ({exc}); falling back to zero keypoints.'
                    )

        decoder_expression = None
        decoder_expr_size = int(getattr(self.model.decoder, 'expression_size', 0))
        if decoder_expr_size > 0:
            decoder_expression = self._fit_expression_vector(
                driving_expression,
                target_dim=decoder_expr_size,
                batch_size=bs,
                device=device,
                dtype=dtype,
                add_view_dim=False,
            )
            if decoder_expression is None:
                decoder_expression = driving_frame.new_zeros((bs, decoder_expr_size))

        return target_kps, decoder_expression

    def compute_loss_rppg(self, data, it):
        device = self.device
        clip_frames = data.get('clip_frames')
        if clip_frames is None:
            raise ValueError("rppg_video batch missing 'clip_frames'.")
        clip_frames = clip_frames.to(device)
        if clip_frames.dim() != 5:
            raise ValueError(f"Expected clip_frames [B,T,C,H,W], got {tuple(clip_frames.shape)}")

        driving_frames = data.get('driving_frames')
        if driving_frames is not None:
            driving_frames = driving_frames.to(device)
            if driving_frames.shape != clip_frames.shape:
                raise ValueError(
                    f"driving_frames shape {tuple(driving_frames.shape)} must match clip_frames {tuple(clip_frames.shape)}."
                )
        else:
            driving_frames = clip_frames

        bs, t_steps, c, h, w = clip_frames.shape
        if c != 3:
            raise ValueError(f"rppg_video expects 3-channel RGB frames, got C={c}.")
        if t_steps < 1:
            raise ValueError("rppg_video clip must contain at least one frame.")

        has_paired_target = self._get_has_paired_target(data, batch_size=bs, device=device)
        num_kp = self._get_num_kp()

        source_frames = clip_frames[:, :1]
        source_kps, encoder_expression = self._build_rppg_source_conditioning(
            source_frames=source_frames,
            num_kp=num_kp,
            device=device,
            dtype=clip_frames.dtype,
        )
        source_pos = self._build_full_image_pos(
            batch_size=bs,
            height=h,
            width=w,
            num_views=1,
            device=device,
            dtype=clip_frames.dtype,
        )

        z = self.model.encoder(source_frames, source_kps, source_pos, expression_vector=encoder_expression)

        target_pos = source_pos[:, 0].reshape(bs, h * w, 2)

        rec_sum = clip_frames.new_zeros(())
        rec_weight_sum = clip_frames.new_zeros(())
        temp_sum = clip_frames.new_zeros(())
        temp_count = 0
        temporal_detail_sum = defaultdict(lambda: clip_frames.new_zeros(()))

        prev_generated = None
        prev_input_frame = None
        prev_state = None

        def decode_frame_chunked(cur_input, prev_input, prev_gen, prev_st):
            """Decode one frame in pixel chunks to reduce peak GPU memory."""
            n_pix = target_pos.shape[1]
            step = max(1, int(getattr(self, 'rppg_decode_chunk', n_pix)))
            pred_chunks = []
            first_extras = {}

            for start in range(0, n_pix, step):
                end = min(n_pix, start + step)
                pos_chunk = target_pos[:, start:end]
                kps_chunk = target_kps[:, start:end]

                temporal_inputs = {
                    'cur_frame': cur_input,
                    'prev_frame': prev_input,
                    'prev_generated': prev_gen,
                    'prev_feat': (prev_st or {}).get('prev_feat'),
                }
                try:
                    pred_chunk, extras_chunk = self.model.decoder(
                        z,
                        pos_chunk,
                        kps_chunk,
                        expression_vector=decoder_expression,
                        temporal_inputs=temporal_inputs,
                        return_temporal_state=True,
                    )
                except TypeError:
                    pred_chunk, extras_chunk = self.model.decoder(
                        z,
                        pos_chunk,
                        kps_chunk,
                        expression_vector=decoder_expression,
                    )
                    if not isinstance(extras_chunk, dict):
                        extras_chunk = {}

                pred_chunks.append(pred_chunk)
                if start == 0:
                    first_extras = extras_chunk

            pred_pixels_full = torch.cat(pred_chunks, dim=1)
            if not isinstance(first_extras, dict):
                first_extras = {}
            return pred_pixels_full, first_extras

        for t in range(t_steps):
            cur_input = driving_frames[:, t]
            cur_target = driving_frames[:, t]
            target_kps, decoder_expression = self._build_rppg_driving_conditioning(
                driving_frame=cur_input,
                n_pix=target_pos.shape[1],
                num_kp=num_kp,
                device=device,
                dtype=clip_frames.dtype,
            )
            pred_pixels, extras = decode_frame_chunked(
                cur_input=cur_input,
                prev_input=prev_input_frame,
                prev_gen=prev_generated,
                prev_st=prev_state,
            )

            pred_frame = pred_pixels.view(bs, h, w, 3).permute(0, 3, 1, 2).contiguous()

            rec_per_sample = ((pred_frame - cur_target) ** 2).mean(dim=(1, 2, 3))
            rec_sum = rec_sum + (rec_per_sample * has_paired_target).sum()
            rec_weight_sum = rec_weight_sum + has_paired_target.sum()

            if t > 0 and prev_generated is not None:
                temporal_aux = extras.get('temporal_aux', {})
                flow_t = temporal_aux.get('flow')
                if flow_t is not None:
                    temp_t, detail_t = multiscale_temporal_consistency_loss(
                        cur_frame=pred_frame,
                        prev_frame=prev_generated,
                        flow=flow_t,
                        scales=self.temporal_scales,
                        scale_weights=self.temporal_scale_weights,
                    )
                    temp_sum = temp_sum + temp_t
                    temp_count += 1
                    for key, value in detail_t.items():
                        temporal_detail_sum[key] = temporal_detail_sum[key] + value

            prev_state = extras.get('temporal_state', None)
            prev_generated = pred_frame.detach()
            prev_input_frame = cur_input

        rec_denom = torch.clamp(rec_weight_sum, min=1.0)
        rec_loss = rec_sum / rec_denom

        if temp_count > 0:
            temp_loss = temp_sum / float(temp_count)
        else:
            temp_loss = clip_frames.new_zeros(())

        loss = self.lambda_rec * rec_loss + self.lambda_temp * temp_loss

        loss_terms = {
            'mse': rec_loss.detach(),
            'rec': rec_loss.detach(),
            'temp': temp_loss.detach(),
            'loss_total': loss.detach(),
            'paired_weight_sum': rec_weight_sum.detach(),
            'temp_steps': torch.tensor(float(temp_count), device=device),
        }
        if temp_count > 0:
            for key, value in temporal_detail_sum.items():
                loss_terms[f'temp_{key}'] = (value / float(temp_count)).detach()

        return loss, loss_terms, {'generated': None, 'real': None, 'target_kp_disc': None}

    def compute_loss(self, data, it):
        if self.is_rppg_video:
            return self.compute_loss_rppg(data, it)

        device = self.device

        input_images_augm = data.get('input_images_augm').to(device)
        input_images_augm2 = data.get('input_images_augm2').to(device)
        input_pos = data.get('input_pos').to(device)
        target_image_augm = data.get('target_image_augm').to(device)
        target_image_augm2 = data.get('target_image_augm2').to(device)
        target_pixels = data.get('target_pixels_augm').to(device)
        target_pos = data.get('target_pos').to(device)
        remaining_pos = data.get('remaining_pos').to(device)
        
        #Randomly select one augemented version for expression extraction
        if torch.rand(1) < 0.5:
            target_image_rand = target_image_augm2
        else:
            target_image_rand = target_image_augm
        #Also select the cropped driving frame (see Paper Sec. 3.2 in ¶ Cropping).
        target_image_augm3_crop = data.get('target_image_augm3_crop').to(device)
        input_kps, target_kps, expression_vector, expression_vector_src_augm2, expression_vector_augm3_crop, expression_vector_src_augm = self.extract_keypoints_and_expression(input_images_augm2, target_image_rand, input_images_augm, target_image_augm3_crop)

        del input_images_augm2
        
        if self.simulate_out_of_frame_motion:
            #Estimate the keypoints coordinates of the uncropped images
            input_kps, target_kps, _, _, _, _ = self.extract_keypoints_and_expression(data.get('input_images_augm2_uncropped').to(device),data.get('target_image_augm2_uncropped').to(device))
            #Transform the keypoints into the coordinate system of the cropped images
            kp_scale = data.get('kp_scale').to(device)
            kp_shift = data.get('kp_shift').to(device)
            input_kps = (input_kps+kp_shift[:,None,None])*kp_scale[:,None,None]
            target_kps = (target_kps+kp_shift[:,None])*kp_scale[:,None]

            
        del target_image_augm
        del target_image_augm2
        del target_image_augm3_crop

        loss_terms = dict()
        
        #Expression vector regularization
        d = expression_vector.shape[-1]
        expression_vector_gathered_1 = torch.cat([expression_vector_src_augm,expression_vector.unsqueeze(1)], dim=1).view(-1,d) 
        expression_vector_gathered_2 = torch.cat([expression_vector_src_augm2,expression_vector_augm3_crop.unsqueeze(1)], dim=1).view(-1,d) 

        if self.statistical_regularization:
            #1. Variance along feature dimension
            weight = expression_vector_gathered_1.shape[0]
            S_1 = torch.sqrt(torch.var(expression_vector_gathered_1,dim=-1)+0.0001)
            S_2 = torch.sqrt(torch.var(expression_vector_gathered_2,dim=-1)+0.0001)
            v_1 = (1./weight)*torch.nn.functional.relu(1.-S_1).sum()
            v_2 = (1./weight)*torch.nn.functional.relu(1.-S_2).sum()

            #2.Covariance and variance (diagonal) 
            cov_sq_1 = torch.cov(expression_vector_gathered_1.T)**2
            cov_sq_2 = torch.cov(expression_vector_gathered_2.T)**2
            c_1 = (1./d)*cov_sq_1.sum()
            c_2 = (1./d)*cov_sq_2.sum()
        else:
            c_1 = 0
            c_2 = 0
            v_1 = 0
            v_2 = 0 
            
        #3.Invariance criterion
        s = 0.5*(((expression_vector_src_augm-expression_vector_src_augm2)**2).mean() + ((expression_vector-expression_vector_augm3_crop)**2).mean())
        reg_loss = self.lambda_invar*s + self.lambda_var*(v_1+v_2) + self.lambda_cov*(c_1+c_2)
        loss_terms['reg_loss'] = reg_loss

        
        #With a probability of 75% we select the expression vector of target_image_augm3_crop
        selected = torch.rand(expression_vector.shape[0]) 
        selected = (selected < 0.25) 
        expression_vector_selected = expression_vector.clone()
        expression_vector_selected[selected.bool()] = expression_vector[selected.bool()]
        expression_vector_selected[(1-selected.type('torch.LongTensor')).bool()] = expression_vector_augm3_crop[(1-selected.type('torch.LongTensor')).bool()]
        
        #Encode input_images_augm along with the expression vector of the same source image with the different color augmentation (expression_vector_src_augm2)
        z = self.model.encoder(input_images_augm, input_kps, input_pos, expression_vector=expression_vector_src_augm2)

        bs, nsrc, c, h, w = input_images_augm.shape
        if data.get('remaining_idxs').shape[1] > 0:
            with torch.no_grad():
                pred_pixels_remaining, extras_remaining = self.model.decoder(z.detach(), remaining_pos, target_kps[:,None].repeat(1,remaining_pos.shape[1],1,1), expression_vector=expression_vector_selected)
                del remaining_pos
            pred_pixels_, extras = self.model.decoder(z, target_pos, target_kps[:,None].repeat(1,target_pos.shape[1],1,1), expression_vector=expression_vector_selected)

            all_idxs = torch.cat([data.get('sampled_idxs').to(device),data.get('remaining_idxs').to(device)], dim = -1)
            all_preds = torch.cat([pred_pixels_,pred_pixels_remaining], dim = -2)
            pred_pixels = torch.zeros_like(all_preds, device=device)
        
            for i in range(bs):
                pred_pixels[i][all_idxs[i]] = all_preds[i]
        else:
            pred_pixels, extras = self.model.decoder(z, target_pos, target_kps[:,None].repeat(1,target_pos.shape[1],1,1), expression_vector=expression_vector_selected)
        
        #Loss functions
        loss = 0.
        loss = loss + self.lambda_mse*((pred_pixels - target_pixels)**2).mean((1, 2))
        loss_terms['mse'] = loss.detach().clone()
        
        generated=None
        real=None
        target_kp_disc=None
        
        if it >= self.phase2_start:
            pred_pixels = pred_pixels.view(bs,h,w,c).permute(0,3,1,2)
            target_pixels = target_pixels.view(bs,h,w,c).permute(0,3,1,2)

            if self.vgg19 is not None and sum(self.perceptual_loss_weight) > 0.:
                perc_loss = 0
                x_vgg = self.vgg19(pred_pixels)
                y_vgg = self.vgg19(target_pixels)

                for i, weight in enumerate(self.perceptual_loss_weight):
                    value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                    perc_loss += weight * value

                loss_terms['perceptual'] = perc_loss
                loss+=perc_loss
                
        if it >= self.phase3_start and self.use_disc:
            generated = self.pyramid(pred_pixels)
            real = self.pyramid(target_pixels)
            target_kp_disc = target_kps.detach()
            
            if it >= self.phase3_start + self.disc_warmup_iters and (self.generator_gan_weight != 0 or sum(self.generator_gan_feature_matching_weight) != 0):
                discriminator_maps_generated = self.discriminator(generated, kp=target_kp_disc)
                discriminator_maps_real = self.discriminator(real, kp=target_kp_disc)
                if self.generator_gan_weight != 0:
                    value_total = 0
                    for scale in self.disc_scales:
                        key = 'prediction_map_%s' % scale
                        value = ((1. - discriminator_maps_generated[key]) ** 2).mean()
                        value_total += self.generator_gan_weight * value
                        loss+=value_total
                        loss_terms['gen_gan'] = value_total.detach().clone()

                if sum(self.generator_gan_feature_matching_weight) != 0:
                    value_total = 0
                    for scale in self.disc_scales:
                        key = 'feature_maps_%s' % scale
                        for i, (a, b) in enumerate(zip(discriminator_maps_real[key], discriminator_maps_generated[key])):
                            if self.generator_gan_feature_matching_weight[i] == 0:
                                continue
                            value = torch.abs(a - b).mean()
                            value_total += self.generator_gan_feature_matching_weight[i] * value
                        loss+=value_total
                        loss_terms['feature_matching'] = value_total.detach().clone()            

        return loss, loss_terms, {'generated': generated, 'real': real, 'target_kp_disc': target_kp_disc }
    
    def eval_step(self, data):
        self.model.eval()
        with torch.no_grad():
            loss, loss_terms, disc_items = self.compute_loss(data, self.phase2_start)

        mse = loss_terms['mse']
        psnr = mse2psnr(mse)
        return {'psnr': psnr, 'mse': mse, **loss_terms}


    def render_face(self, z, target_kps, target_pos, expression_vector=None):
        batch_size, height, width = target_pos.shape[:3]
        target_pos = target_pos.flatten(1, 2)
        target_kps = target_kps.unsqueeze(1).repeat(1, target_pos.shape[1], 1,1)

        max_num_rays = self.config['data']['num_pixels'] * \
                self.config['training']['batch_size'] // (target_pos.shape[0] * get_world_size())
        num_rays = target_pos.shape[1]
        img = torch.zeros((target_pos.shape[0],target_pos.shape[1],3))
        all_extras = []
        for i in range(0, num_rays, max_num_rays):
            img[:, i:i+max_num_rays], extras = self.model.decoder(
                z, target_pos[:, i:i+max_num_rays], target_kps[:, i:i+max_num_rays], expression_vector=expression_vector,
            )
         
        img = img.view(img.shape[0], height, width, 3)
        return img


    def visualize_face(self, data, mode='val'):
        if self.is_rppg_video:
            # Step-2 keeps training/eval plumbing first; image visualization for rppg clips is added later.
            return
        self.model.eval()

        with torch.no_grad():
            device = self.device
            input_images_augm = data.get('input_images_augm').to(device)
            input_pos = data.get('input_pos').to(device)
            target_pos = input_pos[:,0].clone().to(device)
            target_image_augm = data.get('target_image_augm').to(device)
            
            input_kps, target_kps, expression_vector_augm, expression_vector_src_augm,_,_ = self.extract_keypoints_and_expression(input_images_augm, target_image_augm)
            
            input_images_np = np.transpose(input_images_augm.cpu().numpy(), (0, 1, 3, 4, 2))

            z = self.model.encoder(input_images_augm, input_kps, input_pos, expression_vector=expression_vector_src_augm)

            batch_size, num_input_images, height, width, _ = input_pos.shape

            columns = []
            for i in range(num_input_images):
                header = 'input' if num_input_images == 1 else f'input {i+1}'
                columns.append((header, input_images_np[:, i], 'image'))
                
            img = self.render_face(z, target_kps, target_pos, expression_vector=expression_vector_augm)
            name = 'driving'
            columns.append((f'render {name}', img.cpu().numpy(), 'image'))
            t_im = target_image_augm.cpu().numpy().transpose(0,2,3,1)
            columns.append((f'GT {name}', t_im, 'image'))

            output_img_path = os.path.join(self.out_dir, f'renders-{mode}')
            vis.draw_visualization_grid(columns, output_img_path)
