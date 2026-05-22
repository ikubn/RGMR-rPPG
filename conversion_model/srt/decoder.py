import numpy as np
import torch
import torch.nn as nn

from srt.layers import  Transformer, FSRTPosEncoder
from modules.temporal_align import TemporalAlignFuse


class FSRTPixelPredictor(nn.Module): 
    def __init__(self, num_att_blocks=2,pix_octaves=16, pix_start_octave=-1, out_dims=3,
                 z_dim=768, input_mlp=True, output_mlp=False, num_kp=10, expression_size=0, kp_octaves=4, kp_start_octave=-1, attn_checkpoint=False):
        super().__init__()

        self.positional_encoder = FSRTPosEncoder(kp_octaves=kp_octaves,kp_start_octave=kp_start_octave,
                                        pix_octaves=pix_octaves,pix_start_octave=pix_start_octave)
        self.expression_size = expression_size
        self.num_kp = num_kp
        self.feat_dim = pix_octaves*4+num_kp*kp_octaves*4+self.expression_size

        if input_mlp:  # Input MLP added with OSRT improvements
            self.input_mlp = nn.Sequential(
                nn.Linear(self.feat_dim, 720),
                nn.ReLU(),
                nn.Linear(720, self.feat_dim))
        else:
            self.input_mlp = None
        

        self.transformer = Transformer(
            self.feat_dim,
            depth=num_att_blocks,
            heads=12,
            dim_head=z_dim // 12,
            mlp_dim=z_dim * 2,
            selfatt=False,
            kv_dim=z_dim,
            checkpoint_blocks=attn_checkpoint,
        )

        if output_mlp:
            self.output_mlp = nn.Sequential(
                nn.Linear(self.feat_dim, 128),
                nn.ReLU(),
                nn.Linear(128, out_dims))
        else:
            self.output_mlp = None

    def forward(self, z, pixels, keypoints, expression_vector=None):
        """
        Args:
            z: set-latent scene repres. [batch_size, num_patches, patch_dim]
            pixels: query pixels [batch_size, num_pixels, 2]
            keypoints: facial query keypoints [batch_size, num_pixels, num_kp, 2]
            expression_vector: latent repres. of the query expression [batch_size, expression_size]
        """
        bs = pixels.shape[0]
        nr = pixels.shape[1]
        nkp = keypoints.shape[-2]
        queries = self.positional_encoder(pixels, keypoints.view(bs,nr,nkp*2))
        
        if expression_vector is not None:
            queries = torch.cat([queries,expression_vector[:,None].repeat(1,queries.shape[1],1)],dim=-1)

        if self.input_mlp is not None:
            queries = self.input_mlp(queries)

        output = self.transformer(queries, z)
        
        if self.output_mlp is not None:
            output = self.output_mlp(output)
            
        return output
    

class ImprovedFSRTDecoder(nn.Module):
    """ Scene Representation Transformer Decoder with the improvements from Appendix A.4 in the OSRT paper."""
    def __init__(
        self,
        num_att_blocks=2,
        pix_octaves=16,
        pix_start_octave=-1,
        num_kp=10,
        kp_octaves=4,
        kp_start_octave=-1,
        expression_size=0,
        temporal_align_cfg=None,
        attn_checkpoint=False,
    ):
        super().__init__()
        self.allocation_transformer = FSRTPixelPredictor(num_att_blocks=num_att_blocks,
                                                   pix_start_octave=pix_start_octave,
                                                   pix_octaves=pix_octaves,
                                                   z_dim=768,
                                                   input_mlp=True,
                                                   output_mlp=False,
                                                   expression_size=expression_size,
                                                   kp_octaves=kp_octaves,
                                                   kp_start_octave = kp_start_octave,
                                                   attn_checkpoint=attn_checkpoint,
                                                )
        self.expression_size = expression_size 
        self.feat_dim = pix_octaves*4+num_kp*kp_octaves*4+self.expression_size
        self.render_mlp = nn.Sequential(
            nn.Linear(self.feat_dim, 1536),
            nn.ReLU(),
            nn.Linear(1536, 1536),
            nn.ReLU(),
            nn.Linear(1536, 1536),
            nn.ReLU(),
            nn.Linear(1536, 1536),
            nn.ReLU(),
            nn.Linear(1536, 3),
        )
        temporal_align_cfg = temporal_align_cfg or {}
        self.use_temporal_align = bool(temporal_align_cfg.get("enabled", False))
        self.temporal_align = None
        if self.use_temporal_align:
            self.temporal_align = TemporalAlignFuse(
                feat_dim=self.feat_dim,
                ref_mode=temporal_align_cfg.get("ref_mode", "both"),
                flow_backend=temporal_align_cfg.get("flow_backend", "lite"),
                max_disp=float(temporal_align_cfg.get("max_disp", 1.5)),
                spynet_path=temporal_align_cfg.get("spynet_path"),
                spynet_num_levels=int(temporal_align_cfg.get("spynet_num_levels", 6)),
                spynet_strict_load=bool(temporal_align_cfg.get("spynet_strict_load", False)),
                spynet_auto_levels=bool(temporal_align_cfg.get("spynet_auto_levels", True)),
                freeze_flow=bool(temporal_align_cfg.get("freeze_flow", True)),
                confidence_beta=float(temporal_align_cfg.get("confidence_beta", 10.0)),
            )

    def forward(
        self,
        z,
        x,
        pixels,
        expression_vector=None,
        temporal_inputs=None,
        return_temporal_state=False,
    ):
        x = self.allocation_transformer(z, x, pixels, expression_vector = expression_vector)
        extras = {}
        temporal_state = {}
        if self.temporal_align is not None:
            temporal_inputs = temporal_inputs or {}
            x, temporal_aux = self.temporal_align(
                cur_feat=x,
                prev_feat=temporal_inputs.get("prev_feat"),
                cur_frame=temporal_inputs.get("cur_frame"),
                prev_frame=temporal_inputs.get("prev_frame"),
                prev_generated=temporal_inputs.get("prev_generated"),
            )
            extras["temporal_aux"] = temporal_aux
            temporal_state["prev_feat"] = x.detach()

        pixels = self.render_mlp(x)
        if self.temporal_align is not None:
            temporal_state["prev_generated"] = pixels.detach()
        if return_temporal_state:
            extras["temporal_state"] = temporal_state
        return pixels, extras

