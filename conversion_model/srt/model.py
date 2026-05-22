from torch import nn

from srt.encoder import ImprovedFSRTEncoder
from srt.decoder import ImprovedFSRTDecoder
from srt.small_decoder import ImprovedFSRTDecoder as SmallImprovedFSRTDecoder

class FSRT(nn.Module):
    def __init__(self, cfg, expression_encoder=None):
        super().__init__()
            
        self.encoder = ImprovedFSRTEncoder(expression_size=cfg['expression_size'],  **cfg['encoder_kwargs'])
        temporal_align_cfg = cfg.get("temporal_align", None)
        
        if cfg['small_decoder']:
            self.decoder = SmallImprovedFSRTDecoder(expression_size=cfg['expression_size'], **cfg['decoder_kwargs'])
            print('Loading small decoder')
            if temporal_align_cfg and temporal_align_cfg.get("enabled", False):
                print("Warning: temporal_align is ignored when small_decoder=True.")
        else:
            self.decoder = ImprovedFSRTDecoder(
                expression_size=cfg['expression_size'],
                temporal_align_cfg=temporal_align_cfg,
                **cfg['decoder_kwargs'],
            )
            
        self.expression_encoder = expression_encoder

    def forward(
        self,
        cur_frame,
        driving_frame=None,
        prev_generated=None,
        prev_state=None,
        mode='m2s',
        **kwargs,
    ):
        """Forward interface for video-to-video temporal rollout.

        Required kwargs:
            encoder_images, encoder_kps, encoder_pos, decoder_pos, decoder_kps
        Optional kwargs:
            encoder_expression_vector, expression_vector
        """
        encoder_images = kwargs.get("encoder_images")
        encoder_kps = kwargs.get("encoder_kps")
        encoder_pos = kwargs.get("encoder_pos")
        decoder_pos = kwargs.get("decoder_pos")
        decoder_kps = kwargs.get("decoder_kps")
        encoder_expression_vector = kwargs.get("encoder_expression_vector")
        expression_vector = kwargs.get("expression_vector")

        required = {
            "encoder_images": encoder_images,
            "encoder_kps": encoder_kps,
            "encoder_pos": encoder_pos,
            "decoder_pos": decoder_pos,
            "decoder_kps": decoder_kps,
        }
        missing = [k for k, v in required.items() if v is None]
        if missing:
            missing_joined = ", ".join(missing)
            raise ValueError(f"Missing required FSRT.forward kwargs: {missing_joined}")

        z = self.encoder(
            encoder_images,
            encoder_kps,
            encoder_pos,
            expression_vector=encoder_expression_vector,
        )

        temporal_inputs = {
            "cur_frame": cur_frame,
            "prev_frame": driving_frame,
            "prev_generated": prev_generated,
            "prev_feat": (prev_state or {}).get("prev_feat"),
        }
        pred_pixels, extras = self.decoder(
            z,
            decoder_pos,
            decoder_kps,
            expression_vector=expression_vector,
            temporal_inputs=temporal_inputs,
            return_temporal_state=True,
        )

        temporal_state = extras.get("temporal_state", {})
        if "prev_generated" not in temporal_state:
            temporal_state["prev_generated"] = pred_pixels.detach()

        return {
            "pred_pixels": pred_pixels,
            "temporal_state": temporal_state,
            "extras": extras,
            "mode": mode,
        }
