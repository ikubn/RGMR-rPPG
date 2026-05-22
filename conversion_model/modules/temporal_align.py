import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.util import AntiAliasInterpolation2d


def _is_frame_tensor(x):
    return isinstance(x, torch.Tensor) and x.dim() == 4 and x.shape[1] > 0


def _to_three_channels(x):
    if x is None:
        return None
    if not _is_frame_tensor(x):
        return None
    if x.shape[1] == 3:
        return x
    if x.shape[1] == 1:
        return x.repeat(1, 3, 1, 1)
    if x.shape[1] > 3:
        return x[:, :3]
    pad_channels = 3 - x.shape[1]
    pad = torch.zeros(
        (x.shape[0], pad_channels, x.shape[2], x.shape[3]),
        device=x.device,
        dtype=x.dtype,
    )
    return torch.cat([x, pad], dim=1)


def _resize_frame(frame, out_hw):
    if frame is None:
        return None
    if frame.dim() != 4:
        raise ValueError(f"Expected frame with shape [B,C,H,W], got {tuple(frame.shape)}")
    if frame.shape[-2:] == out_hw:
        return frame
    return F.interpolate(frame, size=out_hw, mode="bilinear", align_corners=False)


def _meshgrid_xy(height, width, device, dtype):
    ys = torch.arange(0, height, device=device, dtype=dtype)
    xs = torch.arange(0, width, device=device, dtype=dtype)
    try:
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    except TypeError:
        grid_y, grid_x = torch.meshgrid(ys, xs)
    return grid_x, grid_y


def warp_with_flow(x, flow, padding_mode="border"):
    """Warp tensor x with optical flow.

    Args:
        x: [B,C,H,W]
        flow: [B,2,H,W] in pixel displacement (dx, dy)
              or [B,H,W,2]
    """
    if x is None:
        return None
    if x.dim() != 4:
        raise ValueError(f"Expected x with shape [B,C,H,W], got {tuple(x.shape)}")
    if flow is None:
        return x

    if flow.dim() == 4 and flow.shape[1] == 2:
        flow_hw2 = flow.permute(0, 2, 3, 1)
    elif flow.dim() == 4 and flow.shape[-1] == 2:
        flow_hw2 = flow
    else:
        raise ValueError(f"Expected flow with shape [B,2,H,W] or [B,H,W,2], got {tuple(flow.shape)}")

    b, _, h, w = x.shape
    if flow_hw2.shape[0] != b or flow_hw2.shape[1:3] != (h, w):
        raise ValueError(f"Flow shape {tuple(flow_hw2.shape)} is incompatible with x shape {tuple(x.shape)}")

    grid_x, grid_y = _meshgrid_xy(h, w, x.device, x.dtype)
    grid = torch.stack((grid_x, grid_y), dim=2).unsqueeze(0).expand(b, -1, -1, -1)
    vgrid = grid + flow_hw2
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    return F.grid_sample(x, vgrid_scaled, mode="bilinear", padding_mode=padding_mode, align_corners=True)


def resize_flow(flow, out_hw):
    if flow is None:
        return None
    if flow.dim() == 4 and flow.shape[1] == 2:
        flow_chw = flow
    elif flow.dim() == 4 and flow.shape[-1] == 2:
        flow_chw = flow.permute(0, 3, 1, 2)
    else:
        raise ValueError(f"Expected flow with shape [B,2,H,W] or [B,H,W,2], got {tuple(flow.shape)}")

    in_h, in_w = flow_chw.shape[-2:]
    out_h, out_w = out_hw
    if (in_h, in_w) == (out_h, out_w):
        return flow_chw

    flow_resized = F.interpolate(flow_chw, size=out_hw, mode="bilinear", align_corners=False)
    flow_resized[:, 0] *= float(out_w) / float(max(in_w, 1))
    flow_resized[:, 1] *= float(out_h) / float(max(in_h, 1))
    return flow_resized


def estimate_lite_flow(cur_frame, ref_frame, out_hw, max_disp=1.5):
    cur = _resize_frame(cur_frame, out_hw)
    ref = _resize_frame(ref_frame, out_hw)
    if cur is None or ref is None:
        return None

    cur_gray = cur.mean(dim=1, keepdim=True)
    ref_gray = ref.mean(dim=1, keepdim=True)
    diff = cur_gray - ref_gray

    grad_x = torch.zeros_like(diff)
    grad_y = torch.zeros_like(diff)
    grad_x[:, :, :, 1:] = diff[:, :, :, 1:] - diff[:, :, :, :-1]
    grad_y[:, :, 1:, :] = diff[:, :, 1:, :] - diff[:, :, :-1, :]

    flow_x = torch.tanh(grad_x) * max_disp
    flow_y = torch.tanh(grad_y) * max_disp
    return torch.cat([flow_x, flow_y], dim=1)


def _unwrap_state_dict(state):
    if isinstance(state, dict) and "params" in state and isinstance(state["params"], dict):
        params = state["params"]
    elif isinstance(state, dict):
        params = state
    else:
        raise ValueError("Unsupported checkpoint format for SPyNet.")

    cleaned = {}
    for key, value in params.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        cleaned[key] = value
    return cleaned


def infer_spynet_num_levels(load_path, default_levels=6):
    try:
        state = torch.load(load_path, map_location="cpu")
        params = _unwrap_state_dict(state)
    except Exception:
        return int(default_levels)

    pattern = re.compile(r"^basic_module\.(\d+)\.")
    levels = []
    for key in params.keys():
        match = pattern.match(key)
        if match:
            levels.append(int(match.group(1)))
    if len(levels) == 0:
        return int(default_levels)
    return max(levels) + 1


class _SpyNetBasicModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(8, 32, 7, 1, 3),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 64, 7, 1, 3),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 32, 7, 1, 3),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 16, 7, 1, 3),
            nn.ReLU(inplace=False),
            nn.Conv2d(16, 2, 7, 1, 3),
        )

    def forward(self, x):
        return self.net(x)


class SpyNet(nn.Module):
    """SPyNet flow network adapted from Arbitrary_Resolution_rPPG/model/TFA.py."""

    def __init__(self, load_path=None, num_levels=6, strict_load=False):
        super().__init__()
        self.basic_module = nn.ModuleList([_SpyNetBasicModule() for _ in range(num_levels)])
        if load_path:
            state = torch.load(load_path, map_location=lambda storage, loc: storage)
            params = _unwrap_state_dict(state)
            load_info = self.load_state_dict(params, strict=strict_load)
            if not strict_load:
                # If core SPyNet module keys mismatch heavily, fail fast with a clear message.
                missing_core = [k for k in load_info.missing_keys if k.startswith("basic_module.")]
                unexpected_core = [k for k in load_info.unexpected_keys if k.startswith("basic_module.")]
                if len(missing_core) > 0 and len(unexpected_core) > 0:
                    raise RuntimeError(
                        "SPyNet checkpoint shape mismatch. "
                        f"missing_core={len(missing_core)}, unexpected_core={len(unexpected_core)}"
                    )

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def preprocess(self, tensor_input):
        return (tensor_input - self.mean) / self.std

    def process(self, ref, supp):
        ref_pyramid = [self.preprocess(ref)]
        supp_pyramid = [self.preprocess(supp)]

        for _ in range(len(self.basic_module) - 1):
            ref_pyramid.insert(
                0,
                F.avg_pool2d(ref_pyramid[0], kernel_size=2, stride=2, count_include_pad=False),
            )
            supp_pyramid.insert(
                0,
                F.avg_pool2d(supp_pyramid[0], kernel_size=2, stride=2, count_include_pad=False),
            )

        flow = ref_pyramid[0].new_zeros(
            (
                ref_pyramid[0].shape[0],
                2,
                int(math.floor(ref_pyramid[0].shape[2] / 2.0)),
                int(math.floor(ref_pyramid[0].shape[3] / 2.0)),
            )
        )

        for level in range(len(ref_pyramid)):
            upsampled_flow = F.interpolate(flow, scale_factor=2, mode="bilinear", align_corners=True) * 2.0
            if upsampled_flow.shape[2] != ref_pyramid[level].shape[2]:
                upsampled_flow = F.pad(upsampled_flow, pad=[0, 0, 0, 1], mode="replicate")
            if upsampled_flow.shape[3] != ref_pyramid[level].shape[3]:
                upsampled_flow = F.pad(upsampled_flow, pad=[0, 1, 0, 0], mode="replicate")

            warped_supp = warp_with_flow(supp_pyramid[level], upsampled_flow, padding_mode="border")
            flow = self.basic_module[level](torch.cat([ref_pyramid[level], warped_supp, upsampled_flow], dim=1))
            flow = flow + upsampled_flow
        return flow

    def forward(self, ref, supp):
        assert ref.shape == supp.shape, "SPyNet input shape mismatch"
        h, w = ref.shape[2], ref.shape[3]
        flow = F.interpolate(self.process(ref, supp), size=(h, w), mode="bilinear", align_corners=False)
        scale = 2 ** len(self.basic_module)
        flow[:, 0] *= float(w) / float(scale)
        flow[:, 1] *= float(h) / float(scale)
        return flow


def confidence_from_photometric(cur_frame, warped_ref_frame, beta=10.0):
    if cur_frame is None or warped_ref_frame is None:
        return None
    cur = _to_three_channels(cur_frame)
    ref = _to_three_channels(warped_ref_frame)
    if cur is None or ref is None:
        return None
    cur = _resize_frame(cur, ref.shape[-2:])
    error = (cur - ref).abs().mean(dim=1, keepdim=True)
    confidence = torch.exp(-beta * error)
    return confidence.clamp_(0.0, 1.0)


def multiscale_temporal_consistency_loss(
    cur_frame,
    prev_frame,
    flow,
    scales=(1.0, 0.5, 0.25),
    scale_weights=None,
):
    """Anti-aliased multi-scale temporal consistency loss.

    Returns:
        total_loss (Tensor), loss_per_scale (dict)
    """
    if cur_frame is None or prev_frame is None:
        zero = torch.tensor(0.0, device=flow.device if flow is not None else "cpu")
        return zero, {}

    flow = resize_flow(flow, prev_frame.shape[-2:])
    warped_prev = warp_with_flow(prev_frame, flow)
    if scale_weights is None:
        scale_weights = [1.0 for _ in scales]
    if len(scale_weights) != len(scales):
        raise ValueError("scale_weights length must match scales length.")

    total = 0.0
    detail = {}
    for scale, weight in zip(scales, scale_weights):
        if float(scale) == 1.0:
            cur_s = cur_frame
            prev_s = warped_prev
        else:
            down_cur = AntiAliasInterpolation2d(cur_frame.shape[1], scale).to(cur_frame.device)
            down_prev = AntiAliasInterpolation2d(prev_frame.shape[1], scale).to(prev_frame.device)
            cur_s = down_cur(cur_frame)
            prev_s = down_prev(warped_prev)

        loss_s = (cur_s - prev_s).abs().mean()
        detail[f"scale_{scale}"] = loss_s
        total = total + float(weight) * loss_s
    return total, detail


class TemporalAlignFuse(nn.Module):
    """Flow-guided temporal alignment and confidence-gated fusion.

    Supports:
    - flow_backend='lite' | 'spynet' | 'spynet_stub'
    - ref_mode='prev_frame' | 'prev_feature' | 'both'
    - online single-step fusion and offline bidirectional sequence fusion
    """

    def __init__(
        self,
        feat_dim,
        ref_mode="both",
        flow_backend="lite",
        max_disp=1.5,
        spynet_path=None,
        spynet_num_levels=6,
        spynet_strict_load=False,
        spynet_auto_levels=True,
        freeze_flow=True,
        confidence_beta=10.0,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.ref_mode = ref_mode
        self.flow_backend = flow_backend
        self.max_disp = max_disp
        self.freeze_flow = freeze_flow
        self.confidence_beta = confidence_beta
        self._eps = 1e-6
        self.spynet_num_levels = int(spynet_num_levels)
        self.spynet_strict_load = bool(spynet_strict_load)
        self.spynet_auto_levels = bool(spynet_auto_levels)

        self.spynet = None
        if self.flow_backend == "spynet":
            if spynet_path is None:
                raise ValueError("flow_backend='spynet' requires spynet_path.")
            num_levels = self.spynet_num_levels
            if self.spynet_auto_levels:
                num_levels = infer_spynet_num_levels(spynet_path, default_levels=self.spynet_num_levels)
            self.spynet = SpyNet(
                load_path=spynet_path,
                num_levels=num_levels,
                strict_load=self.spynet_strict_load,
            )
            if self.freeze_flow:
                for p in self.spynet.parameters():
                    p.requires_grad = False
                self.spynet.eval()

        self.gate_conv = nn.Sequential(
            nn.Conv2d(feat_dim * 2 + 3, feat_dim, kernel_size=1),
            nn.Sigmoid(),
        )
        map_fuse_in_dim = feat_dim * 3 + 6
        self.map_fuser = nn.Conv2d(map_fuse_in_dim, feat_dim, kernel_size=1)
        self.token_fuser = nn.Sequential(
            nn.Linear(feat_dim * 2 + 9, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, feat_dim),
        )

        self.token_bi_fuser = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, feat_dim),
        )
        self.map_bi_fuser = nn.Conv2d(feat_dim * 2, feat_dim, kernel_size=1)

    def _valid_ref(self, x):
        x3 = _to_three_channels(x)
        return x3 if x3 is not None else None

    def _pick_refs(self, prev_frame, prev_generated):
        prev_frame_ref = self._valid_ref(prev_frame)
        prev_generated_ref = self._valid_ref(prev_generated)

        if self.ref_mode == "prev_frame":
            return [("prev_frame", prev_frame_ref)] if prev_frame_ref is not None else []

        if self.ref_mode == "prev_feature":
            if prev_generated_ref is not None:
                return [("prev_generated", prev_generated_ref)]
            return [("prev_frame", prev_frame_ref)] if prev_frame_ref is not None else []

        if self.ref_mode != "both":
            raise ValueError(f"Unsupported ref_mode={self.ref_mode}")

        refs = []
        if prev_frame_ref is not None:
            refs.append(("prev_frame", prev_frame_ref))
        if prev_generated_ref is not None and prev_generated_ref.data_ptr() != (
            prev_frame_ref.data_ptr() if prev_frame_ref is not None else -1
        ):
            refs.append(("prev_generated", prev_generated_ref))
        return refs

    def _compute_single_flow(self, cur_frame, ref_frame, out_hw):
        if cur_frame is None or ref_frame is None:
            return None
        cur = _to_three_channels(_resize_frame(cur_frame, out_hw))
        ref = _to_three_channels(_resize_frame(ref_frame, out_hw))
        if cur is None or ref is None:
            return None

        if self.flow_backend == "lite":
            return estimate_lite_flow(cur, ref, out_hw=out_hw, max_disp=self.max_disp)

        if self.flow_backend == "spynet":
            if self.spynet is None:
                raise RuntimeError("SPyNet backend requested but self.spynet is None.")
            with torch.no_grad() if self.freeze_flow else torch.enable_grad():
                flow = self.spynet(cur, ref)
            return flow

        if self.flow_backend == "spynet_stub":
            # Deterministic stub backend for plumbing/smoke only.
            return torch.zeros((cur.shape[0], 2, out_hw[0], out_hw[1]), device=cur.device, dtype=cur.dtype)

        raise ValueError(f"Unsupported flow_backend={self.flow_backend}")

    def _blend_refs(self, cur_frame, refs, out_hw):
        if cur_frame is None or len(refs) == 0:
            return None, None, None, {}
        if out_hw is None and _is_frame_tensor(cur_frame):
            out_hw = cur_frame.shape[-2:]
        if out_hw is None:
            return None, None, None, {}

        cur = _to_three_channels(_resize_frame(cur_frame, out_hw))
        flow_list = []
        warped_frame_list = []
        conf_list = []
        ref_names = []

        for ref_name, ref_frame in refs:
            ref = _to_three_channels(_resize_frame(ref_frame, out_hw))
            flow = self._compute_single_flow(cur, ref, out_hw)
            if flow is None:
                continue
            warped_ref = warp_with_flow(ref, flow)
            conf = confidence_from_photometric(cur, warped_ref, beta=self.confidence_beta)
            if conf is None:
                conf = torch.ones(
                    (cur.shape[0], 1, out_hw[0], out_hw[1]),
                    device=cur.device,
                    dtype=cur.dtype,
                )
            flow_list.append(flow)
            warped_frame_list.append(warped_ref)
            conf_list.append(conf)
            ref_names.append(ref_name)

        if len(flow_list) == 0:
            return None, None, None, {}

        if len(flow_list) == 1:
            return flow_list[0], warped_frame_list[0], conf_list[0], {
                "ref_names": ref_names,
                "ref_weights": torch.ones_like(conf_list[0]),
            }

        conf_stack = torch.stack(conf_list, dim=1)
        weight = conf_stack / (conf_stack.sum(dim=1, keepdim=True) + self._eps)
        flow_stack = torch.stack(flow_list, dim=1)
        frame_stack = torch.stack(warped_frame_list, dim=1)

        flow_blended = (weight * flow_stack).sum(dim=1)
        frame_blended = (weight * frame_stack).sum(dim=1)
        conf_blended = (weight * conf_stack).sum(dim=1)

        return flow_blended, frame_blended, conf_blended, {
            "ref_names": ref_names,
            "ref_weights": weight,
        }

    def _prepare_prev_feat_map(self, prev_feat, target_shape):
        b, c, h, w = target_shape
        if prev_feat is None or prev_feat.dim() != 4 or prev_feat.shape[0] != b:
            return torch.zeros((b, c, h, w), device=self.map_fuser.weight.device, dtype=self.map_fuser.weight.dtype)
        prev_feat_map = _resize_frame(prev_feat, (h, w))
        if prev_feat_map.shape[1] != c:
            return torch.zeros_like(prev_feat_map[:, :1]).repeat(1, c, 1, 1)
        return prev_feat_map

    def _forward_maps(self, cur_feat, prev_feat, cur_frame, prev_frame, prev_generated):
        b, c, h, w = cur_feat.shape
        prev_feat_map = self._prepare_prev_feat_map(prev_feat, target_shape=(b, c, h, w))
        prev_feat_map = prev_feat_map.to(device=cur_feat.device, dtype=cur_feat.dtype)

        refs = self._pick_refs(prev_frame=prev_frame, prev_generated=prev_generated)
        if len(refs) == 0:
            aux = {
                "flow": None,
                "warped_prev_frame": None,
                "warped_prev_feat": prev_feat_map,
                "confidence": None,
                "gate": None,
                "ref_meta": {"ref_names": [], "ref_weights": None},
                "bypass": True,
                "bypass_reason": "no_reference_frame",
            }
            return cur_feat, aux

        flow, warped_prev_frame, confidence, ref_meta = self._blend_refs(cur_frame, refs, out_hw=(h, w))

        if flow is None or warped_prev_frame is None or confidence is None:
            aux = {
                "flow": flow,
                "warped_prev_frame": warped_prev_frame,
                "warped_prev_feat": prev_feat_map,
                "confidence": confidence,
                "gate": None,
                "ref_meta": ref_meta,
                "bypass": True,
                "bypass_reason": "invalid_flow_or_confidence",
            }
            return cur_feat, aux

        flow = flow.to(device=cur_feat.device, dtype=cur_feat.dtype)
        warped_prev_frame = warped_prev_frame.to(device=cur_feat.device, dtype=cur_feat.dtype)
        confidence = confidence.to(device=cur_feat.device, dtype=cur_feat.dtype)

        warped_prev_feat = warp_with_flow(prev_feat_map, flow)
        gate_input = torch.cat([cur_feat, warped_prev_feat, flow, confidence], dim=1)
        gate = self.gate_conv(gate_input)
        fused_base = gate * warped_prev_feat + (1.0 - gate) * cur_feat

        fused_input = torch.cat(
            [fused_base, cur_feat, warped_prev_feat, warped_prev_frame, flow, confidence],
            dim=1,
        )
        fused_feat = self.map_fuser(fused_input)

        aux = {
            "flow": flow,
            "warped_prev_frame": warped_prev_frame,
            "warped_prev_feat": warped_prev_feat,
            "confidence": confidence,
            "gate": gate,
            "ref_meta": ref_meta,
            "bypass": False,
        }
        return fused_feat, aux

    def _forward_tokens(self, cur_feat, prev_feat, cur_frame, prev_frame, prev_generated):
        b, n, c = cur_feat.shape
        side = int(math.sqrt(n))

        if side * side == n:
            cur_map = cur_feat.transpose(1, 2).contiguous().view(b, c, side, side)
            if prev_feat is not None and prev_feat.shape == cur_feat.shape:
                prev_map = prev_feat.transpose(1, 2).contiguous().view(b, c, side, side)
            else:
                prev_map = None
            fused_map, aux = self._forward_maps(
                cur_feat=cur_map,
                prev_feat=prev_map,
                cur_frame=cur_frame,
                prev_frame=prev_frame,
                prev_generated=prev_generated,
            )
            fused_tokens = fused_map.flatten(2).transpose(1, 2).contiguous()
            aux["token_mode"] = "spatial_align"
            return fused_tokens, aux

        if prev_feat is not None and prev_feat.shape == cur_feat.shape:
            prev_feat_tokens = prev_feat
        else:
            prev_feat_tokens = torch.zeros_like(cur_feat)

        out_hw = None
        if _is_frame_tensor(cur_frame):
            out_hw = cur_frame.shape[-2:]
        refs = self._pick_refs(prev_frame=prev_frame, prev_generated=prev_generated)
        if len(refs) == 0:
            aux = {
                "flow": None,
                "warped_prev_frame": None,
                "warped_prev_feat": prev_feat_tokens,
                "confidence": None,
                "gate": None,
                "ref_meta": {"ref_names": [], "ref_weights": None},
                "token_mode": "global_fallback",
                "bypass": True,
                "bypass_reason": "no_reference_frame",
            }
            return cur_feat, aux

        flow, warped_prev_frame, confidence, ref_meta = self._blend_refs(
            cur_frame=cur_frame,
            refs=refs,
            out_hw=out_hw,
        )
        if flow is None or warped_prev_frame is None or confidence is None:
            aux = {
                "flow": flow,
                "warped_prev_frame": warped_prev_frame,
                "warped_prev_feat": prev_feat_tokens,
                "confidence": confidence,
                "gate": None,
                "ref_meta": ref_meta,
                "token_mode": "global_fallback",
                "bypass": True,
                "bypass_reason": "invalid_flow_or_confidence",
            }
            return cur_feat, aux

        flow_mean = torch.zeros((b, 2), device=cur_feat.device, dtype=cur_feat.dtype)
        flow_std = torch.zeros((b, 2), device=cur_feat.device, dtype=cur_feat.dtype)
        conf_mean = torch.zeros((b, 1), device=cur_feat.device, dtype=cur_feat.dtype)
        conf_std = torch.zeros((b, 1), device=cur_feat.device, dtype=cur_feat.dtype)
        frame_mean = torch.zeros((b, 3), device=cur_feat.device, dtype=cur_feat.dtype)

        if flow is not None:
            flow = flow.to(device=cur_feat.device, dtype=cur_feat.dtype)
            flow_mean = flow.mean(dim=(2, 3))
            flow_std = flow.std(dim=(2, 3), unbiased=False)
        if confidence is not None:
            confidence = confidence.to(device=cur_feat.device, dtype=cur_feat.dtype)
            conf_mean = confidence.mean(dim=(2, 3))
            conf_std = confidence.std(dim=(2, 3), unbiased=False)
        if warped_prev_frame is not None:
            warped_prev_frame = warped_prev_frame.to(device=cur_feat.device, dtype=cur_feat.dtype)
            frame_mean = warped_prev_frame.mean(dim=(2, 3))

        gate_scalar = conf_mean.clamp(0.0, 1.0)
        fused_base = gate_scalar[:, None, :] * prev_feat_tokens + (1.0 - gate_scalar[:, None, :]) * cur_feat
        ctx = torch.cat([frame_mean, flow_mean, flow_std, conf_mean, conf_std], dim=-1)
        fused_input = torch.cat([fused_base, cur_feat, ctx[:, None, :].expand(-1, n, -1)], dim=-1)
        fused_feat = self.token_fuser(fused_input)

        aux = {
            "flow": flow,
            "warped_prev_frame": warped_prev_frame,
            "warped_prev_feat": prev_feat_tokens,
            "confidence": confidence,
            "gate": gate_scalar[:, None, :],
            "ref_meta": ref_meta,
            "token_mode": "global_fallback",
            "bypass": False,
        }
        return fused_feat, aux

    def forward(self, cur_feat, prev_feat=None, cur_frame=None, prev_frame=None, prev_generated=None):
        if cur_feat.dim() == 3:
            return self._forward_tokens(
                cur_feat=cur_feat,
                prev_feat=prev_feat,
                cur_frame=cur_frame,
                prev_frame=prev_frame,
                prev_generated=prev_generated,
            )

        if cur_feat.dim() == 4:
            return self._forward_maps(
                cur_feat=cur_feat,
                prev_feat=prev_feat,
                cur_frame=cur_frame,
                prev_frame=prev_frame,
                prev_generated=prev_generated,
            )

        raise ValueError(f"Unsupported cur_feat rank {cur_feat.dim()} for TemporalAlignFuse")

    def _slice_time(self, x, idx):
        if x is None:
            return None
        return x[:, idx]

    def _align_one_direction(self, feat_seq, frame_seq=None, generated_seq=None):
        t_steps = feat_seq.shape[1]
        outputs = []
        aux_list = []
        prev_feat = None
        prev_frame = None
        prev_generated = None

        for t in range(t_steps):
            cur_feat = feat_seq[:, t]
            cur_frame = self._slice_time(frame_seq, t)
            if generated_seq is not None and t > 0:
                prev_generated = generated_seq[:, t - 1]

            fused_t, aux_t = self(
                cur_feat=cur_feat,
                prev_feat=prev_feat,
                cur_frame=cur_frame,
                prev_frame=prev_frame,
                prev_generated=prev_generated,
            )
            outputs.append(fused_t)
            aux_list.append(aux_t)
            prev_feat = fused_t.detach()
            prev_frame = cur_frame
        return torch.stack(outputs, dim=1), aux_list

    def align_sequence(self, feat_seq, frame_seq=None, generated_seq=None, mode="online"):
        """Sequence alignment entry.

        Args:
            feat_seq: [B,T,N,C] or [B,T,C,H,W]
            frame_seq: [B,T,3,H,W], optional
            generated_seq: [B,T,3,H,W], optional
            mode: 'online' or 'offline_bidirectional'
        """
        if feat_seq.dim() not in (4, 5):
            raise ValueError(f"Expected feat_seq rank 4/5, got {feat_seq.dim()}")

        if mode == "online":
            fused, aux = self._align_one_direction(
                feat_seq=feat_seq,
                frame_seq=frame_seq,
                generated_seq=generated_seq,
            )
            return fused, {"mode": mode, "forward_aux": aux}

        if mode != "offline_bidirectional":
            raise ValueError(f"Unsupported mode={mode}. Use online or offline_bidirectional.")

        fwd, fwd_aux = self._align_one_direction(feat_seq=feat_seq, frame_seq=frame_seq, generated_seq=generated_seq)

        rev_feat = torch.flip(feat_seq, dims=[1])
        rev_frame = torch.flip(frame_seq, dims=[1]) if frame_seq is not None else None
        rev_gen = torch.flip(generated_seq, dims=[1]) if generated_seq is not None else None
        bwd_rev, bwd_aux_rev = self._align_one_direction(
            feat_seq=rev_feat,
            frame_seq=rev_frame,
            generated_seq=rev_gen,
        )
        bwd = torch.flip(bwd_rev, dims=[1])
        bwd_aux = list(reversed(bwd_aux_rev))

        if feat_seq.dim() == 4:
            fused = self.token_bi_fuser(torch.cat([fwd, bwd], dim=-1))
        else:
            b, t, c, h, w = fwd.shape
            fused = torch.cat([fwd, bwd], dim=2).reshape(b * t, c * 2, h, w)
            fused = self.map_bi_fuser(fused).reshape(b, t, c, h, w)

        return fused, {
            "mode": mode,
            "forward_aux": fwd_aux,
            "backward_aux": bwd_aux,
        }
