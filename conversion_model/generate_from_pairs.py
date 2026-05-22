#!/usr/bin/env python3
import argparse
import csv
import re
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import yaml

from modules.expression_encoder import ExpressionEncoder
from modules.keypoint_detector import KPDetector
from srt.checkpoint import Checkpoint
from srt.model import FSRT

try:
    from srt.data.rppg_video import _load_video
except ModuleNotFoundError:
    def _read_video_any(path):
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix == ".mat":
            import scipy.io as sio

            payload = sio.loadmat(str(path))
            if "video" not in payload:
                raise ValueError(f"'video' key not found in mat file: {path}")
            arr = np.asarray(payload["video"])
            # Common MMPD layout is [T,H,W,3]; fallback for [H,W,3,T].
            if arr.ndim == 4 and arr.shape[-1] != 3 and arr.shape[2] == 3:
                arr = np.transpose(arr, (3, 0, 1, 2))
        else:
            try:
                import cv2

                cap = cv2.VideoCapture(str(path))
                frames = []
                ok, frame = cap.read()
                while ok:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    ok, frame = cap.read()
                cap.release()
                if not frames:
                    raise ValueError(f"No frames decoded via cv2 from {path}")
                arr = np.stack(frames, axis=0)
            except Exception:
                arr = np.asarray(imageio.mimread(str(path), memtest=False))

        if arr.ndim != 4:
            raise ValueError(f"Expected 4D video tensor [T,H,W,C], got shape {arr.shape} from {path}")
        if arr.shape[-1] != 3:
            raise ValueError(f"Expected RGB channels=3, got shape {arr.shape} from {path}")
        arr = arr.astype(np.float32)
        if arr.max() > 1.5:
            arr /= 255.0
        return np.clip(arr, 0.0, 1.0)

    def _resize_video_frames(video, frame_shape):
        h, w = int(frame_shape[0]), int(frame_shape[1])
        if video.shape[1] == h and video.shape[2] == w:
            return video.astype(np.float32)
        try:
            import cv2
        except Exception as e:
            raise RuntimeError(f"OpenCV is required for resizing fallback _load_video: {e}")
        out = np.empty((video.shape[0], h, w, 3), dtype=np.float32)
        for i in range(video.shape[0]):
            frame = video[i]
            resized = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            out[i] = np.clip(resized, 0.0, 1.0)
        return out

    def _load_video(path, frame_shape):
        video = _read_video_any(path)
        return _resize_video_frames(video, frame_shape=frame_shape)


def parse_args():
    parser = argparse.ArgumentParser("Batch FSRT generation from pair CSV")
    parser.add_argument("--pairs-csv", required=True, help="Pair CSV path.")
    parser.add_argument("--config", required=True, help="FSRT config yaml path.")
    parser.add_argument("--checkpoint", required=True, help="FSRT checkpoint path (model.pt/model_best.pt/model_*.pt).")
    parser.add_argument(
        "--kp-checkpoint",
        default="",
        help=(
            "Keypoint detector checkpoint. Defaults to "
            "../checkpoints/conversion/kp_detector.pt relative to this script."
        ),
    )
    parser.add_argument("--out-dir", required=True, help="Output directory for generated videos.")

    parser.add_argument("--dataset", default="auto", choices=["auto", "mmpd", "vipl"], help="Dataset type for relative path resolution.")
    parser.add_argument("--data-root", default="", help="Root directory containing dataset folders, e.g. <project>/Data.")
    parser.add_argument("--source-col", default="", help="Optional source column name override.")
    parser.add_argument("--driving-col", default="driving_path", help="Driving column name.")

    parser.add_argument("--start-index", type=int, default=0, help="Start row index in CSV (inclusive).")
    parser.add_argument("--end-index", type=int, default=-1, help="End row index in CSV (inclusive), -1 means until end.")
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of pairs to process after slicing.")

    parser.add_argument("--source-index", type=int, default=0, help="Source frame index when source is a video (supports negative index).")
    parser.add_argument(
        "--framewise-source",
        action="store_true",
        help="Use source-video frames that move with time instead of a single fixed source frame.",
    )
    parser.add_argument(
        "--framewise-source-mode",
        default="window",
        choices=["window", "duplicate"],
        help="How to build multi-source views when framewise-source is enabled.",
    )
    parser.add_argument(
        "--framewise-source-interval",
        type=int,
        default=1,
        help=(
            "When framewise-source is enabled, keep the same source-view bundle for this many "
            "driving timesteps before refreshing (1=per-frame, 30=about once per second at 30 FPS)."
        ),
    )
    parser.add_argument("--frame-stride", type=int, default=1, help="Driving frame stride.")
    parser.add_argument("--max-frames", type=int, default=0, help="Maximum driving frames after stride; 0 means no cap.")
    parser.add_argument("--fps", type=float, default=30.0, help="Output FPS.")
    parser.add_argument("--decode-chunk", type=int, default=0, help="Decoder pixel chunk size (0 -> from config training.rppg_decode_chunk, else full frame).")
    parser.add_argument("--device", default="cuda", help="Torch device, e.g. cuda or cpu.")

    parser.add_argument("--skip-existing", action="store_true", help="Skip when output video already exists.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output video.")
    parser.add_argument("--dry-run", action="store_true", help="Only resolve paths and print plan; do not run inference.")
    parser.add_argument("--fail-on-error", action="store_true", help="Exit non-zero if any pair fails.")
    parser.add_argument("--log-csv", default="", help="Optional generation log csv path (default: <out-dir>/generation_log.csv).")
    return parser.parse_args()


def sanitize_name(text, max_len=160):
    text = str(text).strip()
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = text.strip("_")
    if not text:
        text = "pair"
    return text[:max_len]


def infer_dataset(row, forced):
    if forced != "auto":
        return forced

    row_ds = str(row.get("dataset", "")).strip().lower()
    if row_ds in ("mmpd", "vipl"):
        return row_ds

    text = " ".join(
        str(row.get(k, ""))
        for k in ("video_path", "source_path", "driving_path", "source", "driving")
    ).lower()
    if ".mat" in text or "subject" in text or "/mmpd/" in text:
        return "mmpd"
    if "/v1/" in text or "/v2/" in text or "vipl" in text or "video.avi" in text:
        return "vipl"
    return "mmpd"


def resolve_video_path(raw_value, csv_dir, dataset, data_root):
    raw = str(raw_value).strip()
    if not raw:
        raise ValueError("Empty path value in pair row.")

    p = Path(raw)
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append((csv_dir / p).resolve())

        if data_root is not None:
            if dataset == "mmpd":
                mmpd_root = data_root / "MMPD"
                if p.suffix:
                    candidates.append((mmpd_root / p).resolve())
                else:
                    candidates.append((mmpd_root / f"{raw}.mat").resolve())
                    candidates.append((mmpd_root / p).resolve())
            elif dataset == "vipl":
                vipl_root = data_root / "VIPL-HR-V1" / "data"
                if p.suffix:
                    candidates.append((vipl_root / p).resolve())
                else:
                    candidates.append((vipl_root / p / "video.avi").resolve())
                    candidates.append((vipl_root / p / "video.mp4").resolve())
                    candidates.append((vipl_root / p).resolve())

    dedup = []
    seen = set()
    for c in candidates:
        key = str(c)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(c)

    for cand in dedup:
        if cand.is_file():
            return cand
        if cand.is_dir():
            for name in ("video.avi", "video.mp4"):
                maybe = cand / name
                if maybe.is_file():
                    return maybe

    msg = "Unable to resolve video path for '{}' (dataset={}). Tried: {}".format(
        raw,
        dataset,
        ", ".join(str(c) for c in dedup[:8]),
    )
    raise FileNotFoundError(msg)


def resolve_frame_shape(cfg):
    frame_shape = cfg.get("data", {}).get("frame_shape", None)
    if frame_shape is None:
        return (256, 256, 3)
    if len(frame_shape) == 2:
        return (int(frame_shape[0]), int(frame_shape[1]), 3)
    return tuple(int(v) for v in frame_shape)


def build_full_image_pos(batch_size, height, width, num_views, device, dtype):
    ys = torch.arange(height, device=device, dtype=dtype)
    xs = torch.arange(width, device=device, dtype=dtype)
    try:
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    except TypeError:
        grid_y, grid_x = torch.meshgrid(ys, xs)
    grid_x = (grid_x + 0.5 - (width / 2.0)) / (width / 2.0)
    grid_y = (grid_y + 0.5 - (height / 2.0)) / (height / 2.0)
    grid = torch.stack([grid_x, grid_y], dim=-1)
    return grid.unsqueeze(0).unsqueeze(0).repeat(batch_size, num_views, 1, 1, 1)


def fit_expression_vector(expression, target_dim, batch_size, device, dtype, add_view_dim=False):
    if target_dim <= 0:
        return None
    if expression is None:
        return None
    if not torch.is_tensor(expression):
        expression = torch.tensor(expression, device=device)
    expression = expression.to(device=device, dtype=dtype)
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


def align_indices_like_dataset(num_frames, target_length):
    if num_frames <= 0:
        raise ValueError("Video has no frames.")
    if target_length <= 0:
        raise ValueError("target_length must be positive.")

    source_indices = np.arange(target_length, dtype=np.int64)
    if num_frames == len(source_indices):
        return source_indices
    if num_frames > np.max(source_indices):
        return np.clip(source_indices, 0, num_frames - 1)
    return np.round(np.linspace(0, num_frames - 1, len(source_indices))).astype(np.int64)


def normalize_source_index(source_index, source_length):
    src_idx = int(source_index)
    if src_idx < 0:
        src_idx = source_length + src_idx
    if src_idx < 0 or src_idx >= source_length:
        raise IndexError(f"source_index {source_index} out of range for source length {source_length}")
    return src_idx


def resolve_source_view_indices(base_idx, num_src, source_length, mode):
    if num_src <= 1:
        return np.asarray([int(base_idx)], dtype=np.int64)
    if mode == "duplicate":
        return np.full((num_src,), int(base_idx), dtype=np.int64)
    offsets = np.arange(num_src, dtype=np.int64)
    return np.clip(int(base_idx) + offsets, 0, source_length - 1)


class _Cv2WriterWrapper:
    def __init__(self, writer):
        self._writer = writer

    def append_data(self, frame):
        import cv2

        arr = np.asarray(frame)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        # cv2 expects BGR
        arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        self._writer.write(arr_bgr)

    def close(self):
        self._writer.release()


def open_video_writer(out_path, fps, height, width):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import cv2
    except Exception as e:
        raise RuntimeError(f"OpenCV is required for video writing in this environment: {e}")

    candidates = [out_path]
    if out_path.suffix.lower() != ".avi":
        candidates.append(out_path.with_suffix(".avi"))

    errors = []
    for cand in candidates:
        suffix = cand.suffix.lower()
        if suffix == ".avi":
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
        else:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        writer = cv2.VideoWriter(str(cand), fourcc, float(fps), (int(width), int(height)))
        if writer.isOpened():
            return _Cv2WriterWrapper(writer), cand
        writer.release()
        errors.append(f"VideoWriter open failed: {cand}")

    raise RuntimeError("Unable to open video writer. " + " | ".join(errors))


@torch.no_grad()
def generate_pair_video(model, cfg, source_video, driving_video, out_path, args, device, kp_detector=None):
    frame_shape = resolve_frame_shape(cfg)
    h, w = int(frame_shape[0]), int(frame_shape[1])
    num_kp = int(cfg["model"]["encoder_kwargs"].get("num_kp", 10))
    num_src = int(cfg.get("data", {}).get("num_src", 1))
    source_length = int(source_video.shape[0])
    static_src_idx = normalize_source_index(args.source_index, source_length)

    driving_tensor = torch.from_numpy(driving_video).permute(0, 3, 1, 2).unsqueeze(0).to(device=device, dtype=torch.float32)
    bs, t_steps, c, _, _ = driving_tensor.shape
    if bs != 1:
        raise ValueError("Only bs=1 inference is supported in generate_pair_video.")
    if c != 3:
        raise ValueError(f"Expected RGB input, got C={c}")

    expression_encoder = getattr(model, "expression_encoder", None)

    def extract_frame_conditioning(frame):
        if kp_detector is None:
            return None, None
        keypoints, latent = kp_detector(frame)
        expression = None
        if expression_encoder is not None:
            feature_map = latent.get("feature_map")
            heatmap = latent.get("heatmap")
            if feature_map is not None and heatmap is not None:
                if heatmap.dim() == 5 and heatmap.shape[2] == 1:
                    heatmap = heatmap.squeeze(2)
                expression = expression_encoder(feature_map, heatmap)
        return keypoints, expression
    encode_with_expression = bool(getattr(model.encoder, "encode_with_expression", False))
    encoder_expr_size = int(getattr(model.encoder, "expression_size", 0))
    decoder_expr_size = int(getattr(model.decoder, "expression_size", 0))
    dtype = driving_tensor.dtype

    def build_source_bundle(view_indices):
        source_frame_tensors = []
        source_kp_tensors = []
        source_expr_tensors = []
        for src_idx in view_indices:
            source_frame = source_video[int(src_idx)]
            source_tensor = (
                torch.from_numpy(source_frame)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(device=device, dtype=dtype)
            )
            source_frame_tensors.append(source_tensor)

            source_kp_frame, source_expression = extract_frame_conditioning(source_tensor)
            if source_kp_frame is None:
                source_kp_frame = torch.zeros((bs, num_kp, 2), device=device, dtype=dtype)
            else:
                source_kp_frame = source_kp_frame.to(device=device, dtype=dtype)
            source_kp_tensors.append(source_kp_frame)

            if encode_with_expression and encoder_expr_size > 0:
                source_expression = fit_expression_vector(
                    source_expression,
                    target_dim=encoder_expr_size,
                    batch_size=bs,
                    device=device,
                    dtype=dtype,
                    add_view_dim=False,
                )
                if source_expression is None:
                    source_expression = source_tensor.new_zeros((bs, encoder_expr_size))
                source_expr_tensors.append(source_expression)

        source_frames = torch.stack(source_frame_tensors, dim=1)
        source_kps = torch.stack(source_kp_tensors, dim=1)
        source_pos = build_full_image_pos(bs, h, w, len(view_indices), device=device, dtype=dtype)

        encoder_expression = None
        if encode_with_expression and encoder_expr_size > 0:
            encoder_expression = torch.stack(source_expr_tensors, dim=1)

        z = model.encoder(source_frames, source_kps, source_pos, expression_vector=encoder_expression)
        return z, source_frames

    if args.framewise_source:
        source_schedule = align_indices_like_dataset(source_length, t_steps)
        source_refresh_interval = max(1, int(args.framewise_source_interval))
        static_z = None
        static_source_frames = None
    else:
        source_schedule = None
        source_refresh_interval = 1
        static_view_indices = resolve_source_view_indices(
            static_src_idx,
            num_src=num_src,
            source_length=source_length,
            mode="duplicate",
        )
        static_z, static_source_frames = build_source_bundle(static_view_indices)

    target_pos = build_full_image_pos(bs, h, w, 1, device=device, dtype=dtype)[:, 0].reshape(bs, h * w, 2)
    n_pix = target_pos.shape[1]

    decode_chunk = int(args.decode_chunk)
    if decode_chunk <= 0:
        decode_chunk = int(cfg.get("training", {}).get("rppg_decode_chunk", n_pix))
    decode_chunk = max(1, min(decode_chunk, n_pix))

    prev_generated = None
    prev_input_frame = None
    prev_state = None
    supports_temporal = True

    writer, final_path = open_video_writer(out_path, args.fps, h, w)
    frame_written = 0
    try:
        for t in range(t_steps):
            if args.framewise_source:
                refresh_t = min((t // source_refresh_interval) * source_refresh_interval, t_steps - 1)
                frame_src_idx = int(source_schedule[refresh_t])
                view_indices = resolve_source_view_indices(
                    frame_src_idx,
                    num_src=num_src,
                    source_length=source_length,
                    mode=args.framewise_source_mode,
                )
                z, source_frames = build_source_bundle(view_indices)
            else:
                z = static_z
                source_frames = static_source_frames

            cur_input = driving_tensor[:, t]
            target_kps = torch.zeros((bs, n_pix, num_kp, 2), device=device, dtype=source_frames.dtype)
            driving_kp_frame, driving_expression = extract_frame_conditioning(cur_input)
            if driving_kp_frame is not None:
                target_kps = driving_kp_frame[:, None].repeat(1, n_pix, 1, 1).to(
                    device=device, dtype=source_frames.dtype
                )
            decoder_expression = None
            if decoder_expr_size > 0:
                decoder_expression = fit_expression_vector(
                    driving_expression,
                    target_dim=decoder_expr_size,
                    batch_size=bs,
                    device=device,
                    dtype=source_frames.dtype,
                    add_view_dim=False,
                )
                if decoder_expression is None:
                    decoder_expression = source_frames.new_zeros((bs, decoder_expr_size))

            pred_flat = torch.empty((bs, n_pix, 3), device=device, dtype=source_frames.dtype)
            extras_first = {}

            for start in range(0, n_pix, decode_chunk):
                end = min(n_pix, start + decode_chunk)
                pos_chunk = target_pos[:, start:end]
                kps_chunk = target_kps[:, start:end]

                if supports_temporal:
                    temporal_inputs = {
                        "cur_frame": cur_input,
                        "prev_frame": prev_input_frame,
                        "prev_generated": prev_generated,
                        "prev_feat": (prev_state or {}).get("prev_feat"),
                    }
                    try:
                        pred_chunk, extras_chunk = model.decoder(
                            z,
                            pos_chunk,
                            kps_chunk,
                            expression_vector=decoder_expression,
                            temporal_inputs=temporal_inputs,
                            return_temporal_state=True,
                        )
                    except TypeError:
                        supports_temporal = False

                if not supports_temporal:
                    pred_chunk, extras_chunk = model.decoder(
                        z,
                        pos_chunk,
                        kps_chunk,
                        expression_vector=decoder_expression,
                    )
                    if not isinstance(extras_chunk, dict):
                        extras_chunk = {}

                if start == 0 and isinstance(extras_chunk, dict):
                    extras_first = extras_chunk
                pred_flat[:, start:end] = pred_chunk

            pred_frame = pred_flat.view(bs, h, w, 3).permute(0, 3, 1, 2).contiguous()
            prev_state = extras_first.get("temporal_state", None) if isinstance(extras_first, dict) else None
            prev_generated = pred_frame.detach()
            prev_input_frame = cur_input

            frame_np = pred_frame[0].permute(1, 2, 0).clamp(0.0, 1.0).cpu().numpy()
            frame_u8 = np.clip(frame_np * 255.0, 0, 255).astype(np.uint8)
            writer.append_data(frame_u8)
            frame_written += 1
    finally:
        writer.close()

    return final_path, frame_written


def append_log_row(log_path, row):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "timestamp",
        "index",
        "pair_id",
        "status",
        "dataset",
        "source_raw",
        "driving_raw",
        "source_resolved",
        "driving_resolved",
        "output_path",
        "num_frames",
        "elapsed_sec",
        "error",
    ]
    exists = log_path.exists()
    with log_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fields})


def main():
    args = parse_args()

    pairs_csv = Path(args.pairs_csv).resolve()
    cfg_path = Path(args.config).resolve()
    ckpt_path = Path(args.checkpoint).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pairs_csv.exists():
        raise FileNotFoundError(f"pairs-csv not found: {pairs_csv}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"config not found: {cfg_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    if args.skip_existing and args.overwrite:
        raise ValueError("skip-existing and overwrite cannot both be set.")

    log_csv = Path(args.log_csv).resolve() if args.log_csv else (out_dir / "generation_log.csv")
    data_root = Path(args.data_root).resolve() if args.data_root else None

    with cfg_path.open("r") as f:
        cfg = yaml.load(f, Loader=yaml.CLoader)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    device = torch.device(args.device)

    if args.kp_checkpoint:
        kp_detector_path = Path(args.kp_checkpoint).resolve()
    else:
        kp_detector_path = Path(__file__).resolve().parent.parent / "checkpoints" / "conversion" / "kp_detector.pt"
    kp_detector = None
    if kp_detector_path.exists():
        kp_detector = KPDetector().to(device)
        kp_detector.load_state_dict(torch.load(kp_detector_path, map_location=device))
        kp_detector.eval()
    else:
        print(
            f"[WARN] kp_detector checkpoint not found at {kp_detector_path}; "
            "falling back to zero keypoint conditioning."
        )

    expression_encoder = None
    checkpoint_modules = {}
    if kp_detector is not None:
        expression_encoder = ExpressionEncoder(
            expression_size=cfg["model"]["expression_size"],
            in_channels=kp_detector.predictor.out_filters,
        )
        checkpoint_modules["expression_encoder"] = expression_encoder

    model = FSRT(cfg["model"], expression_encoder=expression_encoder).to(device)
    model.eval()
    if expression_encoder is not None:
        model.expression_encoder.eval()

    checkpoint_modules.update({"encoder": model.encoder, "decoder": model.decoder})
    checkpoint = Checkpoint("/", device=device, **checkpoint_modules)
    checkpoint.load(str(ckpt_path))

    with pairs_csv.open("r", newline="") as f:
        rows = list(csv.DictReader(f))

    if len(rows) == 0:
        print(f"No rows in {pairs_csv}")
        return

    start = max(0, int(args.start_index))
    end = len(rows) - 1 if int(args.end_index) < 0 else min(int(args.end_index), len(rows) - 1)
    if end < start:
        raise ValueError(f"Invalid index range: start={start}, end={end}, total_rows={len(rows)}")

    selected = list(range(start, end + 1))
    if args.limit > 0:
        selected = selected[: int(args.limit)]

    frame_shape = resolve_frame_shape(cfg)
    csv_dir = pairs_csv.parent

    ok_count = 0
    skip_count = 0
    fail_count = 0

    print(f"[INFO] total_rows={len(rows)} selected={len(selected)} start={start} end={end}")
    print(f"[INFO] config={cfg_path}")
    print(f"[INFO] checkpoint={ckpt_path}")
    print(f"[INFO] out_dir={out_dir}")
    print(
        "[INFO] source_index={} framewise_source={} framewise_source_mode={} framewise_source_interval={}".format(
            int(args.source_index),
            bool(args.framewise_source),
            args.framewise_source_mode,
            int(args.framewise_source_interval),
        )
    )

    for n, idx in enumerate(selected, start=1):
        row = rows[idx]
        t0 = time.time()
        dataset = infer_dataset(row, args.dataset)

        source_col = args.source_col.strip() if args.source_col.strip() else ("video_path" if row.get("video_path") else "source_path")
        driving_col = args.driving_col.strip() if args.driving_col.strip() else "driving_path"
        source_raw = row.get(source_col, "")
        driving_raw = row.get(driving_col, "")
        pair_id = row.get("pair_id", "") or f"{idx:06d}_{sanitize_name(Path(str(source_raw)).stem)}_to_{sanitize_name(Path(str(driving_raw)).stem)}"
        pair_name = f"{idx:06d}_{sanitize_name(pair_id)}"
        out_path = out_dir / f"{pair_name}.mp4"

        result_row = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "index": idx,
            "pair_id": pair_id,
            "dataset": dataset,
            "source_raw": source_raw,
            "driving_raw": driving_raw,
            "source_resolved": "",
            "driving_resolved": "",
            "output_path": str(out_path),
            "num_frames": 0,
            "elapsed_sec": 0.0,
            "status": "",
            "error": "",
        }

        try:
            if out_path.exists():
                if args.skip_existing and not args.overwrite:
                    result_row["status"] = "skipped_existing"
                    skip_count += 1
                    append_log_row(log_csv, result_row)
                    print(f"[{n}/{len(selected)}] skip existing: {out_path}")
                    continue
                if args.overwrite:
                    out_path.unlink()

            src_path = resolve_video_path(source_raw, csv_dir=csv_dir, dataset=dataset, data_root=data_root)
            drv_path = resolve_video_path(driving_raw, csv_dir=csv_dir, dataset=dataset, data_root=data_root)
            result_row["source_resolved"] = str(src_path)
            result_row["driving_resolved"] = str(drv_path)

            if args.dry_run:
                result_row["status"] = "dry_run"
                append_log_row(log_csv, result_row)
                print(f"[{n}/{len(selected)}] dry-run idx={idx} src={src_path} drv={drv_path}")
                continue

            source_video = _load_video(src_path, frame_shape=frame_shape)
            driving_video = _load_video(drv_path, frame_shape=frame_shape)

            stride = max(1, int(args.frame_stride))
            if stride > 1:
                driving_video = driving_video[::stride]
            if int(args.max_frames) > 0:
                driving_video = driving_video[: int(args.max_frames)]
            if driving_video.shape[0] <= 0:
                raise ValueError("Driving video has no frames after stride/max_frames.")

            final_path, num_frames = generate_pair_video(
                model=model,
                cfg=cfg,
                source_video=source_video,
                driving_video=driving_video,
                out_path=out_path,
                args=args,
                device=device,
                kp_detector=kp_detector,
            )

            result_row["output_path"] = str(final_path)
            result_row["num_frames"] = int(num_frames)
            result_row["status"] = "ok"
            ok_count += 1
            print(f"[{n}/{len(selected)}] ok idx={idx} frames={num_frames} out={final_path}")
        except Exception as e:
            fail_count += 1
            result_row["status"] = "failed"
            result_row["error"] = str(e)
            print(f"[{n}/{len(selected)}] failed idx={idx}: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        finally:
            result_row["elapsed_sec"] = round(time.time() - t0, 3)
            append_log_row(log_csv, result_row)

    print(
        "[SUMMARY] ok={} skipped={} failed={} log={}".format(
            ok_count, skip_count, fail_count, log_csv
        )
    )
    if fail_count > 0 and args.fail_on_error:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
