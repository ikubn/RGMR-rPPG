# Motion Replacement for Video-Based Remote Physiological Measurement

This repository accompanies an anonymous submission. Remote
photoplethysmography (rPPG) extracts pulse waveforms from facial videos,
but head and body motion easily overwhelm the weak chromatic signal. We
propose **Reference-Guided Motion Replacement (RGMR)**, a data-centric
pipeline that converts high-motion clips into lower-motion counterparts
using a stable reference video from the same subject, via frame-by-frame
motion replacement and temporal stabilization. On MMPD and VIPL-HR with
RhythmMamba, TS-CAN, and PhysFormer, RGMR reduces MAE by up to 67.82%.

## Demo

| Source (motion) | Reference (stable) | Converted |
|:---:|:---:|:---:|
| ![source](assets/source.gif) | ![reference](assets/reference.gif) | ![converted](assets/converted.gif) |

## Datasets

We evaluate RGMR on two public rPPG benchmarks. This repository does not
redistribute any dataset content; obtain the datasets from the official
sources and follow their licenses and access conditions:

- MMPD: <https://github.com/McJackTang/MMPD_rPPG_dataset>
- VIPL-HR: <https://vipl.ict.ac.cn/resources/databases/201811/t20181129_32716.html>

## FSRT Conversion Checkpoint

The FSRT-based RGMR conversion model is loaded from
`checkpoints/conversion/vox256_2Source.pt` (~182 MB). Because the file
exceeds GitHub's single-file limit, it is hosted externally and not
committed to this repository. The download link is provided separately;
once obtained, place the file at
`checkpoints/conversion/vox256_2Source.pt` before running the conversion
pipeline.

## Downstream Checkpoints

`checkpoints/downstream/` contains 16 self-trained rPPG checkpoints used
for the paper's RGMR-D and RGMR-S rows, organized as
`<backbone>/<variant>/<dataset>/model.pth` for all combinations of
{RhythmMamba, TS-CAN, PhysNet, PhysFormer} × {RGMR-D, RGMR-S} ×
{MMPD, VIPL-HR}.

## Pair Manifests

Source–reference pair manifests enumerate dataset instances and are
therefore not included. The construction protocol is documented in
`pairing_code/build_{mmpd,vipl}_target_manifests.py`:

- **MMPD**: each `Walking` clip is paired with a `Stationary` clip from
  the same subject under matching light and exercise conditions
  (132 pairs).
- **VIPL-HR**: each `v2` (head-motion) clip is paired with a `v1`
  (stable) clip from the same subject and camera source (306 pairs).
