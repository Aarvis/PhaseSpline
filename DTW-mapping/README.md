# DINOv3 frame embeddings for DTW

`extract_dinov3_embeddings.py` extracts one global DINOv3 ViT-S+/16 embedding for every frame in the human and simulation top-view videos.

## Install

The DINOv3 checkpoint is gated. Accept its license on the [official model page](https://huggingface.co/facebook/dinov3-vits16plus-pretrain-lvd1689m), authenticate locally, and install the required packages. `torchvision` is required by the checkpoint's official fast image processor:

```powershell
cd D:\LeHome-Challenge\Lehome-Spline-ICRA2027\DTW-mapping
py -3.12 -m pip install -r requirements.txt
hf auth login
```

## Extract both videos

The two requested top-view videos are the defaults:

```powershell
py -3.12 extract_dinov3_embeddings.py --device auto --batch-size 16
```

On a CPU-only machine, reduce the batch size if memory is limited:

```powershell
py -3.12 extract_dinov3_embeddings.py --device cpu --batch-size 4
```

Provide explicit videos when needed:

```powershell
py -3.12 extract_dinov3_embeddings.py `
  Source-Human-Episode\episode_000000_top_rgb.mp4 `
  Sim-Robot-Episode\episode_000033_top_rgb.mp4
```

Pass `--overwrite` to replace existing embedding artifacts.

## Outputs

Each input video receives an `.npz` and `.json` beside it. The `.npz` contains:

- `embeddings`: raw DINOv3 pooled embeddings, shape `[frames, 384]`, `float32`.
- `embeddings_l2`: L2-normalized embeddings for cosine-distance DTW, shape `[frames, 384]`.
- `frame_indices`: exact decoded frame positions.
- `timestamps`: frame times computed from the source FPS.
- `fps`, `source_video`, and `model_name`.

The extractor decodes every frame sequentially, shows per-video and per-frame tqdm progress, writes atomically, rejects non-finite results, and validates the saved shapes.

## Transfer robot checkpoints to the human episode with DTW

`dtw_checkpoint_transfer.py` aligns the simulated robot reference to the human target using every top-camera DINOv3 embedding:

- Reference: `Sim-Robot-Episode/episode_000033`, 269 frames at 30 FPS.
- Target: `Source-Human-Episode/episode_000000`, 229 frames at 23 FPS.
- Distance: cosine distance over L2-normalized 384D DINOv3 embeddings.
- Alignment: monotonic endpoint-anchored DTW.
- Boundary mapping: separate median votes immediately before and after each robot checkpoint boundary.
- Continuity: all human segments are generated from one strictly increasing boundary vector, guaranteeing no gaps or overlaps.

Run from this folder:

```powershell
.\.venv\Scripts\python.exe dtw_checkpoint_transfer.py
```

Pass `--overwrite` to intentionally replace an existing transferred file:

```powershell
.\.venv\Scripts\python.exe dtw_checkpoint_transfer.py --overwrite
```

Parameters and paths are in `configs/dtw_top_rgb.yaml`. Relative paths in that file are resolved from the `DTW-mapping` folder.

The main result is written to:

```text
Source-Human-Episode/checkpoints.json
```

Detailed alignment artifacts are written under:

```text
DTW-results/episode_000033_to_episode_000000/
```

These include the DTW path, frame mapping CSV, boundary-transfer CSV, compressed distance matrix, alignment plot, and a copy of the complete result JSON.

Run the unit tests with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
