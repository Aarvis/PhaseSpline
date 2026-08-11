# DTW-Transfer

This folder contains a multi-reference DTW transfer tool for LeHome temporal checkpoint labels using precomputed 2048-D per-frame embeddings.

Main entry point:

- `multi_reference_embedding_dtw_transfer.py`

Default config:

- `configs/lehome_multi_reference_frame_embeddings.yaml`

Reference manifest example:

- `manifests/example_reference_manifest.json`

Typical usage:

```powershell
python multi_reference_embedding_dtw_transfer.py `
  --dataset sim `
  --garment shorts `
  --reference-manifest manifests/example_reference_manifest.json
```

For the three garment categories used by the paired-dataset viewer helper, the default config already includes built-in episode blocks:

- sim:
  - `shorts`: `250:499`
  - `top_long_sleeve`: `500:749`
  - `top_short_sleeve`: `750:999`
- human:
  - `shorts`: `1018:2038`
  - `top_long_sleeve`: `2039:2874`
  - `top_short_sleeve`: `2875:4179`

So a simple run without a manifest is valid as long as the manual reference annotations already exist for that garment:

```powershell
python multi_reference_embedding_dtw_transfer.py `
  --dataset sim `
  --garment shorts
```

If you want to override those blocks or run on a different dataset layout, you can still pass candidate ranges directly:

```powershell
python multi_reference_embedding_dtw_transfer.py `
  --dataset sim `
  --garment shorts `
  --candidate-range shorts=250:499
```

Notes:

- References are always expected to be manually annotated checkpoint files.
- Output checkpoints are written under `outputs/<dataset>/<garment>/transferred_checkpoints/...`.
- Transferred checkpoints are marked with `template_status: dtw_transferred_requires_review`.
- If the full dataset metadata does not encode garment identity for every episode, use the built-in ranges from the config or provide a per-garment candidate pool explicitly in the manifest or via `--candidate-range`.
