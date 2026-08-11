# LeHome Visual Variational Episode Splines — ICRA 2027

This self-contained repository component learns a continuous visual representation of human garment-folding episodes from:

```text
E:\Lehome-Dataset\lehome_round_2_dataset\pretrain_dataset\pretrain_lehome_all_garment_data_z180
```

The inspected dataset contains 4,180 episodes and 724,399 frames at 23 FPS. Each Parquet row contains a top-view RGB image, a 16D state, and a 16D action. Only state/action dimensions `0, 1, 8, 9` are used. For valid frames:

```text
action[t, [0,1,8,9]] == state[t+1, [0,1,8,9]]
```

Consequently, there is no future-state prediction head and no separate action spline. Actions are the supplied future hand trajectory used to condition future visual prediction.

## Representation

The visual pipeline is:

```text
top-view image
  → frozen DINOv3 ViT-S+/16 patch features
  → 8 learned attention-resampler slots
  → posterior mean/log-variance [8,256]
  → flattened deterministic export mean [2048]
  → adaptive shared-knot PCHIP per episode
```

The DINO backbone is frozen. The trainable variational resampler defines:

```text
q(z_t | I_t) = Normal(mu_t, diag(exp(logvar_t)))
```

Training samples with the reparameterization trick, while validation, export, and spline fitting always use `mu_t`.

## Loss heads

The implementation has:

1. A shared patch-token decoder that reconstructs frozen DINO patch features from current and predicted-future latent slots.
2. A shared global head that reconstructs the frozen DINO global token.
3. A current-only 4D state probe.
4. A standard-normal posterior KL with warm-up.
5. An action-conditioned temporal prior producing future mean/log-variance at horizons `[1,2,4,8,16,32]`.
6. Conditional future-posterior/prior KL, future-mean alignment, and frozen-DINO future reconstruction.

Training objectives are cumulative. The deterministic stage optimizes spatial reconstruction, the variational stage adds standard-normal KL, and the temporal stage jointly optimizes the variational spatial objective plus the action-conditioned future objective. Future posterior targets are stop-gradient targets from the current spatial encoder; they can move between optimizer steps, while frozen-DINO reconstruction anchors the representation.

## Installation

Python 3.10+ and a CUDA-enabled PyTorch installation are recommended.

```powershell
cd D:\LeHome-Challenge\Lehome-Spline-ICRA2027\lehome_spline_generation
python -m pip install -e .
```

DINOv3 support requires `transformers>=4.56`. Access to the configured DINOv3 weights may require accepting the model license on Hugging Face and authenticating with `hf auth login`.

All settings are in [configs/default.yaml](configs/default.yaml). Relative output paths are resolved from this component directory, even when a command is launched from elsewhere. Any setting can be overridden without editing the file:

```powershell
python train.py --config configs/default.yaml --set training.batch_size=4
```

## 1. Validate the dataset

Five-episode smoke validation:

```powershell
python validate_dataset.py --config configs/default.yaml --max-episodes 5
```

Full validation:

```powershell
python validate_dataset.py `
  --config configs/default.yaml `
  --output outputs\dataset_validation.json
```

Validation uses an episode tqdm and checks image bytes, row counts, and the valid-dimension next-state invariant.

## 2. Train

```powershell
python train.py --config configs/default.yaml
```

Training is split by complete episodes according to `dataset.train_ratio` and `dataset.val_ratio`; the remainder is the optional test split. It never constructs a window across an episode boundary.

One training epoch is divided into three consecutive, non-overlapping step ranges. The number of complete batches in the training split is calculated first, then allocated using the configured fractions:

1. First 30%: `spatial_deterministic`, reconstruct frozen DINO features using the posterior mean.
2. Next 30%: `spatial_variational`, enable posterior sampling and warm the standard-prior KL.
3. Final 40%: `temporal_prior`, jointly train the variational spatial VAE and n-step action-conditioned future Gaussian prior with an additive objective.

The final-stage loss is `joint_spatial * spatial_total + joint_temporal * temporal_total`. Both weights default to `1.0`.

Largest-remainder rounding guarantees that the integer stage-step counts sum to the one-epoch total. Windows are randomly assigned to exactly one stage, while each stage's read order is grouped by shuffled episode. This preserves a disjoint randomized partition but avoids repeatedly reloading Parquet episodes from slow storage. The resolved counts are written to `outputs/visual_vae/training_schedule.json` before optimization starts.

Training and validation strides are independent. `dataset.window_stride` controls the complete training pass; `dataset.val_window_stride` subsamples validation start frames to reduce validation cost without changing the held-out episode split.

Progress display includes:

```text
training/stages
<stage>/epochs
<stage>/train/epoch-N
<stage>/val/epoch-N
```

The training loss postfix is refreshed every `training.log_frequency` optimizer steps, and a durable `[train]` record is printed to the console. A full validation pass runs every `training.val_frequency` optimizer steps and prints a `[validation]` record when complete. Validation also runs at each stage boundary when that step has not already triggered validation.

All interval records are appended as structured JSON to `outputs/visual_vae/training_metrics.jsonl`. Stage summaries and step-triggered validation results are also maintained in `outputs/visual_vae/metrics.json`.

Checkpoints remain epoch-based and are written atomically after the end-of-epoch validation to:

```text
outputs/visual_vae/checkpoints/
├── best.pt
├── final.pt
└── <stage>_epoch_<N>.pt
```

Resume an interrupted run:

```powershell
python train.py `
  --config configs/default.yaml `
  --resume outputs\visual_vae\checkpoints\spatial_variational_epoch_002.pt
```

Run a fresh schedule initialized from completed model weights:

```powershell
python train.py `
  --config configs\joint_full_epoch.yaml `
  --init-checkpoint outputs\visual_vae\checkpoints\final.pt
```

`--init-checkpoint` restores model weights and the source checkpoint's state normalization, verifies
that the deterministic episode split matches the new config, and then starts at stage 0 / epoch 0 /
global step 0 with a fresh optimizer. This is the correct mode for the full-epoch joint continuation.
`--resume` instead restores stage, epoch, optimizer, and global-step state and must only be used with
the same stage layout as the interrupted run. The two options are mutually exclusive.

## 3. Export per-frame posterior embeddings

Smoke export:

```powershell
python export_embeddings.py `
  --config configs/default.yaml `
  --checkpoint outputs\visual_vae\checkpoints\final.pt `
  --max-episodes 2
```

Full export:

```powershell
python export_embeddings.py `
  --config configs/default.yaml `
  --checkpoint outputs\visual_vae\checkpoints\final.pt
```

Use `--overwrite` to replace complete existing episode files. Writes use `.partial` files followed by atomic replacement and post-write validation.

Each episode produces:

```text
outputs/visual_vae/embeddings/episode_XXXXXX.npz
```

with:

```text
mean             [T,2048] float16
log_variance     [T,2048] float16
state            [T,4]    float32
action           [T,4]    float32
timestamps       [T]      float64
frame_indices    [T]      int64
```

The action array is retained only for alignment/auditing. It is not independently splined.

Training-split latent mean/std are saved as `embedding_normalization.npz` and used for spline error measurement.

## 4. Fit adaptive PCHIP episode splines

Smoke fit:

```powershell
python fit_splines.py --config configs/default.yaml --max-episodes 2
```

Full fit:

```powershell
python fit_splines.py --config configs/default.yaml
```

The outer tqdm reports episode progress. Every episode also has an adaptive-knot tqdm reporting knot count and maximum normalized error.

The fitter uses one shared knot sequence for the 2048D posterior mean and aligned 4D state. Initial mandatory knots include:

- Episode endpoints.
- The top 5% of combined visual/state transitions.
- A maximum eight-frame gap.

It repeatedly adds the worst-error frames until all three criteria pass or the knot limit is reached:

```text
standardized latent RMSE <= 0.05
latent cosine distance    <= 0.01
standardized state RMSE   <= 0.05
```

These values are starting points and must be calibrated using downstream reconstruction and policy performance.

Each spline file contains:

```text
knot_timestamps
knot_frame_indices
knot_embeddings       [K,2048]
knot_states           [K,4]
mandatory_knot_mask
per-frame validation errors
```

PCHIP coefficients are reconstructed from the compact knots rather than stored. Episode metadata records whether tolerance was satisfied; failures are never silently reported as successful fits.

## Tests

The unit tests use a small fake DINO backbone and do not download model weights:

```powershell
python -m pytest
```

They verify posterior/prior tensor shapes, shared decoder outputs, and adaptive PCHIP tolerance behavior.

## Important interpretation

The output spline is a continuous trajectory through learned visual-feature posterior means. It is not a pixel-perfect video codec. Its validity is measured by latent, state, and ultimately frozen-DINO reconstruction quality.

The complete episode is used to fit the offline PCHIP. Therefore, querying that spline as an online policy input would leak future information. Online deployment should use the current encoder output or a separately designed causal history representation.
