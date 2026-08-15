# Spline runtime server workflow

This folder now contains the runtime stack for:

- prompt preprocessing
- spline server
- OpenPI VLA server wrapper
- LeHome parallel-eval websocket client wiring

The runtime flow is:

1. preprocess human prompt videos / dataset prompt episodes into a prompt bank
2. start one or more spline servers
3. start one or more OpenPI spline-conditioned websocket servers
4. run `parallel_eval` with `--policy_type openpi_spline_ws`

## 1. Prompt preprocessing

Code:

- `D:/LeHome-Challenge/Lehome-Spline-ICRA2027/Spline-Server/Prompt-Prepossser/preprocess_prompts.py`
- `D:/LeHome-Challenge/Lehome-Spline-ICRA2027/Spline-Server/Prompt-Prepossser/configs/default.yaml`

Inputs required in the prompt-preprocessor config:

- `paths.prompt_bank_root`
- `paths.human_dataset_root`
- `paths.human_bspline_root`
- `paths.human_frame_embedding_root` for dataset prompts
- `paths.human_embedder_config_path`
- `paths.human_embedder_checkpoint_path`
- `paths.human_embedding_normalization_path`
- `prompts`

Each `prompts` entry must be one of:

1. dataset prompt
   - `source.type: dataset_episode`
   - `prompt_id`
   - `category_id`
   - `human_episode_index`

2. raw video prompt
   - `source.type: video_file`
   - `prompt_id`
   - `category_id`
   - `video_path`
   - optional `fps`
   - optional `frame_stride`
   - optional `max_frames`

Command:

```powershell
cd D:\LeHome-Challenge\Lehome-Spline-ICRA2027\Spline-Server\Prompt-Prepossser
python .\preprocess_prompts.py --config .\configs\default.yaml
```

Output:

- prompt bank root from `paths.prompt_bank_root`
- `manifest.json`
- one prompt package directory per prompt

Each prompt package contains:

- `spline.npz`
- `localizer_cache.npz`
- optional `frame_embeddings.npz`
- optional `annotation_checkpoints.json`

`predicted_width` runtime mode needs prompt packages with `annotation_checkpoints.json`.

## 2. Spline server

Code:

- `D:/LeHome-Challenge/Lehome-Spline-ICRA2027/Spline-Server/serve_spline_policy.py`
- `D:/LeHome-Challenge/Lehome-Spline-ICRA2027/Spline-Server/multi_serve_spline_policy.py`
- `D:/LeHome-Challenge/Lehome-Spline-ICRA2027/Spline-Server/configs/default.yaml`

The spline server does:

- current robot multiview visual embedding
- human localizer inference
- human local interval selection
  - `predicted_width`, or
  - `fixed_future_frames`
- human-to-robot local spline translation

Single server:

```powershell
cd D:\LeHome-Challenge\Lehome-Spline-ICRA2027\Spline-Server
python .\serve_spline_policy.py --config .\configs\default.yaml
```

Single server with a different port:

```powershell
python .\serve_spline_policy.py --config .\configs\default.yaml --port 9100
```

Multi-server on one GPU:

```powershell
python .\multi_serve_spline_policy.py `
  --config .\configs\default.yaml `
  --num-servers 4 `
  --start-port 9100 `
  --gpu-id 0
```

Important config blocks in `Spline-Server/configs/default.yaml`:

- `paths.prompt_bank_root`
- `runtime.device`
- `prompt_selection`
- `end_mode`
- `robot_embedder.checkpoint_path`
- `localizer.checkpoint_path`
- `translator.checkpoint_path`

Runtime request requirements:

- `session_id`
- `observation/top_rgb`
- `observation/left_rgb`
- `observation/right_rgb`
- `observation/state`
- one of:
  - `prompt_id`
  - `prompt_category_id`
  - `garment_type`
  - `garment_name`

The spline server returns:

- `predicted_robot_coefficients`
- `predicted_robot_knots`
- `predicted_human_start_u`
- `predicted_human_end_u`
- validity / fallback flags
- timing metadata

## 3. OpenPI spline-conditioned VLA server

Code:

- `D:/LeHome-Challenge/openpi/scripts/serve_lehome_spline_policy.py`
- `D:/LeHome-Challenge/openpi/scripts/multi_serve_lehome_spline_policy.py`

The VLA server:

- receives robot observations over websocket
- calls the spline server
- injects:
  - `robot_spline_coefficients`
  - `robot_spline_knots`
  - `observation/state`
  - `prompt`
- runs the trained OpenPI spline-conditioned policy

Single VLA server:

```powershell
cd D:\LeHome-Challenge\openpi
uv run scripts/serve_lehome_spline_policy.py `
  --port 8000 `
  --spline-server-url ws://127.0.0.1:9100 `
  policy:checkpoint `
  --policy.config pi05_lehome_robot_spline_joint_delta_finetune `
  --policy.dir D:/LeHome-Challenge/openpi/checkpoints/pi05_lehome_robot_spline_joint_delta_finetune
```

Multi VLA server, one-to-one with spline servers:

```powershell
cd D:\LeHome-Challenge\openpi
uv run scripts/multi_serve_lehome_spline_policy.py `
  --num-servers 4 `
  --start-port 8000 `
  --gpu-id 0 `
  --total-gpu-fraction 0.90 `
  --spline-base-ws-url ws://127.0.0.1 `
  --spline-start-port 9100 `
  -- policy:checkpoint `
  --policy.config pi05_lehome_robot_spline_joint_delta_finetune `
  --policy.dir D:/LeHome-Challenge/openpi/checkpoints/pi05_lehome_robot_spline_joint_delta_finetune
```

Notes:

- `--start-port 8000` means VLA endpoints are `ws://127.0.0.1:8000`, `:8001`, `:8002`, `:8003`
- `--spline-start-port 9100` means spline endpoints are `ws://127.0.0.1:9100`, `:9101`, `:9102`, `:9103`
- server index `i` uses spline server index `i`

## 4. LeHome parallel eval

Client policy type added:

- `openpi_spline_ws`

Code:

- `D:/LeHome-Challenge/lehome-challenge/scripts/eval_policy/openpi_spline_ws_policy.py`

Run `parallel_eval` against the VLA websocket servers:

```powershell
cd D:\LeHome-Challenge\lehome-challenge
python .\parallel_eval\main.py `
  --policy_type openpi_spline_ws `
  --policy_base_ws_url ws://127.0.0.1 `
  --policy_start_port 8000 `
  --policy_server_count 4 `
  --garment_type top_long_sleeve `
  --num_episodes 10 `
  --headless
```

`parallel_eval` now forwards:

- `garment_name`
- `garment_type`

to the websocket policy, and that metadata is passed through to the spline server for prompt selection.

## Recommended startup order

1. preprocess prompts
2. start spline servers
3. start VLA servers
4. run `parallel_eval`

## Practical notes

- `predicted_width` mode requires a width-enabled localizer checkpoint and dataset-derived prompt packages with semantic checkpoint annotations
- `fixed_future_frames` mode works for raw prompt videos too
- if the current human interval is invalid near the prompt tail, the spline server can reuse the session’s last valid predicted robot spline when `runtime.use_last_valid_fallback=true`
- prompt-bank categories must match the categories the localizer / translator checkpoints were trained on
