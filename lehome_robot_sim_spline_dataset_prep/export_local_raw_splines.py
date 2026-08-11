from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import BSpline
from tqdm.auto import tqdm

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "local_raw_splines_config.yaml"
EPSILON = 1e-8


@dataclass(frozen=True)
class PathsConfig:
    dataset_root: Path
    fitted_spline_root: Path
    output_root: Path


@dataclass(frozen=True)
class ProcessingConfig:
    num_future_knots: int
    padding_mode: str
    coefficient_dtype: str
    knot_dtype: str
    u_dtype: str
    index_dtype: str
    overwrite: bool
    write_episode_metadata_json: bool
    max_episodes: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export per-frame exact local cubic B-spline segments from the already fitted robot sim B-spline dataset. "
            "Each local segment starts exactly at the current frame u and ends at the Nth future knot or the last "
            "available knot, while also saving a fixed-length padded future-knot view for model training."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--num-future-knots", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--episode", action="append", type=int, default=[], help="Repeat to export only specific episodes.")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if yaml is None:
        raise ImportError("PyYAML is required for YAML configs.")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid config: {path}")
    return data


def as_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def resolve_paths(config: dict[str, Any], args: argparse.Namespace) -> PathsConfig:
    paths = config.get("paths")
    if not isinstance(paths, dict):
        raise KeyError("Missing paths block in config.")
    dataset_root = as_path(paths["dataset_root"])
    fitted_spline_root = as_path(paths["fitted_spline_root"])
    if args.output_root is not None:
        output_root = as_path(args.output_root)
    else:
        template = paths.get("output_root_template") or paths.get("output_root")
        if not template:
            raise KeyError("Config must define paths.output_root_template or paths.output_root.")
        num_future_knots = (
            int(args.num_future_knots)
            if args.num_future_knots is not None
            else int(config.get("processing", {}).get("num_future_knots", 12))
        )
        output_root = as_path(str(template).format(num_future_knots=num_future_knots))
    return PathsConfig(
        dataset_root=dataset_root,
        fitted_spline_root=fitted_spline_root,
        output_root=output_root,
    )


def resolve_processing(config: dict[str, Any], args: argparse.Namespace) -> ProcessingConfig:
    processing = config.get("processing")
    if not isinstance(processing, dict):
        raise KeyError("Missing processing block in config.")
    num_future_knots = int(args.num_future_knots if args.num_future_knots is not None else processing["num_future_knots"])
    if num_future_knots <= 0:
        raise ValueError("num_future_knots must be positive.")
    padding_mode = str(processing.get("padding_mode", "repeat_last"))
    if padding_mode != "repeat_last":
        raise ValueError(f"Unsupported padding_mode: {padding_mode}")
    return ProcessingConfig(
        num_future_knots=num_future_knots,
        padding_mode=padding_mode,
        coefficient_dtype=str(processing.get("coefficient_dtype", processing.get("embedding_dtype", "float32"))),
        knot_dtype=str(processing.get("knot_dtype", "float32")),
        u_dtype=str(processing.get("u_dtype", "float32")),
        index_dtype=str(processing.get("index_dtype", "int32")),
        overwrite=bool(processing.get("overwrite", False) or args.overwrite),
        write_episode_metadata_json=bool(processing.get("write_episode_metadata_json", True)),
        max_episodes=int(args.max_episodes) if args.max_episodes is not None else (
            int(processing["max_episodes"]) if processing.get("max_episodes") is not None else None
        ),
    )


def episode_index_from_path(path: Path) -> int:
    return int(path.parent.name.split("_")[-1])


def episode_output_dir(output_root: Path, episode_index: int) -> Path:
    return output_root / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}"


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory already contains files, pass overwrite to replace it: {path}")
    path.mkdir(parents=True, exist_ok=True)


def cast_array(array: np.ndarray, dtype_name: str) -> np.ndarray:
    return np.asarray(array, dtype=np.dtype(dtype_name))


def pad_1d(values: np.ndarray, target_len: int, pad_value: float | int) -> np.ndarray:
    values = np.asarray(values)
    if values.shape[0] >= target_len:
        return values[:target_len]
    pad_count = target_len - values.shape[0]
    pad = np.full((pad_count,), pad_value, dtype=values.dtype if values.size else np.asarray([pad_value]).dtype)
    return np.concatenate([values, pad], axis=0)


def _count_knot_multiplicity(knots: np.ndarray, value: float) -> int:
    return int(np.count_nonzero(np.isclose(knots, value, atol=EPSILON, rtol=0.0)))


def _insert_to_boundary_multiplicity(spline: BSpline, value: float) -> BSpline:
    current_mult = _count_knot_multiplicity(np.asarray(spline.t, dtype=np.float64), value)
    required_additions = max(0, spline.k + 1 - current_mult)
    if required_additions == 0:
        return spline
    return spline.insert_knot(value, m=required_additions)


def restrict_bspline_exact(
    global_knots: np.ndarray,
    global_coefficients: np.ndarray,
    degree: int,
    u_start: float,
    u_end: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if u_end <= u_start + EPSILON:
        return None
    spline = BSpline(np.asarray(global_knots, dtype=np.float64), np.asarray(global_coefficients, dtype=np.float64), degree, extrapolate=False)
    spline = _insert_to_boundary_multiplicity(spline, float(u_start))
    spline = _insert_to_boundary_multiplicity(spline, float(u_end))

    knots = np.asarray(spline.t, dtype=np.float64)
    coefficients = np.asarray(spline.c, dtype=np.float64)
    start_matches = np.flatnonzero(np.isclose(knots, u_start, atol=EPSILON, rtol=0.0))
    end_matches = np.flatnonzero(np.isclose(knots, u_end, atol=EPSILON, rtol=0.0))
    if start_matches.size == 0 or end_matches.size == 0:
        raise RuntimeError("Failed to insert local interval boundaries into B-spline knot vector.")

    start_idx = int(start_matches[0])
    end_idx = int(end_matches[-1])
    local_knots = knots[start_idx : end_idx + 1]
    local_num_control_points = int(local_knots.shape[0] - degree - 1)
    if local_num_control_points <= 0:
        return None
    local_coefficients = coefficients[start_idx : start_idx + local_num_control_points]
    local_knots_local_u = (local_knots - float(u_start)) / max(float(u_end - u_start), EPSILON)
    return local_knots, local_knots_local_u, local_coefficients


def build_episode_windows(
    spline_npz_path: Path,
    processing: ProcessingConfig,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(spline_npz_path, allow_pickle=False) as archive:
        frame_indices = np.asarray(archive["frame_indices"], dtype=np.int64)
        frame_u = np.asarray(archive["frame_u"], dtype=np.float64)
        global_knots = np.asarray(archive["global_knots"], dtype=np.float64)
        global_coefficients = np.asarray(archive["global_coefficients"], dtype=np.float64)
        degree = int(np.asarray(archive["global_degree"]).reshape(-1)[0])
        internal_knot_u = np.asarray(archive["internal_knot_u"], dtype=np.float64)

    if frame_indices.ndim != 1 or frame_u.ndim != 1:
        raise ValueError(f"Unexpected frame ranks in {spline_npz_path}")
    if global_knots.ndim != 1 or global_coefficients.ndim != 2:
        raise ValueError(f"Unexpected B-spline arrays in {spline_npz_path}")
    if frame_indices.shape[0] != frame_u.shape[0]:
        raise ValueError(f"frame_indices/frame_u mismatch in {spline_npz_path}")
    if global_knots.shape[0] < 2 * (degree + 1):
        raise ValueError(f"Invalid B-spline knot vector in {spline_npz_path}")

    num_frames = frame_indices.shape[0]
    num_future_knots = processing.num_future_knots
    coefficient_dim = int(global_coefficients.shape[1])

    unique_knot_positions = np.unique(global_knots)
    support_start = float(global_knots[degree])
    support_end = float(global_knots[-degree - 1])
    if unique_knot_positions.size == 0:
        raise ValueError(f"No unique knot positions in {spline_npz_path}")

    local_start_frame_index = cast_array(frame_indices, processing.index_dtype)
    local_end_frame_index = np.empty((num_frames,), dtype=np.dtype(processing.index_dtype))
    global_current_u = np.empty((num_frames,), dtype=np.dtype(processing.u_dtype))
    global_local_start_u = np.empty((num_frames,), dtype=np.dtype(processing.u_dtype))
    global_local_end_u = np.empty((num_frames,), dtype=np.dtype(processing.u_dtype))
    local_end_frame_u = np.empty((num_frames,), dtype=np.dtype(processing.u_dtype))
    valid_knot_count = np.empty((num_frames,), dtype=np.dtype(processing.index_dtype))
    valid_knot_mask = np.zeros((num_frames, num_future_knots), dtype=bool)
    future_knot_u = np.empty((num_frames, num_future_knots), dtype=np.dtype(processing.u_dtype))
    future_knot_local_u = np.empty((num_frames, num_future_knots), dtype=np.dtype(processing.u_dtype))
    future_knot_covering_frame_index = np.empty((num_frames, num_future_knots), dtype=np.dtype(processing.index_dtype))
    exact_local_spline_valid = np.zeros((num_frames,), dtype=bool)
    exact_local_spline_num_knots = np.empty((num_frames,), dtype=np.dtype(processing.index_dtype))
    exact_local_spline_num_control_points = np.empty((num_frames,), dtype=np.dtype(processing.index_dtype))

    local_knot_offsets = [0]
    local_coeff_offsets = [0]
    local_knot_values_global: list[np.ndarray] = []
    local_knot_values_local: list[np.ndarray] = []
    local_coefficients_values: list[np.ndarray] = []

    frames_needing_padding = 0
    frames_zero_future_knots = 0
    frames_exact_degenerate = 0

    for frame_pos in tqdm(
        range(num_frames),
        desc=f"{spline_npz_path.parent.name}: frames",
        unit="frame",
        leave=False,
    ):
        current_u = float(np.clip(frame_u[frame_pos], support_start, support_end))
        future_start = int(np.searchsorted(unique_knot_positions, current_u, side="right"))
        future_positions = unique_knot_positions[future_start:]
        valid_count = int(min(num_future_knots, future_positions.shape[0]))
        valid_future_knot_u = future_positions[:valid_count]

        if valid_count > 0:
            global_end_u = float(valid_future_knot_u[-1])
            pad_u_value = float(valid_future_knot_u[-1])
            end_frame_pos = int(np.searchsorted(frame_u, global_end_u, side="right") - 1)
            end_frame_pos = max(frame_pos, min(end_frame_pos, num_frames - 1))
            covering_frame_indices_valid = frame_indices[
                np.searchsorted(frame_u, valid_future_knot_u, side="right") - 1
            ].astype(np.int64, copy=False)
            local_future_u_valid = np.clip(
                (valid_future_knot_u - current_u) / max(global_end_u - current_u, EPSILON),
                0.0,
                1.0,
            )
            pad_local_u_value = 1.0
        else:
            global_end_u = current_u
            pad_u_value = current_u
            end_frame_pos = frame_pos
            covering_frame_indices_valid = np.zeros((0,), dtype=np.int64)
            local_future_u_valid = np.zeros((0,), dtype=np.float64)
            pad_local_u_value = 1.0

        if valid_count < num_future_knots:
            frames_needing_padding += 1
        if valid_count == 0:
            frames_zero_future_knots += 1

        padded_future_knot_u = pad_1d(valid_future_knot_u, num_future_knots, pad_u_value)
        padded_future_knot_local_u = pad_1d(local_future_u_valid, num_future_knots, pad_local_u_value)
        padded_future_knot_covering_frame_index = pad_1d(
            covering_frame_indices_valid,
            num_future_knots,
            int(frame_indices[end_frame_pos]),
        )

        valid_knot_mask[frame_pos, :valid_count] = True
        valid_knot_count[frame_pos] = valid_count
        future_knot_u[frame_pos] = cast_array(padded_future_knot_u, processing.u_dtype)
        future_knot_local_u[frame_pos] = cast_array(padded_future_knot_local_u, processing.u_dtype)
        future_knot_covering_frame_index[frame_pos] = cast_array(
            padded_future_knot_covering_frame_index,
            processing.index_dtype,
        )
        local_end_frame_index[frame_pos] = int(frame_indices[end_frame_pos])
        local_end_frame_u[frame_pos] = float(frame_u[end_frame_pos])
        global_current_u[frame_pos] = current_u
        global_local_start_u[frame_pos] = current_u
        global_local_end_u[frame_pos] = global_end_u

        restricted = restrict_bspline_exact(global_knots, global_coefficients, degree, current_u, global_end_u)
        if restricted is None:
            exact_local_spline_valid[frame_pos] = False
            exact_local_spline_num_knots[frame_pos] = 0
            exact_local_spline_num_control_points[frame_pos] = 0
            local_knot_offsets.append(local_knot_offsets[-1])
            local_coeff_offsets.append(local_coeff_offsets[-1])
            frames_exact_degenerate += 1
            continue

        local_knots_global_u, local_knots_local_u, local_coefficients = restricted
        exact_local_spline_valid[frame_pos] = True
        exact_local_spline_num_knots[frame_pos] = int(local_knots_global_u.shape[0])
        exact_local_spline_num_control_points[frame_pos] = int(local_coefficients.shape[0])
        local_knot_values_global.append(cast_array(local_knots_global_u, processing.knot_dtype))
        local_knot_values_local.append(cast_array(local_knots_local_u, processing.knot_dtype))
        local_coefficients_values.append(cast_array(local_coefficients, processing.coefficient_dtype))
        local_knot_offsets.append(local_knot_offsets[-1] + int(local_knots_global_u.shape[0]))
        local_coeff_offsets.append(local_coeff_offsets[-1] + int(local_coefficients.shape[0]))

    if local_knot_values_global:
        exact_local_spline_knot_u = np.concatenate(local_knot_values_global, axis=0)
        exact_local_spline_knot_local_u = np.concatenate(local_knot_values_local, axis=0)
        exact_local_spline_coefficients = np.concatenate(local_coefficients_values, axis=0)
    else:
        exact_local_spline_knot_u = np.zeros((0,), dtype=np.dtype(processing.knot_dtype))
        exact_local_spline_knot_local_u = np.zeros((0,), dtype=np.dtype(processing.knot_dtype))
        exact_local_spline_coefficients = np.zeros((0, coefficient_dim), dtype=np.dtype(processing.coefficient_dtype))

    arrays = {
        "frame_indices": cast_array(frame_indices, processing.index_dtype),
        "local_start_frame_index": local_start_frame_index,
        "local_end_frame_index": local_end_frame_index,
        "global_current_u": cast_array(global_current_u, processing.u_dtype),
        "global_local_start_u": cast_array(global_local_start_u, processing.u_dtype),
        "global_local_end_u": cast_array(global_local_end_u, processing.u_dtype),
        "local_end_frame_u": cast_array(local_end_frame_u, processing.u_dtype),
        "future_knot_u": future_knot_u,
        "future_knot_local_u": future_knot_local_u,
        "future_knot_covering_frame_index": future_knot_covering_frame_index,
        "valid_knot_mask": valid_knot_mask,
        "valid_knot_count": valid_knot_count,
        "exact_local_spline_valid": exact_local_spline_valid,
        "exact_local_spline_num_knots": exact_local_spline_num_knots,
        "exact_local_spline_num_control_points": exact_local_spline_num_control_points,
        "exact_local_spline_knot_offsets": cast_array(np.asarray(local_knot_offsets, dtype=np.int64), processing.index_dtype),
        "exact_local_spline_coeff_offsets": cast_array(np.asarray(local_coeff_offsets, dtype=np.int64), processing.index_dtype),
        "exact_local_spline_knot_u": exact_local_spline_knot_u,
        "exact_local_spline_knot_local_u": exact_local_spline_knot_local_u,
        "exact_local_spline_coefficients": exact_local_spline_coefficients,
        "local_spline_degree": np.asarray([degree], dtype=np.int64),
    }

    metadata = {
        "num_frames": int(num_frames),
        "num_total_unique_knot_positions": int(unique_knot_positions.shape[0]),
        "num_total_internal_knots": int(internal_knot_u.shape[0]),
        "num_future_knots": int(num_future_knots),
        "frames_needing_padding": int(frames_needing_padding),
        "frames_zero_future_knots": int(frames_zero_future_knots),
        "frames_exact_degenerate": int(frames_exact_degenerate),
        "frames_full_context": int(num_frames - frames_needing_padding),
        "coefficient_dimension": int(coefficient_dim),
        "bspline_degree": int(degree),
        "padding_mode": processing.padding_mode,
        "exact_local_spline_definition": (
            "exact local cubic B-spline restriction of the global B-spline on [global_local_start_u, global_local_end_u]; "
            "no refit and no dense resampling"
        ),
        "end_frame_definition": "largest dataset frame index whose frame_u <= global_local_end_u",
        "future_knot_definition": "next N unique future knot positions after current frame u; padded by terminal repeat near episode end",
        "stored_fields": sorted(arrays),
    }
    return arrays, metadata


def select_spline_files(root: Path, requested_episodes: set[int], max_episodes: int | None) -> list[Path]:
    files = sorted(root.rglob("spline.npz"))
    if requested_episodes:
        files = [path for path in files if episode_index_from_path(path) in requested_episodes]
    if max_episodes is not None:
        files = files[:max_episodes]
    return files


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    paths = resolve_paths(config, args)
    processing = resolve_processing(config, args)

    if not paths.dataset_root.exists():
        raise FileNotFoundError(paths.dataset_root)
    if not paths.fitted_spline_root.exists():
        raise FileNotFoundError(paths.fitted_spline_root)

    requested_episodes = set(args.episode)
    spline_files = select_spline_files(paths.fitted_spline_root, requested_episodes, processing.max_episodes)
    if not spline_files:
        raise RuntimeError("No spline files selected for export.")

    paths.output_root.mkdir(parents=True, exist_ok=True)

    dataset_summary: dict[str, Any] = {
        "dataset_root": str(paths.dataset_root),
        "fitted_spline_root": str(paths.fitted_spline_root),
        "output_root": str(paths.output_root),
        "episodes_selected": len(spline_files),
        "num_future_knots": processing.num_future_knots,
        "padding_mode": processing.padding_mode,
        "coefficient_dtype": processing.coefficient_dtype,
        "knot_dtype": processing.knot_dtype,
        "u_dtype": processing.u_dtype,
        "index_dtype": processing.index_dtype,
        "episodes": [],
    }

    total_frames = 0
    total_padding_frames = 0
    total_zero_future_knot_frames = 0
    total_exact_degenerate_frames = 0

    print("Exporting local raw B-spline windows")
    print(f"  fitted_spline_root : {paths.fitted_spline_root}")
    print(f"  output_root        : {paths.output_root}")
    print(f"  num_future_knots   : {processing.num_future_knots}")
    print(f"  episodes_selected  : {len(spline_files)}")

    for path in tqdm(spline_files, desc="episodes", unit="episode"):
        episode_index = episode_index_from_path(path)
        out_dir = episode_output_dir(paths.output_root, episode_index)
        ensure_output_dir(out_dir, overwrite=processing.overwrite)
        arrays, metadata = build_episode_windows(path, processing)

        np.savez_compressed(out_dir / "local_raw_bspline_windows.npz", **arrays)

        if processing.write_episode_metadata_json:
            episode_metadata = {
                "episode_index": episode_index,
                "source_spline_path": str(path),
                "output_npz_path": str((out_dir / "local_raw_bspline_windows.npz").resolve()),
                **metadata,
            }
            (out_dir / "local_raw_bspline_metadata.json").write_text(
                json.dumps(episode_metadata, indent=2) + "\n",
                encoding="utf-8",
            )
        else:
            episode_metadata = {
                "episode_index": episode_index,
                **metadata,
            }

        dataset_summary["episodes"].append(
            {
                "episode_index": episode_index,
                "num_frames": metadata["num_frames"],
                "num_total_unique_knot_positions": metadata["num_total_unique_knot_positions"],
                "frames_needing_padding": metadata["frames_needing_padding"],
                "frames_zero_future_knots": metadata["frames_zero_future_knots"],
                "frames_exact_degenerate": metadata["frames_exact_degenerate"],
                "frames_full_context": metadata["frames_full_context"],
                "bspline_degree": metadata["bspline_degree"],
                "output_dir": str(out_dir),
            }
        )
        total_frames += int(metadata["num_frames"])
        total_padding_frames += int(metadata["frames_needing_padding"])
        total_zero_future_knot_frames += int(metadata["frames_zero_future_knots"])
        total_exact_degenerate_frames += int(metadata["frames_exact_degenerate"])

    dataset_summary["total_frames"] = total_frames
    dataset_summary["total_frames_needing_padding"] = total_padding_frames
    dataset_summary["total_zero_future_knot_frames"] = total_zero_future_knot_frames
    dataset_summary["total_exact_degenerate_frames"] = total_exact_degenerate_frames
    dataset_summary["frames_with_full_context"] = total_frames - total_padding_frames
    dataset_summary["padding_frame_fraction"] = (
        float(total_padding_frames / total_frames) if total_frames > 0 else 0.0
    )

    (paths.output_root / "run_summary.json").write_text(
        json.dumps(dataset_summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
