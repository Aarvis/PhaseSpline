from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import BSpline, make_lsq_spline
from tqdm.auto import tqdm

from .config import save_resolved_config
from .spline import _cosine_distance, _frame_u, _range_violation_stats
from .utils import atomic_json_dump, atomic_npz


@dataclass
class BSplineFitResult:
    frames: int
    degree: int
    num_internal_knots: int
    num_knots_total: int
    num_control_points: int
    compression_ratio: float
    tolerance_satisfied: bool
    fit_status: str
    maximum_epsilon_ratio: float
    limiting_error_type: str
    maximum_latent_epsilon_ratio: float
    maximum_cosine_epsilon_ratio: float
    maximum_state_epsilon_ratio: float
    maximum_action_epsilon_ratio: float
    maximum_latent_rmse: float
    mean_latent_rmse: float
    p95_latent_rmse: float
    p99_latent_rmse: float
    maximum_cosine_distance: float
    mean_cosine_distance: float
    p95_cosine_distance: float
    p99_cosine_distance: float
    maximum_state_rmse: float
    mean_state_rmse: float
    p95_state_rmse: float
    p99_state_rmse: float
    maximum_action_rmse: float
    mean_action_rmse: float
    p95_action_rmse: float
    p99_action_rmse: float
    latent_overshoot_percent: float
    latent_undershoot_percent: float
    latent_out_of_range_percent: float
    maximum_latent_overshoot_std: float
    state_overshoot_percent: float
    state_undershoot_percent: float
    state_out_of_range_percent: float
    maximum_state_overshoot_std: float
    action_overshoot_percent: float
    action_undershoot_percent: float
    action_out_of_range_percent: float
    maximum_action_overshoot_std: float
    iterations: int


def _percentile_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "min": 0.0,
            "p1": 0.0,
            "p5": 0.0,
            "p10": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p80": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p98": 0.0,
            "p99": 0.0,
            "p99_9": 0.0,
            "p99_99": 0.0,
            "max": 0.0,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "p1": float(np.percentile(arr, 1)),
        "p5": float(np.percentile(arr, 5)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p80": float(np.percentile(arr, 80)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p98": float(np.percentile(arr, 98)),
        "p99": float(np.percentile(arr, 99)),
        "p99_9": float(np.percentile(arr, 99.9)),
        "p99_99": float(np.percentile(arr, 99.99)),
        "max": float(np.max(arr)),
    }


def _episode_bspline_paths(source: Path, embedding_root: Path, spline_root: Path) -> tuple[Path, Path]:
    if source.name == "frame_embeddings.npz":
        relative_episode_dir = source.parent.relative_to(embedding_root)
        episode_output_dir = spline_root / relative_episode_dir
        return episode_output_dir / "spline.npz", episode_output_dir / "spline_metadata.json"
    return spline_root / source.name, spline_root / f"{source.stem}.json"


def _make_clamped_knots(x: np.ndarray, internal_indices: list[int], degree: int) -> np.ndarray:
    internal = [float(x[i]) for i in sorted(set(internal_indices)) if 0 < i < len(x) - 1]
    start = [float(x[0])] * (degree + 1)
    end = [float(x[-1])] * (degree + 1)
    return np.asarray(start + internal + end, dtype=np.float64)


def _initial_internal_indices(frame_count: int, count: int, degree: int) -> list[int]:
    max_internal = max(0, frame_count - degree - 1)
    count = min(max(0, count), max_internal)
    if count == 0 or frame_count <= 2:
        return []
    values = np.linspace(1, frame_count - 2, count + 2, dtype=np.float64)[1:-1]
    return sorted({int(round(value)) for value in values if 0 < int(round(value)) < frame_count - 1})


def _fit_once(
    x: np.ndarray,
    y: np.ndarray,
    internal_indices: list[int],
    degree: int,
    endpoint_weight: float,
) -> BSpline:
    knots = _make_clamped_knots(x, internal_indices, degree)
    weights = np.ones(len(x), dtype=np.float64)
    weights[0] = endpoint_weight
    weights[-1] = endpoint_weight
    return make_lsq_spline(x, y, knots, k=degree, w=weights)


def _choose_next_knot_index(errors: np.ndarray, existing: set[int], min_spacing: int) -> int | None:
    for candidate in np.argsort(errors)[::-1]:
        candidate = int(candidate)
        if candidate <= 0 or candidate >= len(errors) - 1:
            continue
        if candidate in existing:
            continue
        if any(abs(candidate - old) < min_spacing for old in existing):
            continue
        return candidate
    return None


def _resolve_bspline_root(config: dict[str, Any], save_mode: str | None) -> tuple[Path, str]:
    mode = str(save_mode or config["bspline"].get("save_mode", "dataset")).lower()
    if mode not in {"dataset", "external"}:
        raise ValueError(f"Unsupported save mode {mode!r}. Expected 'dataset' or 'external'.")
    if mode == "dataset":
        root_value = config["output"].get("bspline_dataset_dir")
        if not root_value:
            raise ValueError("output.bspline_dataset_dir is required for save_mode=dataset")
    else:
        root_value = config["output"].get("bspline_external_dir")
        if not root_value:
            raise ValueError("output.bspline_external_dir is required for save_mode=external")
    root = Path(root_value).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root, mode


def fit_episode_bspline(
    mean: np.ndarray,
    state: np.ndarray,
    action: np.ndarray,
    timestamps: np.ndarray,
    frame_indices: np.ndarray | None,
    latent_normalization_mean: np.ndarray,
    latent_normalization_std: np.ndarray,
    state_normalization_mean: np.ndarray,
    state_normalization_std: np.ndarray,
    action_normalization_mean: np.ndarray,
    action_normalization_std: np.ndarray,
    config: dict[str, Any],
    progress_description: str | None = None,
) -> tuple[dict[str, np.ndarray], BSplineFitResult]:
    mean = np.asarray(mean, dtype=np.float64)
    state = np.asarray(state, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if frame_indices is None:
        frame_indices = np.arange(len(mean), dtype=np.int64)
    frame_indices = np.asarray(frame_indices, dtype=np.int64)
    frame_u = _frame_u(timestamps).astype(np.float64)
    frame_count = len(mean)
    if frame_count < 2:
        raise ValueError("An episode requires at least two frames for spline fitting")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("Episode timestamps must be strictly increasing")

    latent_standardized = (mean - latent_normalization_mean) / latent_normalization_std
    state_standardized = (state - state_normalization_mean) / state_normalization_std
    action_standardized = (action - action_normalization_mean) / action_normalization_std
    epsilon_latent = float(config["epsilon_latent_rmse"])
    epsilon_cosine = float(config["epsilon_cosine_distance"])
    epsilon_state = float(config["epsilon_state_rmse"])
    epsilon_action = float(config["epsilon_action_rmse"])
    degree = min(int(config.get("degree", 3)), frame_count - 1)
    if degree < 1:
        raise ValueError(f"Need at least 2 samples to fit a spline, got {frame_count}")
    max_possible_internal = max(0, frame_count - degree - 1)
    max_internal = min(int(config["max_internal_knots"]), max_possible_internal)
    internal = _initial_internal_indices(frame_count, int(config["initial_internal_knots"]), degree)
    internal = internal[:max_internal]
    endpoint_weight = float(config.get("endpoint_weight", 1000.0))
    min_spacing = max(1, int(config.get("min_knot_spacing_frames", 1)))

    progress = tqdm(
        total=max_internal,
        initial=len(internal),
        desc=progress_description or "bspline/adaptive-knots",
        unit="iknot",
        leave=False,
    )

    latent_spline: BSpline | None = None
    state_spline: BSpline | None = None
    action_spline: BSpline | None = None
    latent_rmse: np.ndarray | None = None
    cosine_distance: np.ndarray | None = None
    state_rmse: np.ndarray | None = None
    action_rmse: np.ndarray | None = None
    reconstructed_latent_standardized: np.ndarray | None = None
    reconstructed_state_standardized: np.ndarray | None = None
    reconstructed_action_standardized: np.ndarray | None = None
    maximum_severity = float("nan")
    status = "maximum_internal_knots_reached"
    iterations = 0

    while True:
        try:
            latent_spline = _fit_once(frame_u, latent_standardized, internal, degree, endpoint_weight)
            state_spline = _fit_once(frame_u, state_standardized, internal, degree, endpoint_weight)
            action_spline = _fit_once(frame_u, action_standardized, internal, degree, endpoint_weight)

            reconstructed_latent_standardized = np.asarray(latent_spline(frame_u), dtype=np.float64)
            reconstructed_state_standardized = np.asarray(state_spline(frame_u), dtype=np.float64)
            reconstructed_action_standardized = np.asarray(action_spline(frame_u), dtype=np.float64)
            reconstructed_mean = (
                reconstructed_latent_standardized * latent_normalization_std + latent_normalization_mean
            )

            latent_rmse = np.sqrt(np.mean((reconstructed_latent_standardized - latent_standardized) ** 2, axis=1))
            cosine_distance = _cosine_distance(mean, reconstructed_mean)
            state_rmse = np.sqrt(np.mean((reconstructed_state_standardized - state_standardized) ** 2, axis=1))
            action_rmse = np.sqrt(np.mean((reconstructed_action_standardized - action_standardized) ** 2, axis=1))
            severity = np.maximum.reduce(
                [
                    latent_rmse / max(epsilon_latent, 1e-12),
                    cosine_distance / max(epsilon_cosine, 1e-12),
                    state_rmse / max(epsilon_state, 1e-12),
                    action_rmse / max(epsilon_action, 1e-12),
                ]
            )
            maximum_severity = float(np.max(severity))
            progress.set_postfix(max_error=f"{maximum_severity:.3f}")
            if maximum_severity <= 1.0:
                status = "epsilon_met"
                break
            if len(internal) >= max_internal:
                status = "maximum_internal_knots_reached"
                break
            next_idx = _choose_next_knot_index(severity, set(internal), min_spacing)
            if next_idx is None:
                status = "no_valid_knot_candidate"
                break
            previous_count = len(internal)
            internal.append(next_idx)
            internal = sorted(set(internal))
            if len(internal) > previous_count:
                progress.update(len(internal) - previous_count)
                iterations += 1
        except Exception:
            if internal:
                internal = internal[:-1]
                status = "fit_failed_after_knot_removal"
                continue
            raise
    progress.close()

    if (
        latent_spline is None
        or state_spline is None
        or action_spline is None
        or latent_rmse is None
        or cosine_distance is None
        or state_rmse is None
        or action_rmse is None
        or reconstructed_latent_standardized is None
        or reconstructed_state_standardized is None
        or reconstructed_action_standardized is None
    ):
        raise RuntimeError("B-spline fitting failed before producing a valid spline.")

    internal_indices = np.asarray(sorted(internal), dtype=np.int64)
    latent_range_stats = _range_violation_stats(latent_standardized, reconstructed_latent_standardized)
    state_range_stats = _range_violation_stats(state_standardized, reconstructed_state_standardized)
    action_range_stats = _range_violation_stats(action_standardized, reconstructed_action_standardized)
    maximum_latent_ratio = float(np.max(latent_rmse) / max(epsilon_latent, 1e-12))
    maximum_cosine_ratio = float(np.max(cosine_distance) / max(epsilon_cosine, 1e-12))
    maximum_state_ratio = float(np.max(state_rmse) / max(epsilon_state, 1e-12))
    maximum_action_ratio = float(np.max(action_rmse) / max(epsilon_action, 1e-12))
    limiting_error_type = max(
        [
            ("latent", maximum_latent_ratio),
            ("cosine", maximum_cosine_ratio),
            ("state", maximum_state_ratio),
            ("action", maximum_action_ratio),
        ],
        key=lambda item: item[1],
    )[0]

    coefficients = np.asarray(latent_spline.c, dtype=np.float32)
    state_coefficients = np.asarray(state_spline.c, dtype=np.float32)
    action_coefficients = np.asarray(action_spline.c, dtype=np.float32)
    num_control_points = int(coefficients.shape[0])
    severity_payload = np.maximum.reduce(
        [
            latent_rmse / max(epsilon_latent, 1e-12),
            cosine_distance / max(epsilon_cosine, 1e-12),
            state_rmse / max(epsilon_state, 1e-12),
            action_rmse / max(epsilon_action, 1e-12),
        ]
    ).astype(np.float32)
    payload = {
        "global_knots": np.asarray(latent_spline.t, dtype=np.float64),
        "global_coefficients": coefficients,
        "global_degree": np.asarray([degree], dtype=np.int64),
        "global_state_coefficients": state_coefficients,
        "global_action_coefficients": action_coefficients,
        "frame_indices": frame_indices,
        "frame_timestamps": timestamps.astype(np.float64),
        "frame_u": frame_u.astype(np.float32),
        "u": frame_u.astype(np.float32),
        "internal_knot_frame_indices": frame_indices[internal_indices],
        "internal_knot_source_positions": internal_indices,
        "internal_knot_u": frame_u[internal_indices].astype(np.float64),
        "target_dim": np.asarray([mean.shape[1]], dtype=np.int64),
        "state_dim": np.asarray([state.shape[1]], dtype=np.int64),
        "action_dim": np.asarray([action.shape[1]], dtype=np.int64),
        "frame_latent_rmse": latent_rmse.astype(np.float32),
        "frame_cosine_distance": cosine_distance.astype(np.float32),
        "frame_state_rmse": state_rmse.astype(np.float32),
        "frame_action_rmse": action_rmse.astype(np.float32),
        "frame_epsilon_ratio": severity_payload,
        "frame_latent_out_of_range_fraction": np.mean(
            (reconstructed_latent_standardized < np.min(latent_standardized, axis=0))
            | (reconstructed_latent_standardized > np.max(latent_standardized, axis=0)),
            axis=1,
        ).astype(np.float32),
        "frame_state_out_of_range_fraction": np.mean(
            (reconstructed_state_standardized < np.min(state_standardized, axis=0))
            | (reconstructed_state_standardized > np.max(state_standardized, axis=0)),
            axis=1,
        ).astype(np.float32),
        "frame_action_out_of_range_fraction": np.mean(
            (reconstructed_action_standardized < np.min(action_standardized, axis=0))
            | (reconstructed_action_standardized > np.max(action_standardized, axis=0)),
            axis=1,
        ).astype(np.float32),
    }
    result = BSplineFitResult(
        frames=frame_count,
        degree=degree,
        num_internal_knots=int(len(internal_indices)),
        num_knots_total=int(len(latent_spline.t)),
        num_control_points=num_control_points,
        compression_ratio=float(frame_count / max(num_control_points, 1)),
        tolerance_satisfied=status == "epsilon_met",
        fit_status=status,
        maximum_epsilon_ratio=maximum_severity,
        limiting_error_type=limiting_error_type,
        maximum_latent_epsilon_ratio=maximum_latent_ratio,
        maximum_cosine_epsilon_ratio=maximum_cosine_ratio,
        maximum_state_epsilon_ratio=maximum_state_ratio,
        maximum_action_epsilon_ratio=maximum_action_ratio,
        maximum_latent_rmse=float(np.max(latent_rmse)),
        mean_latent_rmse=float(np.mean(latent_rmse)),
        p95_latent_rmse=float(np.percentile(latent_rmse, 95)),
        p99_latent_rmse=float(np.percentile(latent_rmse, 99)),
        maximum_cosine_distance=float(np.max(cosine_distance)),
        mean_cosine_distance=float(np.mean(cosine_distance)),
        p95_cosine_distance=float(np.percentile(cosine_distance, 95)),
        p99_cosine_distance=float(np.percentile(cosine_distance, 99)),
        maximum_state_rmse=float(np.max(state_rmse)),
        mean_state_rmse=float(np.mean(state_rmse)),
        p95_state_rmse=float(np.percentile(state_rmse, 95)),
        p99_state_rmse=float(np.percentile(state_rmse, 99)),
        maximum_action_rmse=float(np.max(action_rmse)),
        mean_action_rmse=float(np.mean(action_rmse)),
        p95_action_rmse=float(np.percentile(action_rmse, 95)),
        p99_action_rmse=float(np.percentile(action_rmse, 99)),
        latent_overshoot_percent=latent_range_stats["overshoot_percent"],
        latent_undershoot_percent=latent_range_stats["undershoot_percent"],
        latent_out_of_range_percent=latent_range_stats["out_of_range_percent"],
        maximum_latent_overshoot_std=latent_range_stats["maximum_overshoot_std"],
        state_overshoot_percent=state_range_stats["overshoot_percent"],
        state_undershoot_percent=state_range_stats["undershoot_percent"],
        state_out_of_range_percent=state_range_stats["out_of_range_percent"],
        maximum_state_overshoot_std=state_range_stats["maximum_overshoot_std"],
        action_overshoot_percent=action_range_stats["overshoot_percent"],
        action_undershoot_percent=action_range_stats["undershoot_percent"],
        action_out_of_range_percent=action_range_stats["out_of_range_percent"],
        maximum_action_overshoot_std=action_range_stats["maximum_overshoot_std"],
        iterations=iterations,
    )
    return payload, result


def fit_all_bspline_splines(
    config: dict[str, Any],
    overwrite: bool = False,
    max_episodes: int | None = None,
    save_mode: str | None = None,
) -> Path:
    embedding_root = Path(config["output"]["embeddings_dir"]).expanduser().resolve()
    spline_root, resolved_save_mode = _resolve_bspline_root(config, save_mode)
    normalization_path = embedding_root / "embedding_normalization.npz"
    if not normalization_path.is_file():
        raise FileNotFoundError(f"Missing embedding normalization: {normalization_path}")
    with np.load(normalization_path, allow_pickle=False) as normalization:
        latent_mean = normalization["mean"].astype(np.float64)
        latent_std = normalization["std"].astype(np.float64)
    state_action_normalization_path = Path(config["output"]["root"]) / "state_action_normalization.json"
    with state_action_normalization_path.open("r", encoding="utf-8") as handle:
        state_action_normalization = json.load(handle)
    state_mean = np.asarray(state_action_normalization["state_mean"], dtype=np.float64)
    state_std = np.asarray(state_action_normalization["state_std"], dtype=np.float64)
    action_mean = np.asarray(state_action_normalization["action_mean"], dtype=np.float64)
    action_std = np.asarray(state_action_normalization["action_std"], dtype=np.float64)

    sources = sorted(embedding_root.glob("chunk-*/episode_*/frame_embeddings.npz"))
    if not sources:
        sources = sorted(embedding_root.glob("episode_*.npz"))
    if max_episodes is None:
        configured_max = config["bspline"].get("max_episodes")
        if configured_max is not None:
            max_episodes = int(configured_max)
    if max_episodes:
        sources = sources[:max_episodes]

    save_resolved_config(config, spline_root / "resolved_config.yaml")
    summaries: list[dict[str, Any]] = []
    for source in tqdm(sources, desc="bspline/episodes", unit="episode"):
        destination, metadata_destination = _episode_bspline_paths(source, embedding_root, spline_root)
        if destination.is_file() and metadata_destination.is_file() and not overwrite:
            summaries.append(json.loads(metadata_destination.read_text(encoding="utf-8")))
            continue
        with np.load(source, allow_pickle=False) as data:
            payload, result = fit_episode_bspline(
                data["mean"],
                data["state"],
                data["action"],
                data["timestamps"],
                data["frame_indices"],
                latent_mean,
                latent_std,
                state_mean,
                state_std,
                action_mean,
                action_std,
                config["bspline"],
                progress_description=f"bspline/{source.stem}",
            )
        atomic_npz(destination, compressed=bool(config["bspline"].get("compressed", True)), **payload)
        metadata = asdict(result)
        metadata["source"] = str(source)
        metadata["output"] = str(destination)
        metadata["method"] = "cubic_bspline"
        metadata["representation"] = "global knot vector + coefficients"
        metadata["parameterization"] = "u = normalized episode timestamp, [0, 1]"
        metadata["save_mode"] = resolved_save_mode
        atomic_json_dump(metadata, metadata_destination)
        summaries.append(metadata)

    total_frames = sum(int(item["frames"]) for item in summaries)
    maximum_latent_rmse_percentiles = _percentile_summary(
        [float(item.get("maximum_latent_rmse", 0.0)) for item in summaries]
    )
    maximum_latent_epsilon_ratio_percentiles = _percentile_summary(
        [float(item.get("maximum_latent_epsilon_ratio", 0.0)) for item in summaries]
    )
    run_summary = {
        "episodes": len(summaries),
        "tolerance_satisfied": sum(bool(item["tolerance_satisfied"]) for item in summaries),
        "total_frames": total_frames,
        "total_internal_knots": sum(int(item["num_internal_knots"]) for item in summaries),
        "total_knots": sum(int(item["num_internal_knots"]) for item in summaries),
        "total_knots_vector_entries": sum(int(item["num_knots_total"]) for item in summaries),
        "total_control_points": sum(int(item["num_control_points"]) for item in summaries),
        "maximum_epsilon_ratio": max((float(item.get("maximum_epsilon_ratio", 0.0)) for item in summaries), default=0.0),
        "maximum_latent_rmse": max((float(item.get("maximum_latent_rmse", 0.0)) for item in summaries), default=0.0),
        "maximum_cosine_distance": max(
            (float(item.get("maximum_cosine_distance", 0.0)) for item in summaries), default=0.0
        ),
        "maximum_state_rmse": max((float(item.get("maximum_state_rmse", 0.0)) for item in summaries), default=0.0),
        "maximum_action_rmse": max((float(item.get("maximum_action_rmse", 0.0)) for item in summaries), default=0.0),
        "mean_latent_out_of_range_percent": (
            sum(float(item.get("latent_out_of_range_percent", 0.0)) * int(item["frames"]) for item in summaries)
            / max(total_frames, 1)
        ),
        "maximum_latent_out_of_range_percent": max(
            (float(item.get("latent_out_of_range_percent", 0.0)) for item in summaries), default=0.0
        ),
        "maximum_latent_overshoot_std": max(
            (float(item.get("maximum_latent_overshoot_std", 0.0)) for item in summaries), default=0.0
        ),
        "mean_state_out_of_range_percent": (
            sum(float(item.get("state_out_of_range_percent", 0.0)) * int(item["frames"]) for item in summaries)
            / max(total_frames, 1)
        ),
        "maximum_state_out_of_range_percent": max(
            (float(item.get("state_out_of_range_percent", 0.0)) for item in summaries), default=0.0
        ),
        "maximum_state_overshoot_std": max(
            (float(item.get("maximum_state_overshoot_std", 0.0)) for item in summaries), default=0.0
        ),
        "mean_action_out_of_range_percent": (
            sum(float(item.get("action_out_of_range_percent", 0.0)) * int(item["frames"]) for item in summaries)
            / max(total_frames, 1)
        ),
        "maximum_action_out_of_range_percent": max(
            (float(item.get("action_out_of_range_percent", 0.0)) for item in summaries), default=0.0
        ),
        "maximum_action_overshoot_std": max(
            (float(item.get("maximum_action_overshoot_std", 0.0)) for item in summaries), default=0.0
        ),
        "maximum_latent_rmse_percentiles": maximum_latent_rmse_percentiles,
        "maximum_latent_epsilon_ratio_percentiles": maximum_latent_epsilon_ratio_percentiles,
        "save_mode": resolved_save_mode,
        "episode_results": summaries,
    }
    atomic_json_dump(run_summary, spline_root / "run_summary.json")
    atomic_json_dump(
        {
            "episodes": len(summaries),
            "total_frames": total_frames,
            "total_internal_knots": run_summary["total_internal_knots"],
            "total_knots_vector_entries": run_summary["total_knots_vector_entries"],
            "total_control_points": run_summary["total_control_points"],
            "maximum_epsilon_ratio": run_summary["maximum_epsilon_ratio"],
            "maximum_latent_rmse": run_summary["maximum_latent_rmse"],
            "maximum_cosine_distance": run_summary["maximum_cosine_distance"],
            "maximum_state_rmse": run_summary["maximum_state_rmse"],
            "maximum_action_rmse": run_summary["maximum_action_rmse"],
            "mean_latent_out_of_range_percent": run_summary["mean_latent_out_of_range_percent"],
            "maximum_latent_out_of_range_percent": run_summary["maximum_latent_out_of_range_percent"],
            "maximum_latent_overshoot_std": run_summary["maximum_latent_overshoot_std"],
            "mean_state_out_of_range_percent": run_summary["mean_state_out_of_range_percent"],
            "maximum_state_out_of_range_percent": run_summary["maximum_state_out_of_range_percent"],
            "maximum_state_overshoot_std": run_summary["maximum_state_overshoot_std"],
            "mean_action_out_of_range_percent": run_summary["mean_action_out_of_range_percent"],
            "maximum_action_out_of_range_percent": run_summary["maximum_action_out_of_range_percent"],
            "maximum_action_overshoot_std": run_summary["maximum_action_overshoot_std"],
            "maximum_latent_rmse_percentiles": maximum_latent_rmse_percentiles,
            "maximum_latent_epsilon_ratio_percentiles": maximum_latent_epsilon_ratio_percentiles,
            "spline_root": str(spline_root),
            "embedding_root": str(embedding_root),
            "episode_layout": "chunk-XXX/episode_XXXXXX/spline.npz",
            "parameterization": "per-frame frame_u in [0, 1]",
            "method": "cubic_bspline",
            "representation": "global knot vector + coefficients",
            "save_mode": resolved_save_mode,
        },
        spline_root / "splines_summary.json",
    )
    return spline_root
