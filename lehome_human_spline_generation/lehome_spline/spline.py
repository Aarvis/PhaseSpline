from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from tqdm.auto import tqdm

from .utils import atomic_json_dump, atomic_npz


@dataclass
class SplineFitResult:
    frames: int
    knots: int
    compression_ratio: float
    tolerance_satisfied: bool
    fit_status: str
    maximum_epsilon_ratio: float
    limiting_error_type: str
    maximum_latent_epsilon_ratio: float
    maximum_cosine_epsilon_ratio: float
    maximum_state_epsilon_ratio: float
    maximum_latent_rmse: float
    maximum_cosine_distance: float
    maximum_state_rmse: float
    latent_overshoot_percent: float
    latent_undershoot_percent: float
    latent_out_of_range_percent: float
    maximum_latent_overshoot_std: float
    state_overshoot_percent: float
    state_undershoot_percent: float
    state_out_of_range_percent: float
    maximum_state_overshoot_std: float
    iterations: int


def _cosine_distance(reference: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    numerator = np.sum(reference * prediction, axis=-1)
    reference_norm = np.linalg.norm(reference, axis=-1)
    prediction_norm = np.linalg.norm(prediction, axis=-1)
    denominator = reference_norm * prediction_norm
    similarity = numerator / np.maximum(denominator, 1e-12)
    both_zero = (reference_norm < 1e-12) & (prediction_norm < 1e-12)
    similarity = np.where(both_zero, 1.0, similarity)
    return np.clip(1.0 - similarity, 0.0, 2.0)


def _maximum_gap_knots(frame_count: int, maximum_gap: int) -> set[int]:
    return set(range(0, frame_count, max(1, maximum_gap))) | {frame_count - 1}


def _frame_u(timestamps: np.ndarray) -> np.ndarray:
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if len(timestamps) < 2:
        return np.zeros(len(timestamps), dtype=np.float32)
    duration = float(timestamps[-1] - timestamps[0])
    if duration <= 0:
        return np.linspace(0.0, 1.0, len(timestamps), dtype=np.float32)
    return ((timestamps - timestamps[0]) / duration).astype(np.float32)


def _episode_spline_paths(source: Path, embedding_root: Path, spline_root: Path) -> tuple[Path, Path]:
    if source.name == "frame_embeddings.npz":
        relative_episode_dir = source.parent.relative_to(embedding_root)
        episode_output_dir = spline_root / relative_episode_dir
        return episode_output_dir / "spline.npz", episode_output_dir / "spline_metadata.json"
    return spline_root / source.name, spline_root / f"{source.stem}.json"


def _build_interpolator(
    method: str,
    x: np.ndarray,
    y: np.ndarray,
    axis: int,
    cubic_bc_type: str,
):
    method = method.lower()
    if method == "pchip":
        return PchipInterpolator(x, y, axis=axis, extrapolate=False)
    if method == "cubic":
        return CubicSpline(x, y, axis=axis, bc_type=cubic_bc_type, extrapolate=False)
    raise ValueError(f"Unsupported spline method {method!r}. Expected 'pchip' or 'cubic'.")


def _range_violation_stats(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    lower = np.min(reference, axis=0)
    upper = np.max(reference, axis=0)
    scale = np.maximum(upper - lower, 1e-12)
    below = np.maximum(lower - prediction, 0.0) / scale
    above = np.maximum(prediction - upper, 0.0) / scale
    below_mask = below > 0.0
    above_mask = above > 0.0
    out_mask = below_mask | above_mask
    element_count = prediction.size
    return {
        "overshoot_percent": float(100.0 * np.count_nonzero(above_mask) / max(element_count, 1)),
        "undershoot_percent": float(100.0 * np.count_nonzero(below_mask) / max(element_count, 1)),
        "out_of_range_percent": float(100.0 * np.count_nonzero(out_mask) / max(element_count, 1)),
        "maximum_overshoot_std": float(max(float(above.max(initial=0.0)), float(below.max(initial=0.0)))),
    }


def fit_episode_spline(
    mean: np.ndarray,
    state: np.ndarray,
    timestamps: np.ndarray,
    frame_indices: np.ndarray | None,
    latent_normalization_mean: np.ndarray,
    latent_normalization_std: np.ndarray,
    state_normalization_mean: np.ndarray,
    state_normalization_std: np.ndarray,
    config: dict[str, Any],
    progress_description: str | None = None,
) -> tuple[dict[str, np.ndarray], SplineFitResult]:
    mean = np.asarray(mean, dtype=np.float64)
    state = np.asarray(state, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if frame_indices is None:
        frame_indices = np.arange(len(mean), dtype=np.int64)
    frame_indices = np.asarray(frame_indices, dtype=np.int64)
    frame_u = _frame_u(timestamps)
    frame_count = len(mean)
    if frame_count < 2:
        raise ValueError("An episode requires at least two frames for spline fitting")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("Episode timestamps must be strictly increasing")

    latent_standardized = (mean - latent_normalization_mean) / latent_normalization_std
    state_standardized = (state - state_normalization_mean) / state_normalization_std
    latent_transition = np.sqrt(np.mean(np.diff(latent_standardized, axis=0) ** 2, axis=1))
    state_transition = np.sqrt(np.mean(np.diff(state_standardized, axis=0) ** 2, axis=1))
    combined_transition = np.maximum(latent_transition, state_transition)
    transition_quantile = float(config["mandatory_transition_quantile"])
    threshold = float(np.quantile(combined_transition, transition_quantile))
    sharp = np.flatnonzero(combined_transition >= threshold) + 1

    knots = _maximum_gap_knots(frame_count, int(config["maximum_gap_frames"]))
    knots.update({0, frame_count - 1})
    for position in sharp:
        knots.add(int(position - 1))
        knots.add(int(position))
    mandatory = set(knots)
    maximum_knots = min(frame_count, int(config["maximum_knots"]))
    additions = max(1, int(config["knots_per_iteration"]))
    epsilon_latent = float(config["epsilon_latent_rmse"])
    epsilon_cosine = float(config["epsilon_cosine_distance"])
    epsilon_state = float(config["epsilon_state_rmse"])
    spline_method = str(config.get("method", "pchip")).lower()
    cubic_bc_type = str(config.get("cubic_bc_type", "natural"))
    iterations = 0
    progress = tqdm(
        total=maximum_knots,
        initial=len(knots),
        desc=progress_description or "spline/adaptive-knots",
        unit="knot",
        leave=False,
    )

    while True:
        indices = np.asarray(sorted(knots), dtype=np.int64)
        latent_interpolator = _build_interpolator(
            spline_method,
            timestamps[indices],
            latent_standardized[indices],
            axis=0,
            cubic_bc_type=cubic_bc_type,
        )
        state_interpolator = _build_interpolator(
            spline_method,
            timestamps[indices],
            state_standardized[indices],
            axis=0,
            cubic_bc_type=cubic_bc_type,
        )
        reconstructed_latent_standardized = latent_interpolator(timestamps)
        reconstructed_state_standardized = state_interpolator(timestamps)
        reconstructed_mean = (
            reconstructed_latent_standardized * latent_normalization_std + latent_normalization_mean
        )

        latent_rmse = np.sqrt(np.mean((reconstructed_latent_standardized - latent_standardized) ** 2, axis=1))
        cosine_distance = _cosine_distance(mean, reconstructed_mean)
        state_rmse = np.sqrt(np.mean((reconstructed_state_standardized - state_standardized) ** 2, axis=1))
        severity = np.maximum.reduce(
            [
                latent_rmse / max(epsilon_latent, 1e-12),
                cosine_distance / max(epsilon_cosine, 1e-12),
                state_rmse / max(epsilon_state, 1e-12),
            ]
        )
        maximum_severity = float(severity.max())
        progress.set_postfix(max_error=f"{maximum_severity:.3f}")
        if maximum_severity <= 1.0:
            status = "epsilon_met"
            break
        if len(knots) >= maximum_knots:
            status = "maximum_knots_reached"
            break

        candidates = np.argsort(severity)[::-1]
        new_knots: list[int] = []
        for candidate in candidates:
            candidate = int(candidate)
            if candidate not in knots:
                new_knots.append(candidate)
                if len(new_knots) >= additions or len(knots) + len(new_knots) >= maximum_knots:
                    break
        if not new_knots:
            status = "no_new_knots"
            break
        knots.update(new_knots)
        progress.update(len(new_knots))
        iterations += 1
    progress.close()

    indices = np.asarray(sorted(knots), dtype=np.int64)
    mandatory_mask = np.asarray([int(index) in mandatory for index in indices], dtype=np.bool_)
    latent_range_stats = _range_violation_stats(latent_standardized, reconstructed_latent_standardized)
    state_range_stats = _range_violation_stats(state_standardized, reconstructed_state_standardized)
    maximum_latent_ratio = float(latent_rmse.max() / max(epsilon_latent, 1e-12))
    maximum_cosine_ratio = float(cosine_distance.max() / max(epsilon_cosine, 1e-12))
    maximum_state_ratio = float(state_rmse.max() / max(epsilon_state, 1e-12))
    limiting_error_type = max(
        [
            ("latent", maximum_latent_ratio),
            ("cosine", maximum_cosine_ratio),
            ("state", maximum_state_ratio),
        ],
        key=lambda item: item[1],
    )[0]
    payload = {
        "frame_indices": frame_indices,
        "frame_timestamps": timestamps.astype(np.float64),
        "frame_u": frame_u,
        "u": frame_u,
        "knot_timestamps": timestamps[indices],
        "knot_u": frame_u[indices],
        "knot_frame_indices": frame_indices[indices],
        "knot_source_positions": indices,
        "knot_embeddings": mean[indices].astype(np.float16),
        "knot_states": state[indices].astype(np.float32),
        "mandatory_knot_mask": mandatory_mask,
        "frame_latent_rmse": latent_rmse.astype(np.float32),
        "frame_cosine_distance": cosine_distance.astype(np.float32),
        "frame_state_rmse": state_rmse.astype(np.float32),
        "frame_epsilon_ratio": severity.astype(np.float32),
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
    }
    result = SplineFitResult(
        frames=frame_count,
        knots=len(indices),
        compression_ratio=frame_count / len(indices),
        tolerance_satisfied=status == "epsilon_met",
        fit_status=status,
        maximum_epsilon_ratio=float(maximum_severity),
        limiting_error_type=limiting_error_type,
        maximum_latent_epsilon_ratio=maximum_latent_ratio,
        maximum_cosine_epsilon_ratio=maximum_cosine_ratio,
        maximum_state_epsilon_ratio=maximum_state_ratio,
        maximum_latent_rmse=float(latent_rmse.max()),
        maximum_cosine_distance=float(cosine_distance.max()),
        maximum_state_rmse=float(state_rmse.max()),
        latent_overshoot_percent=latent_range_stats["overshoot_percent"],
        latent_undershoot_percent=latent_range_stats["undershoot_percent"],
        latent_out_of_range_percent=latent_range_stats["out_of_range_percent"],
        maximum_latent_overshoot_std=latent_range_stats["maximum_overshoot_std"],
        state_overshoot_percent=state_range_stats["overshoot_percent"],
        state_undershoot_percent=state_range_stats["undershoot_percent"],
        state_out_of_range_percent=state_range_stats["out_of_range_percent"],
        maximum_state_overshoot_std=state_range_stats["maximum_overshoot_std"],
        iterations=iterations,
    )
    return payload, result


def fit_all_splines(
    config: dict[str, Any], overwrite: bool = False, max_episodes: int | None = None
) -> Path:
    embedding_root = Path(config["output"]["embeddings_dir"]).expanduser().resolve()
    spline_root = Path(config["output"]["splines_dir"]).expanduser().resolve()
    spline_root.mkdir(parents=True, exist_ok=True)
    normalization_path = embedding_root / "embedding_normalization.npz"
    if not normalization_path.is_file():
        raise FileNotFoundError(f"Missing embedding normalization: {normalization_path}")
    with np.load(normalization_path, allow_pickle=False) as normalization:
        latent_mean = normalization["mean"].astype(np.float64)
        latent_std = normalization["std"].astype(np.float64)
    state_normalization_path = Path(config["output"]["root"]) / "state_normalization.json"
    with state_normalization_path.open("r", encoding="utf-8") as handle:
        state_normalization = json.load(handle)
    state_mean = np.asarray(state_normalization["mean"], dtype=np.float64)
    state_std = np.asarray(state_normalization["std"], dtype=np.float64)

    sources = sorted(embedding_root.glob("chunk-*/episode_*/frame_embeddings.npz"))
    if not sources:
        sources = sorted(embedding_root.glob("episode_*.npz"))
    if max_episodes is None:
        configured_max_episodes = config["spline"].get("max_episodes")
        if configured_max_episodes is not None:
            max_episodes = int(configured_max_episodes)
    if max_episodes:
        sources = sources[:max_episodes]
    summaries: list[dict[str, Any]] = []
    for source in tqdm(sources, desc="spline/episodes", unit="episode"):
        destination, metadata_destination = _episode_spline_paths(source, embedding_root, spline_root)
        if destination.is_file() and metadata_destination.is_file() and not overwrite:
            summaries.append(json.loads(metadata_destination.read_text(encoding="utf-8")))
            continue
        with np.load(source, allow_pickle=False) as data:
            payload, result = fit_episode_spline(
                data["mean"],
                data["state"],
                data["timestamps"],
                data["frame_indices"],
                latent_mean,
                latent_std,
                state_mean,
                state_std,
                config["spline"],
                progress_description=f"spline/{source.stem}",
            )
        atomic_npz(destination, compressed=bool(config["spline"]["compressed"]), **payload)
        metadata = asdict(result)
        metadata["source"] = str(source)
        metadata["output"] = str(destination)
        metadata["method"] = str(config["spline"].get("method", "pchip")).lower()
        if metadata["method"] == "cubic":
            metadata["cubic_bc_type"] = str(config["spline"].get("cubic_bc_type", "natural"))
        metadata["parameterization"] = "u = normalized episode timestamp, [0, 1]"
        atomic_json_dump(metadata, metadata_destination)
        summaries.append(metadata)

    atomic_json_dump(
        {
            "episodes": len(summaries),
            "tolerance_satisfied": sum(bool(item["tolerance_satisfied"]) for item in summaries),
            "total_frames": sum(int(item["frames"]) for item in summaries),
            "total_knots": sum(int(item["knots"]) for item in summaries),
            "maximum_epsilon_ratio": max(
                (float(item.get("maximum_epsilon_ratio", 0.0)) for item in summaries), default=0.0
            ),
            "maximum_latent_rmse": max(
                (float(item.get("maximum_latent_rmse", 0.0)) for item in summaries), default=0.0
            ),
            "maximum_cosine_distance": max(
                (float(item.get("maximum_cosine_distance", 0.0)) for item in summaries), default=0.0
            ),
            "maximum_state_rmse": max(
                (float(item.get("maximum_state_rmse", 0.0)) for item in summaries), default=0.0
            ),
            "mean_latent_out_of_range_percent": (
                sum(float(item.get("latent_out_of_range_percent", 0.0)) * int(item["frames"]) for item in summaries)
                / max(sum(int(item["frames"]) for item in summaries), 1)
            ),
            "maximum_latent_out_of_range_percent": max(
                (float(item.get("latent_out_of_range_percent", 0.0)) for item in summaries), default=0.0
            ),
            "maximum_latent_overshoot_std": max(
                (float(item.get("maximum_latent_overshoot_std", 0.0)) for item in summaries), default=0.0
            ),
            "mean_state_out_of_range_percent": (
                sum(float(item.get("state_out_of_range_percent", 0.0)) * int(item["frames"]) for item in summaries)
                / max(sum(int(item["frames"]) for item in summaries), 1)
            ),
            "maximum_state_out_of_range_percent": max(
                (float(item.get("state_out_of_range_percent", 0.0)) for item in summaries), default=0.0
            ),
            "maximum_state_overshoot_std": max(
                (float(item.get("maximum_state_overshoot_std", 0.0)) for item in summaries), default=0.0
            ),
            "episode_results": summaries,
        },
        spline_root / "run_summary.json",
    )
    atomic_json_dump(
        {
            "episodes": len(summaries),
            "total_frames": sum(int(item["frames"]) for item in summaries),
            "total_knots": sum(int(item["knots"]) for item in summaries),
            "maximum_epsilon_ratio": max(
                (float(item.get("maximum_epsilon_ratio", 0.0)) for item in summaries), default=0.0
            ),
            "maximum_latent_rmse": max(
                (float(item.get("maximum_latent_rmse", 0.0)) for item in summaries), default=0.0
            ),
            "maximum_cosine_distance": max(
                (float(item.get("maximum_cosine_distance", 0.0)) for item in summaries), default=0.0
            ),
            "maximum_state_rmse": max(
                (float(item.get("maximum_state_rmse", 0.0)) for item in summaries), default=0.0
            ),
            "mean_latent_out_of_range_percent": (
                sum(float(item.get("latent_out_of_range_percent", 0.0)) * int(item["frames"]) for item in summaries)
                / max(sum(int(item["frames"]) for item in summaries), 1)
            ),
            "maximum_latent_out_of_range_percent": max(
                (float(item.get("latent_out_of_range_percent", 0.0)) for item in summaries), default=0.0
            ),
            "maximum_latent_overshoot_std": max(
                (float(item.get("maximum_latent_overshoot_std", 0.0)) for item in summaries), default=0.0
            ),
            "mean_state_out_of_range_percent": (
                sum(float(item.get("state_out_of_range_percent", 0.0)) * int(item["frames"]) for item in summaries)
                / max(sum(int(item["frames"]) for item in summaries), 1)
            ),
            "maximum_state_out_of_range_percent": max(
                (float(item.get("state_out_of_range_percent", 0.0)) for item in summaries), default=0.0
            ),
            "maximum_state_overshoot_std": max(
                (float(item.get("maximum_state_overshoot_std", 0.0)) for item in summaries), default=0.0
            ),
            "spline_root": str(spline_root),
            "embedding_root": str(embedding_root),
            "episode_layout": "chunk-XXX/episode_XXXXXX/spline.npz",
            "parameterization": "per-frame frame_u / knot_u in [0, 1]",
            "method": str(config["spline"].get("method", "pchip")).lower(),
            "cubic_bc_type": str(config["spline"].get("cubic_bc_type", "natural")),
        },
        spline_root / "splines_summary.json",
    )
    return spline_root
