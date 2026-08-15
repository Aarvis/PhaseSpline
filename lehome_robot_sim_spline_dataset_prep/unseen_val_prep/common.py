from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


EPSILON = 1e-8
INVALID_REASON_TO_CODE = {
    "ok": 0,
    "missing_segment": 1,
    "missing_frame_u": 2,
    "degenerate_interval": 3,
    "zero_covered_frames": 4,
    "malformed_interval": 5,
}
INVALID_CODE_TO_REASON = {value: key for key, value in INVALID_REASON_TO_CODE.items()}
SUMMARY_PERCENTILES = (1, 2, 5, 15, 25, 50, 75, 85, 95, 99, 99.9)


@dataclass(frozen=True)
class PairConfig:
    pair_id: str
    category_id: str
    robot_episode_index: int
    human_episode_index: int
    robot_episode_dir: Path
    human_episode_dir: Path


@dataclass(frozen=True)
class PathsConfig:
    output_root: Path


@dataclass(frozen=True)
class ProcessingConfig:
    overwrite: bool
    index_dtype: str
    progress_dtype: str
    u_dtype: str
    num_pairings_per_frame: int
    write_episode_metadata_json: bool


@dataclass(frozen=True)
class AnnotationIndex:
    episode_index: int
    category_id: str
    num_frames: int
    frame_to_segment_id: np.ndarray
    frame_to_progress: np.ndarray
    segment_frames: dict[int, np.ndarray]
    segment_progress: dict[int, np.ndarray]
    segment_frames_sorted_by_progress: dict[int, np.ndarray]
    segment_progress_sorted: dict[int, np.ndarray]


@dataclass(frozen=True)
class HumanSplineEpisode:
    episode_index: int
    frame_indices: np.ndarray
    frame_u: np.ndarray
    global_knots: np.ndarray
    global_coefficients: np.ndarray
    degree: int
    frame_index_to_position: dict[int, int]


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        if yaml is None:
            raise ImportError("PyYAML is required for YAML configs.")
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid config payload in {path}")
    return data


def as_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def canonical_category_id(value: str) -> str:
    return str(value).strip().lower().replace("_", "-")


def load_paths(config: dict[str, Any]) -> PathsConfig:
    paths = config.get("paths")
    if not isinstance(paths, dict):
        raise KeyError("Missing paths block in config.")
    return PathsConfig(output_root=as_path(paths["output_root"]))


def load_processing(config: dict[str, Any]) -> ProcessingConfig:
    processing = config.get("processing") or {}
    if not isinstance(processing, dict):
        raise KeyError("processing block must be a mapping.")
    num_pairings_per_frame = int(processing.get("num_pairings_per_frame", 1))
    if num_pairings_per_frame <= 0:
        raise ValueError("processing.num_pairings_per_frame must be positive.")
    return ProcessingConfig(
        overwrite=bool(processing.get("overwrite", False)),
        index_dtype=str(processing.get("index_dtype", "int32")),
        progress_dtype=str(processing.get("progress_dtype", "float32")),
        u_dtype=str(processing.get("u_dtype", "float32")),
        num_pairings_per_frame=num_pairings_per_frame,
        write_episode_metadata_json=bool(processing.get("write_episode_metadata_json", True)),
    )


def load_pairs(config: dict[str, Any]) -> list[PairConfig]:
    pair_entries = config.get("pairs")
    if not isinstance(pair_entries, list) or not pair_entries:
        raise KeyError("Config must define a non-empty pairs list.")
    out: list[PairConfig] = []
    seen_pair_ids: set[str] = set()
    for entry in pair_entries:
        if not isinstance(entry, dict):
            raise ValueError(f"Bad pair entry: {entry!r}")
        pair_id = str(entry["pair_id"]).strip()
        if not pair_id:
            raise ValueError("pair_id cannot be empty.")
        if pair_id in seen_pair_ids:
            raise ValueError(f"Duplicate pair_id: {pair_id}")
        seen_pair_ids.add(pair_id)
        out.append(
            PairConfig(
                pair_id=pair_id,
                category_id=canonical_category_id(entry["category_id"]),
                robot_episode_index=int(entry["robot_episode_index"]),
                human_episode_index=int(entry["human_episode_index"]),
                robot_episode_dir=as_path(entry["robot_episode_dir"]),
                human_episode_dir=as_path(entry["human_episode_dir"]),
            )
        )
    return out


def pair_output_dir(root: Path, pair: PairConfig) -> Path:
    return root / pair.category_id / pair.pair_id / f"robot_episode_{pair.robot_episode_index:06d}"


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory already contains files, pass overwrite to replace it: {path}")
    path.mkdir(parents=True, exist_ok=True)


def load_frame_indices_from_episode_dir(episode_dir: Path) -> np.ndarray:
    frame_embeddings_path = episode_dir / "frame_embeddings.npz"
    with np.load(frame_embeddings_path, allow_pickle=False) as archive:
        return np.asarray(archive["frame_indices"], dtype=np.int64)


def load_annotation_index(checkpoint_path: Path, episode_index: int) -> AnnotationIndex:
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    category_id = canonical_category_id(payload.get("category_id", ""))
    frame_indexing = payload.get("frame_indexing") or {}
    end_frame_exclusive = int(frame_indexing.get("end_frame_exclusive", 0))
    segments = payload.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError(f"No segments found in {checkpoint_path}")
    if end_frame_exclusive <= 0:
        end_frame_exclusive = max(int(item["end_frame_exclusive"]) for item in segments)

    frame_to_segment_id = np.full((end_frame_exclusive,), fill_value=-1, dtype=np.int32)
    frame_to_progress = np.full((end_frame_exclusive,), fill_value=np.nan, dtype=np.float32)
    segment_frames: dict[int, np.ndarray] = {}
    segment_progress: dict[int, np.ndarray] = {}
    segment_frames_sorted_by_progress: dict[int, np.ndarray] = {}
    segment_progress_sorted: dict[int, np.ndarray] = {}

    for segment in segments:
        segment_id = int(segment["segment_id"])
        per_frame = (segment.get("progress") or {}).get("per_frame")
        if not isinstance(per_frame, list) or not per_frame:
            raise ValueError(f"Segment {segment_id} in {checkpoint_path} has no progress.per_frame entries.")
        frames = np.asarray([int(item["frame"]) for item in per_frame], dtype=np.int32)
        progresses = np.asarray([float(item["progress"]) for item in per_frame], dtype=np.float32)
        if np.any(frames < 0) or np.any(frames >= end_frame_exclusive):
            raise ValueError(f"Segment {segment_id} in {checkpoint_path} has out-of-range frame indices.")
        frame_to_segment_id[frames] = segment_id
        frame_to_progress[frames] = progresses
        segment_frames[segment_id] = frames
        segment_progress[segment_id] = progresses
        order = np.argsort(progresses.astype(np.float64), kind="stable")
        segment_frames_sorted_by_progress[segment_id] = frames[order]
        segment_progress_sorted[segment_id] = progresses[order]

    if np.any(frame_to_segment_id < 0):
        missing_count = int(np.count_nonzero(frame_to_segment_id < 0))
        raise ValueError(f"{checkpoint_path} leaves {missing_count} frames without segment IDs.")
    if np.any(np.isnan(frame_to_progress)):
        missing_count = int(np.count_nonzero(np.isnan(frame_to_progress)))
        raise ValueError(f"{checkpoint_path} leaves {missing_count} frames without progress values.")

    return AnnotationIndex(
        episode_index=episode_index,
        category_id=category_id,
        num_frames=end_frame_exclusive,
        frame_to_segment_id=frame_to_segment_id,
        frame_to_progress=frame_to_progress,
        segment_frames=segment_frames,
        segment_progress=segment_progress,
        segment_frames_sorted_by_progress=segment_frames_sorted_by_progress,
        segment_progress_sorted=segment_progress_sorted,
    )


def load_human_spline_episode(path: Path, episode_index: int) -> HumanSplineEpisode:
    with np.load(path, allow_pickle=False) as archive:
        frame_indices = np.asarray(archive["frame_indices"], dtype=np.int64)
        frame_u = np.asarray(archive["frame_u"], dtype=np.float64)
        global_knots = np.asarray(archive["global_knots"], dtype=np.float64)
        global_coefficients = np.asarray(archive["global_coefficients"], dtype=np.float64)
        degree = int(np.asarray(archive["global_degree"]).reshape(-1)[0])
    if frame_indices.shape[0] != frame_u.shape[0]:
        raise ValueError(f"frame_indices/frame_u mismatch in {path}")
    return HumanSplineEpisode(
        episode_index=episode_index,
        frame_indices=frame_indices,
        frame_u=frame_u,
        global_knots=global_knots,
        global_coefficients=global_coefficients,
        degree=degree,
        frame_index_to_position={int(frame_index): pos for pos, frame_index in enumerate(frame_indices.tolist())},
    )


def interpolate_human_u_with_details(
    human_annotation: AnnotationIndex,
    human_spline: HumanSplineEpisode,
    segment_id: int,
    target_progress: float,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    if segment_id not in human_annotation.segment_progress:
        return {
            "u": float("nan"),
            "valid": False,
            "reason": "missing_segment",
            "lower_frame": -1,
            "upper_frame": -1,
            "lower_progress": float("nan"),
            "upper_progress": float("nan"),
            "alpha": float("nan"),
            "exact_match": False,
            "boundary_clamped": False,
        }
    sorted_progress = human_annotation.segment_progress_sorted[segment_id]
    sorted_frames = human_annotation.segment_frames_sorted_by_progress[segment_id]
    if sorted_progress.size == 0:
        return {
            "u": float("nan"),
            "valid": False,
            "reason": "missing_segment",
            "lower_frame": -1,
            "upper_frame": -1,
            "lower_progress": float("nan"),
            "upper_progress": float("nan"),
            "alpha": float("nan"),
            "exact_match": False,
            "boundary_clamped": False,
        }

    insert = int(np.searchsorted(sorted_progress, target_progress, side="left"))
    if insert <= 0:
        frame_index = int(sorted_frames[0])
        frame_pos = human_spline.frame_index_to_position.get(frame_index)
        if frame_pos is None:
            return {
                "u": float("nan"),
                "valid": False,
                "reason": "missing_frame_u",
                "lower_frame": frame_index,
                "upper_frame": frame_index,
                "lower_progress": float(sorted_progress[0]),
                "upper_progress": float(sorted_progress[0]),
                "alpha": 0.0,
                "exact_match": True,
                "boundary_clamped": True,
            }
        u_value = float(human_spline.frame_u[frame_pos])
        return {
            "u": u_value,
            "valid": np.isfinite(u_value),
            "reason": "ok" if np.isfinite(u_value) else "missing_frame_u",
            "lower_frame": frame_index,
            "upper_frame": frame_index,
            "lower_progress": float(sorted_progress[0]),
            "upper_progress": float(sorted_progress[0]),
            "alpha": 0.0,
            "exact_match": True,
            "boundary_clamped": True,
        }
    if insert >= sorted_progress.size:
        frame_index = int(sorted_frames[-1])
        frame_pos = human_spline.frame_index_to_position.get(frame_index)
        if frame_pos is None:
            return {
                "u": float("nan"),
                "valid": False,
                "reason": "missing_frame_u",
                "lower_frame": frame_index,
                "upper_frame": frame_index,
                "lower_progress": float(sorted_progress[-1]),
                "upper_progress": float(sorted_progress[-1]),
                "alpha": 0.0,
                "exact_match": True,
                "boundary_clamped": True,
            }
        u_value = float(human_spline.frame_u[frame_pos])
        return {
            "u": u_value,
            "valid": np.isfinite(u_value),
            "reason": "ok" if np.isfinite(u_value) else "missing_frame_u",
            "lower_frame": frame_index,
            "upper_frame": frame_index,
            "lower_progress": float(sorted_progress[-1]),
            "upper_progress": float(sorted_progress[-1]),
            "alpha": 0.0,
            "exact_match": True,
            "boundary_clamped": True,
        }

    lower_progress = float(sorted_progress[insert - 1])
    upper_progress = float(sorted_progress[insert])
    lower_frame = int(sorted_frames[insert - 1])
    upper_frame = int(sorted_frames[insert])
    lower_pos = human_spline.frame_index_to_position.get(lower_frame)
    upper_pos = human_spline.frame_index_to_position.get(upper_frame)
    if lower_pos is None or upper_pos is None:
        return {
            "u": float("nan"),
            "valid": False,
            "reason": "missing_frame_u",
            "lower_frame": lower_frame,
            "upper_frame": upper_frame,
            "lower_progress": lower_progress,
            "upper_progress": upper_progress,
            "alpha": float("nan"),
            "exact_match": False,
            "boundary_clamped": False,
        }
    lower_u = float(human_spline.frame_u[lower_pos])
    upper_u = float(human_spline.frame_u[upper_pos])
    if not np.isfinite(lower_u) or not np.isfinite(upper_u):
        return {
            "u": float("nan"),
            "valid": False,
            "reason": "missing_frame_u",
            "lower_frame": lower_frame,
            "upper_frame": upper_frame,
            "lower_progress": lower_progress,
            "upper_progress": upper_progress,
            "alpha": float("nan"),
            "exact_match": False,
            "boundary_clamped": False,
        }

    if abs(upper_progress - lower_progress) <= tolerance:
        u_value = lower_u
        return {
            "u": u_value,
            "valid": np.isfinite(u_value),
            "reason": "ok" if np.isfinite(u_value) else "missing_frame_u",
            "lower_frame": lower_frame,
            "upper_frame": upper_frame,
            "lower_progress": lower_progress,
            "upper_progress": upper_progress,
            "alpha": 0.0,
            "exact_match": True,
            "boundary_clamped": False,
        }

    alpha = (float(target_progress) - lower_progress) / (upper_progress - lower_progress)
    u_value = float((1.0 - alpha) * lower_u + alpha * upper_u)
    exact_match = abs(float(target_progress) - lower_progress) <= tolerance or abs(float(target_progress) - upper_progress) <= tolerance
    return {
        "u": u_value,
        "valid": np.isfinite(u_value),
        "reason": "ok" if np.isfinite(u_value) else "missing_frame_u",
        "lower_frame": lower_frame,
        "upper_frame": upper_frame,
        "lower_progress": lower_progress,
        "upper_progress": upper_progress,
        "alpha": float(alpha),
        "exact_match": bool(exact_match),
        "boundary_clamped": False,
    }


def save_config_snapshot(config: dict[str, Any], output_root: Path, filename: str) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / filename).write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def summarize_distribution(values: np.ndarray) -> dict[str, float | int | None]:
    if values.size == 0:
        out: dict[str, float | int | None] = {"count": 0, "mean": None, "min": None}
        for percentile in SUMMARY_PERCENTILES:
            key = f"p{str(percentile).replace('.', '_')}"
            out[key] = None
        out["max"] = None
        return out
    valid = values.astype(np.float64, copy=False)
    out = {
        "count": int(valid.size),
        "mean": float(valid.mean()),
        "min": float(valid.min()),
    }
    for percentile in SUMMARY_PERCENTILES:
        key = f"p{str(percentile).replace('.', '_')}"
        out[key] = float(np.percentile(valid, percentile))
    out["max"] = float(valid.max())
    return out

