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
DEFAULT_CONFIG = ROOT / "4_human_local_bspline_knot_count_stats_config.yaml"
EPSILON = 1e-8
SUMMARY_PERCENTILES = (5, 10, 15, 25, 50, 75, 90, 95, 99)


@dataclass(frozen=True)
class PathsConfig:
    sim_dataset_root: Path
    human_dataset_root: Path
    human_bspline_root: Path
    frame_human_match_root: Path
    output_root: Path


@dataclass(frozen=True)
class ProcessingConfig:
    seed: int
    num_future_knots: int
    overwrite: bool
    index_dtype: str
    u_dtype: str
    skip_missing_match_episodes: bool
    skip_missing_human_bspline_episodes: bool
    write_episode_metadata_json: bool
    max_robot_episodes_per_category: int | None


@dataclass(frozen=True)
class CategoryConfig:
    category_id: str
    sim_start: int
    sim_end_exclusive: int
    human_start: int
    human_end_exclusive: int


@dataclass(frozen=True)
class HumanSplineEpisode:
    episode_index: int
    frame_indices: np.ndarray
    frame_u: np.ndarray
    global_knots: np.ndarray
    global_coefficients: np.ndarray
    degree: int
    frame_index_to_position: dict[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Step 4: for each valid robot-frame/human-pairing match from step 3, restrict the matched human global "
            "B-spline to the matched start/end frame interval and compute local knot-count statistics."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--frame-human-match-root", type=Path, default=None)
    parser.add_argument("--human-bspline-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-future-knots", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--category", action="append", default=[], help="Repeat to limit export to named categories.")
    parser.add_argument("--max-robot-episodes-per-category", type=int, default=None)
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


def canonical_category_id(value: str) -> str:
    return str(value).strip().lower()


def load_paths(config: dict[str, Any], args: argparse.Namespace) -> PathsConfig:
    paths = config.get("paths")
    if not isinstance(paths, dict):
        raise KeyError("Missing paths block in config.")
    processing = config.get("processing", {})
    seed = int(args.seed) if args.seed is not None else int(processing.get("seed", 0))
    num_future_knots = (
        int(args.num_future_knots)
        if args.num_future_knots is not None
        else int(processing.get("num_future_knots", 12))
    )
    if args.frame_human_match_root is not None:
        frame_human_match_root = as_path(args.frame_human_match_root)
    else:
        template = paths.get("frame_human_match_root_template") or paths.get("frame_human_match_root")
        if not template:
            raise KeyError("Config must define paths.frame_human_match_root_template or paths.frame_human_match_root.")
        frame_human_match_root = as_path(str(template).format(seed=seed, num_future_knots=num_future_knots))
    if args.human_bspline_root is not None:
        human_bspline_root = as_path(args.human_bspline_root)
    else:
        human_bspline_root = as_path(paths["human_bspline_root"])
    if args.output_root is not None:
        output_root = as_path(args.output_root)
    else:
        template = paths.get("output_root_template") or paths.get("output_root")
        if not template:
            raise KeyError("Config must define paths.output_root_template or paths.output_root.")
        output_root = as_path(str(template).format(seed=seed, num_future_knots=num_future_knots))
    return PathsConfig(
        sim_dataset_root=as_path(paths["sim_dataset_root"]),
        human_dataset_root=as_path(paths["human_dataset_root"]),
        human_bspline_root=human_bspline_root,
        frame_human_match_root=frame_human_match_root,
        output_root=output_root,
    )


def load_processing(config: dict[str, Any], args: argparse.Namespace) -> ProcessingConfig:
    processing = config.get("processing")
    if not isinstance(processing, dict):
        raise KeyError("Missing processing block in config.")
    seed = int(args.seed) if args.seed is not None else int(processing.get("seed", 0))
    num_future_knots = (
        int(args.num_future_knots)
        if args.num_future_knots is not None
        else int(processing.get("num_future_knots", 12))
    )
    max_robot_episodes = (
        int(args.max_robot_episodes_per_category)
        if args.max_robot_episodes_per_category is not None
        else (int(processing["max_robot_episodes_per_category"]) if processing.get("max_robot_episodes_per_category") is not None else None)
    )
    return ProcessingConfig(
        seed=seed,
        num_future_knots=num_future_knots,
        overwrite=bool(processing.get("overwrite", False) or args.overwrite),
        index_dtype=str(processing.get("index_dtype", "int32")),
        u_dtype=str(processing.get("u_dtype", "float32")),
        skip_missing_match_episodes=bool(processing.get("skip_missing_match_episodes", True)),
        skip_missing_human_bspline_episodes=bool(processing.get("skip_missing_human_bspline_episodes", False)),
        write_episode_metadata_json=bool(processing.get("write_episode_metadata_json", True)),
        max_robot_episodes_per_category=max_robot_episodes,
    )


def load_categories(config: dict[str, Any], requested: list[str]) -> list[CategoryConfig]:
    categories_block = config.get("categories")
    if not isinstance(categories_block, list):
        raise KeyError("Missing categories list in config.")
    categories = []
    for item in categories_block:
        if not isinstance(item, dict):
            raise ValueError(f"Bad category entry: {item!r}")
        categories.append(
            CategoryConfig(
                category_id=canonical_category_id(item["id"]),
                sim_start=int(item["sim_start"]),
                sim_end_exclusive=int(item["sim_end_exclusive"]),
                human_start=int(item["human_start"]),
                human_end_exclusive=int(item["human_end_exclusive"]),
            )
        )
    if requested:
        wanted = {canonical_category_id(value) for value in requested}
        categories = [item for item in categories if item.category_id in wanted]
        if len(categories) != len(wanted):
            missing = sorted(wanted - {item.category_id for item in categories})
            raise ValueError(f"Unknown requested categories: {missing}")
    return categories


def frame_human_match_npz_path(root: Path, episode_index: int) -> Path:
    return root / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}" / "frame_human_segment_progress_matches.npz"


def human_spline_npz_path(root: Path, episode_index: int) -> Path:
    return root / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}" / "spline.npz"


def output_episode_dir(output_root: Path, episode_index: int) -> Path:
    return output_root / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}"


def ensure_output_episode_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory already contains files, pass overwrite to replace it: {path}")
    path.mkdir(parents=True, exist_ok=True)


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
) -> tuple[np.ndarray, np.ndarray] | None:
    if u_end <= u_start + EPSILON:
        return None
    spline = BSpline(
        np.asarray(global_knots, dtype=np.float64),
        np.asarray(global_coefficients, dtype=np.float64),
        degree,
        extrapolate=False,
    )
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
    return local_knots, local_coefficients


def summarize_distribution(values: np.ndarray) -> dict[str, float | int | None]:
    if values.size == 0:
        out: dict[str, float | int | None] = {"count": 0, "mean": None, "min": None}
        for percentile in SUMMARY_PERCENTILES:
            out[f"p{percentile}"] = None
        out["max"] = None
        return out
    valid = values.astype(np.float64, copy=False)
    out = {
        "count": int(valid.size),
        "mean": float(valid.mean()),
        "min": float(valid.min()),
    }
    for percentile in SUMMARY_PERCENTILES:
        out[f"p{percentile}"] = float(np.percentile(valid, percentile))
    out["max"] = float(valid.max())
    return out


def load_human_spline_episode(path: Path, episode_index: int) -> HumanSplineEpisode:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        frame_indices = np.asarray(archive["frame_indices"], dtype=np.int64)
        frame_u = np.asarray(archive["frame_u"], dtype=np.float64)
        global_knots = np.asarray(archive["global_knots"], dtype=np.float64)
        global_coefficients = np.asarray(archive["global_coefficients"], dtype=np.float64)
        degree = int(np.asarray(archive["global_degree"]).reshape(-1)[0])
    if frame_indices.ndim != 1 or frame_u.ndim != 1:
        raise ValueError(f"Unexpected frame arrays in {path}")
    if frame_indices.shape[0] != frame_u.shape[0]:
        raise ValueError(f"frame_indices/frame_u mismatch in {path}")
    if global_knots.ndim != 1 or global_coefficients.ndim != 2:
        raise ValueError(f"Unexpected global spline arrays in {path}")
    return HumanSplineEpisode(
        episode_index=episode_index,
        frame_indices=frame_indices,
        frame_u=frame_u,
        global_knots=global_knots,
        global_coefficients=global_coefficients,
        degree=degree,
        frame_index_to_position={int(frame_index): pos for pos, frame_index in enumerate(frame_indices.tolist())},
    )


def robot_episode_ids_for_category(
    category: CategoryConfig,
    frame_human_match_root: Path,
    max_robot_episodes_per_category: int | None,
    skip_missing_match_episodes: bool,
) -> list[int]:
    episode_ids = list(range(category.sim_start, category.sim_end_exclusive))
    if max_robot_episodes_per_category is not None:
        episode_ids = episode_ids[:max_robot_episodes_per_category]
    out: list[int] = []
    for episode_index in tqdm(
        episode_ids,
        desc=f"{category.category_id}: robot episodes",
        unit="episode",
        leave=False,
    ):
        path = frame_human_match_npz_path(frame_human_match_root, episode_index)
        if not path.exists():
            if skip_missing_match_episodes:
                continue
            raise FileNotFoundError(path)
        out.append(episode_index)
    return out


def build_episode_knot_counts(
    match_npz_path: Path,
    human_bspline_root: Path,
    human_spline_cache: dict[int, HumanSplineEpisode | None],
    processing: ProcessingConfig,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(match_npz_path, allow_pickle=False) as archive:
        frame_indices = np.asarray(archive["frame_indices"], dtype=np.int64)
        paired_human_episode_indices = np.asarray(archive["paired_human_episode_indices"], dtype=np.int64)
        human_start_frame_index = np.asarray(archive["human_start_frame_index"], dtype=np.int64)
        human_end_frame_index = np.asarray(archive["human_end_frame_index"], dtype=np.int64)
        start_match_valid_mask = np.asarray(archive["start_match_valid_mask"], dtype=bool)
        end_match_valid_mask = np.asarray(archive["end_match_valid_mask"], dtype=bool)

    if paired_human_episode_indices.ndim != 2:
        raise ValueError(f"paired_human_episode_indices must be rank-2 in {match_npz_path}")

    num_frames, num_pairings = paired_human_episode_indices.shape
    local_interval_valid_mask = np.zeros((num_frames, num_pairings), dtype=bool)
    local_exact_knot_count = np.full((num_frames, num_pairings), fill_value=-1, dtype=np.dtype(processing.index_dtype))
    local_unique_knot_count = np.full((num_frames, num_pairings), fill_value=-1, dtype=np.dtype(processing.index_dtype))
    local_control_point_count = np.full((num_frames, num_pairings), fill_value=-1, dtype=np.dtype(processing.index_dtype))
    human_local_start_u = np.full((num_frames, num_pairings), fill_value=np.nan, dtype=np.dtype(processing.u_dtype))
    human_local_end_u = np.full((num_frames, num_pairings), fill_value=np.nan, dtype=np.dtype(processing.u_dtype))

    missing_human_spline_count = 0
    missing_human_frame_u_count = 0
    degenerate_interval_count = 0

    for frame_pos in tqdm(
        range(num_frames),
        desc=f"{match_npz_path.parent.name}: frames",
        unit="frame",
        leave=False,
    ):
        for pairing_slot in range(num_pairings):
            if not (start_match_valid_mask[frame_pos, pairing_slot] and end_match_valid_mask[frame_pos, pairing_slot]):
                continue
            human_episode_index = int(paired_human_episode_indices[frame_pos, pairing_slot])
            if human_episode_index not in human_spline_cache:
                path = human_spline_npz_path(human_bspline_root, human_episode_index)
                if path.exists():
                    human_spline_cache[human_episode_index] = load_human_spline_episode(path, human_episode_index)
                else:
                    if processing.skip_missing_human_bspline_episodes:
                        human_spline_cache[human_episode_index] = None
                    else:
                        raise FileNotFoundError(path)
            human_spline = human_spline_cache[human_episode_index]
            if human_spline is None:
                missing_human_spline_count += 1
                continue

            start_frame = int(human_start_frame_index[frame_pos, pairing_slot])
            end_frame = int(human_end_frame_index[frame_pos, pairing_slot])
            start_pos = human_spline.frame_index_to_position.get(start_frame)
            end_pos = human_spline.frame_index_to_position.get(end_frame)
            if start_pos is None or end_pos is None:
                missing_human_frame_u_count += 1
                continue

            start_u = float(human_spline.frame_u[start_pos])
            end_u = float(human_spline.frame_u[end_pos])
            human_local_start_u[frame_pos, pairing_slot] = np.asarray(start_u, dtype=np.dtype(processing.u_dtype))
            human_local_end_u[frame_pos, pairing_slot] = np.asarray(end_u, dtype=np.dtype(processing.u_dtype))
            if end_u <= start_u + EPSILON:
                degenerate_interval_count += 1
                continue

            local = restrict_bspline_exact(
                human_spline.global_knots,
                human_spline.global_coefficients,
                human_spline.degree,
                start_u,
                end_u,
            )
            if local is None:
                degenerate_interval_count += 1
                continue
            local_knots, local_coefficients = local
            local_interval_valid_mask[frame_pos, pairing_slot] = True
            local_exact_knot_count[frame_pos, pairing_slot] = np.asarray(local_knots.shape[0], dtype=np.dtype(processing.index_dtype))
            local_unique_knot_count[frame_pos, pairing_slot] = np.asarray(np.unique(local_knots).shape[0], dtype=np.dtype(processing.index_dtype))
            local_control_point_count[frame_pos, pairing_slot] = np.asarray(local_coefficients.shape[0], dtype=np.dtype(processing.index_dtype))

    exact_values = local_exact_knot_count[local_interval_valid_mask].astype(np.int64, copy=False)
    unique_values = local_unique_knot_count[local_interval_valid_mask].astype(np.int64, copy=False)
    control_values = local_control_point_count[local_interval_valid_mask].astype(np.int64, copy=False)
    arrays = {
        "frame_indices": np.asarray(frame_indices, dtype=np.dtype(processing.index_dtype)),
        "paired_human_episode_indices": np.asarray(paired_human_episode_indices, dtype=np.dtype(processing.index_dtype)),
        "human_start_frame_index": np.asarray(human_start_frame_index, dtype=np.dtype(processing.index_dtype)),
        "human_end_frame_index": np.asarray(human_end_frame_index, dtype=np.dtype(processing.index_dtype)),
        "local_interval_valid_mask": local_interval_valid_mask,
        "human_local_start_u": human_local_start_u,
        "human_local_end_u": human_local_end_u,
        "local_exact_knot_count": local_exact_knot_count,
        "local_unique_knot_count": local_unique_knot_count,
        "local_control_point_count": local_control_point_count,
    }
    metadata = {
        "robot_episode_index": int(match_npz_path.parent.name.split("_")[-1]),
        "num_frames": num_frames,
        "num_pairings_per_frame": num_pairings,
        "total_frame_pairings": int(num_frames * num_pairings),
        "valid_local_interval_count": int(np.count_nonzero(local_interval_valid_mask)),
        "missing_human_spline_count": int(missing_human_spline_count),
        "missing_human_frame_u_count": int(missing_human_frame_u_count),
        "degenerate_interval_count": int(degenerate_interval_count),
        "local_exact_knot_count_stats": summarize_distribution(exact_values),
        "local_unique_knot_count_stats": summarize_distribution(unique_values),
        "local_control_point_count_stats": summarize_distribution(control_values),
        "stored_fields": sorted(arrays),
    }
    return arrays, metadata


def save_episode_output(
    output_root: Path,
    episode_index: int,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    processing: ProcessingConfig,
) -> None:
    out_dir = output_episode_dir(output_root, episode_index)
    ensure_output_episode_dir(out_dir, overwrite=processing.overwrite)
    np.savez_compressed(out_dir / "human_local_bspline_knot_counts.npz", **arrays)
    if processing.write_episode_metadata_json:
        output_metadata = dict(metadata)
        output_metadata["output_npz_path"] = str((out_dir / "human_local_bspline_knot_counts.npz").resolve())
        (out_dir / "human_local_bspline_knot_counts_metadata.json").write_text(
            json.dumps(output_metadata, indent=2) + "\n",
            encoding="utf-8",
        )


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    paths = load_paths(config, args)
    processing = load_processing(config, args)
    categories = load_categories(config, args.category)

    if not paths.frame_human_match_root.exists():
        raise FileNotFoundError(paths.frame_human_match_root)
    if not paths.human_bspline_root.exists():
        raise FileNotFoundError(paths.human_bspline_root)
    paths.output_root.mkdir(parents=True, exist_ok=True)

    print("Exporting human local B-spline knot-count statistics")
    print(f"  frame_human_match_root : {paths.frame_human_match_root}")
    print(f"  human_bspline_root     : {paths.human_bspline_root}")
    print(f"  output_root            : {paths.output_root}")
    print(f"  seed                   : {processing.seed}")
    print(f"  num_future_knots       : {processing.num_future_knots}")
    print(f"  categories             : {', '.join(item.category_id for item in categories)}")

    human_spline_cache: dict[int, HumanSplineEpisode | None] = {}
    all_exact_counts: list[np.ndarray] = []
    all_unique_counts: list[np.ndarray] = []
    all_control_counts: list[np.ndarray] = []

    run_summary: dict[str, Any] = {
        "frame_human_match_root": str(paths.frame_human_match_root),
        "human_bspline_root": str(paths.human_bspline_root),
        "output_root": str(paths.output_root),
        "seed": processing.seed,
        "num_future_knots": processing.num_future_knots,
        "categories": [],
    }

    for category in tqdm(categories, desc="categories", unit="category"):
        robot_episode_ids = robot_episode_ids_for_category(
            category,
            paths.frame_human_match_root,
            processing.max_robot_episodes_per_category,
            processing.skip_missing_match_episodes,
        )
        if not robot_episode_ids:
            run_summary["categories"].append(
                {
                    "category_id": category.category_id,
                    "robot_episode_count": 0,
                    "robot_frame_count": 0,
                    "total_frame_pairings": 0,
                    "valid_local_interval_count": 0,
                    "episodes": [],
                    "note": "No step-3 match episodes available for this category.",
                }
            )
            continue

        category_episode_outputs: list[dict[str, Any]] = []
        category_exact_counts: list[np.ndarray] = []
        category_unique_counts: list[np.ndarray] = []
        category_control_counts: list[np.ndarray] = []
        category_robot_frames = 0
        category_total_pairings = 0
        category_valid_local_intervals = 0

        for episode_index in tqdm(
            robot_episode_ids,
            desc=f"{category.category_id}: save knot counts",
            unit="episode",
            leave=False,
        ):
            match_npz_path = frame_human_match_npz_path(paths.frame_human_match_root, episode_index)
            arrays, metadata = build_episode_knot_counts(
                match_npz_path,
                paths.human_bspline_root,
                human_spline_cache,
                processing,
            )
            save_episode_output(paths.output_root, episode_index, arrays, metadata, processing)
            category_episode_outputs.append(metadata)
            category_robot_frames += int(metadata["num_frames"])
            category_total_pairings += int(metadata["total_frame_pairings"])
            category_valid_local_intervals += int(metadata["valid_local_interval_count"])
            exact_mask = arrays["local_interval_valid_mask"]
            category_exact_counts.append(arrays["local_exact_knot_count"][exact_mask].astype(np.int64, copy=False))
            category_unique_counts.append(arrays["local_unique_knot_count"][exact_mask].astype(np.int64, copy=False))
            category_control_counts.append(arrays["local_control_point_count"][exact_mask].astype(np.int64, copy=False))

        category_exact = np.concatenate(category_exact_counts) if category_exact_counts else np.asarray([], dtype=np.int64)
        category_unique = np.concatenate(category_unique_counts) if category_unique_counts else np.asarray([], dtype=np.int64)
        category_control = np.concatenate(category_control_counts) if category_control_counts else np.asarray([], dtype=np.int64)
        all_exact_counts.append(category_exact)
        all_unique_counts.append(category_unique)
        all_control_counts.append(category_control)

        run_summary["categories"].append(
            {
                "category_id": category.category_id,
                "robot_episode_count": len(robot_episode_ids),
                "robot_frame_count": category_robot_frames,
                "total_frame_pairings": category_total_pairings,
                "valid_local_interval_count": category_valid_local_intervals,
                "local_exact_knot_count_stats": summarize_distribution(category_exact),
                "local_unique_knot_count_stats": summarize_distribution(category_unique),
                "local_control_point_count_stats": summarize_distribution(category_control),
                "episodes": category_episode_outputs,
            }
        )

    overall_exact = np.concatenate(all_exact_counts) if all_exact_counts else np.asarray([], dtype=np.int64)
    overall_unique = np.concatenate(all_unique_counts) if all_unique_counts else np.asarray([], dtype=np.int64)
    overall_control = np.concatenate(all_control_counts) if all_control_counts else np.asarray([], dtype=np.int64)
    run_summary["overall"] = {
        "local_exact_knot_count_stats": summarize_distribution(overall_exact),
        "local_unique_knot_count_stats": summarize_distribution(overall_unique),
        "local_control_point_count_stats": summarize_distribution(overall_control),
    }

    (paths.output_root / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
