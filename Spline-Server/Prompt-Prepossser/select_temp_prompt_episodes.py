from __future__ import annotations

import argparse
import io
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import imageio.v2 as iio
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageOps
from tqdm.auto import tqdm


IMAGE_COLUMN = "observation.images.top_rgb"
DEFAULT_HUMAN_DATASET_ROOT = Path("D:/pretrain_lehome_all_garment_data_z180")
DEFAULT_OUTPUT_ROOT = Path(
    "D:/LeHome-Challenge/Lehome-Spline-ICRA2027/Spline-Server/Prompt-Prepossser/temp_prompts"
)
DEFAULT_CATEGORY_ORDER = [
    "pants",
    "shorts",
    "top_long_sleeve",
    "top_short_sleeve",
]
DEFAULT_CATEGORY_RANGES: dict[str, tuple[int, int]] = {
    "pants": (0, 1018),
    "shorts": (1018, 2039),
    "top_long_sleeve": (2039, 2875),
    "top_short_sleeve": (2875, 4180),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample one annotated human episode per category and export only the top-view MP4 plus metadata "
            "into a temp prompt folder. This does not run spline preprocessing."
        )
    )
    parser.add_argument("--human-dataset-root", type=Path, default=DEFAULT_HUMAN_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--category",
        action="append",
        default=[],
        help="Repeat to limit sampling to specific category ids.",
    )
    parser.add_argument(
        "--prefer-annotated",
        type=lambda value: str(value).strip().lower() in {"1", "true", "yes", "y"},
        default=True,
        help="If true, prefer annotated episodes when available. If false, sample from the full category range.",
    )
    return parser.parse_args()


def canonical_category_id(value: str) -> str:
    return str(value).strip().lower()


def checkpoint_root(dataset_root: Path) -> Path:
    return dataset_root / "annotations" / "temporal_checkpoints"


def checkpoint_json_path(dataset_root: Path, episode_index: int) -> Path:
    return (
        checkpoint_root(dataset_root)
        / f"chunk-{episode_index // 1000:03d}"
        / f"episode_{episode_index:06d}"
        / "checkpoints.json"
    )


def dataset_episode_parquet_path(dataset_root: Path, episode_index: int) -> Path:
    return dataset_root / "data" / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}.parquet"


def load_checkpoint_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def discover_annotated_episodes(dataset_root: Path) -> dict[str, list[dict[str, Any]]]:
    root = checkpoint_root(dataset_root)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    checkpoint_paths = sorted(root.glob("chunk-*/episode_*/checkpoints.json"))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoint JSON files found under {root}")

    for path in tqdm(checkpoint_paths, desc="scan/checkpoints", unit="episode"):
        payload = load_checkpoint_payload(path)
        category_id = canonical_category_id(payload.get("category_id", ""))
        if not category_id:
            continue
        episode_dir = path.parent.name
        episode_index = int(episode_dir.split("_")[-1])
        grouped[category_id].append(
            {
                "episode_index": episode_index,
                "checkpoint_path": path,
                "payload": payload,
            }
        )
    if not grouped:
        raise RuntimeError(f"No categorized annotated episodes found under {root}")
    return grouped


def discover_dataset_episode_indices(dataset_root: Path) -> list[int]:
    meta_path = dataset_root / "meta" / "episodes.jsonl"
    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)
    episode_indices: list[int] = []
    with meta_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            episode_indices.append(int(payload["episode_index"]))
    if not episode_indices:
        raise RuntimeError(f"No episodes found in {meta_path}")
    return sorted(episode_indices)


def fallback_candidates_for_category(category_id: str, dataset_episode_indices: list[int]) -> list[int]:
    if category_id not in DEFAULT_CATEGORY_RANGES:
        return []
    start, end_exclusive = DEFAULT_CATEGORY_RANGES[category_id]
    return [episode_index for episode_index in dataset_episode_indices if start <= episode_index < end_exclusive]


def annotated_episode_lookup(grouped: dict[str, list[dict[str, Any]]]) -> dict[int, dict[str, Any]]:
    lookup: dict[int, dict[str, Any]] = {}
    for entries in grouped.values():
        for entry in entries:
            lookup[int(entry["episode_index"])] = entry
    return lookup


def decode_rgb_frame(payload: bytes | dict[str, Any]) -> np.ndarray:
    if isinstance(payload, dict):
        payload = payload.get("bytes")
    if not payload:
        raise ValueError("Top-view image row does not contain encoded image bytes.")
    with Image.open(io.BytesIO(payload)) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return np.asarray(image, dtype=np.uint8)


def infer_fps_from_timestamps(timestamps: np.ndarray, fallback: float = 30.0) -> float:
    if timestamps.ndim != 1 or timestamps.size < 2:
        return float(fallback)
    deltas = np.diff(timestamps.astype(np.float64, copy=False))
    positive = deltas[deltas > 0.0]
    if positive.size == 0:
        return float(fallback)
    median_delta = float(np.median(positive))
    if not np.isfinite(median_delta) or median_delta <= 0.0:
        return float(fallback)
    return max(1.0 / median_delta, 1.0)


def export_top_view_video(dataset_root: Path, episode_index: int, destination_path: Path) -> dict[str, Any]:
    parquet_path = dataset_episode_parquet_path(dataset_root, episode_index)
    if not parquet_path.is_file():
        raise FileNotFoundError(parquet_path)
    table = pq.read_table(parquet_path, columns=[IMAGE_COLUMN, "timestamp", "frame_index"])
    encoded_frames = table[IMAGE_COLUMN].to_pylist()
    timestamps = np.asarray(table["timestamp"].to_numpy(), dtype=np.float64)
    frame_indices = np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)
    fps = infer_fps_from_timestamps(timestamps, fallback=30.0)

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    writer = iio.get_writer(destination_path, fps=fps, codec="libx264")
    try:
        for payload in tqdm(encoded_frames, desc=f"video/episode_{episode_index:06d}", unit="frame", leave=False):
            writer.append_data(decode_rgb_frame(payload))
    finally:
        writer.close()

    return {
        "fps": float(fps),
        "frame_count": int(len(encoded_frames)),
        "frame_index_start": int(frame_indices[0]) if frame_indices.size else None,
        "frame_index_end": int(frame_indices[-1]) if frame_indices.size else None,
        "parquet_path": str(parquet_path),
    }


def main() -> None:
    args = parse_args()
    dataset_root = args.human_dataset_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    requested_categories = {canonical_category_id(item) for item in args.category}
    rng = np.random.default_rng(int(args.seed))

    grouped = discover_annotated_episodes(dataset_root)
    annotated_lookup = annotated_episode_lookup(grouped)
    dataset_episode_indices = discover_dataset_episode_indices(dataset_root)
    default_categories = [category_id for category_id in DEFAULT_CATEGORY_ORDER if category_id in DEFAULT_CATEGORY_RANGES]
    selected_categories = (
        [category_id for category_id in DEFAULT_CATEGORY_ORDER if category_id in requested_categories]
        if requested_categories
        else default_categories
    )
    if requested_categories:
        unknown = sorted(requested_categories - set(DEFAULT_CATEGORY_RANGES))
        if unknown:
            raise ValueError(f"Unknown requested categories: {unknown}")

    output_root.mkdir(parents=True, exist_ok=True)
    selections: list[dict[str, Any]] = []

    for category_id in tqdm(selected_categories, desc="categories", unit="category"):
        candidates = grouped.get(category_id, [])
        selection_mode = "annotated" if args.prefer_annotated else "category_range"
        choice: dict[str, Any] | None = None
        if args.prefer_annotated and candidates:
            choice = candidates[int(rng.integers(0, len(candidates)))]
            episode_index = int(choice["episode_index"])
        else:
            fallback_candidates = fallback_candidates_for_category(category_id, dataset_episode_indices)
            if not fallback_candidates:
                raise RuntimeError(f"No selectable episodes found for category {category_id!r}")
            episode_index = int(fallback_candidates[int(rng.integers(0, len(fallback_candidates)))])
            choice = annotated_lookup.get(episode_index)
            if choice is not None:
                selection_mode = "category_range_with_annotation"
            elif args.prefer_annotated:
                selection_mode = "range_fallback"
        prompt_dir = output_root / f"{category_id}_episode_{episode_index:06d}"
        if prompt_dir.exists():
            if args.overwrite:
                for child in prompt_dir.iterdir():
                    if child.is_file():
                        child.unlink()
                    elif child.is_dir():
                        raise RuntimeError(f"Refusing to overwrite nested directory inside {prompt_dir}")
            else:
                raise FileExistsError(f"{prompt_dir} already exists. Use --overwrite to replace it.")
        prompt_dir.mkdir(parents=True, exist_ok=True)

        video_filename = f"{category_id}_episode_{episode_index:06d}_top_view.mp4"
        video_summary = export_top_view_video(dataset_root, episode_index, prompt_dir / video_filename)
        checkpoint_payload = choice["payload"] if choice is not None else {}
        metadata = {
            "category_id": category_id,
            "episode_index": episode_index,
            "prompt_id": f"temp_{category_id}_{episode_index:06d}",
            "video_filename": video_filename,
            "selection_mode": selection_mode,
            "checkpoint_path": str(choice["checkpoint_path"]) if choice is not None else None,
            "source_video_file": checkpoint_payload.get("video_file"),
            "task_name": checkpoint_payload.get("task", {}).get("name"),
            "task_garment_type": checkpoint_payload.get("task", {}).get("garment_type", category_id),
            "supports_predicted_width": bool(choice is not None),
            "supports_fixed_future_frames": True,
            **video_summary,
        }
        (prompt_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        selections.append(metadata)

    summary = {
        "human_dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "seed": int(args.seed),
        "selected_categories": selected_categories,
        "num_prompts": len(selections),
        "prompts": selections,
    }
    (output_root / "selector_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved {len(selections)} temp prompt episodes to {output_root}")


if __name__ == "__main__":
    main()
