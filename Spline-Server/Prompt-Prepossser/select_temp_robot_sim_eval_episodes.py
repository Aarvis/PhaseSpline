from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any

import imageio.v2 as iio
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageOps
from tqdm.auto import tqdm


IMAGE_COLUMN = "observation.images.top_rgb"
DEFAULT_ROBOT_SIM_DATASET_ROOT = Path(
    "E:/Lehome-Dataset/lehome_round_2_dataset/sim_dataset/robot_sim_ft_lehome_all_garment_data_z180"
)
DEFAULT_OUTPUT_ROOT = Path(
    "D:/LeHome-Challenge/Lehome-Spline-ICRA2027/Spline-Server/Prompt-Prepossser/temp_robot_sim_unseen_eval"
)
DEFAULT_CATEGORY_ORDER = ["pants", "top_short_sleeve"]
DEFAULT_CATEGORY_RANGES: dict[str, tuple[int, int]] = {
    "pants": (0, 250),
    "shorts": (250, 500),
    "top_long_sleeve": (500, 750),
    "top_short_sleeve": (750, 1000),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample robot-sim episodes for unseen-category annotation/eval prep and export only top-view MP4s "
            "plus metadata. This does not run prompt preprocessing."
        )
    )
    parser.add_argument("--robot-sim-dataset-root", type=Path, default=DEFAULT_ROBOT_SIM_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--category",
        action="append",
        default=[],
        help="Repeat to limit sampling to specific category ids. Defaults to pants and top_short_sleeve.",
    )
    return parser.parse_args()


def canonical_category_id(value: str) -> str:
    return str(value).strip().lower()


def dataset_episode_parquet_path(dataset_root: Path, episode_index: int) -> Path:
    return dataset_root / "data" / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}.parquet"


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


def candidates_for_category(category_id: str, dataset_episode_indices: list[int]) -> list[int]:
    if category_id not in DEFAULT_CATEGORY_RANGES:
        return []
    start, end_exclusive = DEFAULT_CATEGORY_RANGES[category_id]
    return [episode_index for episode_index in dataset_episode_indices if start <= episode_index < end_exclusive]


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
    dataset_root = args.robot_sim_dataset_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    requested_categories = {canonical_category_id(item) for item in args.category}
    rng = np.random.default_rng(int(args.seed))

    dataset_episode_indices = discover_dataset_episode_indices(dataset_root)
    selected_categories = (
        [category_id for category_id in DEFAULT_CATEGORY_ORDER if category_id in requested_categories]
        if requested_categories
        else list(DEFAULT_CATEGORY_ORDER)
    )
    if requested_categories:
        unknown = sorted(requested_categories - set(DEFAULT_CATEGORY_RANGES))
        if unknown:
            raise ValueError(f"Unknown requested categories: {unknown}")

    output_root.mkdir(parents=True, exist_ok=True)
    selections: list[dict[str, Any]] = []

    for category_id in tqdm(selected_categories, desc="categories", unit="category"):
        candidates = candidates_for_category(category_id, dataset_episode_indices)
        if not candidates:
            raise RuntimeError(f"No selectable robot-sim episodes found for category {category_id!r}")
        episode_index = int(candidates[int(rng.integers(0, len(candidates)))])

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
        metadata = {
            "category_id": category_id,
            "episode_index": episode_index,
            "selection_mode": "category_range_random",
            "video_filename": video_filename,
            "supports_predicted_width": False,
            "supports_fixed_future_frames": True,
            **video_summary,
        }
        (prompt_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        selections.append(metadata)

    summary = {
        "robot_sim_dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "seed": int(args.seed),
        "selected_categories": selected_categories,
        "num_prompts": len(selections),
        "prompts": selections,
    }
    (output_root / "selector_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved {len(selections)} robot-sim unseen-eval episodes to {output_root}")


if __name__ == "__main__":
    main()
