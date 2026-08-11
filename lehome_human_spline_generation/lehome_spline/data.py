from __future__ import annotations

import io
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from .utils import RunningMoments, atomic_json_dump


IMAGE_COLUMN = "observation.images.top_rgb"
STATE_COLUMN = "observation.state"
ACTION_COLUMN = "actions"
STATE_DIMS = (0, 1, 8, 9)


@dataclass(frozen=True)
class EpisodeRecord:
    episode_index: int
    length: int
    path: Path


def read_dataset_info(root: str | Path) -> dict[str, Any]:
    path = Path(root) / "meta" / "info.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def discover_episodes(root: str | Path) -> list[EpisodeRecord]:
    root = Path(root)
    metadata_path = root / "meta" / "episodes.jsonl"
    lengths: dict[int, int] = {}
    with metadata_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            lengths[int(item["episode_index"])] = int(item["length"])

    records: list[EpisodeRecord] = []
    for episode_index, length in sorted(lengths.items()):
        chunk = episode_index // 1000
        path = root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"Episode metadata points to missing file: {path}")
        records.append(EpisodeRecord(episode_index, length, path))
    return records


def split_episodes(
    episodes: list[EpisodeRecord], seed: int, train_ratio: float, val_ratio: float
) -> dict[str, list[EpisodeRecord]]:
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio > 1:
        raise ValueError("Expected train_ratio > 0, val_ratio >= 0, and train_ratio + val_ratio <= 1")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(episodes))
    train_end = round(len(episodes) * train_ratio)
    val_end = train_end + round(len(episodes) * val_ratio)
    return {
        "train": [episodes[index] for index in order[:train_end]],
        "val": [episodes[index] for index in order[train_end:val_end]],
        "test": [episodes[index] for index in order[val_end:]],
    }


def save_splits(splits: dict[str, list[EpisodeRecord]], path: str | Path) -> None:
    atomic_json_dump(
        {name: [record.episode_index for record in records] for name, records in splits.items()},
        path,
    )


def compute_state_normalization(
    episodes: list[EpisodeRecord], output_path: str | Path | None = None
) -> dict[str, Any]:
    moments = RunningMoments(len(STATE_DIMS))
    for record in tqdm(episodes, desc="normalization/episodes", unit="episode"):
        table = pq.read_table(record.path, columns=[STATE_COLUMN])
        states = np.asarray(table[STATE_COLUMN].to_pylist(), dtype=np.float32)[:, STATE_DIMS]
        moments.update(states)
    mean, std = moments.result()
    result = {
        "dimensions": list(STATE_DIMS),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "episodes": len(episodes),
        "frames": moments.count,
    }
    if output_path is not None:
        atomic_json_dump(result, output_path)
    return result


class TopViewTransform:
    """Aspect-preserving letterbox followed by DINO/ImageNet normalization."""

    def __init__(self, image_size: int) -> None:
        self.image_size = int(image_size)
        self.mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)[:, None, None]
        self.std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)[:, None, None]

    def __call__(self, payload: bytes | dict[str, Any]) -> torch.Tensor:
        if isinstance(payload, dict):
            payload = payload.get("bytes")
        if not payload:
            raise ValueError("Top-view image row does not contain encoded image bytes")
        with Image.open(io.BytesIO(payload)) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((self.image_size, self.image_size), Image.Resampling.BICUBIC)
            # ImageNet-mean padding becomes approximately zero after normalization.
            canvas = Image.new("RGB", (self.image_size, self.image_size), color=(124, 116, 104))
            offset = ((self.image_size - image.width) // 2, (self.image_size - image.height) // 2)
            canvas.paste(image, offset)
            array = np.asarray(canvas, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        return (tensor - self.mean) / self.std


class EpisodeCache:
    def __init__(self, capacity: int) -> None:
        self.capacity = max(1, int(capacity))
        self._items: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def get(self, path: Path) -> dict[str, Any]:
        key = str(path)
        if key in self._items:
            value = self._items.pop(key)
            self._items[key] = value
            return value
        table = pq.read_table(
            path,
            columns=[IMAGE_COLUMN, STATE_COLUMN, ACTION_COLUMN, "timestamp", "frame_index"],
        )
        value = {
            "images": table[IMAGE_COLUMN].to_pylist(),
            "states": np.asarray(table[STATE_COLUMN].to_pylist(), dtype=np.float32),
            "actions": np.asarray(table[ACTION_COLUMN].to_pylist(), dtype=np.float32),
            "timestamps": np.asarray(table["timestamp"].to_numpy(), dtype=np.float64),
            "frame_indices": np.asarray(table["frame_index"].to_numpy(), dtype=np.int64),
        }
        self._items[key] = value
        while len(self._items) > self.capacity:
            self._items.popitem(last=False)
        return value


class TemporalWindowDataset(Dataset[dict[str, torch.Tensor]]):
    """Episode-safe windows with current/future images and the complete intervening action path."""

    def __init__(
        self,
        episodes: list[EpisodeRecord],
        horizons: list[int],
        stride: int,
        image_size: int,
        state_mean: list[float],
        state_std: list[float],
        cache_episodes: int = 2,
        include_future_images: bool = True,
    ) -> None:
        self.episodes = episodes
        self.horizons = sorted({int(value) for value in horizons})
        if not self.horizons or self.horizons[0] <= 0:
            raise ValueError("Horizons must contain positive frame offsets")
        self.max_horizon = self.horizons[-1]
        self.transform = TopViewTransform(image_size)
        self.state_mean = np.asarray(state_mean, dtype=np.float32)
        self.state_std = np.asarray(state_std, dtype=np.float32)
        self.cache = EpisodeCache(cache_episodes)
        self.include_future_images = bool(include_future_images)
        self.windows: list[tuple[int, int]] = []
        for episode_position, record in enumerate(episodes):
            # The largest requested image index is start + max_horizon and must
            # remain strictly smaller than record.length. The range endpoint is
            # therefore exclusive at record.length - max_horizon.
            last_start_exclusive = record.length - self.max_horizon
            for start in range(0, max(0, last_start_exclusive), max(1, stride)):
                self.windows.append((episode_position, start))

    def __len__(self) -> int:
        return len(self.windows)

    def _normalize_state(self, values: np.ndarray) -> np.ndarray:
        return (values - self.state_mean) / self.state_std

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode_position, start = self.windows[index]
        record = self.episodes[episode_position]
        episode = self.cache.get(record.path)

        image_positions = [start]
        if self.include_future_images:
            image_positions.extend(start + horizon for horizon in self.horizons)
        if image_positions[-1] >= len(episode["images"]):
            raise IndexError(
                f"Episode {record.episode_index} window start {start} requests image position "
                f"{image_positions[-1]}, but the episode has {len(episode['images'])} rows"
            )
        images = torch.stack([self.transform(episode["images"][position]) for position in image_positions])

        states = episode["states"][:, STATE_DIMS]
        actions = episode["actions"][:, STATE_DIMS]
        current_state = self._normalize_state(states[start])
        action_path = self._normalize_state(actions[start : start + self.max_horizon])

        return {
            "images": images,
            "state": torch.from_numpy(current_state.astype(np.float32)),
            "actions": torch.from_numpy(action_path.astype(np.float32)),
            "episode_index": torch.tensor(record.episode_index, dtype=torch.int64),
            "frame_index": torch.tensor(int(episode["frame_indices"][start]), dtype=torch.int64),
        }


def validate_dataset(
    root: str | Path,
    max_episodes: int | None = None,
    atol: float = 1e-6,
) -> dict[str, Any]:
    episodes = discover_episodes(root)
    selected = episodes[:max_episodes] if max_episodes else episodes
    failures: list[dict[str, Any]] = []
    frames = 0
    max_error = 0.0
    for record in tqdm(selected, desc="validate-dataset/episodes", unit="episode"):
        table = pq.read_table(record.path, columns=[IMAGE_COLUMN, STATE_COLUMN, ACTION_COLUMN])
        states = np.asarray(table[STATE_COLUMN].to_pylist(), dtype=np.float32)
        actions = np.asarray(table[ACTION_COLUMN].to_pylist(), dtype=np.float32)
        images = table[IMAGE_COLUMN].to_pylist()
        frames += len(states)
        if len(states) != record.length or len(actions) != record.length or len(images) != record.length:
            failures.append({"episode": record.episode_index, "reason": "length mismatch"})
            continue
        if len(states) > 1:
            error = float(np.max(np.abs(actions[:-1, STATE_DIMS] - states[1:, STATE_DIMS])))
            max_error = max(max_error, error)
            if error > atol:
                failures.append(
                    {"episode": record.episode_index, "reason": "action != next state", "max_error": error}
                )
        if any(not isinstance(item, dict) or not item.get("bytes") for item in images):
            failures.append({"episode": record.episode_index, "reason": "missing top-view bytes"})
    return {
        "root": str(Path(root).resolve()),
        "episodes_checked": len(selected),
        "frames_checked": frames,
        "state_dimensions": list(STATE_DIMS),
        "max_action_to_next_state_error": max_error,
        "tolerance": atol,
        "failures": failures,
        "valid": not failures,
    }
