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


VIEW_COLUMNS = {
    "top": "observation.images.top_rgb",
    "left": "observation.images.left_rgb",
    "right": "observation.images.right_rgb",
}
STATE_COLUMN = "observation.state"
ACTION_COLUMN = "actions"


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


def _select_dimensions(array: np.ndarray, dimensions: list[int] | str) -> np.ndarray:
    if dimensions == "all":
        return array
    return array[:, [int(value) for value in dimensions]]


def _dimension_count(info: dict[str, Any], column: str) -> int:
    shape = info["features"][column]["shape"]
    if not shape:
        raise ValueError(f"Feature {column!r} does not expose a vector shape")
    return int(shape[0])


def resolve_state_action_dims(config: dict[str, Any], info: dict[str, Any]) -> tuple[list[int], list[int]]:
    dataset = config["dataset"]
    state_dims = dataset.get("state_dims", "all")
    action_dims = dataset.get("action_dims", "all")
    if state_dims == "all":
        state_dims = list(range(_dimension_count(info, STATE_COLUMN)))
    if action_dims == "all":
        action_dims = list(range(_dimension_count(info, ACTION_COLUMN)))
    return [int(value) for value in state_dims], [int(value) for value in action_dims]


def compute_state_action_normalization(
    episodes: list[EpisodeRecord],
    state_dims: list[int],
    action_dims: list[int],
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    state_moments = RunningMoments(len(state_dims))
    action_moments = RunningMoments(len(action_dims))
    for record in tqdm(episodes, desc="normalization/episodes", unit="episode"):
        table = pq.read_table(record.path, columns=[STATE_COLUMN, ACTION_COLUMN])
        states = np.asarray(table[STATE_COLUMN].to_pylist(), dtype=np.float32)[:, state_dims]
        actions = np.asarray(table[ACTION_COLUMN].to_pylist(), dtype=np.float32)[:, action_dims]
        state_moments.update(states)
        action_moments.update(actions)
    state_mean, state_std = state_moments.result()
    action_mean, action_std = action_moments.result()
    result = {
        "state_dimensions": list(state_dims),
        "action_dimensions": list(action_dims),
        "state_mean": state_mean.tolist(),
        "state_std": state_std.tolist(),
        "action_mean": action_mean.tolist(),
        "action_std": action_std.tolist(),
        "episodes": len(episodes),
        "state_frames": state_moments.count,
        "action_frames": action_moments.count,
    }
    if output_path is not None:
        atomic_json_dump(result, output_path)
    return result


class MultiViewTransform:
    """Aspect-preserving letterbox followed by DINO/ImageNet normalization."""

    def __init__(self, image_size: int) -> None:
        self.image_size = int(image_size)
        self.mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)[:, None, None]
        self.std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)[:, None, None]

    def __call__(self, payload: bytes | dict[str, Any]) -> torch.Tensor:
        if isinstance(payload, dict):
            payload = payload.get("bytes")
        if not payload:
            raise ValueError("Image row does not contain encoded image bytes")
        with Image.open(io.BytesIO(payload)) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image.thumbnail((self.image_size, self.image_size), Image.Resampling.BICUBIC)
            canvas = Image.new("RGB", (self.image_size, self.image_size), color=(124, 116, 104))
            offset = ((self.image_size - image.width) // 2, (self.image_size - image.height) // 2)
            canvas.paste(image, offset)
            array = np.asarray(canvas, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        return (tensor - self.mean) / self.std


class EpisodeCache:
    def __init__(self, capacity: int, views: list[str]) -> None:
        self.capacity = max(1, int(capacity))
        self.views = list(views)
        self._items: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def get(self, path: Path) -> dict[str, Any]:
        key = str(path)
        if key in self._items:
            value = self._items.pop(key)
            self._items[key] = value
            return value

        image_columns = [VIEW_COLUMNS[view] for view in self.views]
        table = pq.read_table(
            path,
            columns=[*image_columns, STATE_COLUMN, ACTION_COLUMN, "timestamp", "frame_index"],
        )
        value = {
            "images": {view: table[VIEW_COLUMNS[view]].to_pylist() for view in self.views},
            "states": np.asarray(table[STATE_COLUMN].to_pylist(), dtype=np.float32),
            "actions": np.asarray(table[ACTION_COLUMN].to_pylist(), dtype=np.float32),
            "timestamps": np.asarray(table["timestamp"].to_numpy(), dtype=np.float64),
            "frame_indices": np.asarray(table["frame_index"].to_numpy(), dtype=np.int64),
        }
        self._items[key] = value
        while len(self._items) > self.capacity:
            self._items.popitem(last=False)
        return value


class MultiViewTemporalWindowDataset(Dataset[dict[str, torch.Tensor]]):
    """Episode-safe current/future windows for top, left-wrist, and right-wrist cameras."""

    def __init__(
        self,
        episodes: list[EpisodeRecord],
        horizons: list[int],
        stride: int,
        image_size: int,
        state_dims: list[int],
        action_dims: list[int],
        state_mean: list[float],
        state_std: list[float],
        action_mean: list[float],
        action_std: list[float],
        views: list[str] | None = None,
        cache_episodes: int = 2,
        include_future_images: bool = True,
    ) -> None:
        self.episodes = episodes
        self.horizons = sorted({int(value) for value in horizons})
        if not self.horizons or self.horizons[0] <= 0:
            raise ValueError("Horizons must contain positive frame offsets")
        self.max_horizon = self.horizons[-1]
        self.views = views or ["top", "left", "right"]
        unknown_views = set(self.views) - set(VIEW_COLUMNS)
        if unknown_views:
            raise ValueError(f"Unknown view names: {sorted(unknown_views)}")
        self.transform = MultiViewTransform(image_size)
        self.state_dims = [int(value) for value in state_dims]
        self.action_dims = [int(value) for value in action_dims]
        self.state_mean = np.asarray(state_mean, dtype=np.float32)
        self.state_std = np.asarray(state_std, dtype=np.float32)
        self.action_mean = np.asarray(action_mean, dtype=np.float32)
        self.action_std = np.asarray(action_std, dtype=np.float32)
        self.cache = EpisodeCache(cache_episodes, self.views)
        self.include_future_images = bool(include_future_images)
        self.windows: list[tuple[int, int]] = []
        for episode_position, record in enumerate(episodes):
            last_start_exclusive = record.length - self.max_horizon
            for start in range(0, max(0, last_start_exclusive), max(1, stride)):
                self.windows.append((episode_position, start))

    def __len__(self) -> int:
        return len(self.windows)

    def _normalize_state(self, values: np.ndarray) -> np.ndarray:
        return (values - self.state_mean) / self.state_std

    def _normalize_action(self, values: np.ndarray) -> np.ndarray:
        return (values - self.action_mean) / self.action_std

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode_position, start = self.windows[index]
        record = self.episodes[episode_position]
        episode = self.cache.get(record.path)

        image_positions = [start]
        if self.include_future_images:
            image_positions.extend(start + horizon for horizon in self.horizons)
        if image_positions[-1] >= len(episode["states"]):
            raise IndexError(
                f"Episode {record.episode_index} window start {start} requests position "
                f"{image_positions[-1]}, but the episode has {len(episode['states'])} rows"
            )

        images = torch.stack(
            [
                torch.stack([self.transform(episode["images"][view][position]) for view in self.views])
                for position in image_positions
            ]
        )
        states = episode["states"][:, self.state_dims]
        actions = episode["actions"][:, self.action_dims]
        current_state = self._normalize_state(states[start])
        current_action = self._normalize_action(actions[start])
        future_states = self._normalize_state(states[[start + horizon for horizon in self.horizons]])
        future_actions = self._normalize_action(actions[[start + horizon for horizon in self.horizons]])
        action_path = self._normalize_action(actions[start : start + self.max_horizon])

        return {
            "images": images,
            "state": torch.from_numpy(current_state.astype(np.float32)),
            "action": torch.from_numpy(current_action.astype(np.float32)),
            "future_states": torch.from_numpy(future_states.astype(np.float32)),
            "future_actions": torch.from_numpy(future_actions.astype(np.float32)),
            "action_path": torch.from_numpy(action_path.astype(np.float32)),
            "episode_index": torch.tensor(record.episode_index, dtype=torch.int64),
            "frame_index": torch.tensor(int(episode["frame_indices"][start]), dtype=torch.int64),
        }


def validate_dataset(root: str | Path, max_episodes: int | None = None) -> dict[str, Any]:
    root = Path(root)
    info = read_dataset_info(root)
    episodes = discover_episodes(root)
    selected = episodes[:max_episodes] if max_episodes else episodes
    failures: list[dict[str, Any]] = []
    frames = 0
    required_columns = [*VIEW_COLUMNS.values(), STATE_COLUMN, ACTION_COLUMN, "timestamp", "frame_index"]
    for record in tqdm(selected, desc="validate-dataset/episodes", unit="episode"):
        schema = pq.read_schema(record.path)
        missing = [column for column in required_columns if column not in schema.names]
        if missing:
            failures.append({"episode": record.episode_index, "reason": "missing columns", "columns": missing})
            continue
        table = pq.read_table(record.path, columns=required_columns)
        frames += table.num_rows
        if table.num_rows != record.length:
            failures.append({"episode": record.episode_index, "reason": "length mismatch"})
        states = np.asarray(table[STATE_COLUMN].to_pylist(), dtype=np.float32)
        actions = np.asarray(table[ACTION_COLUMN].to_pylist(), dtype=np.float32)
        if states.ndim != 2 or actions.ndim != 2:
            failures.append({"episode": record.episode_index, "reason": "state/action not matrix-shaped"})
        for view, column in VIEW_COLUMNS.items():
            images = table[column].to_pylist()
            if any(not isinstance(item, dict) or not item.get("bytes") for item in images):
                failures.append({"episode": record.episode_index, "reason": f"missing {view} image bytes"})
    return {
        "root": str(root.resolve()),
        "episodes_checked": len(selected),
        "frames_checked": frames,
        "features": info.get("features", {}),
        "failures": failures,
        "valid": not failures,
    }

