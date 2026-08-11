from __future__ import annotations

import json
import os
import random
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_json_dump(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        tmp_name = handle.name
    os.replace(tmp_name, path)


class RunningMoments:
    def __init__(self, dimension: int) -> None:
        self.dimension = int(dimension)
        self.count = 0
        self.mean = np.zeros((self.dimension,), dtype=np.float64)
        self.m2 = np.zeros((self.dimension,), dtype=np.float64)

    def update(self, batch: np.ndarray) -> None:
        batch = np.asarray(batch, dtype=np.float64)
        if batch.ndim != 2 or batch.shape[1] != self.dimension:
            raise ValueError(f"Expected batch shape [N,{self.dimension}], got {batch.shape}")
        for row in batch:
            self.count += 1
            delta = row - self.mean
            self.mean += delta / self.count
            delta_2 = row - self.mean
            self.m2 += delta * delta_2

    def result(self) -> tuple[np.ndarray, np.ndarray]:
        if self.count == 0:
            raise ValueError("Cannot finalize normalization statistics without samples.")
        variance = self.m2 / max(1, self.count)
        std = np.sqrt(np.maximum(variance, 1e-12))
        return self.mean.astype(np.float32), std.astype(np.float32)


class MetricAverages:
    def __init__(self) -> None:
        self.sums: dict[str, float] = {}
        self.count = 0

    def update(self, metrics: dict[str, float | torch.Tensor]) -> None:
        self.count += 1
        for key, value in metrics.items():
            scalar = float(value.detach().item()) if torch.is_tensor(value) else float(value)
            self.sums[key] = self.sums.get(key, 0.0) + scalar

    def result(self) -> dict[str, float]:
        return {key: value / max(1, self.count) for key, value in self.sums.items()}


class EpisodeLRUCache:
    def __init__(self, capacity: int) -> None:
        self.capacity = max(1, int(capacity))
        self._items: OrderedDict[int, Any] = OrderedDict()

    def get(self, key: int) -> Any | None:
        if key not in self._items:
            return None
        value = self._items.pop(key)
        self._items[key] = value
        return value

    def put(self, key: int, value: Any) -> None:
        self._items[key] = value
        self._items.move_to_end(key)
        while len(self._items) > self.capacity:
            self._items.popitem(last=False)
