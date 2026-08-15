from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .bootstrap import ensure_repo_imports

ensure_repo_imports()

from human_spline_localizer.data import (  # noqa: E402
    AnnotationEpisode,
    interpolate_human_u,
    load_annotation_episode,
    load_human_frame_u_by_frame_index,
)


@dataclass(frozen=True)
class PromptPackage:
    prompt_id: str
    category_id: str
    package_dir: Path
    frame_count: int
    supports_fixed_future_frames: bool
    supports_predicted_width: bool
    source_type: str
    spline_path: Path
    localizer_cache_path: Path
    frame_embeddings_path: Path | None
    annotation_path: Path | None


class LoadedPromptPackage:
    def __init__(self, package: PromptPackage) -> None:
        self.package = package
        with np.load(package.spline_path, allow_pickle=False) as archive:
            self.global_knots = np.asarray(archive["global_knots"], dtype=np.float32)
            self.global_coefficients = np.asarray(archive["global_coefficients"], dtype=np.float32)
            self.global_degree = int(np.asarray(archive["global_degree"]).reshape(-1)[0])
            self.frame_indices = np.asarray(archive["frame_indices"], dtype=np.int64)
            self.frame_u = np.asarray(archive["frame_u"], dtype=np.float32)
        with np.load(package.localizer_cache_path, allow_pickle=False) as cache_archive:
            self.left_support = np.asarray(cache_archive["left_support"], dtype=np.float32)
            self.right_support = np.asarray(cache_archive["right_support"], dtype=np.float32)
            self.support_midpoint = np.asarray(cache_archive["support_midpoint"], dtype=np.float32)
            self.support_width = np.asarray(cache_archive["support_width"], dtype=np.float32)
            self.greville_phase = np.asarray(cache_archive["greville_phase"], dtype=np.float32)
            self.basis = np.asarray(cache_archive["basis_200"], dtype=np.float32)
            self.coefficient_count = int(np.asarray(cache_archive["coefficient_count"]).reshape(-1)[0])
        self._annotation: AnnotationEpisode | None = None
        self._frame_u_by_frame_index: np.ndarray | None = None

    @property
    def annotation(self) -> AnnotationEpisode | None:
        if self.package.annotation_path is None:
            return None
        if self._annotation is None:
            self._annotation = load_annotation_episode(self.package.annotation_path, episode_index=0)
        return self._annotation

    @property
    def frame_u_by_frame_index(self) -> np.ndarray:
        if self._frame_u_by_frame_index is None:
            self._frame_u_by_frame_index = load_human_frame_u_by_frame_index(self.package.spline_path)
        return self._frame_u_by_frame_index

    def nearest_frame_row_for_u(self, value: float) -> int:
        index = int(np.argmin(np.abs(self.frame_u.astype(np.float64) - float(value))))
        return max(0, min(index, self.frame_u.shape[0] - 1))

    def future_end_u_from_frame_offset(self, start_u: float, future_frames: int) -> float:
        start_row = self.nearest_frame_row_for_u(start_u)
        end_row = min(start_row + max(1, int(future_frames)), self.frame_u.shape[0] - 1)
        return float(self.frame_u[end_row])

    def semantic_end_u(self, checkpoint_index: int, progress: float) -> float | None:
        annotation = self.annotation
        if annotation is None:
            return None
        end_u, valid = interpolate_human_u(
            annotation,
            self.frame_u_by_frame_index,
            int(checkpoint_index),
            float(progress),
        )
        return float(end_u) if valid else None


class PromptBank:
    def __init__(self, root: str | Path, seed: int = 2027) -> None:
        self.root = Path(root).expanduser().resolve()
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Prompt-bank manifest not found: {manifest_path}")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = payload.get("prompts", [])
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"Prompt-bank manifest has no prompt entries: {manifest_path}")
        self._packages: dict[str, PromptPackage] = {}
        self._packages_by_category: dict[str, list[PromptPackage]] = {}
        self._loaded_cache: dict[str, LoadedPromptPackage] = {}
        self._rng = random.Random(int(seed))

        for entry in entries:
            prompt_id = str(entry["prompt_id"])
            category_id = str(entry["category_id"])
            package_dir = (self.root / str(entry["relative_dir"])).resolve()
            package = PromptPackage(
                prompt_id=prompt_id,
                category_id=category_id,
                package_dir=package_dir,
                frame_count=int(entry["frame_count"]),
                supports_fixed_future_frames=bool(entry.get("supports_fixed_future_frames", True)),
                supports_predicted_width=bool(entry.get("supports_predicted_width", False)),
                source_type=str(entry.get("source_type", "unknown")),
                spline_path=(package_dir / "spline.npz").resolve(),
                localizer_cache_path=(package_dir / "localizer_cache.npz").resolve(),
                frame_embeddings_path=(package_dir / "frame_embeddings.npz").resolve()
                if (package_dir / "frame_embeddings.npz").is_file()
                else None,
                annotation_path=(package_dir / "annotation_checkpoints.json").resolve()
                if (package_dir / "annotation_checkpoints.json").is_file()
                else None,
            )
            self._packages[prompt_id] = package
            self._packages_by_category.setdefault(category_id, []).append(package)

    def categories(self) -> list[str]:
        return sorted(self._packages_by_category.keys())

    def prompt_ids(self) -> list[str]:
        return sorted(self._packages.keys())

    def choose(self, *, category_id: str | None = None, prompt_id: str | None = None) -> LoadedPromptPackage:
        if prompt_id is not None:
            package = self._packages.get(str(prompt_id))
            if package is None:
                raise KeyError(f"Prompt id {prompt_id!r} not found in prompt bank.")
            return self._load(package)
        if category_id is None:
            raise ValueError("Prompt selection requires either prompt_id or category_id.")
        category_packages = self._packages_by_category.get(str(category_id))
        if not category_packages:
            raise KeyError(f"No prompt packages available for category {category_id!r}.")
        package = self._rng.choice(category_packages)
        return self._load(package)

    def _load(self, package: PromptPackage) -> LoadedPromptPackage:
        loaded = self._loaded_cache.get(package.prompt_id)
        if loaded is not None:
            return loaded
        loaded = LoadedPromptPackage(package)
        self._loaded_cache[package.prompt_id] = loaded
        return loaded
