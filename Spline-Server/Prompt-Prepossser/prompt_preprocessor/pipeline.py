from __future__ import annotations

import io
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as iio_v2
import imageio.v3 as iio
import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image, ImageOps
from torch.amp import autocast
from tqdm.auto import tqdm

_SPLINE_SERVER_ROOT = Path(__file__).resolve().parents[2]
if str(_SPLINE_SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(_SPLINE_SERVER_ROOT))

from spline_server.bootstrap import ensure_repo_imports  # noqa: E402

ensure_repo_imports()

from human_spline_localizer.data import checkpoint_json_path  # noqa: E402
from human_spline_localizer.spline import (  # noqa: E402
    compute_coefficient_geometry,
    evaluate_bspline_basis_matrix,
    normalize_knots_to_unit_domain,
)
from lehome_human_spline_generation.lehome_spline.bspline import fit_episode_bspline  # type: ignore[attr-defined]  # noqa: E402
from lehome_spline.data import TopViewTransform  # noqa: E402
from lehome_spline.model import VisualSplineVAE  # noqa: E402
from lehome_spline.utils import atomic_json_dump, atomic_npz  # noqa: E402


HUMAN_TOP_VIEW_COLUMN = "observation.images.top_rgb"
REFERENCE_VIDEO_FILENAME_DEFAULT = "top_view_reference.mp4"


@dataclass(frozen=True)
class PreprocessorPaths:
    prompt_bank_root: Path
    human_dataset_root: Path
    human_bspline_root: Path
    human_frame_embedding_root: Path | None
    human_embedder_config_path: Path
    human_embedder_checkpoint_path: Path
    human_embedding_normalization_path: Path


def _as_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _resolve_paths(config: dict[str, Any]) -> PreprocessorPaths:
    paths = config.get("paths")
    if not isinstance(paths, dict):
        raise KeyError("Missing paths block in prompt preprocessor config.")
    frame_embedding_root = paths.get("human_frame_embedding_root")
    return PreprocessorPaths(
        prompt_bank_root=_as_path(paths["prompt_bank_root"]),
        human_dataset_root=_as_path(paths["human_dataset_root"]),
        human_bspline_root=_as_path(paths["human_bspline_root"]),
        human_frame_embedding_root=_as_path(frame_embedding_root) if frame_embedding_root else None,
        human_embedder_config_path=_as_path(paths["human_embedder_config_path"]),
        human_embedder_checkpoint_path=_as_path(paths["human_embedder_checkpoint_path"]),
        human_embedding_normalization_path=_as_path(paths["human_embedding_normalization_path"]),
    )


def _phase_bin_count(config: dict[str, Any]) -> int:
    prompt_cfg = config.get("prompt_bank", {})
    value = prompt_cfg.get("phase_bin_count", 400)
    return int(value)


def _save_reference_top_view_video_enabled(config: dict[str, Any]) -> bool:
    prompt_cfg = config.get("prompt_bank", {})
    return bool(prompt_cfg.get("save_reference_top_view_video", False))


def _reference_video_filename(config: dict[str, Any]) -> str:
    prompt_cfg = config.get("prompt_bank", {})
    return str(prompt_cfg.get("reference_video_filename", REFERENCE_VIDEO_FILENAME_DEFAULT))


def _resolve_prompt_source_type(entry: dict[str, Any]) -> str:
    source = entry.get("source")
    if isinstance(source, dict) and "type" in source:
        return str(source["type"]).strip().lower()
    return str(entry.get("source_type", "dataset_episode")).strip().lower()


def _resolve_dataset_episode_index(entry: dict[str, Any]) -> int:
    if "human_episode_index" in entry:
        return int(entry["human_episode_index"])
    if "episode_index" in entry:
        return int(entry["episode_index"])
    raise KeyError("Dataset prompt entry must define either human_episode_index or episode_index.")


def _build_localizer_cache_from_spline(
    *,
    spline_npz_path: Path,
    destination_path: Path,
    phase_bin_count: int,
) -> None:
    with np.load(spline_npz_path, allow_pickle=False) as archive:
        coefficients = np.asarray(archive["global_coefficients"], dtype=np.float32)
        global_knots = np.asarray(archive["global_knots"], dtype=np.float64)
        degree = int(np.asarray(archive["global_degree"]).reshape(-1)[0])
    coefficient_count = int(coefficients.shape[0])
    normalized_knots = normalize_knots_to_unit_domain(global_knots, degree)
    geometry = compute_coefficient_geometry(normalized_knots, degree, coefficient_count)
    bin_centers = ((np.arange(int(phase_bin_count), dtype=np.float64) + 0.5) / float(phase_bin_count)).astype(np.float64)
    basis_200 = evaluate_bspline_basis_matrix(normalized_knots, degree, bin_centers, coefficient_count)
    atomic_npz(
        destination_path,
        compressed=True,
        coefficient_count=np.asarray(coefficient_count, dtype=np.int32),
        normalized_knots=normalized_knots.astype(np.float32),
        basis_200=basis_200.astype(np.float32),
        **geometry,
    )


def _human_spline_episode_path(root: Path, episode_index: int) -> Path:
    return root / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}" / "spline.npz"


def _human_frame_embedding_episode_path(root: Path, episode_index: int) -> Path:
    return root / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}" / "frame_embeddings.npz"


def _human_dataset_episode_parquet_path(root: Path, episode_index: int) -> Path:
    return root / "data" / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}.parquet"


def _copy_if_exists(source: Path | None, destination: Path) -> bool:
    if source is None or not source.is_file():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def _decode_top_view_rgb_frame(payload: bytes | dict[str, Any]) -> np.ndarray:
    if isinstance(payload, dict):
        payload = payload.get("bytes")
    if not payload:
        raise ValueError("Top-view image row does not contain encoded image bytes")
    with Image.open(io.BytesIO(payload)) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return np.asarray(image, dtype=np.uint8)


def _infer_video_fps_from_timestamps(timestamps: np.ndarray, default_fps: float) -> float:
    if timestamps.ndim != 1 or timestamps.size < 2:
        return max(float(default_fps), 1.0)
    deltas = np.diff(timestamps.astype(np.float64, copy=False))
    positive = deltas[deltas > 0.0]
    if positive.size == 0:
        return max(float(default_fps), 1.0)
    median_delta = float(np.median(positive))
    if not np.isfinite(median_delta) or median_delta <= 0.0:
        return max(float(default_fps), 1.0)
    return max(1.0 / median_delta, 1.0)


def _write_reference_video_mp4(
    frame_iterable: Any,
    destination_path: Path,
    *,
    fps: float,
) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    writer = iio_v2.get_writer(destination_path, fps=max(float(fps), 1.0), codec="libx264")
    try:
        for frame in frame_iterable:
            writer.append_data(np.asarray(frame, dtype=np.uint8))
    finally:
        writer.close()


def _export_dataset_episode_reference_video(
    *,
    dataset_root: Path,
    episode_index: int,
    destination_path: Path,
    default_fps: float,
) -> float:
    parquet_path = _human_dataset_episode_parquet_path(dataset_root, episode_index)
    if not parquet_path.is_file():
        raise FileNotFoundError(parquet_path)
    table = pq.read_table(parquet_path, columns=[HUMAN_TOP_VIEW_COLUMN, "timestamp"])
    timestamps = np.asarray(table["timestamp"].to_numpy(), dtype=np.float64)
    encoded_frames = table[HUMAN_TOP_VIEW_COLUMN].to_pylist()
    fps = _infer_video_fps_from_timestamps(timestamps, default_fps)

    def _iter_frames() -> Any:
        for payload in tqdm(
            encoded_frames,
            desc=f"prompt-video/episode-{episode_index:06d}",
            unit="frame",
            leave=False,
        ):
            yield _decode_top_view_rgb_frame(payload)

    _write_reference_video_mp4(_iter_frames(), destination_path, fps=fps)
    return float(fps)


def _export_video_file_reference_video(
    *,
    video_path: Path,
    destination_path: Path,
    fps: float,
) -> None:
    if video_path.suffix.lower() == ".mp4":
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(video_path, destination_path)
        return

    def _iter_frames() -> Any:
        for frame in tqdm(iio.imiter(video_path), desc=f"prompt-video/{video_path.stem}", unit="frame", leave=False):
            yield np.asarray(frame, dtype=np.uint8)

    _write_reference_video_mp4(_iter_frames(), destination_path, fps=fps)


class HumanVideoEmbedderRuntime:
    def __init__(self, config_path: Path, checkpoint_path: Path, device: str, amp: bool) -> None:
        from lehome_spline.config import load_config  # noqa: WPS433

        self.config = load_config(config_path)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.amp = bool(amp) and self.device.type == "cuda"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = VisualSplineVAE(self.config)
        model.load_state_dict(checkpoint["model"])
        self.model = model.to(self.device).eval()
        self.transform = TopViewTransform(int(self.config["model"]["dino"]["image_size"]))

    def encode_video_frames(
        self,
        *,
        video_path: Path,
        batch_size: int,
        fps: float,
        frame_stride: int,
        max_frames: int | None,
    ) -> dict[str, np.ndarray]:
        retained_frames: list[np.ndarray] = []
        for frame_index, frame in enumerate(tqdm(iio.imiter(video_path), desc=f"video/{video_path.stem}", unit="frame", leave=False)):
            if frame_index % max(1, int(frame_stride)) != 0:
                continue
            retained_frames.append(np.asarray(frame))
            if max_frames is not None and len(retained_frames) >= int(max_frames):
                break
        if len(retained_frames) < 2:
            raise ValueError(f"Prompt video must yield at least two retained frames after stride/max_frames filtering: {video_path}")

        means: list[np.ndarray] = []
        log_variances: list[np.ndarray] = []
        with torch.inference_mode():
            for start in tqdm(range(0, len(retained_frames), int(batch_size)), desc=f"embed/{video_path.stem}", unit="batch", leave=False):
                stop = min(len(retained_frames), start + int(batch_size))
                images = torch.stack([self.transform(frame) for frame in retained_frames[start:stop]]).to(self.device, non_blocking=True)
                with autocast(device_type=self.device.type, enabled=self.amp, dtype=torch.bfloat16):
                    dino = self.model.encode_dino(images)
                    posterior = self.model.posterior_from_patches(dino.patches)
                means.append(posterior.mean.flatten(1).float().cpu().numpy())
                log_variances.append(posterior.log_variance.flatten(1).float().cpu().numpy())

        mean = np.concatenate(means, axis=0).astype(np.float32, copy=False)
        log_variance = np.concatenate(log_variances, axis=0).astype(np.float32, copy=False)
        timestamps = (np.arange(len(retained_frames), dtype=np.float64) / max(float(fps), 1.0e-6)).astype(np.float64)
        frame_indices = np.arange(len(retained_frames), dtype=np.int64)
        frame_u = np.linspace(0.0, 1.0, len(retained_frames), dtype=np.float32)
        return {
            "mean": mean,
            "log_variance": log_variance,
            "timestamps": timestamps,
            "frame_indices": frame_indices,
            "frame_u": frame_u,
            "u": frame_u,
        }


def _dummy_state_payload(frame_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    state = np.zeros((frame_count, 4), dtype=np.float32)
    state_mean = np.zeros((4,), dtype=np.float64)
    state_std = np.ones((4,), dtype=np.float64)
    return state, state.copy(), state_mean, state_std


def _prepare_video_prompt(
    *,
    entry: dict[str, Any],
    paths: PreprocessorPaths,
    prompt_bank_config: dict[str, Any],
    embedder: HumanVideoEmbedderRuntime | None,
    package_dir: Path,
    phase_bin_count: int,
    save_reference_video: bool,
    reference_video_filename: str,
) -> dict[str, Any]:
    if embedder is None:
        raise RuntimeError(
            "Video-file prompt preprocessing requires the human video embedder, but it was not initialized."
        )
    video_path = _as_path(entry["video_path"])
    fps = float(entry.get("fps", prompt_bank_config.get("default_video_fps", 30.0)))
    frame_stride = int(entry.get("frame_stride", 1))
    max_frames = entry.get("max_frames")
    batch_size = int(prompt_bank_config.get("video_batch_size", 64))
    encoded = embedder.encode_video_frames(
        video_path=video_path,
        batch_size=batch_size,
        fps=fps,
        frame_stride=frame_stride,
        max_frames=int(max_frames) if max_frames is not None else None,
    )
    frame_count = int(encoded["mean"].shape[0])
    atomic_npz(package_dir / "frame_embeddings.npz", compressed=True, **encoded)

    with np.load(paths.human_embedding_normalization_path, allow_pickle=False) as norm_archive:
        latent_mean = np.asarray(norm_archive["mean"], dtype=np.float64)
        latent_std = np.asarray(norm_archive["std"], dtype=np.float64)

    dummy_state, dummy_action, state_mean, state_std = _dummy_state_payload(frame_count)
    spline_payload, spline_result = fit_episode_bspline(
        mean=encoded["mean"],
        state=dummy_state,
        timestamps=encoded["timestamps"],
        frame_indices=encoded["frame_indices"],
        latent_normalization_mean=latent_mean,
        latent_normalization_std=latent_std,
        state_normalization_mean=state_mean,
        state_normalization_std=state_std,
        config={
            "degree": 3,
            "initial_internal_knots": int(prompt_bank_config.get("initial_internal_knots", 8)),
            "max_internal_knots": int(prompt_bank_config.get("max_internal_knots", 256)),
            "endpoint_weight": float(prompt_bank_config.get("endpoint_weight", 1000.0)),
            "min_knot_spacing_frames": int(prompt_bank_config.get("min_knot_spacing_frames", 1)),
            "epsilon_latent_rmse": float(prompt_bank_config.get("epsilon_latent_rmse", 0.05)),
            "epsilon_cosine_distance": float(prompt_bank_config.get("epsilon_cosine_distance", 0.01)),
            "epsilon_state_rmse": float(prompt_bank_config.get("epsilon_state_rmse", 0.05)),
        },
        progress_description=f"fit/{entry['prompt_id']}",
    )
    atomic_npz(package_dir / "spline.npz", compressed=True, **spline_payload)
    _build_localizer_cache_from_spline(
        spline_npz_path=package_dir / "spline.npz",
        destination_path=package_dir / "localizer_cache.npz",
        phase_bin_count=phase_bin_count,
    )
    saved_reference_video = False
    if save_reference_video:
        _export_video_file_reference_video(
            video_path=video_path,
            destination_path=package_dir / reference_video_filename,
            fps=fps,
        )
        saved_reference_video = True
    atomic_json_dump(
        {
            "prompt_id": str(entry["prompt_id"]),
            "category_id": str(entry["category_id"]),
            "source_type": "video_file",
            "source_path": str(video_path),
            "fps": float(fps),
            "frame_stride": int(frame_stride),
            "max_frames": int(max_frames) if max_frames is not None else None,
            "frame_count": frame_count,
            "saved_reference_top_view_video": bool(saved_reference_video),
            "reference_top_view_video_filename": reference_video_filename if saved_reference_video else None,
            "supports_fixed_future_frames": True,
            "supports_predicted_width": False,
            "fit_summary": spline_result.__dict__,
        },
        package_dir / "metadata.json",
    )
    return {
        "prompt_id": str(entry["prompt_id"]),
        "category_id": str(entry["category_id"]),
        "relative_dir": str(package_dir.name),
        "frame_count": frame_count,
        "saved_reference_top_view_video": bool(saved_reference_video),
        "reference_top_view_video_filename": reference_video_filename if saved_reference_video else None,
        "supports_fixed_future_frames": True,
        "supports_predicted_width": False,
        "source_type": "video_file",
    }


def _prepare_dataset_prompt(
    *,
    entry: dict[str, Any],
    paths: PreprocessorPaths,
    prompt_bank_config: dict[str, Any],
    package_dir: Path,
    phase_bin_count: int,
    save_reference_video: bool,
    reference_video_filename: str,
) -> dict[str, Any]:
    episode_index = _resolve_dataset_episode_index(entry)
    spline_source = _human_spline_episode_path(paths.human_bspline_root, episode_index)
    if not spline_source.is_file():
        raise FileNotFoundError(spline_source)
    shutil.copy2(spline_source, package_dir / "spline.npz")
    _build_localizer_cache_from_spline(
        spline_npz_path=package_dir / "spline.npz",
        destination_path=package_dir / "localizer_cache.npz",
        phase_bin_count=phase_bin_count,
    )

    copied_frame_embeddings = False
    if paths.human_frame_embedding_root is not None:
        copied_frame_embeddings = _copy_if_exists(
            _human_frame_embedding_episode_path(paths.human_frame_embedding_root, episode_index),
            package_dir / "frame_embeddings.npz",
        )
    annotation_source = checkpoint_json_path(paths.human_dataset_root, episode_index)
    copied_annotation = _copy_if_exists(annotation_source, package_dir / "annotation_checkpoints.json")
    with np.load(package_dir / "spline.npz", allow_pickle=False) as archive:
        frame_count = int(np.asarray(archive["frame_indices"]).shape[0])
    saved_reference_video = False
    reference_video_fps: float | None = None
    if save_reference_video:
        reference_video_fps = _export_dataset_episode_reference_video(
            dataset_root=paths.human_dataset_root,
            episode_index=episode_index,
            destination_path=package_dir / reference_video_filename,
            default_fps=float(prompt_bank_config.get("default_video_fps", 30.0)),
        )
        saved_reference_video = True
    atomic_json_dump(
        {
            "prompt_id": str(entry["prompt_id"]),
            "category_id": str(entry["category_id"]),
            "source_type": "dataset_episode",
            "episode_index": episode_index,
            "frame_count": frame_count,
            "copied_frame_embeddings": bool(copied_frame_embeddings),
            "copied_annotation": bool(copied_annotation),
            "saved_reference_top_view_video": bool(saved_reference_video),
            "reference_top_view_video_filename": reference_video_filename if saved_reference_video else None,
            "reference_top_view_video_fps": float(reference_video_fps) if reference_video_fps is not None else None,
            "supports_fixed_future_frames": True,
            "supports_predicted_width": bool(copied_annotation),
        },
        package_dir / "metadata.json",
    )
    return {
        "prompt_id": str(entry["prompt_id"]),
        "category_id": str(entry["category_id"]),
        "relative_dir": str(package_dir.name),
        "frame_count": frame_count,
        "saved_reference_top_view_video": bool(saved_reference_video),
        "reference_top_view_video_filename": reference_video_filename if saved_reference_video else None,
        "supports_fixed_future_frames": True,
        "supports_predicted_width": bool(copied_annotation),
        "source_type": "dataset_episode",
    }


def preprocess_prompt_bank(config: dict[str, Any]) -> Path:
    paths = _resolve_paths(config)
    prompt_bank_cfg = config.get("prompt_bank", {})
    overwrite = bool(prompt_bank_cfg.get("overwrite", False))
    phase_bin_count = _phase_bin_count(config)
    save_reference_video = _save_reference_top_view_video_enabled(config)
    reference_video_filename = _reference_video_filename(config)
    paths.prompt_bank_root.mkdir(parents=True, exist_ok=True)

    prompts = config.get("prompts", [])
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("Config must contain a non-empty prompts list.")

    needs_embedder = False
    for entry in prompts:
        if not isinstance(entry, dict):
            raise ValueError(f"Prompt entry must be a mapping, got {type(entry)!r}")
        if _resolve_prompt_source_type(entry) == "video_file":
            needs_embedder = True
            break

    embedder: HumanVideoEmbedderRuntime | None = None
    if needs_embedder:
        embedder = HumanVideoEmbedderRuntime(
            config_path=paths.human_embedder_config_path,
            checkpoint_path=paths.human_embedder_checkpoint_path,
            device=str(prompt_bank_cfg.get("device", "cuda")),
            amp=bool(prompt_bank_cfg.get("amp", True)),
        )

    manifest_entries: list[dict[str, Any]] = []
    for entry in tqdm(prompts, desc="prompt-bank/prompts", unit="prompt"):
        prompt_id = str(entry["prompt_id"])
        package_dir = paths.prompt_bank_root / prompt_id
        if package_dir.exists() and overwrite:
            shutil.rmtree(package_dir)
        package_dir.mkdir(parents=True, exist_ok=True)
        source_type = _resolve_prompt_source_type(entry)
        if source_type == "dataset_episode":
            manifest_entry = _prepare_dataset_prompt(
                entry=entry,
                paths=paths,
                prompt_bank_config=prompt_bank_cfg,
                package_dir=package_dir,
                phase_bin_count=phase_bin_count,
                save_reference_video=save_reference_video,
                reference_video_filename=reference_video_filename,
            )
        elif source_type == "video_file":
            manifest_entry = _prepare_video_prompt(
                entry=entry,
                paths=paths,
                prompt_bank_config=prompt_bank_cfg,
                embedder=embedder,
                package_dir=package_dir,
                phase_bin_count=phase_bin_count,
                save_reference_video=save_reference_video,
                reference_video_filename=reference_video_filename,
            )
        else:
            raise ValueError(f"Unsupported prompt source_type: {source_type!r}")
        manifest_entries.append(manifest_entry)

    manifest = {
        "prompt_bank_root": str(paths.prompt_bank_root),
        "phase_bin_count": int(phase_bin_count),
        "save_reference_top_view_video": bool(save_reference_video),
        "reference_top_view_video_filename": reference_video_filename if save_reference_video else None,
        "prompts": manifest_entries,
    }
    atomic_json_dump(manifest, paths.prompt_bank_root / "manifest.json")
    return paths.prompt_bank_root
