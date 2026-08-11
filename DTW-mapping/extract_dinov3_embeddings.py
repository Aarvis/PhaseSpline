"""Extract one DINOv3 ViT-S+ global embedding for every video frame."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModel


MODEL_NAME = "facebook/dinov3-vits16plus-pretrain-lvd1689m"
OUTPUT_SUFFIX = "_dinov3_vits16plus_embeddings"
SCRIPT_ROOT = Path(__file__).resolve().parent
DEFAULT_VIDEOS = (
    SCRIPT_ROOT / "Source-Human-Episode" / "episode_000000_top_rgb.mp4",
    SCRIPT_ROOT / "Sim-Robot-Episode" / "episode_000033_top_rgb.mp4",
)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def iter_video_batches(
    video_path: Path,
    batch_size: int,
    progress: tqdm[Any],
) -> Iterable[tuple[list[np.ndarray], np.ndarray]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames: list[np.ndarray] = []
    indices: list[int] = []
    frame_index = 0
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            indices.append(frame_index)
            frame_index += 1
            progress.update(1)
            if len(frames) == batch_size:
                yield frames, np.asarray(indices, dtype=np.int64)
                frames, indices = [], []
        if frames:
            yield frames, np.asarray(indices, dtype=np.int64)
    finally:
        capture.release()


def video_metadata(video_path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    try:
        return {
            "reported_frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
    finally:
        capture.release()


def atomic_save_npz(path: Path, **arrays: Any) -> None:
    partial = path.with_name(path.name + ".partial")
    try:
        with partial.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def atomic_save_json(path: Path, payload: dict[str, Any]) -> None:
    partial = path.with_name(path.name + ".partial")
    try:
        partial.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def extract_video_embeddings(
    video_path: Path,
    processor: Any,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    overwrite: bool,
) -> Path:
    video_path = video_path.expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video does not exist: {video_path}")

    output_path = video_path.with_name(video_path.stem + OUTPUT_SUFFIX + ".npz")
    metadata_path = video_path.with_name(video_path.stem + OUTPUT_SUFFIX + ".json")
    if not overwrite and (output_path.exists() or metadata_path.exists()):
        raise FileExistsError(
            f"Output already exists for {video_path}. Pass --overwrite to replace it."
        )

    source = video_metadata(video_path)
    all_raw: list[np.ndarray] = []
    all_normalized: list[np.ndarray] = []
    all_indices: list[np.ndarray] = []
    amp_enabled = device.type == "cuda"

    with tqdm(
        total=source["reported_frames"],
        desc=f"DINOv3/{video_path.name}",
        unit="frame",
        dynamic_ncols=True,
    ) as progress:
        for rgb_frames, frame_indices in iter_video_batches(video_path, batch_size, progress):
            inputs = processor(images=rgb_frames, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device, non_blocking=amp_enabled)
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                outputs = model(pixel_values=pixel_values)
                pooled = getattr(outputs, "pooler_output", None)
                if pooled is None:
                    pooled = outputs.last_hidden_state[:, 0]
                normalized = torch.nn.functional.normalize(pooled.float(), dim=-1)

            all_raw.append(pooled.float().cpu().numpy())
            all_normalized.append(normalized.cpu().numpy())
            all_indices.append(frame_indices)

    if not all_raw:
        raise RuntimeError(f"Video contains no decodable frames: {video_path}")

    raw_embeddings = np.concatenate(all_raw, axis=0).astype(np.float32, copy=False)
    normalized_embeddings = np.concatenate(all_normalized, axis=0).astype(np.float32, copy=False)
    frame_indices = np.concatenate(all_indices, axis=0)
    decoded_frames, embedding_dim = raw_embeddings.shape
    if decoded_frames != source["reported_frames"]:
        raise RuntimeError(
            f"Decoded {decoded_frames} frames, but the container reports "
            f"{source['reported_frames']} for {video_path}"
        )
    if not np.isfinite(raw_embeddings).all() or not np.isfinite(normalized_embeddings).all():
        raise RuntimeError(f"Non-finite embeddings produced for {video_path}")

    timestamps = frame_indices.astype(np.float64) / source["fps"]
    model_name = getattr(model, "name_or_path", MODEL_NAME)
    metadata = {
        "source_video": str(video_path),
        "model": str(model_name),
        "embedding_type": "DINOv3 pooler output (CLS global image embedding)",
        "normalized_embedding": "L2 normalization of embeddings along the feature dimension",
        "frames": decoded_frames,
        "embedding_dimension": embedding_dim,
        "fps": source["fps"],
        "width": source["width"],
        "height": source["height"],
        "batch_size": batch_size,
        "device": str(device),
        "npz_file": output_path.name,
    }
    atomic_save_npz(
        output_path,
        embeddings=raw_embeddings,
        embeddings_l2=normalized_embeddings,
        frame_indices=frame_indices,
        timestamps=timestamps,
        fps=np.asarray(source["fps"], dtype=np.float64),
        source_video=np.asarray(str(video_path)),
        model_name=np.asarray(str(model_name)),
    )
    atomic_save_json(metadata_path, metadata)

    with np.load(output_path, allow_pickle=False) as saved:
        if saved["embeddings"].shape != (decoded_frames, embedding_dim):
            raise RuntimeError(f"Post-write validation failed for {output_path}")
        if saved["frame_indices"].shape != (decoded_frames,):
            raise RuntimeError(f"Frame-index validation failed for {output_path}")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract per-frame DINOv3 ViT-S+ embeddings from top-view videos."
    )
    parser.add_argument(
        "videos",
        nargs="*",
        type=Path,
        help="Input MP4s; defaults to the human and simulation top-view videos.",
    )
    parser.add_argument("--model", default=MODEL_NAME, help="Hugging Face model ID or local path")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    videos = tuple(args.videos) if args.videos else DEFAULT_VIDEOS
    device = select_device(args.device)
    print(f"Loading {args.model} on {device} ...")
    processor = AutoImageProcessor.from_pretrained(args.model, use_fast=True)
    model = AutoModel.from_pretrained(args.model).to(device).eval()
    print(f"Model hidden size: {model.config.hidden_size}")

    outputs = []
    for video in tqdm(videos, desc="videos", unit="video", dynamic_ncols=True):
        outputs.append(
            extract_video_embeddings(
                video,
                processor=processor,
                model=model,
                device=device,
                batch_size=args.batch_size,
                overwrite=args.overwrite,
            )
        )
    print("Saved embeddings:")
    for output in outputs:
        print(f"  {output}")


if __name__ == "__main__":
    main()
