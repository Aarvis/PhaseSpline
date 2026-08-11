from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import imageio_ffmpeg
except ImportError:  # pragma: no cover - optional dependency
    imageio_ffmpeg = None

import pyarrow.parquet as pq

from multi_reference_embedding_dtw_transfer import (
    DEFAULT_CONFIG,
    EPISODE_PATTERN,
    canonical_garment_name,
    load_config,
    load_dataset_config,
    source_file_for_episode,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = ROOT / "temp_segment_snips"
SAFE_NAME_PATTERN = re.compile(r"[^a-zA-Z0-9._-]+")
IMAGE_COLUMN = "observation.images.top_rgb"


@dataclass(frozen=True)
class TransferredEpisode:
    episode_index: int
    garment_type: str
    annotation_path: Path
    template_status: str
    source_file: Path
    segments: list[dict[str, Any]]
    raw: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Randomly sample transferred DTW-labeled episodes for a garment and export one video snip per segment."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", choices=["sim", "human"], required=True)
    parser.add_argument("--garment", required=True, help="Garment type to sample, e.g. shorts.")
    parser.add_argument("--sample-count", type=int, default=5, help="Number of transferred episodes to sample.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible sampling.")
    parser.add_argument(
        "--transferred-episodes-directory",
        "--transferred_episodes_directory",
        dest="transferred_episodes_directory",
        type=Path,
        default=None,
        help=(
            "Optional root directory containing transferred DTW checkpoints to sample from, "
            "for example DTW-Transfer/outputs/<dataset>/<garment>/transferred_checkpoints."
        ),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true", help="Replace existing exported episode folders.")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Discover candidates and print the sampled episodes without exporting videos.",
    )
    return parser.parse_args()


def sanitize_name(value: str, max_len: int = 80) -> str:
    cleaned = SAFE_NAME_PATTERN.sub("_", value.strip()).strip("._-")
    if not cleaned:
        return "segment"
    if len(cleaned) > max_len:
        return cleaned[:max_len].rstrip("._-")
    return cleaned


def resolve_ffmpeg_executable() -> str:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path
    if imageio_ffmpeg is not None:
        return imageio_ffmpeg.get_ffmpeg_exe()
    raise FileNotFoundError("Could not find ffmpeg. Install ffmpeg or imageio-ffmpeg.")


def parse_episode_index_from_path(path: Path) -> int:
    match = EPISODE_PATTERN.search(str(path))
    if not match:
        raise ValueError(f"Could not parse episode index from annotation path: {path}")
    return int(match.group(1))


def validate_segments(segments: Any, annotation_path: Path) -> list[dict[str, Any]]:
    if not isinstance(segments, list) or not segments:
        raise ValueError(f"Missing segments in {annotation_path}")
    validated: list[dict[str, Any]] = []
    expected_start = 0
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise ValueError(f"Bad segment entry at index {index} in {annotation_path}")
        label = str(segment.get("label", "")).strip()
        start = int(segment["start_frame"])
        end = int(segment["end_frame_exclusive"])
        if not label:
            raise ValueError(f"Segment {index} has no label in {annotation_path}")
        if start != expected_start:
            raise ValueError(
                f"Segment {index} in {annotation_path} starts at {start}, expected contiguous start {expected_start}."
            )
        if end <= start:
            raise ValueError(f"Segment {index} in {annotation_path} has non-positive length.")
        expected_start = end
        validated.append(segment)
    return validated


def load_dataset_info(dataset_root: Path) -> dict[str, Any]:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        return {}
    return json.loads(info_path.read_text(encoding="utf-8"))


def load_dataset_fps(dataset_root: Path) -> float:
    info = load_dataset_info(dataset_root)
    return float(info.get("fps", 30))


def parquet_path_for_episode(dataset_root: Path, episode_index: int) -> Path:
    return Path(source_file_for_episode(dataset_root, episode_index)).expanduser().resolve()


def load_transferred_episode(annotation_path: Path, dataset_root: Path) -> TransferredEpisode:
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    template_status = str(data.get("template_status", "")).strip()
    if not template_status.startswith("dtw_transferred"):
        raise ValueError(f"Annotation is not DTW transferred: {annotation_path}")

    episode_index = parse_episode_index_from_path(annotation_path)
    garment = canonical_garment_name(str(data.get("category_id") or data.get("task", {}).get("garment_type") or ""))
    if not garment:
        raise ValueError(f"Missing garment in {annotation_path}")

    source_file_value = str(data.get("source_file") or "").strip()
    if not source_file_value:
        source_file_value = source_file_for_episode(dataset_root, episode_index)
    if not source_file_value:
        raise FileNotFoundError(f"No source video path found for {annotation_path}")
    source_file = Path(source_file_value).expanduser().resolve()
    if not source_file.exists():
        parquet_fallback = parquet_path_for_episode(dataset_root, episode_index)
        if not parquet_fallback.exists():
            raise FileNotFoundError(f"Source media does not exist for {annotation_path}: {source_file}")
        source_file = parquet_fallback

    segments = validate_segments(data.get("segments"), annotation_path)
    return TransferredEpisode(
        episode_index=episode_index,
        garment_type=garment,
        annotation_path=annotation_path,
        template_status=template_status,
        source_file=source_file,
        segments=segments,
        raw=data,
    )


def discover_transferred_episodes(
    annotations_root: Path,
    dataset_root: Path,
    garment: str,
) -> tuple[list[TransferredEpisode], list[dict[str, str]]]:
    candidates: list[TransferredEpisode] = []
    skipped: list[dict[str, str]] = []
    for annotation_path in sorted(annotations_root.rglob("checkpoints.json")):
        try:
            episode = load_transferred_episode(annotation_path, dataset_root)
        except (ValueError, FileNotFoundError, KeyError, json.JSONDecodeError) as exc:
            skipped.append({"annotation_path": str(annotation_path), "reason": str(exc)})
            continue
        if episode.garment_type != garment:
            continue
        candidates.append(episode)
    return candidates, skipped


def sample_episodes(
    candidates: list[TransferredEpisode],
    sample_count: int,
    seed: int,
) -> list[TransferredEpisode]:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if not candidates:
        raise ValueError("No transferred episodes available for sampling.")
    rng = random.Random(seed)
    if sample_count >= len(candidates):
        selected = list(candidates)
        rng.shuffle(selected)
        return selected
    return rng.sample(candidates, sample_count)


def ensure_clean_output_dir(path: Path, output_root: Path, overwrite: bool) -> None:
    try:
        path.resolve().relative_to(output_root.resolve())
    except ValueError as exc:
        raise ValueError(f"Output path escapes output root: {path}") from exc
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists, pass --overwrite to replace it: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def export_segment_video(
    ffmpeg_executable: str,
    source_file: Path,
    start_frame: int,
    end_frame_exclusive: int,
    output_path: Path,
    fps: float,
    overwrite: bool,
) -> None:
    if end_frame_exclusive <= start_frame:
        raise ValueError(f"Bad segment frame range: {start_frame}:{end_frame_exclusive}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if source_file.suffix.lower() == ".parquet":
        export_segment_video_from_parquet(
            ffmpeg_executable=ffmpeg_executable,
            parquet_path=source_file,
            start_frame=start_frame,
            end_frame_exclusive=end_frame_exclusive,
            output_path=output_path,
            fps=fps,
            overwrite=overwrite,
        )
        return
    export_segment_video_from_video(
        ffmpeg_executable=ffmpeg_executable,
        source_file=source_file,
        start_frame=start_frame,
        end_frame_exclusive=end_frame_exclusive,
        output_path=output_path,
        overwrite=overwrite,
    )


def export_segment_video_from_video(
    ffmpeg_executable: str,
    source_file: Path,
    start_frame: int,
    end_frame_exclusive: int,
    output_path: Path,
    overwrite: bool,
) -> None:
    overwrite_flag = "-y" if overwrite else "-n"
    video_filter = f"trim=start_frame={start_frame}:end_frame={end_frame_exclusive},setpts=PTS-STARTPTS"
    command = [
        ffmpeg_executable,
        overwrite_flag,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source_file),
        "-map",
        "0:v:0",
        "-vf",
        video_filter,
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(
            f"ffmpeg failed for {source_file.name} frames {start_frame}:{end_frame_exclusive} -> {output_path}. {stderr}"
        )


def export_segment_video_from_parquet(
    ffmpeg_executable: str,
    parquet_path: Path,
    start_frame: int,
    end_frame_exclusive: int,
    output_path: Path,
    fps: float,
    overwrite: bool,
) -> None:
    temporary_path = output_path.with_suffix(".building.mp4")
    temporary_path.unlink(missing_ok=True)
    overwrite_flag = "-y" if overwrite else "-n"
    command = [
        ffmpeg_executable,
        "-hide_banner",
        "-loglevel",
        "error",
        overwrite_flag,
        "-f",
        "image2pipe",
        "-framerate",
        str(fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temporary_path),
    ]
    parquet_file = pq.ParquetFile(parquet_path)
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    frame_index = 0
    try:
        for batch in parquet_file.iter_batches(columns=[IMAGE_COLUMN], batch_size=64):
            for value in batch.column(0):
                if frame_index >= end_frame_exclusive:
                    break
                if frame_index >= start_frame:
                    frame = value.as_py()
                    process.stdin.write(frame["bytes"])
                frame_index += 1
            if frame_index >= end_frame_exclusive:
                break
        process.stdin.close()
        return_code = process.wait()
    except BaseException:
        process.kill()
        process.wait()
        temporary_path.unlink(missing_ok=True)
        raise
    if return_code != 0:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"ffmpeg failed for {parquet_path.name} frames {start_frame}:{end_frame_exclusive} -> {output_path}"
        )
    if output_path.exists():
        output_path.unlink()
    temporary_path.replace(output_path)


def build_episode_manifest(
    episode: TransferredEpisode,
    episode_output_dir: Path,
    segment_outputs: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "episode_index": episode.episode_index,
        "garment_type": episode.garment_type,
        "annotation_path": str(episode.annotation_path),
        "template_status": episode.template_status,
        "source_file": str(episode.source_file),
        "output_dir": str(episode_output_dir),
        "num_segments": len(segment_outputs),
        "segments": segment_outputs,
    }


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    dataset_cfg = load_dataset_config(config, args.dataset)
    garment = canonical_garment_name(args.garment)
    output_root = args.output_root.expanduser().resolve()
    garment_output_root = output_root / dataset_cfg.name / garment
    fps = load_dataset_fps(dataset_cfg.dataset_root)
    annotation_source_root = (
        args.transferred_episodes_directory.expanduser().resolve()
        if args.transferred_episodes_directory is not None
        else dataset_cfg.annotations_root
    )
    if not annotation_source_root.exists():
        raise FileNotFoundError(f"Transferred episodes directory does not exist: {annotation_source_root}")

    candidates, skipped = discover_transferred_episodes(
        annotations_root=annotation_source_root,
        dataset_root=dataset_cfg.dataset_root,
        garment=garment,
    )
    selected = sample_episodes(candidates, args.sample_count, args.seed)

    print("Transferred segment snip sampler")
    print(f"  dataset           : {dataset_cfg.name}")
    print(f"  garment           : {garment}")
    print(f"  annotations_root  : {annotation_source_root}")
    print(f"  output_root       : {garment_output_root}")
    print(f"  fps               : {fps}")
    print(f"  transferred_pool  : {len(candidates)}")
    print(f"  sample_count      : {len(selected)}")
    print(f"  seed              : {args.seed}")
    if skipped:
        print(f"  skipped_bad_files : {len(skipped)}")
    print("  sampled_episodes  : " + ", ".join(str(item.episode_index) for item in selected))

    if args.validate_only:
        print("Validation OK.")
        return 0

    ffmpeg_executable = resolve_ffmpeg_executable()
    garment_output_root.mkdir(parents=True, exist_ok=True)

    run_manifest: dict[str, Any] = {
        "dataset": dataset_cfg.name,
        "garment": garment,
        "sample_count_requested": args.sample_count,
        "sample_count_actual": len(selected),
        "seed": args.seed,
        "fps": fps,
        "annotations_root": str(annotation_source_root),
        "output_root": str(garment_output_root),
        "ffmpeg_executable": ffmpeg_executable,
        "skipped_discovery": skipped,
        "episodes": [],
    }

    for episode in selected:
        episode_dir = garment_output_root / f"episode_{episode.episode_index:06d}"
        ensure_clean_output_dir(episode_dir, garment_output_root, overwrite=args.overwrite)
        segment_outputs: list[dict[str, Any]] = []
        for segment in episode.segments:
            segment_id = int(segment["segment_id"])
            start_frame = int(segment["start_frame"])
            end_frame_exclusive = int(segment["end_frame_exclusive"])
            label = str(segment["label"])
            safe_label = sanitize_name(label)
            output_path = episode_dir / f"segment_{segment_id:02d}_{safe_label}.mp4"
            export_segment_video(
                ffmpeg_executable=ffmpeg_executable,
                source_file=episode.source_file,
                start_frame=start_frame,
                end_frame_exclusive=end_frame_exclusive,
                output_path=output_path,
                fps=fps,
                overwrite=args.overwrite,
            )
            segment_outputs.append(
                {
                    "segment_id": segment_id,
                    "label": label,
                    "start_frame": start_frame,
                    "end_frame_exclusive": end_frame_exclusive,
                    "num_frames": int(segment.get("num_frames", end_frame_exclusive - start_frame)),
                    "output_file": str(output_path),
                }
            )

        episode_manifest = build_episode_manifest(episode, episode_dir, segment_outputs)
        (episode_dir / "episode_manifest.json").write_text(
            json.dumps(episode_manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        run_manifest["episodes"].append(episode_manifest)

    (garment_output_root / "sample_manifest.json").write_text(
        json.dumps(run_manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
