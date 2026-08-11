from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import subprocess
from pathlib import Path

import pyarrow.parquet as pq


IMAGE_COLUMN = "observation.images.top_rgb"


DATASETS = {
    "human": {
        "root": Path(r"D:\pretrain_lehome_all_garment_data_z180"),
        "label": "Human",
    },
    "sim": {
        "root": Path(r"E:\Lehome-Dataset\lehome_round_2_dataset\sim_dataset\robot_sim_ft_lehome_all_garment_data_z180"),
        "label": "Sim",
    },
}


def episode_stem(episode_index: int) -> str:
    return f"episode_{episode_index:06d}"


def episode_chunk(episode_index: int) -> str:
    return f"chunk-{episode_index // 1000:03d}"


def sanitize(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return value.strip("._-") or "segment"


def load_fps(dataset_root: Path) -> float:
    info_path = dataset_root / "meta" / "info.json"
    with info_path.open("r", encoding="utf-8") as handle:
        return float(json.load(handle).get("fps", 30))


def find_checkpoint_files(dataset_root: Path) -> list[Path]:
    checkpoint_root = dataset_root / "annotations" / "temporal_checkpoints"
    if not checkpoint_root.exists():
        return []
    return sorted(checkpoint_root.rglob("checkpoints.json"))


def episode_index_from_checkpoint(checkpoint_path: Path) -> int:
    match = re.search(r"episode_(\d{6})", str(checkpoint_path))
    if not match:
        raise ValueError(f"Could not infer episode index from {checkpoint_path}")
    return int(match.group(1))


def parquet_path(dataset_root: Path, episode_index: int) -> Path:
    return dataset_root / "data" / episode_chunk(episode_index) / f"{episode_stem(episode_index)}.parquet"


def validate_segments(segments: list[dict], frame_count: int) -> None:
    expected_start = 0
    for index, segment in enumerate(segments):
        start = int(segment["start_frame"])
        end = int(segment["end_frame_exclusive"])
        if start != expected_start or end <= start:
            raise ValueError(f"Segment {index} is not contiguous/non-empty: [{start}, {end})")
        expected_start = end
    if expected_start != frame_count:
        raise ValueError(f"Segments end at {expected_start}, expected frame_count {frame_count}")


def write_segment_mp4(parquet_file: pq.ParquetFile, output_path: Path, start: int, end: int, fps: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".building.mp4")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
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
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    frame_index = 0
    try:
        for batch in parquet_file.iter_batches(columns=[IMAGE_COLUMN], batch_size=64):
            for value in batch.column(0):
                if frame_index >= end:
                    break
                if frame_index >= start:
                    frame = value.as_py()
                    process.stdin.write(frame["bytes"])
                frame_index += 1
            if frame_index >= end:
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
        raise RuntimeError(f"FFmpeg failed while writing {output_path}")
    temporary_path.replace(output_path)


def export_dataset(dataset_id: str, dataset: dict, output_root: Path, rng: random.Random) -> dict:
    dataset_root = dataset["root"]
    checkpoints = find_checkpoint_files(dataset_root)
    if not checkpoints:
        raise RuntimeError(f"No checkpoints found for {dataset_id}: {dataset_root}")

    rng.shuffle(checkpoints)
    selected_checkpoint = None
    selected_episode_index = None
    selected_parquet = None
    selected_payload = None

    for checkpoint_path in checkpoints:
        episode_index = episode_index_from_checkpoint(checkpoint_path)
        parquet = parquet_path(dataset_root, episode_index)
        if not parquet.exists():
            continue
        with checkpoint_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload.get("segments"), list) and payload["segments"]:
            selected_checkpoint = checkpoint_path
            selected_episode_index = episode_index
            selected_parquet = parquet
            selected_payload = payload
            break

    if selected_checkpoint is None or selected_episode_index is None or selected_parquet is None or selected_payload is None:
        raise RuntimeError(f"No usable checkpoint/parquet pair found for {dataset_id}")

    fps = load_fps(dataset_root)
    parquet_file = pq.ParquetFile(selected_parquet)
    frame_count = parquet_file.metadata.num_rows
    segments = selected_payload["segments"]
    validate_segments(segments, frame_count)

    dataset_output = output_root / dataset_id
    dataset_output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selected_checkpoint, dataset_output / "checkpoints.json")

    outputs = []
    for index, segment in enumerate(segments):
        start = int(segment["start_frame"])
        end = int(segment["end_frame_exclusive"])
        label = str(segment.get("label", f"segment_{index + 1:02d}"))
        output_name = f"{index + 1:02d}_{sanitize(label)}_f{start:06d}_to_f{end - 1:06d}.mp4"
        output_path = dataset_output / output_name
        write_segment_mp4(parquet_file, output_path, start, end, fps)
        outputs.append(
            {
                "segment_index": index,
                "label": label,
                "start_frame": start,
                "end_frame_exclusive": end,
                "end_frame_inclusive": end - 1,
                "num_frames": end - start,
                "video": str(output_path),
            }
        )

    summary = {
        "dataset_id": dataset_id,
        "dataset_label": dataset["label"],
        "dataset_root": str(dataset_root),
        "episode_index": selected_episode_index,
        "episode_stem": episode_stem(selected_episode_index),
        "fps": fps,
        "frame_count": frame_count,
        "checkpoint_path": str(selected_checkpoint),
        "parquet_path": str(selected_parquet),
        "segments": outputs,
    }
    with (dataset_output / "selected_episode_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Export top-view MP4 snippets for one random checkpointed human and sim episode.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    output_root = args.output
    output_root.mkdir(parents=True, exist_ok=True)

    summaries = {
        dataset_id: export_dataset(dataset_id, dataset, output_root, rng)
        for dataset_id, dataset in DATASETS.items()
    }
    with (output_root / "export_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"output_root": str(output_root), "datasets": summaries}, indent=2))


if __name__ == "__main__":
    main()
