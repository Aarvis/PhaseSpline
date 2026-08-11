from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image


IMAGE_COLUMN = "observation.images.top_rgb"


def inspect_episode(parquet_path: Path) -> dict[str, int]:
    parquet = pq.ParquetFile(parquet_path)
    frame_count = parquet.metadata.num_rows
    table = parquet.read_row_group(0, columns=[IMAGE_COLUMN])
    first = table.column(0)[0].as_py()
    from io import BytesIO

    with Image.open(BytesIO(first["bytes"])) as image:
        width, height = image.size
    return {"frame_count": frame_count, "width": width, "height": height}


def extract_video(parquet_path: Path, output_path: Path, fps: float) -> None:
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
    try:
        parquet = pq.ParquetFile(parquet_path)
        for batch in parquet.iter_batches(columns=[IMAGE_COLUMN], batch_size=32):
            for value in batch.column(0):
                frame = value.as_py()
                process.stdin.write(frame["bytes"])
        process.stdin.close()
        return_code = process.wait()
    except BaseException:
        process.kill()
        process.wait()
        temporary_path.unlink(missing_ok=True)
        raise
    if return_code != 0:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(f"FFmpeg exited with status {return_code}")
    temporary_path.replace(output_path)


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--inspect":
        print(json.dumps(inspect_episode(Path(sys.argv[2]))))
        return
    if len(sys.argv) != 4:
        raise SystemExit("usage: extract-episode-video.py PARQUET OUTPUT_MP4 FPS")
    extract_video(Path(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3]))


if __name__ == "__main__":
    main()
