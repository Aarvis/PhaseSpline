from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from tqdm.auto import tqdm


DEFAULT_CONFIG = Path(__file__).resolve().with_name("compute_standardized_latent_gap_rmse_stats.yaml")


def load_config(path: str | Path) -> dict:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute standardized latent RMSE statistics between per-frame embedding pairs "
            "separated by a fixed number of frames in between."
        )
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path to a YAML config file.",
    )
    parser.add_argument(
        "--embedding-root",
        default=None,
        help="Root directory containing chunk-XXX/episode_XXXXXX/frame_embeddings.npz exports.",
    )
    parser.add_argument(
        "--frames-in-between",
        type=int,
        default=None,
        help="Number of frames strictly between the two frames in each pair. Default: 10.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to save the computed summary as JSON.",
    )
    return parser.parse_args()


def percentile_dict(values: np.ndarray) -> dict[str, float]:
    percentiles = {
        "min": float(np.min(values)),
        "p1": float(np.percentile(values, 1)),
        "p2": float(np.percentile(values, 2)),
        "p5": float(np.percentile(values, 5)),
        "p10": float(np.percentile(values, 10)),
        "p15": float(np.percentile(values, 15)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p85": float(np.percentile(values, 85)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "p99_9": float(np.percentile(values, 99.9)),
        "max": float(np.max(values)),
    }
    return percentiles


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    embedding_root_value = args.embedding_root or config.get("embedding_root")
    if not embedding_root_value:
        raise ValueError("embedding_root must be provided either in the config or via --embedding-root")
    embedding_root = Path(embedding_root_value).expanduser().resolve()
    normalization_path = embedding_root / "embedding_normalization.npz"
    if not normalization_path.is_file():
        raise FileNotFoundError(f"Missing embedding normalization file: {normalization_path}")

    with np.load(normalization_path, allow_pickle=False) as normalization:
        latent_mean = np.asarray(normalization["mean"], dtype=np.float32)
        latent_std = np.asarray(normalization["std"], dtype=np.float32)
    latent_std = np.maximum(latent_std, 1e-8)

    frames_in_between = int(
        args.frames_in_between
        if args.frames_in_between is not None
        else config.get("frames_in_between", 10)
    )
    if frames_in_between < 0:
        raise ValueError("--frames-in-between must be non-negative")
    offset = frames_in_between + 1

    sources = sorted(embedding_root.glob("chunk-*/episode_*/frame_embeddings.npz"))
    if not sources:
        sources = sorted(embedding_root.glob("episode_*/frame_embeddings.npz"))
    if not sources:
        raise FileNotFoundError(f"No frame_embeddings.npz files found under {embedding_root}")

    rmse_chunks: list[np.ndarray] = []
    pair_count = 0
    used_episodes = 0
    skipped_short_episodes = 0

    for source in tqdm(sources, desc="episodes", unit="episode"):
        with np.load(source, allow_pickle=False) as data:
            mean = np.asarray(data["mean"], dtype=np.float32)
        if mean.ndim != 2:
            raise ValueError(f"Expected 2D mean array in {source}, got shape {mean.shape}")
        if mean.shape[0] <= offset:
            skipped_short_episodes += 1
            continue

        standardized = (mean.astype(np.float32) - latent_mean[None, :]) / latent_std[None, :]
        delta = standardized[offset:] - standardized[:-offset]
        rmse = np.sqrt(np.mean(np.square(delta, dtype=np.float32), axis=1, dtype=np.float32))
        rmse_chunks.append(rmse.astype(np.float32, copy=False))
        pair_count += int(rmse.shape[0])
        used_episodes += 1

    if not rmse_chunks:
        raise RuntimeError(
            f"No valid frame pairs found for frames_in_between={frames_in_between} under {embedding_root}"
        )

    all_rmse = np.concatenate(rmse_chunks, axis=0).astype(np.float64, copy=False)
    summary = {
        "config_path": str(Path(args.config).expanduser().resolve()),
        "embedding_root": str(embedding_root),
        "normalization_path": str(normalization_path),
        "frames_in_between": frames_in_between,
        "frame_index_offset": offset,
        "episodes_scanned": len(sources),
        "episodes_used": used_episodes,
        "episodes_skipped_too_short": skipped_short_episodes,
        "pair_count": pair_count,
        "standardized_latent_rmse": percentile_dict(all_rmse),
    }

    print(json.dumps(summary, indent=2))

    output_json_value = args.output_json if args.output_json is not None else config.get("output_json")
    if output_json_value:
        output_path = Path(output_json_value).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Saved summary to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
