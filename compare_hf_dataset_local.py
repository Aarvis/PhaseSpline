from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi
from tqdm.auto import tqdm


CHUNK_SIZE = 8 * 1024 * 1024


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha1(path: Path) -> str:
    size = path.stat().st_size
    h = hashlib.sha1()
    header = f"blob {size}\0".encode("utf-8")
    h.update(header)
    with path.open("rb") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def extract_lfs_sha256(lfs_obj: Any) -> str | None:
    if lfs_obj is None:
        return None
    if isinstance(lfs_obj, dict):
        return lfs_obj.get("sha256")
    return getattr(lfs_obj, "sha256", None)


def list_remote_files(repo_id: str, repo_type: str, revision: str | None) -> dict[str, dict[str, Any]]:
    api = HfApi()
    remote: dict[str, dict[str, Any]] = {}
    tree_iter = api.list_repo_tree(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        recursive=True,
        expand=False,
    )

    for item in tqdm(tree_iter, desc="remote/files", unit="file"):
        if not hasattr(item, "size"):
            continue
        path = getattr(item, "path", None)
        if not path:
            continue
        lfs_sha256 = extract_lfs_sha256(getattr(item, "lfs", None))
        remote[path] = {
            "size": int(getattr(item, "size")),
            "blob_id": getattr(item, "blob_id", None),
            "lfs_sha256": lfs_sha256,
            "hash_mode": "sha256" if lfs_sha256 else "git_blob_sha1",
            "remote_hash": lfs_sha256 if lfs_sha256 else getattr(item, "blob_id", None),
        }
    return remote


def list_local_files(local_dir: Path) -> dict[str, Path]:
    local: dict[str, Path] = {}
    all_files = [p for p in local_dir.rglob("*") if p.is_file()]

    for path in tqdm(all_files, desc="local/scan", unit="file"):
        rel = path.relative_to(local_dir)
        if rel.parts and rel.parts[0] == ".cache":
            continue
        local[str(rel).replace("\\", "/")] = path
    return local


def compare_one(rel_path: str, local_path: Path, remote_info: dict[str, Any]) -> tuple[str, str]:
    local_size = local_path.stat().st_size
    remote_size = remote_info["size"]

    if local_size != remote_size:
        return rel_path, "updated"

    remote_hash = remote_info["remote_hash"]
    if not remote_hash:
        return rel_path, "same_size_unknown"

    if remote_info["hash_mode"] == "sha256":
        local_hash = sha256_file(local_path)
    else:
        local_hash = git_blob_sha1(local_path)

    if local_hash == remote_hash:
        return rel_path, "unchanged"
    return rel_path, "updated"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare a local folder with a Hugging Face repo.")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--repo-type", default="dataset", choices=["model", "dataset", "space"])
    parser.add_argument("--revision", default=None)
    parser.add_argument("--local-dir", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    local_dir = Path(args.local_dir).resolve()
    if not local_dir.exists():
        raise FileNotFoundError(local_dir)

    remote = list_remote_files(args.repo_id, args.repo_type, args.revision)
    local = list_local_files(local_dir)

    remote_paths = set(remote.keys())
    local_paths = set(local.keys())

    new_remote_files = sorted(remote_paths - local_paths)
    stale_local_files = sorted(local_paths - remote_paths)

    common_paths = sorted(remote_paths & local_paths)

    unchanged: list[str] = []
    updated: list[str] = []
    same_size_unknown: list[str] = []

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = {
            ex.submit(compare_one, rel_path, local[rel_path], remote[rel_path]): rel_path
            for rel_path in common_paths
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="compare/hash", unit="file"):
            rel_path, status = fut.result()
            if status == "unchanged":
                unchanged.append(rel_path)
            elif status == "updated":
                updated.append(rel_path)
            else:
                same_size_unknown.append(rel_path)

    unchanged.sort()
    updated.sort()
    same_size_unknown.sort()

    report = {
        "repo_id": args.repo_id,
        "repo_type": args.repo_type,
        "revision": args.revision,
        "local_dir": str(local_dir),
        "counts": {
            "remote_total_files": len(remote_paths),
            "local_total_files": len(local_paths),
            "unchanged": len(unchanged),
            "updated": len(updated),
            "new_remote_files": len(new_remote_files),
            "stale_local_files": len(stale_local_files),
            "same_size_unknown": len(same_size_unknown),
        },
        "unchanged": unchanged,
        "updated": updated,
        "new_remote_files": new_remote_files,
        "stale_local_files": stale_local_files,
        "same_size_unknown": same_size_unknown,
    }

    print("\nSummary")
    print(json.dumps(report["counts"], indent=2))

    def print_preview(title: str, values: list[str], limit: int = 20) -> None:
        print(f"\n{title}: {len(values)}")
        for item in values[:limit]:
            print(f"  {item}")
        if len(values) > limit:
            print(f"  ... ({len(values) - limit} more)")

    print_preview("Updated files", updated)
    print_preview("New remote files", new_remote_files)
    print_preview("Stale local files", stale_local_files)

    if args.output_json:
        output_path = Path(args.output_json).resolve()
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved report: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())