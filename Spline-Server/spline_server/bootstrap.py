from __future__ import annotations

import os
import sys
from pathlib import Path


def _first_existing(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except Exception:
            continue
        if resolved.exists():
            return resolved
    return None


def repo_roots() -> dict[str, Path]:
    this_file = Path(__file__).resolve()
    explicit_spline_root = os.environ.get("LEHOME_SPLINE_REPO")
    explicit_openpi_root = os.environ.get("LEHOME_OPENPI_REPO")

    if explicit_spline_root:
        spline_root = Path(explicit_spline_root).expanduser().resolve()
        challenge_root = spline_root.parent
    else:
        spline_root = _first_existing(
            [
                this_file.parents[2],
                this_file.parents[3] / "Lehome-Spline-ICRA2027",
            ]
        )
        if spline_root is None:
            raise FileNotFoundError(
                "Could not determine spline repo root. Set LEHOME_SPLINE_REPO to the repo path containing "
                "'human-spline-localizer', 'human-to-robot-local-spline-translator', and 'lehome_robot_sim_embedding'."
            )
        challenge_root = spline_root.parent

    if explicit_openpi_root:
        openpi_root = Path(explicit_openpi_root).expanduser().resolve()
    else:
        openpi_root = _first_existing(
            [
                challenge_root / "openpi",
                challenge_root / "lehome-openpi",
                spline_root.parent / "openpi",
                spline_root.parent / "lehome-openpi",
            ]
        )
        if openpi_root is None:
            openpi_root = challenge_root / "openpi"
    return {
        "challenge_root": challenge_root,
        "spline_root": spline_root,
        "openpi_root": openpi_root,
    }


def ensure_repo_imports(openpi_root_override: str | None = None) -> None:
    roots = repo_roots()
    spline_root = roots["spline_root"]
    openpi_root = Path(openpi_root_override).expanduser().resolve() if openpi_root_override else roots["openpi_root"]

    candidates = [
        spline_root,
        spline_root / "human-spline-localizer",
        spline_root / "human-to-robot-local-spline-translator",
        spline_root / "lehome_human_spline_generation",
        spline_root / "lehome_robot_sim_embedding",
        openpi_root / "src",
        openpi_root / "packages" / "openpi-client" / "src",
    ]
    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate.exists() and candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)

    if openpi_root.exists():
        os.environ.setdefault("LEHOME_OPENPI_REPO", str(openpi_root))
