from __future__ import annotations

import os
import sys
from pathlib import Path


def repo_roots() -> dict[str, Path]:
    this_file = Path(__file__).resolve()
    challenge_root = this_file.parents[3]
    spline_root = challenge_root / "Lehome-Spline-ICRA2027"
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
