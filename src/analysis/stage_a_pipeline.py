"""End-to-end Stage A: preprocessing through tables and figures (no Markov)."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def run_stage_a(
    project_root: Path,
    input_path: Path,
    *,
    seed: int = 42,
    force: bool = False,
    chunk_rows: int = 500_000,
) -> dict[str, Path]:
    """Полный Stage A: chunked-пайплайн, Parquet, чекпойнты (см. ``stage_a_chunked_pipeline``)."""
    from src.analysis.stage_a_chunked_pipeline import run_stage_a_chunked

    return run_stage_a_chunked(
        Path(project_root),
        Path(input_path),
        seed=seed,
        force=force,
        chunk_rows=chunk_rows,
    )
