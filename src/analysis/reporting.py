"""Assemble narrative sections and link figures for Stage A reports."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def build_stage_a_report(outputs_dir: Path) -> Path | None:
    """Return the path to the Stage A manifest under ``outputs/reports`` if it exists.

    The manifest is written by :func:`src.analysis.stage_a_pipeline.run_stage_a`.

    Parameters
    ----------
    outputs_dir:
        Project ``outputs`` root (parent of ``reports/``).

    Returns
    -------
    pathlib.Path | None
        Path to ``stage_a_manifest.txt``, or ``None`` if not found.
    """
    manifest = Path(outputs_dir) / "reports" / "stage_a_manifest.txt"
    if manifest.is_file():
        logger.info("Stage A manifest: %s", manifest)
        return manifest
    logger.warning("No Stage A manifest at %s", manifest)
    return None
