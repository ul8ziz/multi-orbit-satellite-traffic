"""Core traffic statistics and summaries required by the assignment."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def summarize_traffic(project_root: Path) -> pd.DataFrame:
    """Load per-slice summary statistics produced by Stage A.

    Parameters
    ----------
    project_root:
        Project root containing ``outputs/tables/slice_descriptive_stats.csv``.

    Returns
    -------
    pandas.DataFrame
        Contents of ``slice_descriptive_stats.csv`` if present.

    Raises
    ------
    FileNotFoundError
        If the Stage A statistics file does not exist yet.
    """
    path = Path(project_root) / "outputs" / "tables" / "slice_descriptive_stats.csv"
    if not path.is_file():
        msg = f"Run Stage A first; missing {path}"
        raise FileNotFoundError(msg)
    return pd.read_csv(path)
