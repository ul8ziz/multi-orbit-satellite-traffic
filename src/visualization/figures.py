"""Create and save matplotlib (or other) figures to ``outputs/figures``."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def save_figure(fig: Any, path: Path) -> None:
    """Persist a figure object to ``path`` with project-default styling.

    Parameters
    ----------
    fig:
        Figure or artist root (library-specific).
    path:
        Destination file under ``outputs/figures``.
    """
    ...
