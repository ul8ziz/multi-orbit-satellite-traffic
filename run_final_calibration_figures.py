# -*- coding: utf-8 -*-
"""Generate final thesis figures for calibrated CTMC scenarios."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def _ensure_src_on_path() -> None:
    r = str(PROJECT_ROOT)
    if r not in sys.path:
        sys.path.insert(0, r)


def main() -> int:
    _ensure_src_on_path()
    from src.visualization.final_calibration_figures import generate_final_figures

    paths = generate_final_figures(PROJECT_ROOT)
    for p in paths:
        print(p.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
