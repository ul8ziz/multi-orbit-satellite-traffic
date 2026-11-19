"""Smoke tests: packages import without side effects."""

from __future__ import annotations

from pathlib import Path
import sys


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_src_packages_import() -> None:
    """Ensure ``src`` subpackages are importable when project root is on ``sys.path``."""
    root = str(_project_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    import src.analysis  # noqa: F401
    import src.forecasting  # noqa: F401
    import src.integration  # noqa: F401
    import src.markov  # noqa: F401
    import src.metrics  # noqa: F401
    import src.preprocessing  # noqa: F401
    import src.visualization  # noqa: F401
