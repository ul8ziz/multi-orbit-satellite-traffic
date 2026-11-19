"""3GPP-style slice mapping from ``APP_CLASS`` and hourly aggregation."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from typing import Final

import pandas as pd

logger = logging.getLogger(__name__)

# eMBB / URLLC / mMTC — full APP_CLASS labels as they appear in the dataset.
_SLICE_APPS: Final[dict[str, str]] = {
    "web applications": "eMBB",
    "streaming applications": "eMBB",
    "file transfer": "eMBB",
    "music streaming": "eMBB",
    "mail": "eMBB",
    "p2p applications": "eMBB",
    "audio streaming": "eMBB",
    "instant messaging applications": "URLLC",
    "games": "URLLC",
    "voip": "URLLC",
    "db transactions": "URLLC",
    "others": "mMTC",
    "security": "mMTC",
    "terminals": "mMTC",
    "network operation": "mMTC",
    "file systems": "mMTC",
    "legacy protocols": "mMTC",
    "cloud services": "mMTC",
}


def normalize_app_class(raw: object) -> str:
    """Normalize free-text ``APP_CLASS`` for lookup."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return ""
    s = str(raw).strip()
    s = re.sub(r"\s+", " ", s)
    return s.casefold()


def _normalize_app_class(raw: object) -> str:
    """Backward-compatible wrapper for older internal imports."""
    return normalize_app_class(raw)


def map_app_to_slice(app_class: object) -> str:
    """Map a single ``APP_CLASS`` label to ``eMBB``, ``URLLC``, or ``mMTC``.

    Unknown labels are assigned to ``mMTC`` (same slice as *Others*).
    """
    key = normalize_app_class(app_class)
    if key in _SLICE_APPS:
        return _SLICE_APPS[key]
    logger.debug("Unmapped APP_CLASS %r -> mMTC (Others bucket)", app_class)
    return "mMTC"


def app_class_mapping_table(app_classes: Iterable[object] | None = None) -> pd.DataFrame:
    """Таблица соответствия канонических ``APP_CLASS`` срезам 3GPP (для отчёта)."""
    if app_classes is None:
        keys = sorted(_SLICE_APPS)
    else:
        keys = sorted({normalize_app_class(a) for a in app_classes if normalize_app_class(a)})
    rows = [{"canonical_APP_CLASS": k, "slice_id": _SLICE_APPS.get(k, "mMTC")} for k in keys]
    return pd.DataFrame(rows)


def add_slice_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``slice_id`` column from ``APP_CLASS``."""
    out = df.copy()
    out["slice_id"] = out["APP_CLASS"].map(map_app_to_slice)
    return out


def aggregate_by_hour_slice(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate upload, download, and total by ``hour`` and ``slice_id``.

    Parameters
    ----------
    df:
        Preprocessed frame with ``hour``, ``slice_id``, ``UPLOAD``, ``DOWNLOAD``, ``total_mb``.

    Returns
    -------
    pandas.DataFrame
        Columns: ``hour``, ``slice_id``, ``upload``, ``download``, ``total``.
    """
    g = (
        df.groupby(["hour", "slice_id"], as_index=False)[["UPLOAD", "DOWNLOAD", "total_mb"]]
        .sum()
        .rename(columns={"UPLOAD": "upload", "DOWNLOAD": "download", "total_mb": "total"})
    )
    return g.sort_values(["hour", "slice_id"]).reset_index(drop=True)
