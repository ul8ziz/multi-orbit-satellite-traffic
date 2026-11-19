"""Per-user traffic tiers: Heavy / Medium / Light."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROFILE_ORDER: tuple[str, ...] = ("Light", "Medium", "Heavy")


def classify_user_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Classify each ``MASKED_MSISDN`` by total volume into Light / Medium / Heavy.

    Sorts users by ascending total ``total_mb`` and splits into three as-equal-as-possible
    groups (lowest third Light, highest third Heavy).

    Parameters
    ----------
    df:
        Preprocessed traffic with ``MASKED_MSISDN`` and ``total_mb``.

    Returns
    -------
    pandas.DataFrame
        Columns: ``MASKED_MSISDN``, ``total_mb``, ``profile``.
    """
    per_user = df.groupby("MASKED_MSISDN", as_index=False)["total_mb"].sum()
    n = len(per_user)
    if n == 0:
        return pd.DataFrame(columns=["MASKED_MSISDN", "total_mb", "profile"])

    if n < 3:
        per_user["profile"] = "Medium"
        logger.warning("Fewer than 3 users; assigning Medium to all.")
        return per_user

    pu = per_user.sort_values("total_mb", kind="mergesort").reset_index(drop=True)
    parts = np.array_split(np.arange(n), 3)
    prof = np.empty(n, dtype=object)
    for part, name in zip(parts, PROFILE_ORDER):
        prof[part] = name
    pu["profile"] = prof
    logger.info("User profiles: %s", pu["profile"].value_counts().to_dict())
    return pu
