"""Load raw inputs from ``data/raw`` and emit structured frames for downstream steps."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

from src.preprocessing.dataset_path import REAL_DATASET_FILENAME

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = (
    "START_HOUR",
    "MASKED_MSISDN",
    "APP_CLASS",
    "UPLOAD",
    "DOWNLOAD",
)

# Минимальный размер полного реального датасета (защита от подстановки урезанной выборки).
MIN_FULL_DATASET_ROWS = 1_000_000


def _allow_alternate_dataset_for_dev() -> bool:
    return os.environ.get("PROJECT_ALLOW_ALT_DATASET", "").strip() in ("1", "true", "yes")


def _sniff_csv_separator(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        header = f.readline()
    return ";" if header.count(";") > header.count(",") else ","


def _assert_production_dataset_rules(path: Path) -> None:
    if _allow_alternate_dataset_for_dev():
        return
    if path.name != REAL_DATASET_FILENAME:
        msg = (
            f"Допускается только файл с именем {REAL_DATASET_FILENAME!r}, получено: {path.name!r}. "
            f"Полный путь: {path}"
        )
        raise ValueError(msg)


def load_raw_traffic(path: Path) -> pd.DataFrame:
    """Загрузка CSV с колонками задания.

    Для production обязателен файл с именем ``Applications_data_sample.csv`` и не менее
    ``MIN_FULL_DATASET_ROWS`` строк. Сразу после чтения в stdout печатается блок валидации.

    Разделитель полей определяется по строке заголовка (``;`` или ``,``).

    Для локальных тестов с малым CSV установите ``PROJECT_ALLOW_ALT_DATASET=1`` (не для production).
    """
    path = Path(path).resolve()
    if not path.is_file():
        msg = f"Input file not found: {path}"
        raise FileNotFoundError(msg)

    _assert_production_dataset_rules(path)

    sep = _sniff_csv_separator(path)
    df = pd.read_csv(
        path,
        sep=sep,
        parse_dates=["START_HOUR"],
        engine="c",
        low_memory=False,
    )
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        msg = f"Missing required columns {missing}; found {list(df.columns)}"
        raise ValueError(msg)

    n = len(df)
    if not _allow_alternate_dataset_for_dev() and n < MIN_FULL_DATASET_ROWS:
        msg = (
            f"Слишком мало строк в датасете: {n} < {MIN_FULL_DATASET_ROWS}. "
            f"Ожидается полный файл {REAL_DATASET_FILENAME}."
        )
        raise ValueError(msg)

    uniq_msisdn = int(df["MASKED_MSISDN"].nunique())
    ts = df["START_HOUR"]
    t_min, t_max = ts.min(), ts.max()

    print("=== DATASET VALIDATION (load_raw_traffic) ===", flush=True)
    print(f"file path: {path}", flush=True)
    print(f"number of rows: {n}", flush=True)
    print(f"number of unique MASKED_MSISDN: {uniq_msisdn}", flush=True)
    print(f"time range (min/max START_HOUR): {t_min} / {t_max}", flush=True)
    print("=== END DATASET VALIDATION ===", flush=True)

    logger.info("Loaded %s rows from %s (unique MSISDN=%s)", n, path, uniq_msisdn)
    return df
