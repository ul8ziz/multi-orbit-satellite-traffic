# -*- coding: utf-8 -*-
"""Чтение очищенного трафика: предпочтительно Parquet (полный датасет), иначе CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def cleaned_traffic_path(project_root: Path) -> tuple[Path | None, Path | None]:
    """Возвращает (parquet_path или None, csv_path или None), если файл существует."""
    proc = Path(project_root) / "data" / "processed"
    pq = proc / "cleaned_traffic.parquet"
    csv = proc / "cleaned_traffic.csv"
    return (pq if pq.is_file() else None, csv if csv.is_file() else None)


def read_cleaned_traffic(project_root: Path) -> pd.DataFrame:
    """Читает ``cleaned_traffic.parquet`` при наличии, иначе ``cleaned_traffic.csv``."""
    proc = Path(project_root) / "data" / "processed"
    pq = proc / "cleaned_traffic.parquet"
    csv = proc / "cleaned_traffic.csv"
    if pq.is_file():
        return pd.read_parquet(pq)
    if csv.is_file():
        return pd.read_csv(csv)
    msg = f"Нет ни {pq}, ни {csv}. Запустите Stage A."
    raise FileNotFoundError(msg)


def read_cleaned_traffic_file(path: Path) -> pd.DataFrame:
    """Читает один файл очищенного трафика по расширению."""
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)
