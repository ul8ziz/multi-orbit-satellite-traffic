# -*- coding: utf-8 -*-
"""Единственный допустимый сырой датасет для production-конвейера: Applications_data_sample.csv."""

from __future__ import annotations

from pathlib import Path

REAL_DATASET_FILENAME = "Applications_data_sample.csv"


def resolve_real_dataset_path(project_root: Path) -> Path:
    """Возвращает путь к ``Applications_data_sample.csv`` (родитель каталога ``project/`` или ``data/raw``).

    Файл ожидается на уровне рабочей области (рядом с ``project``) или в ``project/data/raw/``.
    """
    project_root = Path(project_root).resolve()
    candidates = [
        project_root / "data" / REAL_DATASET_FILENAME,
        project_root / "data" / "raw" / REAL_DATASET_FILENAME,
        project_root.parent / REAL_DATASET_FILENAME,
    ]
    for p in candidates:
        if p.is_file():
            return p.resolve()
    searched = ", ".join(str(c) for c in candidates)
    msg = f"Файл {REAL_DATASET_FILENAME} не найден. Проверено: {searched}"
    raise FileNotFoundError(msg)
