"""Шаг 1: инвентаризация выгрузки.

Строится «манифест» — единая таблица файлового уровня, к которой дальше
присоединяется всё остальное (хэши, регионы, зоны, матчинг, таргеты).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config as cfg
from .config import STUDIES_DIR

DICOM_SUFFIX = ".dcm"


def list_dicom_files(studies_dir: Path | str = STUDIES_DIR) -> list[Path]:
    """Все .dcm выгрузки, отсортированные — порядок обхода детерминирован."""
    studies_dir = Path(studies_dir)
    return sorted(p for p in studies_dir.rglob(f"*{DICOM_SUFFIX}") if p.is_file())


def build_inventory(studies_dir: Path | str = STUDIES_DIR) -> pd.DataFrame:
    """Файловый реестр: путь, папка-корень исследования, размер, глубина."""
    studies_dir = Path(studies_dir)
    rows = []
    for path in list_dicom_files(studies_dir):
        relative = path.relative_to(studies_dir)
        rows.append(
            {
                "file_id": "",  # проставляется ниже, когда порядок зафиксирован
                "relative_path": str(relative),
                "abs_path": str(path),
                "study_folder": relative.parts[0],
                "file_name": path.name,
                "depth_in_study": len(relative.parts) - 1,
                "size_bytes": path.stat().st_size,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame.sort_values("relative_path", ignore_index=True)
    frame["file_id"] = [f"f{idx:04d}" for idx in range(len(frame))]
    return frame


def count_all_dicom_under(root: Path | str) -> int:
    """Сколько .dcm лежит под каталогом — для сверки 502 против 499."""
    return sum(1 for p in Path(root).rglob(f"*{DICOM_SUFFIX}") if p.is_file())


def load_manifest(
    path: Path | str | None = None,
    *,
    studies_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Читает манифест (CSV/parquet) и пересчитывает `abs_path` заново.

    `relative_path` в манифесте задан относительно `studies_dir` НА МАШИНЕ,
    где строилась выгрузка (см. `build_inventory`); колонка `abs_path` из
    файла могла быть записана на другой машине (другая ОС/пользователь) и
    там не существует. Поэтому `abs_path` ВСЕГДА пересчитывается из
    `relative_path` относительно `studies_dir` ТЕКУЩЕЙ машины, а не читается
    из файла — единственный загрузчик для всех читателей манифеста (этап 9,
    п. 9.1; паттерн раньше жил отдельно в каждом из `hip_eval`/`hip_export`/
    `hip_grids`).

    `path`/`studies_dir` по умолчанию берутся из `config` в МОМЕНТ ВЫЗОВА
    (не как значение по умолчанию аргумента), чтобы `monkeypatch.setattr`
    в тестах на `cfg.MANIFEST_CSV`/`cfg.STUDIES_DIR` действовал.
    """
    path = Path(path if path is not None else cfg.MANIFEST_CSV)
    studies_dir = Path(studies_dir if studies_dir is not None else cfg.STUDIES_DIR)
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, low_memory=False)
    if not frame.empty and "relative_path" in frame.columns:
        frame["abs_path"] = frame["relative_path"].apply(lambda rel: str(studies_dir / rel))
    return frame
