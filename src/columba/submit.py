"""CLI сабмита (шаг 7 этапа 3).

Запуск: `uv run python -m columba.submit <input_dir> <output.csv>`.

Строит сабмит через `run_inference` — `predictor` не передаётся явно, поэтому
подхватывается значение по умолчанию (`AUTO`): агрегатор этапа 3, когда он
подключён, иначе «нулевой» сабмит для отладки формата. `validate_submission`
уже встроена в `run_inference`, здесь она не дублируется.

После записи печатается сводка: число файлов, разбивка по
`anatomical_region`, число файлов на каждый тип нарушения из
`config.VIOLATION_TYPES`, число `quality_class == 1`, суммарное и среднее
время (на файл и на исследование). Никаких сетевых вызовов — только чтение
входного каталога и запись CSV.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from . import config as cfg
from .inference import FLAT_STUDY_KEY, run_inference
from .inventory import DICOM_SUFFIX


def main(argv: list[str] | None = None) -> int:
    """Точка входа CLI. Возвращает код возврата (0 — успех)."""
    _ensure_utf8_streams()
    args = _build_parser().parse_args(argv)

    input_dir = Path(args.input_dir)
    output_csv = Path(args.output_csv)

    error = _validate_input_dir(input_dir)
    if error is not None:
        print(f"ошибка: {error}", file=sys.stderr)
        return 1

    started = time.perf_counter()
    try:
        submission = run_inference(input_dir, output_csv=output_csv)
    except ValueError as exc:
        # ValueError покрывает и «нет .dcm файлов» из run_inference, и
        # SubmissionError (её подкласс) из validate_submission — без трейсбека.
        print(f"ошибка: {exc}", file=sys.stderr)
        return 1
    elapsed = time.perf_counter() - started

    _print_summary(submission, output_csv, elapsed)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m columba.submit",
        description="Построить сабмит в формате автопроверки по каталогу DICOM и сохранить в CSV.",
    )
    parser.add_argument("input_dir", help="каталог с .dcm — плоский или с папками исследований")
    parser.add_argument("output_csv", help="путь для сохранения сабмита (CSV)")
    return parser


def _validate_input_dir(input_dir: Path) -> str | None:
    """Понятное сообщение об ошибке входа вместо трейсбека, или None если всё ок."""
    if not input_dir.exists():
        return f"каталог не найден: {input_dir}"
    if not input_dir.is_dir():
        return f"это не каталог: {input_dir}"
    if not any(input_dir.rglob(f"*{DICOM_SUFFIX}")):
        return f"во входном каталоге нет файлов {DICOM_SUFFIX}: {input_dir}"
    return None


def _ensure_utf8_streams() -> None:
    """Кириллица в сводке не должна падать в консоли Windows (cp866/cp1251)."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass  # поток не поддерживает переconfig (например, уже закрыт/перенаправлен)


def _print_summary(submission: pd.DataFrame, output_csv: Path, elapsed: float) -> None:
    n_files = len(submission)
    print(f"Файлов обработано: {n_files}")
    print(f"Сабмит сохранён: {output_csv}")

    print("Разбивка по anatomical_region:")
    for region, count in submission["anatomical_region"].value_counts().items():
        print(f"  {region}: {count}")

    print("Файлов с нарушением по типам (config.VIOLATION_TYPES):")
    violation_texts = submission["violation_type"].fillna("").astype(str)
    for violation_type in cfg.VIOLATION_TYPES:
        count = int(
            violation_texts.apply(
                lambda text, vt=violation_type: vt in text.split(cfg.VIOLATION_TYPE_SEPARATOR)
            ).sum()
        )
        print(f"  {violation_type}: {count}")

    n_violation = int((submission["quality_class"] == cfg.QUALITY_CLASS_VIOLATION).sum())
    print(f"quality_class == 1 (нарушение): {n_violation} из {n_files}")

    n_studies, is_flat = _count_studies(submission)
    if is_flat:
        print(
            "Каталог плоский (без папок исследований) — считаем его одним "
            "исследованием для расчёта времени на исследование."
        )
    print(f"Исследований (study_folder): {n_studies}")

    per_file = elapsed / n_files if n_files else 0.0
    per_study = elapsed / n_studies if n_studies else 0.0
    print(
        f"Время: {elapsed:.2f} c всего, {per_file:.3f} c/файл, "
        f"{per_study:.2f} c/исследование"
    )


def _count_studies(submission: pd.DataFrame) -> tuple[int, bool]:
    """Число исследований и флаг «каталог плоский» (study_folder == '.')."""
    if "study_folder" not in submission.columns or submission.empty:
        return len(submission), False
    folders = submission["study_folder"]
    n_studies = int(folders.nunique())
    is_flat = n_studies == 1 and folders.iloc[0] == FLAT_STUDY_KEY
    return n_studies, is_flat


if __name__ == "__main__":
    raise SystemExit(main())
