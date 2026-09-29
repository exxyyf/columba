"""Бонус 5 (этап 8, приоритет 5): DICOM SR как альтернативный формат вывода.

Один DICOM SR (Comprehensive SR, PS3.3 Annex A) документ на входной файл —
та же информация, что уже есть в CSV-сабмите (`anatomical_region`,
`quality_class`, `quality_prob`, `violation_type` — формат автопроверки),
плюс разбивка по отдельным чекерам региона (`score`/`flag`/`status`/
`reason` из `inference.describe_inputs`'s `spine_signals`/`hip_signals`) —
то, чего в плоском CSV нет вообще. Это ДОПОЛНИТЕЛЬНЫЙ артефакт по запросу
бонуса, не замена CSV: формат автопроверки организаторов остаётся
`submission.SUBMISSION_COLUMNS`, здесь не тронут.

**Почему `highdicom`, а не сборка тегов вручную поверх pydicom.** Правила
вложенности content-дерева SR (Value Type / Relationship Type, PS3.3
Annex A) нетривиальны — ручная сборка рисковала бы дать формально
невалидный документ, который парсер DICOM-вьюера отвергнет. `highdicom` —
специализированная, тестируемая библиотека именно для этого (уже де-факто
стандарт в medical imaging AI за пределами этого проекта), добавлена как
зависимость проекта (`pyproject.toml`) для этой единственной задачи —
инференс/сервис её не импортируют.

**Честная оговорка о кодировании понятий.** Поля, специфичные для этого
проекта (`quality_class`, `violation_type`, имена чекеров/сигналов — не из
стандартного DICOM-словаря), кодируются ЧАСТНОЙ схемой `99COLUMBA`
(легитимная DICOM-практика для непокрытых стандартом понятий, PS3.16) — не
подобраны (возможно неточные) существующие SNOMED/DCM-коды: заявить
соответствие стандартному коду, не проверив его точное значение по
таблицам PS3.16, было бы менее честно, чем открыто использовать частную
схему. Корневой контейнер — `(121070, DCM, "Findings")`, общеупотребимый
стандартный DCM-код для корня SR с находками.

**Anonymized-артефакт исходных данных.** Организаторы анонимизировали поля
пациента буквальной строкой `"Anonymized"` независимо от VR — включая
`PatientBirthDate`/`StudyDate`/`StudyTime`/`PatientSex` (DA/DA/TM/CS),
для которых это невалидное значение. `highdicom` строго проверяет VR при
наследовании Patient/Study-модулей от evidence-датасета и падает
`ValueError` без очистки — `_sanitize_evidence` чистит именно эти 4 поля
(на КОПИИ датасета, не трогая файл на диске) перед передачей в
`ComprehensiveSR(evidence=...)`.

**Что НЕ кодируется.** Сырые геометрические `signals` чекеров (координаты
ориентиров, bbox компонент металла и т.п.) не переносятся в SR content-дерево
— это не дискретные находки/измерения, для которых SR предназначен, а
дамп внутренних координат; та же информация уже доступна как JSON через
`/visualize?format=json` (задача 9.6) и как PNG-оверлей (`visualize.py`).

Запуск: `uv run python -m columba.dicom_sr <файл.dcm|каталог> [-o каталог]`
-> по одному `.dcm` (Modality=SR) на входной файл, по умолчанию
`artifacts/sr/<имя>.sr.dcm`.
"""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pydicom
from highdicom.sr import (
    CodeContentItem,
    CodedConcept,
    ComprehensiveSR,
    ContainerContentItem,
    ContentSequence,
    NumContentItem,
    TextContentItem,
)
from pydicom.sr.coding import Code
from pydicom.uid import generate_uid

from . import config as cfg
from .inference import describe_inputs
from .submission import region_to_output

DEFAULT_OUTPUT_DIR = cfg.ARTIFACTS_DIR / "sr"

_ROOT_CONCEPT = Code("121070", "DCM", "Findings")
_NO_UNITS = Code("1", "UCUM", "no units")
_ANONYMIZED_FIELDS_TO_CLEAR = ("PatientBirthDate", "StudyDate", "StudyTime", "PatientSex")


def _private_concept(key: str, meaning: str) -> CodedConcept:
    return CodedConcept(value=key, scheme_designator="99COLUMBA", meaning=meaning)


def _sanitize_evidence(ds: pydicom.Dataset) -> pydicom.Dataset:
    """Копия `ds` с очищенными anonymized-полями, невалидными для своего VR
    (см. докстринг модуля) — `highdicom` иначе падает на `DA("Anonymized")`.
    """
    ds = deepcopy(ds)
    for tag in _ANONYMIZED_FIELDS_TO_CLEAR:
        if getattr(ds, tag, None) == "Anonymized":
            setattr(ds, tag, "")
    return ds


def _checker_container(checker_key: str, result: dict) -> ContainerContentItem:
    container = ContainerContentItem(
        name=_private_concept(checker_key, checker_key),
        relationship_type="CONTAINS",
    )
    items = [
        TextContentItem(
            name=_private_concept("status", "Статус чекера"),
            value=str(result.get("status", "")),
            relationship_type="CONTAINS",
        ),
        CodeContentItem(
            name=_private_concept("flag", "Флаг нарушения"),
            value=Code("1", "99COLUMBA", "Да") if result.get("flag") else Code("0", "99COLUMBA", "Нет"),
            relationship_type="CONTAINS",
        ),
    ]
    score = result.get("score")
    if score is not None and not (isinstance(score, float) and math.isnan(score)):
        items.append(
            NumContentItem(
                name=_private_concept("score", "Скор чекера"),
                value=float(score),
                unit=_NO_UNITS,
                relationship_type="CONTAINS",
            )
        )
    reason = result.get("reason")
    if reason:
        items.append(
            TextContentItem(
                name=_private_concept("reason", "Причина"),
                value=str(reason),
                relationship_type="CONTAINS",
            )
        )
    container.ContentSequence = ContentSequence(items)
    return container


def build_sr_content(row: pd.Series) -> ContainerContentItem:
    """Строка манифеста (`inference.describe_inputs`, с `region_final`,
    `hip_side_final`, `spine_signals`/`hip_signals`, опционально
    `quality_class`/`quality_prob`/`violation_type` из сабмита —
    `write_sr_reports` подмешивает их через `submission_from_manifest`)
    -> корневой контейнер SR-документа для этого файла."""
    root = ContainerContentItem(name=_ROOT_CONCEPT)
    items = [
        TextContentItem(
            name=_private_concept("file_name", "Имя файла"),
            value=str(row.get("file_name", row.get("relative_path", ""))),
            relationship_type="CONTAINS",
        ),
        TextContentItem(
            name=_private_concept("anatomical_region", "Анатомический регион"),
            value=region_to_output(row["region_final"]),
            relationship_type="CONTAINS",
        ),
    ]
    if "quality_class" in row.index and pd.notna(row.get("quality_class")):
        items.append(
            CodeContentItem(
                name=_private_concept("quality_class", "Класс качества"),
                value=Code("1", "99COLUMBA", "Нарушение") if int(row["quality_class"]) == 1 else Code(
                    "0", "99COLUMBA", "Норма"
                ),
                relationship_type="CONTAINS",
            )
        )
    if "quality_prob" in row.index and pd.notna(row.get("quality_prob")):
        items.append(
            NumContentItem(
                name=_private_concept("quality_prob", "Вероятность нарушения"),
                value=float(row["quality_prob"]),
                unit=_NO_UNITS,
                relationship_type="CONTAINS",
            )
        )
    if row.get("violation_type"):
        items.append(
            TextContentItem(
                name=_private_concept("violation_type", "Тип нарушения"),
                value=str(row["violation_type"]),
                relationship_type="CONTAINS",
            )
        )
    if pd.notna(row.get("hip_side_final")):
        items.append(
            TextContentItem(
                name=_private_concept("side", "Сторона"),
                value=str(row["hip_side_final"]),
                relationship_type="CONTAINS",
            )
        )

    signals_column = "spine_signals" if row["region_final"] == cfg.REGION_SPINE else "hip_signals"
    raw_signals = row.get(signals_column)
    checkers: dict = {}
    if isinstance(raw_signals, str) and raw_signals.strip():
        checkers = json.loads(raw_signals)
    for checker_key, result in checkers.items():
        items.append(_checker_container(checker_key, result))

    root.ContentSequence = ContentSequence(items)
    return root


def build_sr_document(dicom_path: Path, row: pd.Series) -> ComprehensiveSR:
    evidence = _sanitize_evidence(pydicom.dcmread(dicom_path))
    content = build_sr_content(row)
    return ComprehensiveSR(
        evidence=[evidence],
        content=content,
        series_instance_uid=generate_uid(),
        series_number=1,
        sop_instance_uid=generate_uid(),
        instance_number=1,
        manufacturer="columba",
        is_complete=True,
        is_final=True,
    )


def write_sr_reports(input_dir: Path | str, output_dir: Path | str = DEFAULT_OUTPUT_DIR) -> list[Path]:
    """Один SR-файл на каждый читаемый DICOM в `input_dir` (рекурсивно).

    Каждый документ несёт и разбивку по чекерам (из манифеста), и итоговый
    вердикт сабмита (`quality_class`/`quality_prob`/`violation_type` из
    `submission_from_manifest` — тот же агрегатор, что и у `/predict`,
    не пересчитан заново другой логикой), смёрженный по `file_name`.
    """
    from .inference import submission_from_manifest

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = describe_inputs(input_dir)
    submission = submission_from_manifest(manifest)
    verdict_by_file = submission.set_index("file_name")[["quality_class", "quality_prob", "violation_type"]]

    written: list[Path] = []
    for row in manifest.itertuples():
        row_dict = row._asdict()
        if row_dict.get("read_status") != "Success":
            continue
        series = pd.Series(row_dict)
        if series["file_name"] in verdict_by_file.index:
            verdict = verdict_by_file.loc[series["file_name"]]
            series = pd.concat([series, verdict[~verdict.index.isin(series.index)]])
        doc = build_sr_document(Path(row_dict["abs_path"]), series)
        out_path = output_dir / f"{Path(row_dict['abs_path']).stem}.sr.dcm"
        doc.save_as(out_path)
        written.append(out_path)
    return written


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="python -m columba.dicom_sr",
        description="Сабмит-результаты + разбивка по чекерам -> DICOM SR (бонус 5).",
    )
    parser.add_argument("input", help="DICOM-файл или каталог (рекурсивно)")
    parser.add_argument("-o", "--output", default=str(DEFAULT_OUTPUT_DIR), help="каталог для .sr.dcm")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ошибка: не найдено: {input_path}", file=sys.stderr)
        return 1
    if input_path.is_file():
        # Изолированный tmp-каталог — иначе describe_inputs подхватил бы и
        # соседние файлы из той же папки, а вызывающий просил ровно один
        # (тот же приём, что `service.py`'s `/visualize`).
        with tempfile.TemporaryDirectory(prefix="columba_dicom_sr_") as tmp:
            shutil.copy(input_path, Path(tmp) / input_path.name)
            written = write_sr_reports(tmp, output_dir=args.output)
    else:
        written = write_sr_reports(input_path, output_dir=args.output)
    for path in written:
        print(path)
    print(f"записано {len(written)} SR-документ(ов)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
