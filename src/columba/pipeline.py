"""Оркестрация этапа 0: от сырой выгрузки до манифеста, таргетов и сплита."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg
from .dedup import assign_dedup_groups, pixel_hash
from .dicom_io import STATUS_FAILURE, STATUS_SUCCESS, read_dicom
from .inventory import build_inventory, count_all_dicom_under
from .markup import load_markup
from .regions import (
    classify_region_from_pixels,
    detect_mask_steps,
    hip_side,
    resolve_sides_within_study,
    zone_key,
)
from .split import build_split, save_split
from .submission import build_submission, save_submission, validate_submission
from .targets import (
    audit_against_legacy_totals,
    build_targets,
    check_no_violation_without_criteria,
)

TAG_TO_COLUMN = {
    "SOPInstanceUID": "sop_instance_uid",
    "StudyInstanceUID": "study_instance_uid",
    "SeriesInstanceUID": "series_instance_uid",
    "SeriesNumber": "series_number",
    "InstanceNumber": "instance_number",
    "Modality": "modality",
    "Manufacturer": "manufacturer",
    "ManufacturerModelName": "model_name",
    "SoftwareVersions": "software_version",
    "StudyDescription": "study_description",
    "SeriesDescription": "series_description",
    "BodyPartExamined": "body_part",
    "ViewPosition": "view_position",
    "Laterality": "laterality_tag",
    "PatientOrientation": "patient_orientation",
    "Rows": "rows",
    "Columns": "cols",
    "BitsAllocated": "bits_allocated",
    "BitsStored": "bits_stored",
    "PixelRepresentation": "pixel_representation",
    "PhotometricInterpretation": "photometric",
}


def run_stage0(
    studies_dir: Path | str = cfg.STUDIES_DIR,
    markup_xlsx: Path | str = cfg.MARKUP_XLSX,
    artifacts_dir: Path | str = cfg.ARTIFACTS_DIR,
    *,
    write_eda: bool = True,
    verbose: bool = True,
) -> dict:
    """Пройти шаги 1-8 и разложить артефакты в `artifacts_dir`."""
    started = time.time()
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    anomalies: list[dict] = []

    def say(message: str) -> None:
        if verbose:
            print(message, flush=True)

    # --- Шаг 1: инвентаризация ------------------------------------------- #
    manifest = build_inventory(studies_dir)
    total_under_data = count_all_dicom_under(cfg.DATA_DIR)
    say(f"[1/8] Инвентаризация: {len(manifest)} файлов в выгрузке, {manifest['study_folder'].nunique()} папок")
    anomalies.append(
        {
            "kind": "file_count_reconciliation",
            "detail": (
                f"Всего .dcm под data/: {total_under_data}. "
                f"В «{Path(studies_dir).name}» (обучающая выгрузка): {len(manifest)}. "
                f"В «{cfg.TEST_DIR.name}»: {count_all_dicom_under(cfg.TEST_DIR)}. "
                "Расхождение 502 против 499 из ответов организаторов объясняется "
                "тремя файлами тестового набора — вопрос закрыт."
            ),
        }
    )

    # --- Шаги 2-4: чтение, хэш, регион, маскирование ---------------------- #
    manifest = _read_all(manifest, say)
    manifest = assign_dedup_groups(manifest)
    manifest = _assign_sides(manifest)
    say(
        f"[3/8] Дедупликация: {manifest['dedup_group_id'].nunique()} уникальных изображений "
        f"из {int(manifest['read_status'].eq(STATUS_SUCCESS).sum())} читаемых файлов"
    )
    say(
        "[4/8] Регионы: "
        + ", ".join(f"{k}={v}" for k, v in manifest["region"].value_counts().items())
    )
    anomalies.extend(_region_anomalies(manifest))

    # --- Шаг 5: матчинг с «Калибровкой» ----------------------------------- #
    markup = load_markup(markup_xlsx)
    manifest, match_anomalies = _match(manifest, markup)
    anomalies.extend(match_anomalies)
    say(f"[5/8] Матчинг: {markup['study_folder'].nunique()} строк таблицы, "
        f"{int(manifest['matched_markup'].sum())} файлов сматчено")

    # --- Шаг 6: таргеты ---------------------------------------------------- #
    targets, target_anomalies = build_targets(manifest, markup)
    anomalies.extend(target_anomalies)
    anomalies.extend(audit_against_legacy_totals(targets, markup))
    broken = check_no_violation_without_criteria(targets)
    if len(broken):
        anomalies.append(
            {
                "kind": "violation_without_criteria",
                "detail": f"{len(broken)} строк с quality_class=1 при нулевых критериях: "
                + ", ".join(broken["dedup_group_id"]),
            }
        )
    say(
        f"[6/8] Таргеты: {int(targets['has_target'].sum())} размеченных изображений, "
        f"positives={int((targets['quality_class'] == 1).sum())}, "
        f"без таргета={int((~targets['has_target']).sum())}"
    )

    # --- Шаг 7: сплит ------------------------------------------------------ #
    split_payload = build_split(targets)
    save_split(split_payload, artifacts_dir / cfg.SPLIT_JSON.name)
    train_ids = set(split_payload["dedup_groups"]["train"])
    val_ids = set(split_payload["dedup_groups"]["val"])
    targets["split"] = targets["dedup_group_id"].map(
        lambda gid: "train" if gid in train_ids else ("val" if gid in val_ids else "excluded")
    )
    say(
        f"[7/8] Сплит: train={len(train_ids)} / val={len(val_ids)} изображений, "
        f"исследований {len(split_payload['groups']['train'])}/{len(split_payload['groups']['val'])}"
    )

    # --- Сохранение артефактов -------------------------------------------- #
    manifest = manifest.merge(
        targets[["dedup_group_id", "zone_annotated", "has_target", "quality_class", "split"]],
        on="dedup_group_id",
        how="left",
    )
    reference = _reference_submission(manifest, targets)
    validate_submission(reference, expected_rows=len(manifest))
    save_submission(reference, artifacts_dir / cfg.SUBMISSION_REFERENCE_CSV.name)
    say(
        f"[8/8] Эталонный сабмит: {len(reference)} строк на {len(manifest)} файлов, "
        f"формат и полярность провалидированы"
    )

    manifest.to_parquet(artifacts_dir / cfg.MANIFEST_PARQUET.name, index=False)
    manifest.to_csv(artifacts_dir / cfg.MANIFEST_CSV.name, index=False)
    targets.to_csv(artifacts_dir / cfg.TARGETS_CSV.name, index=False)
    _write_anomaly_log(anomalies, artifacts_dir / cfg.ANOMALY_LOG.name)

    if write_eda:
        from .eda import write_eda_report

        write_eda_report(manifest, targets, split_payload, artifacts_dir / cfg.EDA_DIR.name)
    say(f"[8/8] Артефакты записаны в {artifacts_dir} за {time.time() - started:.1f} c")

    return {
        "manifest": manifest,
        "submission_reference": reference,
        "targets": targets,
        "markup": markup,
        "split": split_payload,
        "anomalies": anomalies,
    }


# --------------------------------------------------------------------------- #
# Шаги 2-4
# --------------------------------------------------------------------------- #


def _read_all(manifest: pd.DataFrame, say) -> pd.DataFrame:
    records = []
    failures = 0
    for row in manifest.itertuples():
        result = read_dicom(row.abs_path)
        record: dict = {"file_id": row.file_id, "read_status": result.status, "read_error": result.error}
        if result.ok:
            record.update({TAG_TO_COLUMN[k]: v for k, v in result.tags.items() if k in TAG_TO_COLUMN})
            pixels = result.pixels
            record["pixel_hash"] = pixel_hash(pixels)
            record["pixel_min"] = int(np.min(pixels))
            record["pixel_max"] = int(np.max(pixels))
            # Регион — только из пикселей: в закрытом тесте тегов зоны нет.
            record["region"] = classify_region_from_pixels(pixels)
            record["cols_from_pixels"] = int(pixels.shape[1])
            side = hip_side(pixels) if record["region"] == cfg.REGION_HIP else None
            record["hip_side_score"] = side.score if side else np.nan
            record["hip_side_confident"] = bool(side.confident) if side else False
            steps = detect_mask_steps(pixels)
            record["mask_step_count"] = len(steps)
            record["mask_step_area_px"] = int(sum(int(s["area_px"]) for s in steps))
            record["has_mask_rect"] = bool(steps)
        else:
            failures += 1
            record["pixel_hash"] = None
            record["region"] = cfg.REGION_UNKNOWN
            record["hip_side_score"] = np.nan
            record["hip_side_confident"] = False
            record["mask_step_count"] = 0
            record["mask_step_area_px"] = 0
            record["has_mask_rect"] = False
        records.append(record)
    say(f"[2/8] Чтение DICOM: успешно {len(records) - failures}, Failure {failures}")
    return manifest.merge(pd.DataFrame(records), on="file_id", how="left")


def _assign_sides(manifest: pd.DataFrame) -> pd.DataFrame:
    """Сторона бедра: относительное сравнение внутри исследования."""
    manifest = manifest.copy()
    manifest["hip_side"] = pd.NA
    hips = manifest[(manifest["region"] == cfg.REGION_HIP) & manifest["dedup_group_id"].notna()]
    for study, chunk in hips.groupby("study_folder", sort=True):
        groups = (
            chunk.groupby("dedup_group_id", sort=True)["hip_side_score"].first().sort_index()
        )
        sides = resolve_sides_within_study(list(groups.values))
        mapping = dict(zip(groups.index, sides))
        mask = manifest["dedup_group_id"].isin(mapping) & (manifest["study_folder"] == study)
        manifest.loc[mask, "hip_side"] = manifest.loc[mask, "dedup_group_id"].map(mapping)
    manifest["zone_key"] = [
        zone_key(region, side if pd.notna(side) else None)
        for region, side in zip(manifest["region"], manifest["hip_side"])
    ]
    return manifest


def _region_anomalies(manifest: pd.DataFrame) -> list[dict]:
    findings: list[dict] = []
    readable = manifest[manifest["read_status"] == STATUS_SUCCESS]
    tag_mismatch = readable[readable["cols"] != readable["cols_from_pixels"]]
    for row in tag_mismatch.itertuples():
        findings.append(
            {
                "kind": "tag_vs_pixels_mismatch",
                "detail": f"{row.relative_path}: тег Columns={row.cols}, ширина массива={row.cols_from_pixels}",
            }
        )
    unexpected = manifest[manifest["region"] == cfg.REGION_UNKNOWN]
    unexpected = unexpected[unexpected["read_status"] == STATUS_SUCCESS]
    for width, chunk in unexpected.groupby("cols", sort=True):
        findings.append(
            {
                "kind": "unexpected_frame_width",
                "detail": (
                    f"ширина {int(width)} px не описана эвристикой (300 — позвоночник, 280 — бедро): "
                    f"{len(chunk)} файлов, исследования: {', '.join(sorted(set(chunk['study_folder'])))}"
                ),
            }
        )
    weak = manifest[(manifest["region"] == cfg.REGION_HIP) & (~manifest["hip_side_confident"])]
    weak = weak[weak["is_group_representative"]]
    for row in weak.itertuples():
        findings.append(
            {
                "kind": "weak_hip_side_signal",
                "detail": f"{row.study_folder} / {row.dedup_group_id}: |score|={abs(row.hip_side_score):.3f}",
            }
        )
    failures = manifest[manifest["read_status"] == STATUS_FAILURE]
    for row in failures.itertuples():
        findings.append({"kind": "unreadable_file", "detail": f"{row.relative_path}: {row.read_error}"})
    return findings


# --------------------------------------------------------------------------- #
# Шаг 5
# --------------------------------------------------------------------------- #


def _match(manifest: pd.DataFrame, markup: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    findings: list[dict] = []

    duplicated = markup[markup.duplicated("study_folder", keep=False)]
    for study, chunk in duplicated.groupby("study_folder"):
        findings.append(
            {
                "kind": "markup_duplicate_study",
                "detail": f"{study}: строк в таблице {len(chunk)} ({sorted(chunk['markup_row'])})",
            }
        )

    folders = set(manifest["study_folder"])
    table = set(markup["study_folder"])
    for study in sorted(folders - table):
        findings.append({"kind": "folder_without_markup", "detail": study})
    for study in sorted(table - folders):
        findings.append({"kind": "markup_without_folder", "detail": study})

    lookup = markup.drop_duplicates("study_folder").set_index("study_folder")
    manifest = manifest.copy()
    manifest["matched_markup"] = manifest["study_folder"].isin(lookup.index)
    manifest["markup_row"] = manifest["study_folder"].map(lookup["markup_row"])
    manifest["markup_index"] = manifest["study_folder"].map(lookup["markup_index"])
    manifest["markup_comment"] = manifest["study_folder"].map(lookup["comment"]).fillna("")

    findings.extend(_zone_consistency(manifest, lookup))
    return manifest, findings


def _zone_consistency(manifest: pd.DataFrame, lookup: pd.DataFrame) -> list[dict]:
    """Сходится ли набор зон на снимках с тем, что размечено в таблице."""
    findings: list[dict] = []
    representatives = manifest[manifest["is_group_representative"].fillna(False)]
    for study, chunk in representatives.groupby("study_folder", sort=True):
        if study not in lookup.index:
            continue
        row = lookup.loc[study]
        annotated = {
            zone
            for zone in cfg.ZONES
            if any(pd.notna(row[c.key]) for c in cfg.CRITERIA_BY_ZONE[zone])
        }
        imaged = {z for z in chunk["zone_key"].dropna()}
        unknown_images = int(chunk["zone_key"].isna().sum())
        if annotated != imaged or unknown_images:
            findings.append(
                {
                    "kind": "zone_mismatch",
                    "detail": (
                        f"{study}: на снимках {sorted(imaged) or '—'}"
                        + (f" (+{unknown_images} без зоны)" if unknown_images else "")
                        + f", в таблице размечено {sorted(annotated) or '—'}"
                        + (f", комментарий: {row['comment']}" if row["comment"] else "")
                    ),
                }
            )
    return findings


# --------------------------------------------------------------------------- #
# Лог аномалий
# --------------------------------------------------------------------------- #

ANOMALY_TITLES = {
    "file_count_reconciliation": "Сверка числа файлов (502 против 499)",
    "unreadable_file": "Нечитаемые файлы (статус Failure)",
    "unexpected_frame_width": "Кадры с неожиданной шириной",
    "tag_vs_pixels_mismatch": "Тег Columns расходится с формой массива",
    "weak_hip_side_signal": "Слабый сигнал латеральности бедра",
    "markup_duplicate_study": "Папка сматчилась на несколько строк таблицы",
    "folder_without_markup": "Папка выгрузки без строки в таблице",
    "markup_without_folder": "Строка таблицы без папки в выгрузке",
    "zone_mismatch": "Расхождение зон: снимки против таблицы",
    "dedup_group_spans_studies": "Одна dedup-группа в разных исследованиях",
    "dedup_group_ambiguous_zone": "Dedup-группа с неоднозначной зоной",
    "partially_annotated_zone": "Зона размечена частично",
    "legacy_total_mismatch": "«Итог» расходится с перегенерированным quality_class",
    "violation_without_criteria": "quality_class=1 при нулевых критериях",
}


def _write_anomaly_log(anomalies: list[dict], path: Path) -> Path:
    grouped: dict[str, list[dict]] = {}
    for item in anomalies:
        grouped.setdefault(item["kind"], []).append(item)

    lines = [
        "# Лог аномалий этапа 0",
        "",
        "Файл генерируется автоматически: `uv run python main.py`.",
        "",
    ]
    for kind, title in ANOMALY_TITLES.items():
        items = grouped.get(kind, [])
        lines.append(f"## {title} — {len(items)}")
        lines.append("")
        if not items:
            lines.extend(["Не обнаружено.", ""])
            continue
        for item in items:
            prefix = f"`{item['dedup_group_id']}` — " if item.get("dedup_group_id") else ""
            lines.append(f"- {prefix}{item['detail']}")
        lines.append("")
    for kind in sorted(set(grouped) - set(ANOMALY_TITLES)):
        lines.append(f"## {kind} — {len(grouped[kind])}")
        lines.extend([f"- {i['detail']}" for i in grouped[kind]] + [""])

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _reference_submission(manifest: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Сабмит из истинных меток обучающей выгрузки.

    Нужен как сквозная проверка цепочки «манифест -> таргеты -> выходной
    формат»: строка на каждый файл, посимвольные строки словаря, полярность
    `1 = нарушение`. Модели здесь нет — это эталон, а не предсказание.
    """
    annotated = targets[targets["has_target"].fillna(False)]
    predictions = pd.DataFrame(
        {
            "dedup_group_id": annotated["dedup_group_id"],
            "quality_class": annotated["quality_class"].astype(int),
            "quality_prob": annotated["quality_class"].astype(float),
            "violation_type": annotated["violation_type"].fillna(""),
        }
    )
    return build_submission(manifest, predictions)
