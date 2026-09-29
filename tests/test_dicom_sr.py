"""Бонус 5: DICOM SR как альтернативный формат вывода."""

from __future__ import annotations

import json

import pydicom

from columba import config as cfg
from columba.dicom_sr import (
    _sanitize_evidence,
    build_sr_content,
    build_sr_document,
    write_sr_reports,
)

from conftest import requires_data

COMPREHENSIVE_SR_SOP_CLASS_UID = "1.2.840.10008.5.1.4.1.1.88.33"


def _find_text(item, meaning: str) -> str | None:
    for child in item.ContentSequence:
        child_meaning = child.ConceptNameCodeSequence[0].CodeMeaning
        if child_meaning == meaning and child.ValueType == "TEXT":
            return child.TextValue
    return None


def _find_container(item, meaning: str):
    for child in item.ContentSequence:
        child_meaning = child.ConceptNameCodeSequence[0].CodeMeaning
        if child_meaning == meaning and child.ValueType == "CONTAINER":
            return child
    return None


def test_sanitize_evidence_clears_invalid_anonymized_fields():
    ds = pydicom.Dataset()
    ds.PatientBirthDate = "Anonymized"
    ds.StudyDate = "Anonymized"
    ds.StudyTime = "Anonymized"
    ds.PatientSex = "Anonymized"
    ds.PatientID = "Anonymized"  # валиден для LO/SH — не трогаем

    sanitized = _sanitize_evidence(ds)

    assert sanitized.PatientBirthDate == ""
    assert sanitized.StudyDate == ""
    assert sanitized.StudyTime == ""
    assert sanitized.PatientSex == ""
    assert sanitized.PatientID == "Anonymized"  # не тронуто
    # Исходный датасет не изменён (сохраняется копия).
    assert ds.PatientBirthDate == "Anonymized"


def test_build_sr_content_includes_region_side_and_checkers():
    import pandas as pd

    row = pd.Series(
        {
            "file_name": "example.dcm",
            "region_final": cfg.REGION_HIP,
            "hip_side_final": "left",
            "hip_signals": json.dumps(
                {
                    "hip_positioning": {
                        "status": "ok",
                        "score": 0.75,
                        "flag": True,
                        "reason": "",
                    },
                    "hip_roi": {
                        "status": "not_evaluated",
                        "score": float("nan"),
                        "flag": False,
                        "reason": "кадр вырожден",
                    },
                }
            ),
        }
    )
    root = build_sr_content(row)

    assert _find_text(root, "Имя файла") == "example.dcm"
    assert _find_text(root, "Анатомический регион") == "Проксимальный отдел бедра"
    assert _find_text(root, "Сторона") == "left"

    positioning = _find_container(root, "hip_positioning")
    assert positioning is not None
    assert _find_text(positioning, "Статус чекера") == "ok"

    roi = _find_container(root, "hip_roi")
    assert roi is not None
    assert _find_text(roi, "Статус чекера") == "not_evaluated"
    assert _find_text(roi, "Причина") == "кадр вырожден"
    # score=NaN -> NUM-элемент не создаётся (score not None and not NaN check).
    assert not any(
        c.ValueType == "NUM" for c in roi.ContentSequence
    )


@requires_data
def test_build_sr_document_round_trips_as_valid_sr(tmp_path):
    import pandas as pd

    from columba.inference import describe_inputs

    manifest = describe_inputs(cfg.TEST_DIR)
    row = manifest[manifest["region_final"] == cfg.REGION_SPINE].iloc[0]
    doc = build_sr_document(cfg.TEST_DIR / row["file_name"], pd.Series(row))

    out_path = tmp_path / "test.sr.dcm"
    doc.save_as(out_path)

    reread = pydicom.dcmread(out_path)
    assert reread.Modality == "SR"
    assert str(reread.SOPClassUID) == COMPREHENSIVE_SR_SOP_CLASS_UID
    assert _find_text(reread, "Анатомический регион") == "Поясничный отдел позвоночника"
    assert _find_container(reread, "spine_axis") is not None


@requires_data
def test_write_sr_reports_creates_one_file_per_readable_input(tmp_path):
    written = write_sr_reports(cfg.TEST_DIR, output_dir=tmp_path)
    assert len(written) == 3
    for path in written:
        assert path.exists()
        reread = pydicom.dcmread(path)
        assert reread.Modality == "SR"


@requires_data
def test_write_sr_reports_includes_submission_verdict(tmp_path):
    """Регрессия: `write_sr_reports` подмешивает `quality_class`/
    `quality_prob`/`violation_type` из `submission_from_manifest` — та же
    итоговая информация, что в CSV-сабмите, не только разбивка по чекерам."""
    written = write_sr_reports(cfg.TEST_DIR, output_dir=tmp_path)
    hip_doc_path = next(p for p in written if "ППОБ" in p.name)
    reread = pydicom.dcmread(hip_doc_path)

    quality_class_item = next(
        c for c in reread.ContentSequence if c.ConceptNameCodeSequence[0].CodeMeaning == "Класс качества"
    )
    assert quality_class_item.ConceptCodeSequence[0].CodeMeaning in ("Норма", "Нарушение")
    assert any(c.ConceptNameCodeSequence[0].CodeMeaning == "Вероятность нарушения" for c in reread.ContentSequence)
