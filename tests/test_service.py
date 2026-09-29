"""Этап 7: HTTP-сервис (`service.py`) — тонкая обёртка над `run_inference`.

Реальные предсказания не проверяются здесь заново (это территория
`test_submission.py`/`test_aggregate.py`) — только то, что HTTP-слой
корректно передаёт файлы в `run_inference` и отдаёт формат ответа.
"""

from __future__ import annotations

import io
import zipfile

from fastapi.testclient import TestClient

from columba import config as cfg
from columba.service import app

from conftest import requires_data

client = TestClient(app)

TEST_FILES = ("CR000000_ПОП.dcm", "CR000000_ППОБ.dcm", "CR000001_ЛПОБ.dcm")


def test_health_reports_status_device_and_weights():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "device" in body
    assert body["seed"] == cfg.SEED
    assert set(body["weights_loaded"]) == {
        "region_cnn",
        "spine_keypoint_unet",
        "spine_objects_cnn",
        "hip_positioning_cnn",
    }


def test_predict_without_files_is_a_client_error():
    response = client.post("/predict", files=[])
    assert response.status_code in (400, 422)


def test_predict_zip_rejects_invalid_archive():
    response = client.post("/predict/zip", files={"file": ("bad.zip", io.BytesIO(b"not a zip"), "application/zip")})
    assert response.status_code == 400


def test_predict_zip_rejects_archive_without_dicom():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("readme.txt", "no dicom here")
    buf.seek(0)
    response = client.post("/predict/zip", files={"file": ("batch.zip", buf, "application/zip")})
    assert response.status_code == 400


@requires_data
def test_predict_returns_csv_with_one_row_per_file():
    files = [
        ("files", (name, open(cfg.TEST_DIR / name, "rb"), "application/dicom")) for name in TEST_FILES
    ]
    response = client.post("/predict", files=files)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    lines = response.text.strip().splitlines()
    assert len(lines) == 1 + len(TEST_FILES)  # заголовок + строка на файл
    header = lines[0].split(",")
    from columba.submission import SUBMISSION_COLUMNS

    for column in SUBMISSION_COLUMNS:
        assert column in header


@requires_data
def test_predict_json_format_matches_csv_row_count():
    files = [("files", (name, open(cfg.TEST_DIR / name, "rb"), "application/dicom")) for name in TEST_FILES]
    response = client.post("/predict?format=json", files=files)
    assert response.status_code == 200
    body = response.json()
    assert body["n_files"] == len(TEST_FILES)
    assert len(body["rows"]) == len(TEST_FILES)
    assert body["elapsed_seconds"] >= 0.0


@requires_data
def test_predict_zip_preserves_study_folder_structure():
    """ZIP с папкой исследования — `study_folder` в ответе не `.` (плоский),
    а имя папки (парное определение стороны бедра доступно)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        for name in TEST_FILES:
            archive.write(cfg.TEST_DIR / name, arcname=f"study_001/{name}")
    buf.seek(0)
    response = client.post("/predict/zip?format=json", files={"file": ("batch.zip", buf, "application/zip")})
    assert response.status_code == 200
    rows = response.json()["rows"]
    assert all(row["study_folder"] == "study_001" for row in rows)


@requires_data
def test_predict_does_not_lose_files_with_colliding_basenames():
    """Два разных файла, загруженных под ОДНИМ именем, раньше молча теряли
    один друг друга на диске до инференса (задача 9.3, P0): второй перезаписывал
    первый в общем tmp_dir. Оба должны попасть в сабмит и обработаться реально
    (не как дубль одного и того же файла) — тест ловит это через регион: один
    файл позвоночник, другой бедро, значит при перезаписи оба ряда совпали бы."""
    names = ("CR000000_ПОП.dcm", "CR000000_ППОБ.dcm")
    files = [
        ("files", ("same.dcm", open(cfg.TEST_DIR / name, "rb"), "application/dicom")) for name in names
    ]
    response = client.post("/predict?format=json", files=files)
    assert response.status_code == 200
    body = response.json()
    assert body["n_files"] == 2
    rows = body["rows"]
    file_names = {row["file_name"] for row in rows}
    assert len(file_names) == 2  # уникализированы, ни один не потерян
    regions = {row["anatomical_region"] for row in rows}
    assert len(regions) == 2  # реально два разных файла, а не дубль одного


@requires_data
def test_predict_flat_upload_ignores_uploaded_subpaths():
    """Плоский `/predict` — filename без директорий, даже если клиент
    прислал путь с разделителями (защита от path traversal при записи)."""
    name = TEST_FILES[0]
    malicious_name = "../../etc/whatever.dcm"
    files = [("files", (malicious_name, open(cfg.TEST_DIR / name, "rb"), "application/dicom"))]
    response = client.post("/predict?format=json", files=files)
    assert response.status_code == 200
    assert response.json()["n_files"] == 1


# --------------------------------------------------------------------------- #
# `/predict` с `paths` — папочная загрузка веб-интерфейса (задача 9.6)
# --------------------------------------------------------------------------- #


@requires_data
def test_predict_structured_paths_enable_study_folder_pairing():
    """С `paths` (папочный режим UI) `/predict` восстанавливает структуру
    каталогов на диске — та же пара сторон, что у `/predict/zip`
    (регрессия по образцу `test_predict_zip_preserves_study_folder_structure`)."""
    files = [("files", (name, open(cfg.TEST_DIR / name, "rb"), "application/dicom")) for name in TEST_FILES]
    paths = [f"study_001/{name}" for name in TEST_FILES]
    response = client.post("/predict?format=json", files=files, data={"paths": paths})
    assert response.status_code == 200
    rows = response.json()["rows"]
    assert all(row["study_folder"] == "study_001" for row in rows)


@requires_data
def test_predict_structured_paths_rejects_traversal():
    name = TEST_FILES[0]
    files = [("files", (name, open(cfg.TEST_DIR / name, "rb"), "application/dicom"))]
    response = client.post("/predict?format=json", files=files, data={"paths": ["../../etc/whatever.dcm"]})
    assert response.status_code == 400


@requires_data
def test_predict_structured_paths_dedupes_colliding_paths():
    """Тот же случай, что `test_predict_does_not_lose_files_with_colliding_basenames`,
    но в папочном режиме — коллизия ПОЛНОГО относительного пути, не только имени."""
    names = ("CR000000_ПОП.dcm", "CR000000_ППОБ.dcm")
    files = [("files", ("dup.dcm", open(cfg.TEST_DIR / name, "rb"), "application/dicom")) for name in names]
    response = client.post(
        "/predict?format=json", files=files, data={"paths": ["study_001/dup.dcm", "study_001/dup.dcm"]}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["n_files"] == 2
    regions = {row["anatomical_region"] for row in body["rows"]}
    assert len(regions) == 2


def test_predict_structured_paths_count_mismatch_is_a_client_error():
    name = TEST_FILES[0]
    files = [("files", (name, io.BytesIO(b"stub"), "application/dicom"))]
    response = client.post("/predict?format=json", files=files, data={"paths": ["a.dcm", "b.dcm"]})
    assert response.status_code == 400


# --------------------------------------------------------------------------- #
# Веб-интерфейс: GET /, статика, POST /visualize (этап 8, бонусы 1-2)
# --------------------------------------------------------------------------- #


def test_index_page_served():
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "columba" in response.text.lower()


def test_static_assets_served():
    for path in ("/static/style.css", "/static/app.js"):
        response = client.get(path)
        assert response.status_code == 200, path


@requires_data
def test_visualize_returns_png():
    name = "CR000000_ППОБ.dcm"
    with open(cfg.TEST_DIR / name, "rb") as f:
        response = client.post("/visualize", files={"file": (name, f, "application/dicom")})
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_visualize_rejects_unreadable_file():
    response = client.post(
        "/visualize", files={"file": ("bad.dcm", io.BytesIO(b"not a dicom"), "application/dicom")}
    )
    assert response.status_code == 400


@requires_data
def test_visualize_json_matches_png_region_and_reports_checkers():
    name = "CR000000_ППОБ.dcm"
    with open(cfg.TEST_DIR / name, "rb") as f:
        response = client.post("/visualize?format=json", files={"file": (name, f, "application/dicom")})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    assert body["region"] == cfg.REGION_HIP
    assert body["side"] in ("left", "right")
    assert set(body["checkers"]) == {"hip_positioning", "hip_roi"}
    for result in body["checkers"].values():
        assert result["status"] in ("ok", "not_evaluated")


@requires_data
def test_visualize_region_matches_predict_region_for_same_file():
    """Регрессия задачи 9.6: `/visualize` раньше определял регион/сторону
    отдельной упрощённой эвристикой (`classify_region_from_pixels`/
    `hip_side` напрямую), а `/predict` — через `inference.describe_inputs`
    (та же эвристика + CNN-арбитраж этапа 1, если веса на месте). На кадре,
    где эвристика и CNN расходятся, эндпоинты могли бы показать разный
    регион для одного и того же файла. Теперь оба используют один и тот же
    `describe_inputs` — этот тест сравнивает их независимо от того, есть ли
    веса `region_cnn` на этой машине (см. `stage_9_decisions.md`, 9.6)."""
    from columba.submission import region_to_output

    name = "CR000000_ПОП.dcm"  # позвоночник
    with open(cfg.TEST_DIR / name, "rb") as f:
        predict_response = client.post("/predict?format=json", files={"files": (name, f, "application/dicom")})
    assert predict_response.status_code == 200
    predict_region = predict_response.json()["rows"][0]["anatomical_region"]

    with open(cfg.TEST_DIR / name, "rb") as f:
        visualize_response = client.post("/visualize?format=json", files={"file": (name, f, "application/dicom")})
    assert visualize_response.status_code == 200
    visualize_region = visualize_response.json()["region"]

    assert region_to_output(visualize_region) == predict_region
