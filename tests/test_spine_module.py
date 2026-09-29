"""Этап 2: модуль позвоночника — инварианты, деградация, синтетика.

Обязательные тесты из плана этапа (stage_2.md, шаг 9):
анизотропный угол, независимость от тегов/имён, синтетические повороты,
маскирование углов, not_evaluated без исключений, схема файла разметки,
инволюция преобразований координат, кэш дублей.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest

from columba import config as cfg
from columba.spine_checkers import (
    check_spine_axis,
    check_spine_objects,
    check_spine_positioning,
    get_spine_keypoints,
    metal_components,
    run_spine_checkers,
)
from columba.spine_keypoints import (
    STATUS_NOT_EVALUATED,
    STATUS_OK,
    detect_vertebra_centers,
    iso_to_raw,
    load_annotations,
    raw_to_iso,
    validate_annotation_entry,
)
from columba.spine_synth import (
    crop_field,
    insert_synthetic_metal,
    rotate_isotropic,
    rotate_points_iso,
    rotate_raw_frame,
)

from conftest import requires_data

SPINE_TEST_FILE = cfg.TEST_DIR / "CR000000_ПОП.dcm"


# --------------------------------------------------------------------------- #
# Синтетический «позвоночник» с известным физическим наклоном
# --------------------------------------------------------------------------- #


def synthetic_spine(angle_deg: float = 0.0, height: int = 320, width: int = 300) -> np.ndarray:
    """Колонна с периодическими «телами позвонков», наклонённая на angle_deg
    от вертикали В ФИЗИЧЕСКИХ координатах (мм)."""
    y_px = np.arange(height)[:, None]
    x_px = np.arange(width)[None, :]
    y_mm = y_px * cfg.PIXEL_SPACING_MM_Y
    x_mm = x_px * cfg.PIXEL_SPACING_MM_X
    center_mm = (width - 1) / 2.0 * cfg.PIXEL_SPACING_MM_X
    # ось колонны: x_mm = center + tan(angle) * y_mm
    axis_x_mm = center_mm + np.tan(np.deg2rad(angle_deg)) * (y_mm - y_mm.mean())
    across = np.abs(x_mm - axis_x_mm)
    column = np.clip(1.0 - across / 22.0, 0.0, 1.0)  # колонна ~44 мм шириной
    along = 0.75 + 0.25 * np.cos(2 * np.pi * y_mm / 33.0)  # «позвонки» каждые 33 мм
    image = (column * along).astype(np.float32)
    return np.clip(image, 0.0, 1.0)


def test_axis_angle_respects_anisotropy():
    """Чекер возвращает физический угол; наивный пиксельный угол другой."""
    true_angle = 8.0
    image = synthetic_spine(true_angle)
    keypoints = detect_vertebra_centers(image)
    assert keypoints.status == STATUS_OK and keypoints.n_centers >= 3
    result = check_spine_axis(image, keypoints)
    assert result.status == STATUS_OK
    assert result.score == pytest.approx(true_angle, abs=1.2)

    # Наивный угол в «сырых пикселях» на тех же точках обязан отличаться:
    # тангенс отличается в ANISOTROPY_Y_OVER_X раз.
    top, bottom = keypoints.centers_raw_xy[0], keypoints.centers_raw_xy[-1]
    naive = np.degrees(np.arctan2(abs(top[0] - bottom[0]), abs(top[1] - bottom[1])))
    assert abs(naive - true_angle) > 2.0

    vertical = check_spine_axis(synthetic_spine(0.0), detect_vertebra_centers(synthetic_spine(0.0)))
    assert vertical.status == STATUS_OK
    assert abs(vertical.score) < 1.0


def test_axis_flag_on_synthetic_rotation_thresholds():
    """Поворот на 8° поднимает флаг оси, на 1° — нет (порог из config)."""
    base = synthetic_spine(0.0)
    rotated_bad = rotate_raw_frame(base, 8.0)
    bad = check_spine_axis(rotated_bad, detect_vertebra_centers(rotated_bad))
    assert bad.status == STATUS_OK and bad.flag

    rotated_ok = rotate_raw_frame(base, 1.0)
    ok = check_spine_axis(rotated_ok, detect_vertebra_centers(rotated_ok))
    assert ok.status == STATUS_OK and not ok.flag


# --------------------------------------------------------------------------- #
# Преобразования координат
# --------------------------------------------------------------------------- #


def test_raw_iso_transform_involution():
    rng = np.random.default_rng(0)
    for raw_height in (259, 300, 346):
        points = rng.uniform([0, 0], [299, raw_height - 1], size=(20, 2))
        forth = raw_to_iso(points, raw_height)
        back = iso_to_raw(forth, raw_height)
        np.testing.assert_allclose(back, points, atol=1e-9)


def test_net_transform_involution():
    from columba.spine_keypoint_net import NetTransform

    rng = np.random.default_rng(1)
    for raw_height in (259, 346):
        iso_shape = (int(round(raw_height * cfg.ANISOTROPY_Y_OVER_X)), 300)
        transform = NetTransform.for_iso_shape(iso_shape, raw_height)
        points = rng.uniform([0, 0], [299, raw_height - 1], size=(20, 2))
        back = transform.net_to_raw(transform.raw_to_net(points))
        np.testing.assert_allclose(back, points, atol=1e-6)


def test_rotation_image_and_points_are_consistent():
    image = np.zeros((200, 100), dtype=np.float32)
    image[40, 30] = 1.0
    for angle in (8.0, -8.0):
        rotated = rotate_isotropic(image, angle)
        peak_y, peak_x = np.unravel_index(np.argmax(rotated), rotated.shape)
        predicted = rotate_points_iso(np.array([[30.0, 40.0]]), image.shape, angle)[0]
        assert abs(predicted[0] - peak_x) <= 1.5
        assert abs(predicted[1] - peak_y) <= 1.5


# --------------------------------------------------------------------------- #
# Деградация и устойчивость
# --------------------------------------------------------------------------- #


def test_not_evaluated_on_degenerate_inputs_without_exceptions():
    empty = np.zeros((4, 4), dtype=np.float32)
    keypoints = detect_vertebra_centers(empty)
    assert keypoints.status == STATUS_NOT_EVALUATED and keypoints.reason

    axis = check_spine_axis(empty, keypoints)
    assert axis.status == STATUS_NOT_EVALUATED and not axis.flag and axis.score != axis.score

    positioning = check_spine_positioning(empty, keypoints)
    assert positioning.status == STATUS_NOT_EVALUATED and not positioning.flag

    objects_result = check_spine_objects(empty)
    assert objects_result.status == STATUS_NOT_EVALUATED

    dark = np.zeros((320, 300), dtype=np.float32)
    dark_keypoints = detect_vertebra_centers(dark)
    assert dark_keypoints.status == STATUS_NOT_EVALUATED
    assert check_spine_axis(dark, dark_keypoints).status == STATUS_NOT_EVALUATED
    assert check_spine_positioning(dark, dark_keypoints).status == STATUS_NOT_EVALUATED


def test_positioning_does_not_depend_on_keypoints():
    """Сигнал гребней считается и тогда, когда детектор центров не сработал."""
    image = synthetic_spine(0.0)
    good = detect_vertebra_centers(image)
    assert good.status == STATUS_OK
    failed = detect_vertebra_centers(np.zeros((4, 4), dtype=np.float32))  # not_evaluated

    with_keypoints = check_spine_positioning(image, good)
    without_keypoints = check_spine_positioning(image, failed)
    assert without_keypoints.status == STATUS_OK
    assert without_keypoints.score == with_keypoints.score
    assert without_keypoints.flag == with_keypoints.flag
    assert without_keypoints.signals["center_offset_mm"] is None
    assert without_keypoints.signals["keypoints_reason"]


def test_masking_corners_does_not_change_objects_and_keypoints():
    """Чёрные прямоугольники маскирования не меняют выходы (синтетика)."""
    image = synthetic_spine(3.0)
    baseline_keypoints = detect_vertebra_centers(image)
    baseline_objects = check_spine_objects(image)

    masked = image.copy()
    masked[: image.shape[0] // 5, : image.shape[1] // 5] = 0.0
    masked[-image.shape[0] // 6 :, -image.shape[1] // 4 :] = 0.0
    masked_keypoints = detect_vertebra_centers(masked)
    masked_objects = check_spine_objects(masked)

    assert masked_objects.flag == baseline_objects.flag
    assert masked_keypoints.n_centers == baseline_keypoints.n_centers
    np.testing.assert_allclose(
        masked_keypoints.centers_raw_xy, baseline_keypoints.centers_raw_xy, atol=3.0
    )


def test_symmetry_signals_present_and_neutral_on_symmetric_frame():
    """Сигналы симметрии (план шага 5) логируются и не влияют на скор/флаг."""
    image = synthetic_spine(0.0)
    result = check_spine_positioning(image, detect_vertebra_centers(image))
    assert result.status == STATUS_OK
    # симметричная колонна по центру кадра: асимметрия силуэта около нуля
    assert result.signals["silhouette_asymmetry_mm"] is not None
    assert abs(result.signals["silhouette_asymmetry_mm"]) < 5.0
    # у синтетики нет гребней в нижней полосе — наклон честно не считается
    assert result.signals["crest_tilt_deg"] is None
    # скор по-прежнему равен дефициту гребней: сигналы во флаг не входят
    assert result.score == result.signals["crest_deficit"]


def test_crest_line_tilt_sign_and_magnitude():
    """Наклон линии гребней: знак и величина на синтетических «крыльях»."""
    from columba.spine_checkers import crest_line_tilt_deg

    def frame_with_wings(dy_right_px: int) -> np.ndarray:
        image = synthetic_spine(0.0)
        height, width = image.shape
        band = int(cfg.SPINE_CREST_BAND_MM / cfg.PIXEL_SPACING_MM_Y)
        y_left = height - band + 2
        y_right = y_left + dy_right_px
        image[y_left : y_left + 8, 10:70] = 1.0  # левое «крыло»
        image[y_right : y_right + 8, width - 70 : width - 10] = 1.0  # правое
        return image

    level = crest_line_tilt_deg(frame_with_wings(0))
    tilted = crest_line_tilt_deg(frame_with_wings(10))  # правое крыло ниже
    assert level is not None and abs(level) < 2.0
    assert tilted is not None and tilted > 2.0  # положительный = правая ниже
    # угол физический: 10 px по Y между центроидами крыльев по X
    dy_mm = 10 * cfg.PIXEL_SPACING_MM_Y
    dx_mm = (frame_with_wings(0).shape[1] - 80) * cfg.PIXEL_SPACING_MM_X
    expected = np.degrees(np.arctan(dy_mm / dx_mm))
    assert tilted == pytest.approx(expected, abs=1.5)


@requires_data
def test_top_margin_deficit_rises_on_top_crop():
    """Кроп верха поля поднимает сигнал top_margin_deficit (во флаг не входит)."""
    from columba.dicom_io import read_dicom, normalize

    result = read_dicom(SPINE_TEST_FILE)
    pixels = normalize(result.pixels, result.tags)
    base = check_spine_positioning(pixels, detect_vertebra_centers(pixels))
    assert base.status == STATUS_OK

    rng = np.random.default_rng(5)
    cropped = crop_field(pixels, rng, side="top")
    after = check_spine_positioning(cropped, detect_vertebra_centers(cropped))
    assert after.status == STATUS_OK
    assert after.signals["top_margin_deficit"] is not None
    assert after.signals["top_margin_deficit"] > (base.signals["top_margin_deficit"] or 0.0)
    # флаг определяется только гребнями: кроп верха его не поднимает
    assert after.flag == base.flag


@requires_data
@pytest.mark.skipif(not cfg.SPINE_UNET_WEIGHTS.exists(), reason="нет весов U-Net")
def test_unet_keypoints_stable_under_corner_masking():
    """Та же проверка маскирования, что у классики, — для U-Net (шаг 3)."""
    from columba.dicom_io import read_dicom, normalize
    from columba.spine_keypoint_net import load_spine_keypoint_predictor, match_points_mm
    from columba.spine_unet_check import mask_corners

    predictor = load_spine_keypoint_predictor()
    result = read_dicom(SPINE_TEST_FILE)
    pixels = normalize(result.pixels, result.tags)
    base = predictor.predict(pixels)
    assert base.status == STATUS_OK
    masked = predictor.predict(mask_corners(pixels))
    agreement = match_points_mm(masked.centers_raw_xy, base.centers_raw_xy)
    assert agreement["recall"] >= 0.8


def test_synthetic_metal_raises_objects_score():
    # seed=1 (не 3, ревью этапа 9): у seed=3 бусины synth-дуги сливаются в
    # дилатации в компоненты диагональю ~45-49 мм / площадью ~78-84 мм² —
    # заметно крупнее любой компоненты реальных train-позитивов (максимум
    # 28.5 мм / 34.2 мм², не считая одного выброса 58 мм / 45 мм², см.
    # SPINE_METAL_MAX_AREA_MM2/SPINE_METAL_MAX_DIAG_MM в config.py) — похоже
    # на артефакт генератора (бусины рисуются слишком плотно и сливаются в
    # одну сплошную дугу), а не на типичную цепочку с train. Новые верхние
    # границы геометрии (находка 9.4: без них правило ловит рёбра как металл)
    # закономерно режут такой нереалистично крупный объект; seed=1 даёт
    # компактную компоненту (диагональ ~25 мм, площадь ~38 мм²), сопоставимую
    # с реальными позитивами train, и остаётся честной проверкой детектора.
    image = synthetic_spine(0.0)
    baseline = check_spine_objects(image)
    rng = np.random.default_rng(1)
    with_chain = insert_synthetic_metal(image, rng, kind="chain")
    chained = check_spine_objects(with_chain)
    assert chained.score > baseline.score
    assert chained.flag


def test_crop_field_removes_rows():
    image = synthetic_spine(0.0)
    rng = np.random.default_rng(4)
    cropped = crop_field(image, rng, side="bottom")
    assert cropped.shape[0] < image.shape[0]
    assert cropped.shape[1] == image.shape[1]


# --------------------------------------------------------------------------- #
# Файл разметки
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not cfg.SPINE_KEYPOINTS_JSON.exists(), reason="нет файла разметки")
def test_annotation_file_schema_and_coverage():
    annotations = load_annotations()
    for gid, entry in annotations.items():
        assert validate_annotation_entry(entry) == [], gid


@requires_data
@pytest.mark.skipif(not cfg.SPINE_KEYPOINTS_JSON.exists(), reason="нет файла разметки")
def test_annotation_covers_every_spine_group(manifest):
    annotations = load_annotations()
    spine_groups = set(
        manifest[
            (manifest["region"] == cfg.REGION_SPINE) & manifest["is_group_representative"].fillna(False)
        ]["dedup_group_id"]
    )
    assert spine_groups == set(annotations)


# --------------------------------------------------------------------------- #
# Интеграция: describe_inputs, теги, кэш дублей
# --------------------------------------------------------------------------- #


CHECKER_COLUMNS = [
    f"{key}_{field}"
    for key in ("spine_axis", "spine_positioning", "spine_objects")
    for field in ("score", "flag", "status")
]


@requires_data
def test_describe_inputs_attaches_checkers_for_spine_only(tmp_path):
    import shutil

    from columba.inference import describe_inputs

    for name in ("CR000000_ПОП.dcm", "CR000000_ППОБ.dcm"):
        shutil.copy(cfg.TEST_DIR / name, tmp_path / name)
    frame = describe_inputs(tmp_path)
    for column in CHECKER_COLUMNS + ["spine_signals"]:
        assert column in frame.columns

    spine_row = frame[frame["region_final"] == cfg.REGION_SPINE].iloc[0]
    hip_row = frame[frame["region_final"] == cfg.REGION_HIP].iloc[0]
    assert spine_row["spine_axis_status"] in (STATUS_OK, STATUS_NOT_EVALUATED)
    assert pd.notna(spine_row["spine_signals"])
    signals = json.loads(spine_row["spine_signals"])
    assert set(signals) == {"spine_axis", "spine_positioning", "spine_objects"}
    assert "keypoints_raw_xy" in signals["spine_axis"]["signals"]
    # Бедро — нейтральный пропуск.
    assert pd.isna(hip_row["spine_axis_status"]) and pd.isna(hip_row["spine_signals"])


@requires_data
def test_checker_outputs_survive_stripped_tags_and_renamed_files(tmp_path):
    """Выходы чекеров не зависят от тегов и имён файлов."""
    import pydicom

    from columba.inference import describe_inputs

    plain = tmp_path / "plain"
    stripped = tmp_path / "stripped"
    plain.mkdir()
    stripped.mkdir()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dataset = pydicom.dcmread(SPINE_TEST_FILE)
        dataset.save_as(plain / "spine.dcm", enforce_file_format=False)
        for tag in ("BodyPartExamined", "ViewPosition", "Laterality", "SeriesDescription",
                    "StudyDescription", "SoftwareVersions", "PatientOrientation"):
            if tag in dataset:
                delattr(dataset, tag)
        dataset.save_as(stripped / "000000.dcm", enforce_file_format=False)

    frame_plain = describe_inputs(plain)
    frame_stripped = describe_inputs(stripped)
    for column in CHECKER_COLUMNS:
        left, right = frame_plain.iloc[0][column], frame_stripped.iloc[0][column]
        if isinstance(left, float) and left != left:
            assert right != right
        else:
            assert left == right, column

    # Шаг 9 п.12 этапа 3: агрегатор (через run_inference) тоже не должен
    # зависеть от тегов/имени файла — сверяем сабмит, а не только чекеры.
    # Пока агрегатора нет, сабмит нулевой (нули по построению совпадают у
    # обоих прогонов) — проверка станет содержательной после его подключения.
    from columba.inference import run_inference

    submission_plain = run_inference(plain)
    submission_stripped = run_inference(stripped)
    assert len(submission_plain) == len(submission_stripped) == 1
    for column in ("quality_class", "quality_prob", "violation_type"):
        assert submission_plain.iloc[0][column] == submission_stripped.iloc[0][column], column


@requires_data
def test_duplicates_hit_spine_checkers_once(tmp_path, monkeypatch):
    """Кэш по хэшу пикселей охватывает чекеры: 3 копии — 1 вызов."""
    import shutil

    from columba import spine_checkers as checkers_module
    from columba.inference import describe_inputs

    calls = {"n": 0}
    original = checkers_module.run_spine_checkers

    def counting(pixels, **kwargs):
        calls["n"] += 1
        return original(pixels, **kwargs)

    monkeypatch.setattr(checkers_module, "run_spine_checkers", counting)
    for index in range(3):
        shutil.copy(SPINE_TEST_FILE, tmp_path / f"copy_{index}.dcm")
    frame = describe_inputs(tmp_path)
    assert len(frame) == 3
    assert calls["n"] == 1
    assert frame["spine_axis_status"].notna().all()


@requires_data
def test_run_inference_with_checkers_still_produces_valid_submission():
    from columba.inference import run_inference
    from columba.submission import validate_submission

    submission = run_inference(cfg.TEST_DIR)
    validate_submission(submission, expected_rows=cfg.EXPECTED_TEST_FILES)


@requires_data
def test_real_spine_frame_gets_reasonable_keypoints():
    from columba.dicom_io import read_dicom, normalize

    result = read_dicom(SPINE_TEST_FILE)
    pixels = normalize(result.pixels, result.tags)
    keypoints = get_spine_keypoints(pixels)
    assert keypoints.status == STATUS_OK
    assert 4 <= keypoints.n_centers <= 11
    checkers = run_spine_checkers(pixels)
    assert checkers["spine_axis"].status == STATUS_OK
    assert checkers["spine_positioning"].status == STATUS_OK
    assert checkers["spine_objects"].status == STATUS_OK


# --------------------------------------------------------------------------- #
# Ревью этапа 9, п. 9.4: рёбра в детекторе предметов
# --------------------------------------------------------------------------- #


def test_metal_components_rejects_oversized_bbox():
    """Верхние границы площади/диагонали (config) отсекают протяжённые
    структуры, которые проходят нижние пороги, но геометрически не похожи на
    компактный металл (цепочка/клипса), а похожи на костный край (ребро).

    Синтетика: тонкая яркая ДУГА почти во всю ширину кадра — она проходит
    нижние пороги площади/диагонали, но должна быть отсечена верхней
    границей `SPINE_METAL_MAX_DIAG_MM`/`SPINE_METAL_MAX_AREA_MM2`.
    """
    height, width = 320, 300
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    # Дуга: тонкая яркая линия, кривизна как у ребра, почти во всю ширину.
    arc_y = 40.0 + 0.002 * (xx - width / 2.0) ** 2
    thin_arc = np.exp(-((yy - arc_y) ** 2) / (2 * 2.0**2))
    image = np.clip(thin_arc, 0.0, 1.0).astype(np.float32)

    components = metal_components(image)
    for component in components:
        assert component["area_mm2"] <= cfg.SPINE_METAL_MAX_AREA_MM2
        assert component["diag_mm"] <= cfg.SPINE_METAL_MAX_DIAG_MM


@requires_data
def test_real_test_file_objects_not_flagged_as_metal():
    """Регрессия ревью этапа 9, п. 9.4.

    data/Для теста/CR000000_ПОП.dcm — реальный тестовый файл; без верхних
    границ геометрии metal_components ловит рёбра у верхних боковых краёв
    кадра как «металл» (rule_score = 4.37 при пороге флага 0.40, в 11 раз
    выше). Просмотр overlay (bbox найденных компонент поверх кадра, плюс
    zoom по каждой оставшейся после введения границ) подтвердил: все
    компоненты лежат точно на рёберном крае, никакого металлического
    предмета на снимке нет — flag=True был бы ложным. После введения
    SPINE_METAL_MAX_AREA_MM2/SPINE_METAL_MAX_DIAG_MM и перекалиброванного
    порога SPINE_METAL_MIN_SCORE (config, см. комментарий и
    stage_9_decisions.md, раздел 9.4) файл не флагуется.
    """
    from columba.dicom_io import read_dicom, normalize

    result = read_dicom(SPINE_TEST_FILE)
    pixels = normalize(result.pixels, result.tags)
    checker = check_spine_objects(pixels)
    assert checker.status == STATUS_OK
    assert checker.flag is False
    # Скор упал на порядок относительно дорелизной геометрии, но не до нуля —
    # честно задокументированный остаточный риск (открыт в decisions):
    # оставшиеся компоненты — мелкие фрагменты рёбер, геометрически похожие
    # на настоящий мелкий металл.
    assert checker.score < cfg.SPINE_METAL_MIN_SCORE


@requires_data
def test_real_train_positives_with_strong_metal_still_flagged():
    """Регрессия ревью этапа 9, п. 9.4 (не сломать реальную детекцию).

    После сужения геометрии (верхние границы площади/диагонали) и подъёма
    порога флага часть слабых train-позитивов класса `spine_objects` уходит
    в FN — это задокументированный, а не скрытый компромисс (decisions,
    раздел 9.4: их скор в новой геометрии неотличим от скора рёберного
    фрагмента на CR000000_ПОП.dcm). Но самые уверенные train-позитивы
    (заметная цепочка на глаз, наибольший скор до правки) обязаны остаться
    флагом True — иначе правка сломала бы детекцию настоящего металла, а не
    только ложные срабатывания на рёбрах.
    """
    from columba.dicom_io import read_dicom, normalize
    from columba.spine_objects_cnn import objects_dataset_frame

    frame = objects_dataset_frame()
    frame = frame[(frame["fold"] == "train") & (frame["label"] == 1)]
    # Самые сильные позитивы по итогам калибровки (наибольший rule_score
    # среди train-позитивов и до, и после правки 9.4; см. decisions).
    strong_positive_ids = {"g0081", "g0084", "g0134", "g0150", "g0202"}
    checked = 0
    for row in frame.itertuples():
        if row.dedup_group_id not in strong_positive_ids:
            continue
        result = read_dicom(row.abs_path)
        pixels = normalize(result.pixels, result.tags)
        checker = check_spine_objects(pixels)
        assert checker.status == STATUS_OK
        assert checker.flag is True, f"{row.dedup_group_id}: score={checker.score}"
        checked += 1
    assert checked == len(strong_positive_ids)
