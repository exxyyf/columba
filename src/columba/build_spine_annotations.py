"""Сборка разметки центров тел позвонков (шаг 2 этапа 2).

Полуавтоматический процесс, воспроизводимый одной командой:

    uv run python -m columba.build_spine_annotations

1. Классический детектор (`spine_keypoints.detect_vertebra_centers`) даёт
   кандидатов на всех уникальных снимках ПОП.
2. Все 99 снимков просмотрены глазами с оверлеями (этап 2, шаг 2); найденные
   ошибки исправляются точечными правками из `MANUAL_CORRECTIONS` — они
   зафиксированы здесь кодом, чтобы разметка не зависела от ручных действий.
3. Итог пишется в `artifacts/annotations/spine_keypoints.json`.

Атрибуты видимости границ поля (`top_ribs_visible`, `bottom_crest_visible`)
проставлены глазами только для просмотренного подмножества (позитивы
`spine_positioning` + контрольные негативы) — они нужны для калибровки
автоматических сигналов чекера укладки, не как таргет модели.
"""

from __future__ import annotations

from . import config as cfg
from .dicom_io import normalize, read_dicom_strict
from .inventory import load_manifest
from .spine_keypoints import (
    ANNOTATION_SOURCE_AUTO,
    ANNOTATION_SOURCE_REVIEWED,
    detect_vertebra_centers,
    save_annotations,
)

# Точечные правки по итогам просмотра оверлеев (координаты исходных пикселей).
# drop_first / drop_last: точки, зацепившиеся за металл или крестец.
# move: {индекс_после_drop: (x, y)} — смещение точки на середину тела.
MANUAL_CORRECTIONS: dict[str, dict] = {
    # Верхние две точки лежат на металлической клипсе (см. spine_artifacts).
    "g0081": {"drop_first": 2},
    # Последние две точки уехали на крестец; L5 сдвинута на середину тела.
    "g0020": {"drop_last": 2, "move": {4: (157.0, 240.0)}},
    # Вторая точка лежала на поперечном отростке справа от тела.
    "g0241": {"move": {1: (136.0, 83.0)}},
}

# Глазная разметка видимости гребней подвздошных костей у нижнего края кадра
# (подмножество: все позитивы spine_positioning + контрольные негативы).
BOTTOM_CREST_REVIEWED: dict[str, bool] = {
    "g0024": False,
    "g0027": True,
    "g0052": False,
    "g0118": False,
    "g0203": False,
    "g0212": True,
    # негативы для контраста
    "g0013": True,
    "g0014": True,
    "g0016": True,
    "g0020": True,
    "g0032": True,
    "g0040": True,
    "g0051": True,
    "g0062": True,
    "g0108": True,
    "g0123": True,
    "g0154": True,
    "g0160": True,
}


def apply_corrections(gid: str, centers: list[list[float]]) -> tuple[list[list[float]], bool]:
    """Применить правки просмотра; возвращает (центры, была_ли_правка)."""
    correction = MANUAL_CORRECTIONS.get(gid)
    if not correction:
        return centers, False
    result = list(centers)
    if correction.get("drop_first"):
        result = result[correction["drop_first"] :]
    if correction.get("drop_last"):
        result = result[: -correction["drop_last"]]
    for index, (x, y) in correction.get("move", {}).items():
        result[index] = [float(x), float(y)]
    return result, True


def build_annotations(verbose: bool = True) -> dict:
    manifest = load_manifest()
    representatives = manifest[
        manifest["is_group_representative"].fillna(False)
        & (manifest["region"] == cfg.REGION_SPINE)
    ].sort_values("dedup_group_id")

    annotations: dict[str, dict] = {}
    for row in representatives.itertuples():
        result = read_dicom_strict(row.abs_path)
        pixels = normalize(result.pixels, result.tags)
        keypoints = detect_vertebra_centers(pixels)
        centers = [[round(float(x), 1), round(float(y), 1)] for x, y in keypoints.centers_raw_xy]
        centers, corrected = apply_corrections(row.dedup_group_id, centers)
        annotations[row.dedup_group_id] = {
            "centers_xy_raw": centers,
            "raw_shape": [int(pixels.shape[0]), int(pixels.shape[1])],
            "top_ribs_visible": None,
            "bottom_crest_visible": BOTTOM_CREST_REVIEWED.get(row.dedup_group_id),
            "source": ANNOTATION_SOURCE_REVIEWED if corrected else ANNOTATION_SOURCE_AUTO,
        }
    if verbose:
        n_corrected = sum(1 for v in annotations.values() if v["source"] == ANNOTATION_SOURCE_REVIEWED)
        print(f"разметка: {len(annotations)} снимков, ручных правок {n_corrected}", flush=True)
    return annotations


def main() -> None:
    annotations = build_annotations()
    path = save_annotations(annotations)
    print(f"записано: {path}", flush=True)


if __name__ == "__main__":
    main()
