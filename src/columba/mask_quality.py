"""Бонус 4 (этап 8, приоритет 4): оценка КАЧЕСТВА эвристики маскирования от
денситометра (`regions.detect_mask_steps`/`has_mask_rect`).

Не путать с этапом 4, шагом 8 («устойчивость к маскированию») — та проверка
убеждается, что ЧЕКЕРЫ не опираются на маскирование как ложный признак
(синтетическая перестановка не меняет их скор/флаг). Этот модуль проверяет
другое: насколько ТОЧЕН сам геометрический детектор `has_mask_rect` —
находит ли он реальные прямоугольные вырезы денситометра и не путает ли их
с чем-то ещё. Отдельной вожжённой bitmap-разметки денситометра (готовых
рамок ROI как ground truth) в данных нет — это зафиксировала ещё разведка
данных этапа 4 (`stages/stage_4.md`), поэтому единственный доступный
способ оценки — визуальная проверка отрисованных рамок на выборке реальных
кадров (не заменяет размеченный датасет, но честнее, чем некритично
доверять эвристике).

Запуск: `uv run python -m columba.mask_quality` ->
`artifacts/eda/mask_quality/*.png` (рамки найденных шагов поверх кадра) +
`artifacts/eda/mask_quality_report.md` (список кадров и найденные шаги,
координаты — для ручного разбора).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg
from .dicom_io import normalize, read_dicom
from .inventory import load_manifest
from .regions import detect_mask_steps

DEFAULT_OUTPUT_DIR = cfg.EDA_DIR / "mask_quality"
DEFAULT_REPORT = cfg.EDA_DIR / "mask_quality_report.md"


def render_mask_overlay(pixels: np.ndarray, steps: list[dict] | None = None):
    """Кадр + рамки найденных `detect_mask_steps` шагов -> (figure, steps).

    Рамка на каждый шаг: `edge` определяет, вдоль какой стороны кадра лежит
    вырез (`left`/`right` — вертикальная полоса шириной `depth`,
    `top`/`bottom` — горизонтальная полоса высотой `depth`), `start`/`end` —
    диапазон вдоль перпендикулярной оси, `inset`/`depth` — положение и
    толщина среза от края. Никакой новой логики — только геометрия,
    которую уже вычисляет `detect_mask_steps`.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if steps is None:
        steps = detect_mask_steps(pixels)

    height, width = pixels.shape[:2]
    fig, ax = plt.subplots(figsize=(5.0, 7.2))
    ax.imshow(pixels, cmap="gray", aspect=cfg.ANISOTROPY_Y_OVER_X)

    for step in steps:
        edge, start, end, inset, depth = step["edge"], step["start"], step["end"], step["inset"], step["depth"]
        if edge == "left":
            box = plt.Rectangle((0, start), inset + depth, end - start, fill=False, edgecolor="red", linewidth=1.4)
        elif edge == "right":
            box = plt.Rectangle(
                (width - inset - depth, start), inset + depth, end - start, fill=False, edgecolor="red", linewidth=1.4
            )
        elif edge == "top":
            box = plt.Rectangle((start, 0), end - start, inset + depth, fill=False, edgecolor="red", linewidth=1.4)
        else:  # bottom
            box = plt.Rectangle(
                (start, height - inset - depth), end - start, inset + depth, fill=False, edgecolor="red", linewidth=1.4
            )
        ax.add_patch(box)

    ax.axis("off")
    ax.set_title(f"{len(steps)} шаг(ов): " + ", ".join(s["edge"] for s in steps) if steps else "шагов не найдено", fontsize=8)
    fig.tight_layout()
    return fig, steps


def sample_for_qa(
    manifest: pd.DataFrame | None = None,
    n_positive: int = 8,
    n_negative: int = 4,
    seed: int = cfg.SEED,
) -> pd.DataFrame:
    """Стратифицированная выборка представителей dedup-групп для ручного
    просмотра: часть с `has_mask_rect=True` (проверить, что рамка реально
    легла на вырез, а не на что-то другое), часть с `False` (проверить, что
    эвристика не пропустила реальный вырез — ложный негатив)."""
    if manifest is None:
        manifest = load_manifest()
    representatives = manifest[manifest["is_group_representative"].fillna(False) & manifest["read_status"].eq("Success")]
    rng = np.random.default_rng(seed)

    positives = representatives[representatives["has_mask_rect"].fillna(False)]
    negatives = representatives[~representatives["has_mask_rect"].fillna(False)]
    pos_sample = positives.sample(n=min(n_positive, len(positives)), random_state=rng.integers(2**31))
    neg_sample = negatives.sample(n=min(n_negative, len(negatives)), random_state=rng.integers(2**31))
    return pd.concat([pos_sample, neg_sample]).reset_index(drop=True)


def render_qa_sample(
    manifest: pd.DataFrame | None = None,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    report_path: Path | str = DEFAULT_REPORT,
    n_positive: int = 8,
    n_negative: int = 4,
) -> Path:
    """Отрисовать выборку -> PNG на кадр + markdown-отчёт со ссылками и
    геометрией найденных шагов (для ручного разбора)."""
    import matplotlib.pyplot as plt

    if manifest is None:
        manifest = load_manifest()
    sample = sample_for_qa(manifest, n_positive=n_positive, n_negative=n_negative)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Бонус 4 — визуальная QA эвристики `has_mask_rect`",
        "",
        "Красные рамки — шаги, найденные `regions.detect_mask_steps`. Готовой",
        "bitmap-разметки денситометра как ground truth в данных нет",
        "(`stages/stage_4.md`) — оценка только визуальная, по выборке ниже.",
        "",
        "| файл | dedup_group_id | has_mask_rect | найдено шагов | картинка |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in sample.itertuples():
        result = read_dicom(row.abs_path)
        if not result.ok:
            continue
        pixels = normalize(result.pixels, result.tags)
        fig, steps = render_mask_overlay(pixels)
        out_path = output_dir / f"{row.dedup_group_id}.png"
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        lines.append(
            f"| {Path(row.abs_path).name} | {row.dedup_group_id} | {bool(row.has_mask_rect)} | "
            f"{len(steps)} | `{out_path.name}` |"
        )

    report_path = Path(report_path)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def main() -> int:
    path = render_qa_sample()
    print(f"отчёт: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
