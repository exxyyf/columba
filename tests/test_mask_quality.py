"""Бонус 4: визуальная QA эвристики `has_mask_rect` (не сама эвристика —
она уже покрыта косвенно этапом 0; здесь только инструмент отрисовки/сэмплинга)."""

from __future__ import annotations

import numpy as np

from columba.mask_quality import render_mask_overlay, sample_for_qa
from columba.regions import detect_mask_steps

from conftest import requires_data


def _frame_with_left_cutout(height=200, width=150, cut_width=30, cut_start=60, cut_end=140):
    """Синтетический кадр: ткань во весь кадр, кроме прямоугольного чёрного
    выреза слева на ПОДМНОЖЕСТВЕ строк (как реальное маскирование —
    отступ от края одинаков только для части кадра, не для всех строк
    сразу, иначе `detect_mask_steps` не увидит отклонения от медианы)."""
    frame = np.full((height, width), 0.5, dtype=np.float32)
    frame[cut_start:cut_end, :cut_width] = 0.0
    return frame


def test_render_mask_overlay_draws_box_for_each_detected_step():
    frame = _frame_with_left_cutout()
    steps = detect_mask_steps(frame)
    assert len(steps) >= 1
    assert steps[0]["edge"] == "left"

    fig, returned_steps = render_mask_overlay(frame)
    assert returned_steps == steps
    # По одному Rectangle-патчу на найденный шаг.
    ax = fig.axes[0]
    from matplotlib.patches import Rectangle

    boxes = [p for p in ax.patches if isinstance(p, Rectangle)]
    assert len(boxes) == len(steps)
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_render_mask_overlay_handles_no_steps_found():
    frame = np.full((100, 100), 0.5, dtype=np.float32)  # без выреза
    fig, steps = render_mask_overlay(frame)
    assert steps == []
    import matplotlib.pyplot as plt

    plt.close(fig)


@requires_data
def test_sample_for_qa_stratifies_by_has_mask_rect(manifest):
    sample = sample_for_qa(manifest, n_positive=3, n_negative=2)
    assert len(sample) == 5
    assert sample["has_mask_rect"].fillna(False).sum() == 3
    assert (~sample["has_mask_rect"].fillna(False)).sum() == 2


@requires_data
def test_render_qa_sample_creates_report_and_images(manifest, tmp_path):
    from columba.mask_quality import render_qa_sample

    output_dir = tmp_path / "mask_quality"
    report_path = tmp_path / "mask_quality_report.md"
    result = render_qa_sample(
        manifest, output_dir=output_dir, report_path=report_path, n_positive=2, n_negative=2
    )
    assert result == report_path
    assert report_path.exists()
    pngs = list(output_dir.glob("*.png"))
    assert len(pngs) == 4
    for png in pngs:
        assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
