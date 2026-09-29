"""Ревью этапа 1: воспроизводимость сравнения версий софта (шаг 2).

Числа из stage_1_decisions.md (52/192 vs 20/57) должны воспроизводиться из
артефактов, а статистические выводы отчёта — держаться: общая разница долей
не значима, hip_positioning на уровне исследований — пограничен, но выше 0,05.
"""

from __future__ import annotations

import pytest

from columba.compare_software_versions import (
    compare_versions,
    render_report,
    two_proportion_z_test,
    wilson_interval,
)

from conftest import requires_data


# --------------------------------------------------------------------------- #
# Статистические функции (юнит, без данных)
# --------------------------------------------------------------------------- #


def test_wilson_interval_contains_point_estimate_and_stays_in_unit_range():
    eps = 1e-9  # для k=0 и k=n граница совпадает с p̂ с точностью float
    for k, n in [(0, 10), (10, 10), (52, 192), (20, 57), (1, 3)]:
        lo, hi = wilson_interval(k, n)
        assert 0.0 <= lo <= k / n + eps, (k, n)
        assert k / n - eps <= hi <= 1.0, (k, n)


def test_wilson_interval_narrows_with_sample_size():
    lo_small, hi_small = wilson_interval(5, 20)
    lo_big, hi_big = wilson_interval(50, 200)
    assert hi_big - lo_big < hi_small - lo_small


def test_two_proportion_z_test_basics():
    z, p = two_proportion_z_test(10, 100, 10, 100)
    assert z == pytest.approx(0.0)
    assert p == pytest.approx(1.0)
    z_ab, p_ab = two_proportion_z_test(30, 100, 10, 100)
    z_ba, p_ba = two_proportion_z_test(10, 100, 30, 100)
    assert z_ab == pytest.approx(-z_ba)  # антисимметрия по порядку групп
    assert p_ab == pytest.approx(p_ba)
    assert p_ab < 0.05  # 30% vs 10% на n=100 — заведомо значимо


# --------------------------------------------------------------------------- #
# Воспроизводимость чисел decisions из артефактов
# --------------------------------------------------------------------------- #


@requires_data
def test_version_shares_match_decisions_numbers(manifest, targets):
    stats = compare_versions(manifest, targets)
    total = stats["total"]
    # Числа, зафиксированные в stage_1_decisions.md (шаг 2).
    assert (total.k1, total.n1) == (52, 192)
    assert (total.k2, total.n2) == (20, 57)
    assert stats["mixed_version_studies"] == 0
    assert stats["n_studies"] == {"18.41.005": 75, "18.50.082": 25}


@requires_data
def test_statistical_conclusions_hold(manifest, targets):
    """Выводы отчёта: общая разница не значима, hip_positioning выше 0,05
    на честной единице (исследование), несмотря на p<0,01 по изображениям."""
    stats = compare_versions(manifest, targets)
    _, p_total = stats["total"].test
    assert p_total > 0.05

    hip_image = next(c for c in stats["labels"] if c.name.startswith("hip_positioning"))
    _, p_image = hip_image.test
    _, p_study = stats["hip_study"].test
    assert p_image < 0.01  # сигнал уровня изображений реален...
    assert p_study > 0.05  # ...но на уровне исследований не значим
    assert p_study > p_image  # кластеризация ослабляет значимость


@requires_data
def test_report_renders_with_key_sections(manifest, targets):
    report = render_report(compare_versions(manifest, targets))
    for fragment in (
        "18.41.005",
        "18.50.082",
        "ДИ Уилсона",
        "уровне исследований",
        "не нужны",
        "Не путать с ДИ этапа 6",
    ):
        assert fragment in report, fragment
