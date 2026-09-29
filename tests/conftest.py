"""Общие фикстуры: этап 0 прогоняется один раз на всю сессию тестов."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from columba import config as cfg  # noqa: E402
from columba.pipeline import run_stage0  # noqa: E402


def _data_available() -> bool:
    return cfg.STUDIES_DIR.exists() and cfg.MARKUP_XLSX.exists()


requires_data = pytest.mark.skipif(not _data_available(), reason="нет выгрузки data/НД_для_обучения")


@pytest.fixture(scope="session")
def stage0(tmp_path_factory):
    if not _data_available():
        pytest.skip("нет выгрузки data/НД_для_обучения")
    artifacts = tmp_path_factory.mktemp("artifacts")
    return run_stage0(artifacts_dir=artifacts, write_eda=False, verbose=False) | {"artifacts_dir": artifacts}


@pytest.fixture(scope="session")
def manifest(stage0):
    return stage0["manifest"]


@pytest.fixture(scope="session")
def targets(stage0):
    return stage0["targets"]


@pytest.fixture(scope="session")
def markup(stage0):
    return stage0["markup"]


@pytest.fixture(scope="session")
def split(stage0):
    return stage0["split"]
