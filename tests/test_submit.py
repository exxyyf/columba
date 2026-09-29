"""Тесты CLI сабмита (шаг 7 этапа 3, пункт 11 шага 9).

`main()` вызывается напрямую (без `subprocess`) — быстрее и достаточно, CLI
не делает ничего специфичного для процесса.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from columba import config as cfg
from columba.submission import validate_submission
from columba.submit import main

from conftest import requires_data


@requires_data
def test_cli_writes_a_valid_submission(tmp_path, capsys):
    output_csv = tmp_path / "out.csv"

    exit_code = main([str(cfg.TEST_DIR), str(output_csv)])

    assert exit_code == 0
    assert output_csv.exists()

    submission = pd.read_csv(output_csv, keep_default_na=False)
    validate_submission(submission, expected_rows=cfg.EXPECTED_TEST_FILES)

    out = capsys.readouterr().out
    assert f"Файлов обработано: {cfg.EXPECTED_TEST_FILES}" in out
    for violation_type in cfg.VIOLATION_TYPES:
        assert violation_type in out


def test_cli_reports_missing_directory_without_traceback(tmp_path, capsys):
    missing = tmp_path / "не_существует"

    exit_code = main([str(missing), str(tmp_path / "out.csv")])

    assert exit_code != 0
    captured = capsys.readouterr()
    assert "traceback" not in captured.err.lower()
    assert str(missing) in captured.err
    assert not (tmp_path / "out.csv").exists()


def test_cli_reports_empty_directory_without_traceback(tmp_path, capsys):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    exit_code = main([str(empty_dir), str(tmp_path / "out.csv")])

    assert exit_code != 0
    captured = capsys.readouterr()
    assert "traceback" not in captured.err.lower()
    assert not (tmp_path / "out.csv").exists()
