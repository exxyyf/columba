"""Замер времени полного инференса на батче.

    uv run python -m columba.benchmark <input_dir> [--output путь.json]

Прогоняет `inference.run_inference` (регион/сторона + чекеры позвоночника и
бедра, весь AUTO-путь — как в `service.py`) по каталогу, пишет отчёт в
`artifacts/benchmark_<дата>.json` по умолчанию: число файлов/исследований,
общее время, с/файл, с/исследование, устройство, какие веса из реестра
`columba.weights` присутствовали, версии python/torch.

Не импортирует `columba.weights` для сетевых операций — только для чтения
локального реестра (`verify_all`), сеть не трогается.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

from . import config as cfg
from .inference import list_input_files, run_inference
from .inventory import DICOM_SUFFIX


def _study_count(input_dir: Path, files: list[Path]) -> int:
    keys = set()
    for path in files:
        relative = path.relative_to(input_dir)
        keys.add(relative.parts[0] if len(relative.parts) > 1 else ".")
    return len(keys)


def _device_report() -> str:
    try:
        from .region_cnn import default_device

        return str(default_device())
    except Exception as exc:  # noqa: BLE001 — отчёт не должен падать из-за диагностики
        return f"unknown ({exc})"


def _torch_version() -> str | None:
    try:
        import torch

        return torch.__version__
    except ImportError:
        return None


def _weights_presence() -> dict[str, bool]:
    try:
        from . import weights as w

        return {r.entry.name: r.status == "ok" for r in w.verify_all()}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def run_benchmark(input_dir: Path | str) -> dict:
    input_dir = Path(input_dir)
    files = list_input_files(input_dir)
    if not files:
        raise ValueError(f"во входном каталоге нет файлов {DICOM_SUFFIX}: {input_dir}")

    n_studies = _study_count(input_dir, files)

    start = time.perf_counter()
    submission = run_inference(input_dir)
    elapsed = time.perf_counter() - start

    n_files = len(submission)
    return {
        "input_dir": str(input_dir),
        "n_files": n_files,
        "n_studies": n_studies,
        "elapsed_seconds": elapsed,
        "seconds_per_file": elapsed / n_files if n_files else None,
        "seconds_per_study": elapsed / n_studies if n_studies else None,
        "device": _device_report(),
        "weights_present": _weights_presence(),
        "python_version": sys.version.split()[0],
        "torch_version": _torch_version(),
        "platform": platform.platform(),
    }


def _default_output_path() -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return cfg.ARTIFACTS_DIR / f"benchmark_{stamp}.json"


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    parser = argparse.ArgumentParser(prog="columba.benchmark", description=__doc__)
    parser.add_argument("input_dir", type=Path, help="Каталог с .dcm файлами (плоский или вложенный)")
    parser.add_argument("--output", type=Path, default=None, help="Куда сохранить JSON-отчёт")
    args = parser.parse_args(argv)

    report = run_benchmark(args.input_dir)
    output = args.output or _default_output_path()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nОтчёт сохранён: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
