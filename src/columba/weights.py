"""Реестр весов CNN-моделей: проверка и (ручная) загрузка.

Источник истины по контрольным суммам — `weights_registry.json` в корне
репозитория (не в `artifacts/`, которая целиком в `.gitignore` — реестр
обязан коммититься вместе с кодом). `WEIGHTS.md` — человекочитаемая проекция
этого файла.

Это ЕДИНСТВЕННЫЙ модуль пакета с сетевым кодом (`urllib`, только stdlib, без
новых зависимостей). Инференс и сервис (`inference.py`, `service.py`,
`aggregate.py`, `submission.py`, чекеры, `*_cnn.py`) не импортируют его и не
должны — воспроизводимый прогон обязан работать полностью офлайн; сетевой
вызов допустим только как явное ручное действие через CLI ниже.

CLI:
    uv run python -m columba.weights verify
    uv run python -m columba.weights download [имя_файла ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from html import unescape as html_unescape
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

from . import config as cfg

REGISTRY_JSON = cfg.PROJECT_ROOT / "weights_registry.json"

_CHUNK_SIZE = 1 << 20  # 1 МиБ
_DOWNLOAD_TIMEOUT_S = 60  # чтобы зависший сокет не вешал download() навсегда


@dataclass(frozen=True)
class WeightEntry:
    """Одна запись реестра весов."""

    name: str
    config_attr: str
    size_bytes: int
    sha256: str
    reproduce: str
    url: str | None
    role: str

    @property
    def path(self) -> Path:
        return getattr(cfg, self.config_attr)


def load_registry(registry_json: Path | str = REGISTRY_JSON) -> tuple[WeightEntry, ...]:
    payload = json.loads(Path(registry_json).read_text(encoding="utf-8"))
    return tuple(
        WeightEntry(
            name=item["name"],
            config_attr=item["config_attr"],
            size_bytes=item["size_bytes"],
            sha256=item["sha256"],
            reproduce=item["reproduce"],
            url=item.get("url"),
            role=item.get("role", ""),
        )
        for item in payload["entries"]
    )


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class VerifyResult:
    entry: WeightEntry
    status: str  # "ok" | "missing" | "mismatch"
    actual_sha256: str | None = None
    actual_size: int | None = None


def verify_entry(entry: WeightEntry) -> VerifyResult:
    if not entry.path.exists():
        return VerifyResult(entry, "missing")
    actual_sha256 = sha256_of(entry.path)
    actual_size = entry.path.stat().st_size
    if actual_sha256 != entry.sha256 or actual_size != entry.size_bytes:
        return VerifyResult(entry, "mismatch", actual_sha256, actual_size)
    return VerifyResult(entry, "ok", actual_sha256, actual_size)


def verify_all(registry: tuple[WeightEntry, ...] | None = None) -> list[VerifyResult]:
    entries = registry if registry is not None else load_registry()
    return [verify_entry(entry) for entry in entries]


class WeightsError(RuntimeError):
    """Ошибка загрузки/проверки весов (нет ссылки, битый хэш и т. п.)."""


_GOOGLE_DRIVE_HOSTS = {"drive.google.com", "docs.google.com"}


def _google_drive_file_id(url: str) -> str | None:
    """Ссылка Google Drive -> id файла, либо `None`, если это не Drive-ссылка.

    Понимает оба распространённых вида: `.../file/d/<id>/view?...` и
    `...?id=<id>&...` (включая уже нормализованный `uc?export=download&id=`)."""
    parsed = urlparse(url)
    if parsed.netloc not in _GOOGLE_DRIVE_HOSTS:
        return None
    match = re.search(r"/file/d/([^/]+)", parsed.path)
    if match:
        return match.group(1)
    query = parse_qs(parsed.query)
    if "id" in query:
        return query["id"][0]
    return None


_DRIVE_FORM_ACTION_RE = re.compile(r'<form[^>]+id="download-form"[^>]+action="([^"]+)"')
_DRIVE_HIDDEN_INPUT_RE = re.compile(r'<input type="hidden" name="([^"]+)" value="([^"]*)"')


def _parse_drive_warning_form(html: str) -> str | None:
    """Достать прямую ссылку из HTML-формы страницы-предупреждения Drive.

    **Проверено на реальной странице** (не только на сфабрикованном
    образце — первый прогон `columba.weights download` против настоящей
    ссылки Drive упал именно потому, что реальный формат оказался другим,
    чем предполагалось при первой версии этой функции): текущий Drive
    отдаёт не `<a href="...confirm=...">`, а `<form id="download-form"
    action="https://drive.usercontent.google.com/download" method="get">`
    со скрытыми полями `id`/`export`/`confirm`/`uuid` — токен `confirm` в
    этой форме не уникальная строка, а буквально `"t"`, а недостающий
    `uuid` без формы никак не получить. Собираем итоговый URL из
    `action` + query-строки всех скрытых полей формы.
    """
    action_match = _DRIVE_FORM_ACTION_RE.search(html)
    if action_match is None:
        return None
    action = html_unescape(action_match.group(1))
    fields = {name: html_unescape(value) for name, value in _DRIVE_HIDDEN_INPUT_RE.findall(html)}
    if not fields:
        return None
    return f"{action}?{urlencode(fields)}"


def _resolve_drive_direct_url(file_id: str) -> str:
    """id файла Google Drive -> прямая ссылка на байты.

    Для файлов этого реестра (2-45 МБ) Drive вместо содержимого отдаёт HTML
    с предупреждением о непроверенном антивирусом файле — обычный GET
    получил бы именно эту HTML-страницу, `download_weight` сравнил бы её
    sha256 с ожидаемым и упал с непонятным «sha256 не совпадает», не
    объяснив, что дело в самом Drive, не в файле.
    """
    base = f"https://drive.google.com/uc?export=download&id={file_id}"
    request = urllib.request.Request(base, headers={"User-Agent": "columba-weights/1.0"})
    with urllib.request.urlopen(request, timeout=_DOWNLOAD_TIMEOUT_S) as response:
        content_type = response.headers.get("Content-Type", "")
        if not content_type.startswith("text/html"):
            return base  # маленький файл — предупреждения не было, ссылка уже рабочая
        html = response.read().decode("utf-8", errors="replace")
    direct_url = _parse_drive_warning_form(html)
    if direct_url is None:
        raise WeightsError(
            f"Google Drive: не удалось разобрать страницу-предупреждение для file_id={file_id} — "
            "похоже, её формат снова изменился (Google уже менял его раньше, см. stages/stage_9.md)"
        )
    return direct_url


def _fetch_url(url: str, dest: Path) -> None:
    """Единственное место, реально ходящее в сеть. Тесты подменяют эту функцию."""
    file_id = _google_drive_file_id(url)
    if file_id is not None:
        url = _resolve_drive_direct_url(file_id)
    with urllib.request.urlopen(url, timeout=_DOWNLOAD_TIMEOUT_S) as response, open(dest, "wb") as out:
        while True:
            chunk = response.read(_CHUNK_SIZE)
            if not chunk:
                break
            out.write(chunk)


def download_weight(entry: WeightEntry, *, fetch=_fetch_url) -> Path:
    """Скачать во временный файл рядом с целью, проверить sha256, атомарно переместить.

    Неверный хэш или отсутствующая ссылка -> `WeightsError`, целевой файл не создаётся
    (и временный файл удаляется).
    """
    if not entry.url:
        raise WeightsError(f"{entry.name}: ссылка не вписана в {REGISTRY_JSON.name} (заполнить после загрузки в хранилище)")

    entry.path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(entry.path.parent), prefix=f".{entry.name}.", suffix=".part")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        fetch(entry.url, tmp_path)
        actual_sha256 = sha256_of(tmp_path)
        if actual_sha256 != entry.sha256:
            raise WeightsError(
                f"{entry.name}: sha256 не совпадает после загрузки "
                f"(ожидалось {entry.sha256}, получено {actual_sha256}) — файл не сохранён"
            )
        os.replace(tmp_path, entry.path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return entry.path


def _print_verify_report(results: list[VerifyResult]) -> int:
    missing = [r for r in results if r.status == "missing"]
    mismatched = [r for r in results if r.status == "mismatch"]
    ok = [r for r in results if r.status == "ok"]

    for r in ok:
        print(f"OK       {r.entry.name}  sha256={r.actual_sha256}")
    for r in mismatched:
        print(
            f"MISMATCH {r.entry.name}  ожидалось={r.entry.sha256} получено={r.actual_sha256} "
            f"(размер ожидался={r.entry.size_bytes} получен={r.actual_size})"
        )
    if missing:
        print("Отсутствуют веса (см. WEIGHTS.md):")
        for r in missing:
            print(f"  - {r.entry.name} -> {r.entry.path}  ({r.entry.role})")

    return 1 if missing or mismatched else 0


def _cli_verify(_args: argparse.Namespace) -> int:
    return _print_verify_report(verify_all())


def _cli_download(args: argparse.Namespace) -> int:
    registry = load_registry()
    by_name = {e.name: e for e in registry}
    names = args.names or list(by_name)

    unknown = [n for n in names if n not in by_name]
    if unknown:
        print(f"Неизвестные имена весов: {unknown}. Доступные: {list(by_name)}", file=sys.stderr)
        return 2

    exit_code = 0
    for name in names:
        entry = by_name[name]
        try:
            path = download_weight(entry)
            print(f"OK    {name} -> {path}")
        except WeightsError as exc:
            print(f"ERROR {exc}", file=sys.stderr)
            exit_code = 1
    return exit_code


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="columba.weights", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    verify_parser = sub.add_parser("verify", help="Проверить sha256 присутствующих весов, перечислить отсутствующие")
    verify_parser.set_defaults(func=_cli_verify)

    download_parser = sub.add_parser("download", help="Скачать веса по ссылкам из реестра")
    download_parser.add_argument("names", nargs="*", help="Имена файлов (по умолчанию — все); напр. region_cnn.pt")
    download_parser.set_defaults(func=_cli_download)

    return parser


def main(argv: list[str] | None = None) -> int:
    # Windows-консоль по умолчанию не в UTF-8 — русский текст иначе превращается в кракозябры.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
