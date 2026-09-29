"""Реестр весов: согласованность с config, сверка sha256, скачивание, изоляция от инференса."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from columba import config as cfg
from columba import weights as w

SRC_DIR = Path(cfg.__file__).resolve().parent

EXPECTED_CONFIG_ATTRS = {
    "REGION_CNN_WEIGHTS",
    "SPINE_UNET_WEIGHTS",
    "SPINE_OBJECTS_CNN_WEIGHTS",
    "HIP_POSITIONING_CNN_WEIGHTS",
}

INFERENCE_MODULE_STEMS = {
    "inference",
    "service",
    "aggregate",
    "submission",
    "spine_checkers",
    "hip_checkers",
}


def test_registry_matches_config_paths():
    registry = w.load_registry()
    attrs = {e.config_attr for e in registry}
    assert attrs == EXPECTED_CONFIG_ATTRS
    for entry in registry:
        assert entry.path == getattr(cfg, entry.config_attr)
        assert entry.sha256.isalnum() and len(entry.sha256) == 64
        assert entry.size_bytes > 0


def test_registry_names_are_unique():
    registry = w.load_registry()
    names = [e.name for e in registry]
    assert len(names) == len(set(names))


@pytest.mark.parametrize(
    "status,exit_code",
    [("ok", 0), ("missing", 1), ("mismatch", 1)],
)
def test_verify_exit_code_requires_all_weights(status, exit_code, capsys):
    entry = w.load_registry()[0]
    result = w.VerifyResult(entry=entry, status=status)
    assert w._print_verify_report([result]) == exit_code
    capsys.readouterr()


@pytest.mark.parametrize("entry_name", [e.name for e in w.load_registry()])
def test_present_weight_matches_registry_sha256(entry_name):
    entry = next(e for e in w.load_registry() if e.name == entry_name)
    if not entry.path.exists():
        pytest.skip(f"{entry.name} отсутствует локально (artifacts/models/ в .gitignore)")
    result = w.verify_entry(entry)
    assert result.status == "ok", (
        f"{entry.name}: ожидался sha256={entry.sha256} size={entry.size_bytes}, "
        f"получено sha256={result.actual_sha256} size={result.actual_size}"
    )


def test_download_weight_correct_hash(tmp_path, monkeypatch):
    payload = b"columba-test-weight-bytes"
    import hashlib

    sha256 = hashlib.sha256(payload).hexdigest()
    dest = tmp_path / "fake_weight.pt"
    entry = w.WeightEntry(
        name="fake_weight.pt",
        config_attr="_FAKE_ATTR",
        size_bytes=len(payload),
        sha256=sha256,
        reproduce="n/a",
        url="https://example.invalid/fake_weight.pt",
        role="test",
    )
    monkeypatch.setattr(cfg, "_FAKE_ATTR", dest, raising=False)

    def fake_fetch(url, dest_path):
        assert url == entry.url
        Path(dest_path).write_bytes(payload)

    result_path = w.download_weight(entry, fetch=fake_fetch)
    assert result_path == dest
    assert dest.exists()
    assert dest.read_bytes() == payload
    # временных файлов не осталось
    assert list(tmp_path.glob(".*part")) == []


def test_download_weight_wrong_hash_does_not_create_file(tmp_path, monkeypatch):
    dest = tmp_path / "fake_weight_bad.pt"
    entry = w.WeightEntry(
        name="fake_weight_bad.pt",
        config_attr="_FAKE_ATTR_BAD",
        size_bytes=5,
        sha256="0" * 64,
        reproduce="n/a",
        url="https://example.invalid/fake_weight_bad.pt",
        role="test",
    )
    monkeypatch.setattr(cfg, "_FAKE_ATTR_BAD", dest, raising=False)

    def fake_fetch(url, dest_path):
        Path(dest_path).write_bytes(b"wrong-bytes")

    with pytest.raises(w.WeightsError):
        w.download_weight(entry, fetch=fake_fetch)

    assert not dest.exists()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "url,expected_id",
    [
        ("https://drive.google.com/file/d/1AbC-XyZ_123/view?usp=sharing", "1AbC-XyZ_123"),
        ("https://drive.google.com/open?id=1AbC-XyZ_123", "1AbC-XyZ_123"),
        ("https://drive.google.com/uc?export=download&id=1AbC-XyZ_123", "1AbC-XyZ_123"),
        ("https://example.invalid/fake_weight.pt", None),
        ("https://docs.google.com/uc?id=42", "42"),
    ],
)
def test_google_drive_file_id_extraction(url, expected_id):
    assert w._google_drive_file_id(url) == expected_id


# Реальная HTML-страница-предупреждение Drive (сохранена curl'ом с живой
# ссылки при первом сквозном прогоне `columba.weights download` — первая
# версия парсера искала `<a href="...confirm=...">` и падала на настоящей
# странице: Drive использует ФОРМУ с action на другой домен и отдельными
# скрытыми полями, не ссылку с confirm в query-строке).
REAL_DRIVE_WARNING_HTML = """
<!DOCTYPE html><html><head><title>Google Drive - Virus scan warning</title></head>
<body><div class="uc-main"><div id="uc-text">
<p class="uc-warning-caption">Google Drive can't scan this file for viruses.</p>
<form id="download-form" action="https://drive.usercontent.google.com/download" method="get">
<input type="submit" id="uc-download-link" class="goog-inline-block jfk-button jfk-button-action" value="Download anyway"/>
<input type="hidden" name="id" value="1y0ghBr0NrVqR8B9SDwx-l-z6IawIHn3j">
<input type="hidden" name="export" value="download">
<input type="hidden" name="confirm" value="t">
<input type="hidden" name="uuid" value="c944db56-38ef-4e88-a75d-e042b510e176">
</form></div></div></body></html>
"""


def test_parse_drive_warning_form_extracts_download_url_from_real_page():
    url = w._parse_drive_warning_form(REAL_DRIVE_WARNING_HTML)
    assert url is not None
    assert url.startswith("https://drive.usercontent.google.com/download?")
    parsed = w.parse_qs(w.urlparse(url).query)
    assert parsed["id"] == ["1y0ghBr0NrVqR8B9SDwx-l-z6IawIHn3j"]
    assert parsed["export"] == ["download"]
    assert parsed["confirm"] == ["t"]
    assert parsed["uuid"] == ["c944db56-38ef-4e88-a75d-e042b510e176"]


def test_parse_drive_warning_form_missing_form_returns_none():
    assert w._parse_drive_warning_form("<html><body>no form here</body></html>") is None


def test_resolve_drive_direct_url_uses_warning_form(monkeypatch):
    """Реальная сеть не трогается — `urllib.request.urlopen` подменён."""

    class FakeResponse:
        headers = {"Content-Type": "text/html; charset=utf-8"}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return REAL_DRIVE_WARNING_HTML.encode("utf-8")

    monkeypatch.setattr(w.urllib.request, "urlopen", lambda *a, **k: FakeResponse())
    url = w._resolve_drive_direct_url("1y0ghBr0NrVqR8B9SDwx-l-z6IawIHn3j")
    assert url.startswith("https://drive.usercontent.google.com/download?")


def test_resolve_drive_direct_url_small_file_skips_warning_page(monkeypatch):
    """Файл без предупреждения (маленький) — Content-Type не text/html,
    базовая ссылка уже рабочая, второй запрос не нужен."""

    class FakeResponse:
        headers = {"Content-Type": "application/octet-stream"}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            raise AssertionError("не должно читаться — это не HTML-предупреждение")

    monkeypatch.setattr(w.urllib.request, "urlopen", lambda *a, **k: FakeResponse())
    url = w._resolve_drive_direct_url("abc")
    assert url == "https://drive.google.com/uc?export=download&id=abc"


def test_resolve_drive_direct_url_unparseable_page_raises(monkeypatch):
    class FakeResponse:
        headers = {"Content-Type": "text/html"}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b"<html><body>Drive changed its format again</body></html>"

    monkeypatch.setattr(w.urllib.request, "urlopen", lambda *a, **k: FakeResponse())
    with pytest.raises(w.WeightsError, match="не удалось разобрать"):
        w._resolve_drive_direct_url("abc")


def test_fetch_url_routes_drive_links_through_warning_form(tmp_path, monkeypatch):
    """`_fetch_url` (не только `download_weight`'s `fetch=` мок) реально
    распознаёт Drive-ссылку и запрашивает URL, извлечённый из формы."""
    requested_urls = []

    class FakeResponse:
        headers = {"Content-Type": "text/html"}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, *args):
            if args:
                return b""  # второй вызов (файл) — конец потока
            return REAL_DRIVE_WARNING_HTML.encode("utf-8")  # первый вызов (страница-предупреждение)

    def fake_urlopen(request_or_url, *a, **k):
        url = request_or_url if isinstance(request_or_url, str) else request_or_url.full_url
        requested_urls.append(url)
        return FakeResponse()

    monkeypatch.setattr(w.urllib.request, "urlopen", fake_urlopen)
    dest = tmp_path / "out.pt"
    w._fetch_url("https://drive.google.com/file/d/abc/view", dest)

    assert requested_urls[0] == "https://drive.google.com/uc?export=download&id=abc"
    assert requested_urls[1].startswith("https://drive.usercontent.google.com/download?")


def test_download_weight_without_url_raises(tmp_path, monkeypatch):
    dest = tmp_path / "no_url.pt"
    entry = w.WeightEntry(
        name="no_url.pt",
        config_attr="_FAKE_ATTR_NOURL",
        size_bytes=1,
        sha256="0" * 64,
        reproduce="n/a",
        url=None,
        role="test",
    )
    monkeypatch.setattr(cfg, "_FAKE_ATTR_NOURL", dest, raising=False)
    with pytest.raises(w.WeightsError, match="ссылка не вписана"):
        w.download_weight(entry)
    assert not dest.exists()


def _imported_module_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module.split(".")[0])
            for alias in node.names:
                names.add(alias.name)
    return names


def _inference_module_paths() -> list[Path]:
    paths = {SRC_DIR / f"{stem}.py" for stem in INFERENCE_MODULE_STEMS}
    paths |= set(SRC_DIR.glob("*cnn*.py"))
    return sorted(p for p in paths if p.exists())


@pytest.mark.parametrize("path", _inference_module_paths(), ids=lambda p: p.name)
def test_inference_modules_do_not_import_weights(path):
    imported = _imported_module_names(path)
    assert "weights" not in imported, f"{path.name} импортирует columba.weights"
