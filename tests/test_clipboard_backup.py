import subprocess

import clipboard_backup
from clipboard_backup import EMPTY, Snapshot


class _FakeRun:
    """Stands in for subprocess.run; answers wl-paste by type, records wl-copy."""

    def __init__(self, types=None, contents=None, list_error=None):
        self.types = types or []
        self.contents = contents or {}
        self.list_error = list_error
        self.calls = []

    def __call__(self, args, **kwargs):
        self.calls.append((args, kwargs))
        if args[:2] == ["wl-paste", "--list-types"]:
            if self.list_error:
                raise self.list_error
            out = "".join(t + "\n" for t in self.types).encode()
            return subprocess.CompletedProcess(args, 0, stdout=out)
        if args[0] == "wl-paste":
            mime = args[args.index("--type") + 1]
            return subprocess.CompletedProcess(args, 0, stdout=self.contents[mime])
        return subprocess.CompletedProcess(args, 0)


def _install(monkeypatch, fake):
    monkeypatch.setattr(clipboard_backup.subprocess, "run", fake)
    return fake


def test_save_reads_text_as_utf8_plain(monkeypatch):
    fake = _install(monkeypatch, _FakeRun(
        types=["UTF8_STRING", "STRING", "text/plain;charset=utf-8", "text/plain"],
        contents={"text/plain;charset=utf-8": "Grüße\n".encode()},
    ))

    assert clipboard_backup.save() == Snapshot("text/plain;charset=utf-8", "Grüße\n".encode())
    paste_args = fake.calls[1][0]
    assert "--no-newline" in paste_args


def test_save_prefers_plain_text_over_html(monkeypatch):
    _install(monkeypatch, _FakeRun(
        types=["text/html", "text/plain;charset=utf-8"],
        contents={"text/plain;charset=utf-8": b"hi"},
    ))

    assert clipboard_backup.save().mime == "text/plain;charset=utf-8"


def test_save_prefers_image_over_text(monkeypatch):
    _install(monkeypatch, _FakeRun(
        types=["text/plain", "image/jpeg", "image/png"],
        contents={"image/png": b"\x89PNG"},
    ))

    assert clipboard_backup.save() == Snapshot("image/png", b"\x89PNG")


def test_save_skips_x11_pseudo_types(monkeypatch):
    _install(monkeypatch, _FakeRun(
        types=["TARGETS", "TIMESTAMP", "application/x-thing"],
        contents={"application/x-thing": b"\x00\x01"},
    ))

    assert clipboard_backup.save() == Snapshot("application/x-thing", b"\x00\x01")


def test_save_reports_empty_clipboard(monkeypatch):
    _install(monkeypatch, _FakeRun(
        list_error=subprocess.CalledProcessError(1, ["wl-paste"], stderr=b"Nothing is copied\n"),
    ))

    assert clipboard_backup.save() is EMPTY


def test_save_returns_none_when_wl_paste_missing(monkeypatch):
    _install(monkeypatch, _FakeRun(list_error=FileNotFoundError("wl-paste")))

    assert clipboard_backup.save() is None


def test_save_returns_none_when_owner_hangs(monkeypatch):
    _install(monkeypatch, _FakeRun(list_error=subprocess.TimeoutExpired("wl-paste", 2)))

    assert clipboard_backup.save() is None


def test_save_returns_none_for_huge_content(monkeypatch):
    big = b"x" * (clipboard_backup.MAX_BYTES + 1)
    _install(monkeypatch, _FakeRun(types=["image/png"], contents={"image/png": big}))

    assert clipboard_backup.save() is None


def test_restore_writes_snapshot_with_its_type(monkeypatch):
    fake = _install(monkeypatch, _FakeRun())

    assert clipboard_backup.restore(Snapshot("image/png", b"\x89PNG")) is True

    args, kwargs = fake.calls[0]
    assert args == ["wl-copy", "--type", "image/png"]
    assert kwargs["input"] == b"\x89PNG"


def test_restore_empty_clears_clipboard(monkeypatch):
    fake = _install(monkeypatch, _FakeRun())

    assert clipboard_backup.restore(EMPTY) is True

    assert fake.calls[0][0] == ["wl-copy", "--clear"]


def test_restore_reports_failure(monkeypatch):
    def broken(args, **kwargs):
        raise subprocess.CalledProcessError(1, args)
    monkeypatch.setattr(clipboard_backup.subprocess, "run", broken)

    assert clipboard_backup.restore(Snapshot("text/plain", b"x")) is False
