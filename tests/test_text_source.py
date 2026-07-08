from types import SimpleNamespace

import text_source


def test_returns_primary_selection(monkeypatch):
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(stdout="hello world")

    monkeypatch.setattr(text_source.subprocess, "run", fake_run)
    assert text_source.get_text_to_read() == "hello world"
    # Clipboard must NOT be queried when the selection has text.
    assert calls == [["wl-paste", "--primary", "--no-newline"]]


def test_falls_back_to_clipboard_when_selection_blank(monkeypatch):
    def fake_run(args, **kwargs):
        if "--primary" in args:
            return SimpleNamespace(stdout="   \n")  # whitespace only
        return SimpleNamespace(stdout="clipboard text")

    monkeypatch.setattr(text_source.subprocess, "run", fake_run)
    assert text_source.get_text_to_read() == "clipboard text"


def test_returns_none_when_both_empty(monkeypatch):
    monkeypatch.setattr(
        text_source.subprocess, "run", lambda args, **kw: SimpleNamespace(stdout="")
    )
    assert text_source.get_text_to_read() is None


def test_selection_error_falls_back_to_clipboard(monkeypatch):
    import subprocess as sp

    def fake_run(args, **kwargs):
        if "--primary" in args:
            raise sp.CalledProcessError(1, args)
        return SimpleNamespace(stdout="from clipboard")

    monkeypatch.setattr(text_source.subprocess, "run", fake_run)
    assert text_source.get_text_to_read() == "from clipboard"
