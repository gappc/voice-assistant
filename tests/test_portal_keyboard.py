import stat

import portal_keyboard as pk

PNG = b"\x89PNG\r\n\x1a\n" + b"rest"


def test_sniff_recognises_images_text_and_junk():
    assert pk.sniff_mime(PNG) == "image/png"
    assert pk.sniff_mime(b"\xff\xd8\xff\xe0jpeg") == "image/jpeg"
    assert pk.sniff_mime(b"GIF89a...") == "image/gif"
    assert pk.sniff_mime("Grüße".encode()) == pk.TEXT
    assert pk.sniff_mime(b"\xff\xfe\x00broken") is None
    assert pk.sniff_mime(b"") is None


class _FakePortal(pk.PortalKeyboard):
    """Replaces the D-Bus traffic with a script.

    `clipboard` maps the MIME type asked for to what SelectionRead returns.
    `fetches[("gnome", n)]` lists the types GNOME requests right after the
    n-th SetSelection; `fetches[("paste", n)]` those the target app requests
    once the paste keys are released while offer n is current."""

    keys_last_released = None

    def __init__(self, tmp_path, clipboard=None, fetches=None):
        super().__init__(tmp_path / "token")
        self.clipboard = clipboard or {}
        self.fetches = fetches or {}
        self.log = []
        self.offers = []
        self.pending = []

    def _ensure_session(self):
        self.log.append("open")

    def _drop(self):
        self.log.append("close")

    def _keycode(self, keycode, state):
        self.log.append(("key", keycode, state))
        if keycode == self.keys_last_released and state == pk.KEY_RELEASED:
            self.pending += self.fetches.get(("paste", len(self.offers)), [])

    def _read(self, mime):
        self.log.append(("read", mime))
        return self.clipboard.get(mime, b"")

    def _set_selection(self, mime_types):
        self.offers.append(list(mime_types))
        self.log.append(("offer", len(self.offers)))
        self.pending += self.fetches.get(("gnome", len(self.offers)), [])

    def _next_transfer(self, timeout):
        if not self.pending:
            return None
        return self.pending.pop(0), 7

    def _write(self, serial, data):
        self.log.append(("served", len(self.offers), data))


def _paste(kb, text="hallo", keys=(29, 47), restore=True):
    kb.keys_last_released = keys[0]
    kb.paste(text, list(keys), restore=restore)


def test_paste_offers_text_presses_keys_then_restores_text(tmp_path):
    kb = _FakePortal(
        tmp_path,
        clipboard={"text/plain;charset=utf-8": b"earlier"},
        fetches={("gnome", 1): ["text/plain;charset=utf-8"],
                 ("paste", 1): ["UTF8_STRING"],
                 ("gnome", 2): ["text/plain;charset=utf-8"]},
    )

    _paste(kb, "Grüße")

    assert kb.offers == [pk.TEXT_TYPES, pk.TEXT_TYPES]
    served = [e for e in kb.log if e[0] == "served"]
    assert served == [("served", 1, "Grüße".encode()), ("served", 1, "Grüße".encode()),
                      ("served", 2, b"earlier")]
    keys = [e for e in kb.log if e[0] == "key"]
    assert keys == [("key", 29, 1), ("key", 47, 1), ("key", 47, 0), ("key", 29, 0)]
    assert kb.log[0] == "open" and kb.log[-1] == "close"


def test_paste_restores_an_image_under_its_own_type(tmp_path):
    kb = _FakePortal(
        tmp_path,
        clipboard={"text/plain;charset=utf-8": PNG},  # the portal ignores the type asked for
        fetches={("gnome", 2): ["image/png"]},
    )

    _paste(kb)

    assert kb.offers[1] == ["image/png"]
    assert ("served", 2, PNG) in kb.log


def test_paste_reads_png_when_text_comes_back_empty(tmp_path):
    kb = _FakePortal(tmp_path, clipboard={"image/png": PNG}, fetches={("gnome", 2): ["image/png"]})

    _paste(kb)

    assert ("read", "image/png") in kb.log
    assert kb.offers[1] == ["image/png"]


def test_paste_leaves_dictation_when_clipboard_was_empty(tmp_path):
    kb = _FakePortal(tmp_path)

    _paste(kb)

    assert len(kb.offers) == 1


def test_paste_without_restore_does_not_read(tmp_path):
    kb = _FakePortal(tmp_path, clipboard={"text/plain;charset=utf-8": b"earlier"})

    _paste(kb, restore=False)

    assert not [e for e in kb.log if e[0] == "read"]
    assert len(kb.offers) == 1


def test_paste_answers_unknown_types_with_failure(tmp_path):
    kb = _FakePortal(tmp_path, fetches={("paste", 1): ["text/html"]})

    _paste(kb, restore=False)

    assert ("served", 1, None) in kb.log


def test_retry_keeps_the_first_backup(tmp_path, monkeypatch):
    """A retry after a dropped session must not back up our own dictation."""
    kb = _FakePortal(tmp_path, clipboard={"text/plain;charset=utf-8": b"earlier"},
                     fetches={("gnome", 3): ["text/plain;charset=utf-8"]})
    original_keycode = kb._keycode
    failed = []

    def flaky(keycode, state):
        if not failed:
            failed.append(1)
            kb.clipboard = {"text/plain;charset=utf-8": b"hallo"}  # dictation now on it
            raise OSError("session dropped")
        original_keycode(keycode, state)
    monkeypatch.setattr(kb, "_keycode", flaky)

    _paste(kb)

    assert ("served", 3, b"earlier") in kb.log


def _recording_keyboard(tmp_path, monkeypatch):
    kb = pk.PortalKeyboard(tmp_path / "token")
    sent = []
    monkeypatch.setattr(kb, "_ensure_session", lambda: sent.append("open"))
    monkeypatch.setattr(kb, "_drop", lambda: sent.append("close"))
    return kb, sent


def test_failed_send_reopens_session_and_retries_once(tmp_path, monkeypatch):
    kb = pk.PortalKeyboard(tmp_path / "token")
    opened = []
    monkeypatch.setattr(kb, "_ensure_session", lambda: opened.append(1))
    closed = []
    monkeypatch.setattr(kb, "_drop", lambda: closed.append(1))
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) == 1:
            raise OSError("session closed")

    kb._send(flaky)
    assert len(attempts) == 2 and len(opened) == 2
    assert closed, "the session must be closed after the retry too"


def test_session_closed_even_when_the_action_fails(tmp_path, monkeypatch):
    kb, sent = _recording_keyboard(tmp_path, monkeypatch)

    def broken():
        raise RuntimeError("boom")

    try:
        kb._send(broken)
    except RuntimeError:
        pass
    assert sent[-1] == "close"


def test_connect_opens_and_closes_again(tmp_path, monkeypatch):
    kb, sent = _recording_keyboard(tmp_path, monkeypatch)
    kb.connect()
    assert sent == ["open", "close"]


def _opening_keyboard(tmp_path, monkeypatch, grants):
    """`grants` lists, per _open_session call, whether clipboard is granted."""
    kb = pk.PortalKeyboard(tmp_path / "token")
    calls = []

    def fake_open(token):
        calls.append(token)
        if not grants[len(calls) - 1]:
            raise pk.ClipboardNotGranted("no clipboard")
    monkeypatch.setattr(kb, "_open_session", fake_open)
    return kb, calls


def test_old_keyboard_only_token_is_dropped_and_asked_again(tmp_path, monkeypatch):
    kb, calls = _opening_keyboard(tmp_path, monkeypatch, grants=[False, True])
    kb._save_token("old")

    kb._ensure_session()

    assert calls == ["old", None]
    assert kb._load_token() is None


def test_no_clipboard_without_token_is_an_error(tmp_path, monkeypatch):
    kb, calls = _opening_keyboard(tmp_path, monkeypatch, grants=[False])

    try:
        kb._ensure_session()
    except pk.ClipboardNotGranted:
        pass
    else:
        raise AssertionError("expected ClipboardNotGranted")
    assert calls == [None]


def test_restore_token_is_saved_private(tmp_path):
    kb = pk.PortalKeyboard(tmp_path / "sub" / "token")
    kb._save_token("abc")
    assert kb._load_token() == "abc"
    mode = stat.S_IMODE((tmp_path / "sub" / "token").stat().st_mode)
    assert mode == 0o600
