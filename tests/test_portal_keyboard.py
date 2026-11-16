import stat

import portal_keyboard as pk


def test_keysym_for_latin1_unicode_and_newline():
    assert pk.keysym_for("a") == ord("a")
    assert pk.keysym_for("ü") == 0xFC
    assert pk.keysym_for("€") == 0x01000000 | 0x20AC
    assert pk.keysym_for("\n") == pk.KEY_RETURN_SYM


def _recording_keyboard(tmp_path, monkeypatch):
    kb = pk.PortalKeyboard(tmp_path / "token")
    sent = []
    monkeypatch.setattr(kb, "_ensure_session", lambda: sent.append("open"))
    monkeypatch.setattr(kb, "_drop", lambda: sent.append("close"))
    monkeypatch.setattr(kb, "_keycode", lambda k, s: sent.append(("code", k, s)))
    monkeypatch.setattr(kb, "_keysym", lambda k, s: sent.append(("sym", k, s)))
    return kb, sent


def test_press_combo_presses_in_order_and_releases_in_reverse(tmp_path, monkeypatch):
    kb, sent = _recording_keyboard(tmp_path, monkeypatch)
    kb.press_combo([29, 42, 47])
    assert sent == ["open", ("code", 29, 1), ("code", 42, 1), ("code", 47, 1),
                    ("code", 47, 0), ("code", 42, 0), ("code", 29, 0), "close"]


def test_type_text_presses_and_releases_each_char(tmp_path, monkeypatch):
    kb, sent = _recording_keyboard(tmp_path, monkeypatch)
    kb.type_text("hi")
    assert sent == ["open", ("sym", ord("h"), 1), ("sym", ord("h"), 0),
                    ("sym", ord("i"), 1), ("sym", ord("i"), 0), "close"]


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


def test_restore_token_is_saved_private(tmp_path):
    kb = pk.PortalKeyboard(tmp_path / "sub" / "token")
    kb._save_token("abc")
    assert kb._load_token() == "abc"
    mode = stat.S_IMODE((tmp_path / "sub" / "token").stat().st_mode)
    assert mode == 0o600
