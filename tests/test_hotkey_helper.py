import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parent.parent / "hotkey_helper" / "voice-hotkeyd.py"
_spec = importlib.util.spec_from_file_location("voice_hotkeyd", _PATH)
hk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hk)

ME = 1000
OTHER = 1001


def _forwarder(active=ME):
    state = {"uid": active}
    fw = hk.Forwarder(ME, active_uid_fn=lambda: state["uid"])
    return fw, state


def test_talk_key_press_and_release_are_forwarded():
    fw, _ = _forwarder()
    assert fw.on_event("event3", hk.KEY_RIGHTALT, 1) == "talk down event3\n"
    assert fw.on_event("event3", hk.KEY_RIGHTALT, 0) == "talk up event3\n"


def test_read_key_is_forwarded():
    fw, _ = _forwarder()
    assert fw.on_event("event3", hk.KEY_F23, 1) == "read down event3\n"
    assert fw.on_event("event3", hk.KEY_F23, 0) == "read up event3\n"


def test_every_other_key_is_dropped():
    fw, _ = _forwarder()
    for code in range(0, 0x300):
        if code in hk.FORWARDED_KEYS:
            continue
        for value in (0, 1, 2):
            assert fw.on_event("event3", code, value) is None


def test_autorepeat_is_dropped():
    fw, _ = _forwarder()
    fw.on_event("event3", hk.KEY_RIGHTALT, 1)
    assert fw.on_event("event3", hk.KEY_RIGHTALT, 2) is None


def test_presses_in_another_users_session_stay_private():
    fw, _ = _forwarder(active=OTHER)
    assert fw.on_event("event3", hk.KEY_RIGHTALT, 1) is None
    assert fw.on_event("event3", hk.KEY_RIGHTALT, 0) is None


def test_unknown_active_session_fails_closed():
    fw, _ = _forwarder(active=None)
    assert fw.on_event("event3", hk.KEY_RIGHTALT, 1) is None


def test_release_after_session_switch_still_ends_the_hold():
    fw, state = _forwarder()
    fw.on_event("event3", hk.KEY_RIGHTALT, 1)
    state["uid"] = OTHER
    assert fw.on_event("event3", hk.KEY_RIGHTALT, 0) == "talk up event3\n"


def test_unplugged_keyboard_releases_what_it_held():
    fw, _ = _forwarder()
    fw.on_event("event3", hk.KEY_RIGHTALT, 1)
    fw.on_event("event5", hk.KEY_RIGHTALT, 1)
    assert fw.on_device_lost("event3") == ["talk up event3\n"]
    assert fw.on_device_lost("event3") == []
    assert fw.on_event("event5", hk.KEY_RIGHTALT, 0) == "talk up event5\n"


def test_active_uid_reads_logind_seat_file(tmp_path):
    seat = tmp_path / "seat0"
    seat.write_text("IS_SEAT0=1\nACTIVE=3\nACTIVE_UID=1000\nSESSIONS=3\n")
    assert hk.active_uid(str(seat)) == 1000


def test_active_uid_is_none_when_missing_or_garbled(tmp_path):
    assert hk.active_uid(str(tmp_path / "nope")) is None
    seat = tmp_path / "seat0"
    seat.write_text("ACTIVE_UID=abc\n")
    assert hk.active_uid(str(seat)) is None
    seat.write_text("IS_SEAT0=1\n")
    assert hk.active_uid(str(seat)) is None
