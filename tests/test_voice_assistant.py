import sys
import types

# Stub heavy/hardware modules so importing voice_assistant is cheap and safe.
for name in ["faster_whisper", "evdev"]:
    sys.modules.setdefault(name, types.ModuleType(name))

# Minimal attributes the module accesses at import time.
sys.modules["faster_whisper"].WhisperModel = object
_evdev = sys.modules["evdev"]
_evdev.InputDevice = object
_evdev.categorize = lambda e: e
_evdev.list_devices = lambda: []
_ecodes = types.SimpleNamespace(KEY_RIGHTALT=100, KEY_F23=193, EV_KEY=1, KEY_A=30)
_evdev.ecodes = _ecodes

import voice_assistant


def test_get_output_devices_filters_by_output_channels(monkeypatch):
    fake_devices = [
        {"name": "Mic Only", "max_input_channels": 2, "max_output_channels": 0},
        {"name": "Speakers", "max_input_channels": 0, "max_output_channels": 2},
        {"name": "Headset", "max_input_channels": 1, "max_output_channels": 2},
    ]
    monkeypatch.setattr(voice_assistant.sd, "query_devices", lambda: fake_devices)

    va = voice_assistant.VoiceAssistant.__new__(voice_assistant.VoiceAssistant)
    result = va.get_output_devices()

    assert result == [
        {"index": 1, "name": "Speakers"},
        {"index": 2, "name": "Headset"},
    ]


def test_resolve_device_matches_by_index():
    devices = [{"index": 0, "name": "Mic"}, {"index": 2, "name": "USB Mic"}]
    assert voice_assistant._resolve_device(2, devices) == 2
    assert voice_assistant._resolve_device("2", devices) == 2


def test_resolve_device_matches_by_name_substring():
    devices = [{"index": 0, "name": "Mic"}, {"index": 2, "name": "USB Microphone"}]
    assert voice_assistant._resolve_device("usb", devices) == 2


def test_resolve_device_returns_none_for_no_match_or_none():
    devices = [{"index": 0, "name": "Mic"}]
    assert voice_assistant._resolve_device("nonexistent", devices) is None
    assert voice_assistant._resolve_device(None, devices) is None


def test_device_name_looks_up_name_by_index():
    devices = [{"index": 0, "name": "Mic"}, {"index": 2, "name": "USB Microphone"}]
    assert voice_assistant._device_name(2, devices) == "USB Microphone"


def test_device_name_returns_none_for_none_index():
    devices = [{"index": 0, "name": "Mic"}]
    assert voice_assistant._device_name(None, devices) is None
    assert voice_assistant._device_name(99, devices) is None


def test_set_output_device_resolves_by_name(monkeypatch):
    fake_devices = [
        {"name": "Mic Only", "max_input_channels": 2, "max_output_channels": 0},
        {"name": "Speakers", "max_input_channels": 0, "max_output_channels": 2},
    ]
    monkeypatch.setattr(voice_assistant.sd, "query_devices", lambda: fake_devices)

    va = voice_assistant.VoiceAssistant.__new__(voice_assistant.VoiceAssistant)
    va.reader = types.SimpleNamespace(set_output_device=lambda idx: None)
    va.set_output_device("speakers")

    assert va.output_device == 1


def test_piper_is_not_a_dependency():
    """piper-tts must be absent from the environment, not merely unimported."""
    import importlib.metadata as md
    import pytest

    with pytest.raises(md.PackageNotFoundError):
        md.version("piper-tts")


def test_resolve_startup_settings_cli_device_overrides_saved(tmp_path):
    path = tmp_path / "settings.json"
    voice_assistant.tray_settings.save(
        voice_assistant.tray_settings.TraySettings(input_device="Saved Mic"), path
    )
    result = voice_assistant._resolve_startup_settings("CLI Mic", path)
    assert result.input_device == "CLI Mic"


def test_resolve_startup_settings_falls_back_to_saved_device(tmp_path):
    path = tmp_path / "settings.json"
    voice_assistant.tray_settings.save(
        voice_assistant.tray_settings.TraySettings(input_device="Saved Mic"), path
    )
    result = voice_assistant._resolve_startup_settings(None, path)
    assert result.input_device == "Saved Mic"


def test_resolve_startup_settings_unknown_voice_falls_back_to_default(tmp_path):
    path = tmp_path / "settings.json"
    voice_assistant.tray_settings.save(
        voice_assistant.tray_settings.TraySettings(voice="no-longer-exists"), path
    )
    result = voice_assistant._resolve_startup_settings(None, path)
    assert result.voice == voice_assistant.DEFAULT_VOICE


def test_resolve_startup_settings_defaults_when_nothing_saved(tmp_path):
    path = tmp_path / "settings.json"  # never written
    result = voice_assistant._resolve_startup_settings(None, path)
    assert result.input_device is None
    assert result.output_device is None
    assert result.voice == voice_assistant.DEFAULT_VOICE
    assert result.speed == voice_assistant.READ_SPEED
    assert result.paste_with_shift is True


class _FakeReader:
    """Minimal stand-in for SpeechReader: mutates _voice_name/_speed like the
    real thing so _save_settings sees the post-change state."""

    def __init__(self, voice_name, speed):
        self._voice_name = voice_name
        self._speed = speed

    def set_speed(self, scale):
        self._speed = scale

    def set_voice(self, name):
        self._voice_name = name


def test_save_settings_snapshots_current_state(monkeypatch, tmp_path):
    va = voice_assistant.VoiceAssistant.__new__(voice_assistant.VoiceAssistant)
    va._settings_path = tmp_path / "settings.json"
    va.current_device = 3
    va.output_device = 1
    va.paste_with_shift = False
    va.reader = _FakeReader("en_US-sarah", 1.25)
    monkeypatch.setattr(va, "get_input_devices", lambda: [{"index": 3, "name": "USB Microphone"}])
    monkeypatch.setattr(va, "get_output_devices", lambda: [{"index": 1, "name": "Speakers"}])

    va._save_settings()

    saved = voice_assistant.tray_settings.load(va._settings_path)
    assert saved == voice_assistant.tray_settings.TraySettings(
        input_device="USB Microphone",
        output_device="Speakers",
        voice="en_US-sarah",
        speed=1.25,
        paste_with_shift=False,
    )


def test_set_paste_with_shift_persists(monkeypatch, tmp_path):
    va = voice_assistant.VoiceAssistant.__new__(voice_assistant.VoiceAssistant)
    va._settings_path = tmp_path / "settings.json"
    va.current_device = None
    va.output_device = None
    va.reader = _FakeReader(voice_assistant.DEFAULT_VOICE, voice_assistant.READ_SPEED)
    monkeypatch.setattr(va, "get_input_devices", lambda: [])
    monkeypatch.setattr(va, "get_output_devices", lambda: [])

    va.set_paste_with_shift(False)

    assert va.paste_with_shift is False
    saved = voice_assistant.tray_settings.load(va._settings_path)
    assert saved.paste_with_shift is False


def test_set_speed_persists(monkeypatch, tmp_path):
    va = voice_assistant.VoiceAssistant.__new__(voice_assistant.VoiceAssistant)
    va._settings_path = tmp_path / "settings.json"
    va.current_device = None
    va.output_device = None
    va.paste_with_shift = True
    va.reader = _FakeReader(voice_assistant.DEFAULT_VOICE, voice_assistant.READ_SPEED)
    monkeypatch.setattr(va, "get_input_devices", lambda: [])
    monkeypatch.setattr(va, "get_output_devices", lambda: [])

    va.set_speed(1.25)

    assert va.reader._speed == 1.25
    saved = voice_assistant.tray_settings.load(va._settings_path)
    assert saved.speed == 1.25


def test_on_select_voice_persists_on_success(monkeypatch, tmp_path):
    va = voice_assistant.VoiceAssistant.__new__(voice_assistant.VoiceAssistant)
    va._settings_path = tmp_path / "settings.json"
    va.current_device = None
    va.output_device = None
    va.paste_with_shift = True
    va.reader = _FakeReader(voice_assistant.DEFAULT_VOICE, voice_assistant.READ_SPEED)
    monkeypatch.setattr(va, "get_input_devices", lambda: [])
    monkeypatch.setattr(va, "get_output_devices", lambda: [])

    va.on_select_voice("en_US-sarah")

    saved = voice_assistant.tray_settings.load(va._settings_path)
    assert saved.voice == "en_US-sarah"


def test_on_select_voice_does_not_persist_on_failure(tmp_path):
    va = voice_assistant.VoiceAssistant.__new__(voice_assistant.VoiceAssistant)
    va._settings_path = tmp_path / "settings.json"
    va.tray = types.SimpleNamespace(showMessage=lambda *a, **k: None)

    class _FailingReader:
        def set_voice(self, name):
            raise RuntimeError("boom")

    va.reader = _FailingReader()

    va.on_select_voice("bad-voice")

    assert not va._settings_path.exists()
