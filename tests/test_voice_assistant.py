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
