import sys
import types

# Stub heavy/hardware modules so importing voice_assistant is cheap and safe.
for name in ["faster_whisper", "piper", "evdev"]:
    sys.modules.setdefault(name, types.ModuleType(name))

# Minimal attributes the module accesses at import time.
sys.modules["faster_whisper"].WhisperModel = object
_piper = sys.modules["piper"]
_piper.PiperVoice = object
_piper.SynthesisConfig = object
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
