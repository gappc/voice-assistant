import json
from pathlib import Path

import pytest

import tray_settings


def test_default_config_dir_uses_xdg(monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", "/xdg/config")
    assert tray_settings.default_config_dir() == Path("/xdg/config/voice-assistant")


def test_default_config_dir_falls_back_to_home(monkeypatch):
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setenv("HOME", "/home/tester")
    assert tray_settings.default_config_dir() == Path("/home/tester/.config/voice-assistant")


def test_load_missing_file_returns_defaults(tmp_path):
    settings = tray_settings.load(tmp_path / "settings.json")
    assert settings == tray_settings.TraySettings()


def test_save_then_load_round_trips(tmp_path):
    path = tmp_path / "settings.json"
    original = tray_settings.TraySettings(
        input_device="USB Microphone",
        output_device="Built-in Audio Analog Stereo",
        voice="en_US-sarah",
        speed=1.25,
        paste_with_shift=False,
        stt_language="de",
    )
    tray_settings.save(original, path)
    assert tray_settings.load(path) == original


def test_load_corrupt_json_returns_defaults(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text("{not valid json")
    assert tray_settings.load(path) == tray_settings.TraySettings()


def test_load_non_object_json_returns_defaults(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(["not", "an", "object"]))
    assert tray_settings.load(path) == tray_settings.TraySettings()


def test_load_wrong_shaped_fields_fall_back_to_defaults(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"speed": "fast", "paste_with_shift": "yes", "voice": 42, "stt_language": 7}))
    settings = tray_settings.load(path)
    assert settings.speed is None
    assert settings.paste_with_shift is True
    assert settings.stt_language is None
    assert settings.voice is None


def test_save_is_atomic_leaves_prior_file_on_failure(tmp_path, monkeypatch):
    path = tmp_path / "settings.json"
    tray_settings.save(tray_settings.TraySettings(voice="en_US-bella"), path)
    original_content = path.read_text()

    def fail_rename(self, target):
        raise OSError("simulated failure")

    monkeypatch.setattr(Path, "rename", fail_rename)
    with pytest.raises(OSError):
        tray_settings.save(tray_settings.TraySettings(voice="en_US-sarah"), path)

    assert path.read_text() == original_content
