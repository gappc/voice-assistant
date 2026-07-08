from pathlib import Path

import speech_reader


def test_default_voices_dir_uses_xdg(monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", "/xdg/data")
    assert speech_reader.default_voices_dir() == Path(
        "/xdg/data/voice-assistant/voices"
    )


def test_default_voices_dir_falls_back_to_home(monkeypatch):
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    monkeypatch.setenv("HOME", "/home/tester")
    assert speech_reader.default_voices_dir() == Path(
        "/home/tester/.local/share/voice-assistant/voices"
    )


def test_ensure_voice_model_downloads_when_missing(monkeypatch, tmp_path):
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        # Simulate the download producing the model file.
        (tmp_path / "en_US-amy-medium.onnx").write_text("model")

    monkeypatch.setattr(speech_reader.subprocess, "run", fake_run)
    path = speech_reader.ensure_voice_model("en_US-amy-medium", tmp_path)

    assert path == tmp_path / "en_US-amy-medium.onnx"
    assert path.exists()
    assert len(calls) == 1
    assert "piper.download_voices" in calls[0]
    assert "en_US-amy-medium" in calls[0]
    assert str(tmp_path) in calls[0]


def test_ensure_voice_model_skips_download_when_present(monkeypatch, tmp_path):
    (tmp_path / "en_US-amy-medium.onnx").write_text("already here")

    def fail_run(args, **kwargs):
        raise AssertionError("should not download when model exists")

    monkeypatch.setattr(speech_reader.subprocess, "run", fail_run)
    path = speech_reader.ensure_voice_model("en_US-amy-medium", tmp_path)
    assert path == tmp_path / "en_US-amy-medium.onnx"
