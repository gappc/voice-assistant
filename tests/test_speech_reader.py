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


from types import SimpleNamespace

import numpy as np


class _FakeStream:
    """Records writes; used to stand in for sd.OutputStream."""

    def __init__(self, sink):
        self.sink = sink

    def start(self):
        self.sink["started"] = True

    def write(self, data):
        self.sink["written"].append(len(data))

    def stop(self):
        self.sink["stopped"] = True

    def abort(self):
        self.sink["aborted"] = True

    def close(self):
        self.sink["closed"] = True


def _make_reader():
    return speech_reader.SpeechReader("en_US-amy-medium", speed=1.0)


def test_toggle_when_idle_calls_start(monkeypatch):
    r = _make_reader()
    events = []
    monkeypatch.setattr(r, "start", lambda t: events.append(("start", t)))
    monkeypatch.setattr(r, "stop", lambda: events.append(("stop",)))
    r._reading = False
    r.toggle("hello")
    assert events == [("start", "hello")]


def test_toggle_when_reading_calls_stop(monkeypatch):
    r = _make_reader()
    events = []
    monkeypatch.setattr(r, "start", lambda t: events.append(("start", t)))
    monkeypatch.setattr(r, "stop", lambda: events.append(("stop",)))
    r._reading = True
    r.toggle("ignored")
    assert events == [("stop",)]


def test_toggle_idle_with_no_text_does_nothing(monkeypatch):
    r = _make_reader()
    events = []
    monkeypatch.setattr(r, "start", lambda t: events.append(("start", t)))
    r._reading = False
    r.toggle(None)
    assert events == []


def test_play_chunks_writes_all_when_not_stopped(monkeypatch):
    r = _make_reader()
    sink = {"written": []}
    monkeypatch.setattr(
        speech_reader.sd, "OutputStream", lambda **kw: _FakeStream(sink)
    )
    r._stop.clear()
    samples = np.zeros(5000, dtype=np.int16)
    r._play_chunks(iter([(samples, 22050)]))
    # 5000 frames written in BLOCK-sized pieces; total equals 5000.
    assert sum(sink["written"]) == 5000
    assert sink.get("started") is True


def test_play_chunks_stops_early_when_flag_set(monkeypatch):
    r = _make_reader()
    sink = {"written": []}
    monkeypatch.setattr(
        speech_reader.sd, "OutputStream", lambda **kw: _FakeStream(sink)
    )
    r._stop.set()  # already asked to stop before any block is written
    r._play_chunks(iter([(np.zeros(9000, dtype=np.int16), 22050)]))
    assert sink["written"] == []
    assert sink.get("aborted") is True


def test_start_then_wait_resets_state_and_notifies(monkeypatch):
    states = []
    r = speech_reader.SpeechReader(
        "en_US-amy-medium", on_state_change=lambda: states.append(True)
    )
    # Replace real synthesis/playback with instant no-ops.
    monkeypatch.setattr(r, "_iter_chunks", lambda text: iter([]))
    monkeypatch.setattr(r, "_play_chunks", lambda chunks: None)
    r.start("hello")
    r.wait()
    assert r.is_reading is False
    assert states  # on_state_change fired at least once
