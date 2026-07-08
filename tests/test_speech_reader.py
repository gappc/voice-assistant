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


def test_split_sentences():
    from speech_reader import split_sentences
    assert split_sentences("This is a sentence. And another! Is it?") == [
        "This is a sentence.",
        "And another!",
        "Is it?",
    ]
    assert split_sentences("Hello world") == ["Hello world"]
    assert split_sentences("") == []
    assert split_sentences("  Spaces at ends.  ") == ["Spaces at ends."]
    # Verify the abbreviation known limitation
    assert split_sentences("Dr. Smith is here.") == ["Dr.", "Smith is here."]


def test_ensure_kokoro_voice_files(monkeypatch, tmp_path):
    calls = []

    def fake_urlretrieve(url, filename):
        calls.append((url, filename))
        # Simulate download producing the files
        Path(filename).write_text("dummy")

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlretrieve", fake_urlretrieve)

    # Use a mock config for KOKORO_VOICE_SOURCES so we don't depend on actual HF config in tests
    mock_sources = {
        "test-voice": {
            "repo": "test-repo/voice",
            "commit": "123456",
            "model_file": "test.onnx",
            "voices_file": "test.npz",
        }
    }
    monkeypatch.setattr(speech_reader, "KOKORO_VOICE_SOURCES", mock_sources)

    model_path, voices_path = speech_reader.ensure_kokoro_voice_files("test-voice", tmp_path)

    assert model_path == tmp_path / "kokoro" / "test.onnx"
    assert voices_path == tmp_path / "kokoro" / "test.npz"
    assert model_path.exists()
    assert voices_path.exists()
    assert len(calls) == 2
    assert "https://huggingface.co/test-repo/voice/resolve/123456/test.onnx" in calls[0][0]
    assert "https://huggingface.co/test-repo/voice/resolve/123456/test.npz" in calls[1][0]


def test_ensure_kokoro_voice_files_skips_when_present(monkeypatch, tmp_path):
    kokoro_dir = tmp_path / "kokoro"
    kokoro_dir.mkdir()
    (kokoro_dir / "test.onnx").write_text("already here")
    (kokoro_dir / "test.npz").write_text("already here")

    def fail_urlretrieve(url, filename):
        raise AssertionError("should not download when files exist")

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlretrieve", fail_urlretrieve)

    mock_sources = {
        "test-voice": {
            "repo": "test-repo/voice",
            "commit": "123456",
            "model_file": "test.onnx",
            "voices_file": "test.npz",
        }
    }
    monkeypatch.setattr(speech_reader, "KOKORO_VOICE_SOURCES", mock_sources)

    model_path, voices_path = speech_reader.ensure_kokoro_voice_files("test-voice", tmp_path)
    assert model_path == kokoro_dir / "test.onnx"
    assert voices_path == kokoro_dir / "test.npz"


def test_kokoro_engine_speed_translation(monkeypatch):
    import sys
    import types
    import numpy as np

    class FakeKokoro:
        def __init__(self, model_path, voices_path):
            self.created_calls = []

        def create(self, text, voice, speed, lang):
            self.created_calls.append((text, voice, speed, lang))
            return np.zeros(10, dtype=np.float32), 24000

    # Inject mock Kokoro class
    monkeypatch.setitem(sys.modules, "kokoro_onnx", types.ModuleType("kokoro_onnx"))
    import kokoro_onnx
    kokoro_onnx.Kokoro = FakeKokoro

    # Avoid downloading files
    monkeypatch.setattr(speech_reader, "ensure_kokoro_voice_files", lambda name, folder: (Path("model"), Path("voices")))

    # Create KokoroEngine
    engine = speech_reader.KokoroEngine(model_name="martin", voice_id="dm_martin", lang="de", voices_dir=Path("/dummy"))
    engine.load()

    # Normal speed (reciprocal of 1.0 is 1.0)
    list(engine.synthesize("test", 1.0))
    assert engine._kokoro.created_calls[-1][2] == 1.0

    # Slow speed (reciprocal of 1.3 is ~0.769)
    list(engine.synthesize("test", 1.3))
    assert abs(engine._kokoro.created_calls[-1][2] - 1.0 / 1.3) < 1e-5

    # Fast speed (reciprocal of 0.8 is 1.25)
    list(engine.synthesize("test", 0.8))
    assert engine._kokoro.created_calls[-1][2] == 1.25

    # Extreme slow (Piper length_scale 3.0 -> reciprocal 0.33 -> clamped to 0.5)
    list(engine.synthesize("test", 3.0))
    assert engine._kokoro.created_calls[-1][2] == 0.5

    # Extreme fast (Piper length_scale 0.2 -> reciprocal 5.0 -> clamped to 2.0)
    list(engine.synthesize("test", 0.2))
    assert engine._kokoro.created_calls[-1][2] == 2.0


def test_speech_reader_stops_between_sentences(monkeypatch):
    from speech_reader import AudioChunk
    import numpy as np
    synthesized_sentences = []

    class MockEngine:
        def load(self):
            pass

        def synthesize(self, text, speed):
            synthesized_sentences.append(text)
            yield AudioChunk(np.zeros(2048, dtype=np.int16), 22050)

    r = speech_reader.SpeechReader("en_US-amy-medium")
    # Set the active engine to MockEngine directly
    mock_engine = MockEngine()
    monkeypatch.setattr(r, "_get_or_create_engine", lambda spec: mock_engine)

    # Let _play_chunks stop the reader immediately when it plays the first chunk
    def mock_play(chunks):
        # We consume one chunk, then trigger a stop on the reader
        for samples, rate in chunks:
            r.stop()  # set the stop event
            break

    monkeypatch.setattr(r, "_play_chunks", mock_play)

    # We start with a multi-sentence text
    r.start("Sentence one. Sentence two.")
    r.wait()

    # Sentence one should have been synthesized, but sentence two should NOT be synthesized
    # because of the early stop check in the sentence loop of _iter_chunks
    assert synthesized_sentences == ["Sentence one."]
