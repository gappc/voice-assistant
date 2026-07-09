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
            "url_model": "https://example.com/test.onnx",
            "url_voices": "https://example.com/test.npz",
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
    assert "https://example.com/test.onnx" == calls[0][0]
    assert "https://example.com/test.npz" == calls[1][0]


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
            "url_model": "https://example.com/test.onnx",
            "url_voices": "https://example.com/test.npz",
            "model_file": "test.onnx",
            "voices_file": "test.npz",
        }
    }
    monkeypatch.setattr(speech_reader, "KOKORO_VOICE_SOURCES", mock_sources)

    model_path, voices_path = speech_reader.ensure_kokoro_voice_files("test-voice", tmp_path)
    assert model_path == kokoro_dir / "test.onnx"
    assert voices_path == kokoro_dir / "test.npz"


def test_build_kokoro_bounds_threads_and_disables_spinning(monkeypatch):
    import kokoro_onnx

    captured = {}

    def fake_session(path, sess_options, providers):
        captured["path"] = path
        captured["opts"] = sess_options
        captured["providers"] = providers
        return object()

    monkeypatch.setattr(speech_reader.ort, "InferenceSession", fake_session)
    monkeypatch.setattr(
        kokoro_onnx.Kokoro, "from_session",
        classmethod(lambda cls, session, voices: ("kokoro", session, voices)),
    )

    result = speech_reader.build_kokoro(Path("m.onnx"), Path("v.bin"))

    assert result[0] == "kokoro"
    assert result[2] == "v.bin"
    assert captured["path"] == "m.onnx"
    assert captured["providers"] == ["CPUExecutionProvider"]
    opts = captured["opts"]
    assert opts.intra_op_num_threads == speech_reader.KOKORO_INTRA_OP_THREADS
    assert opts.inter_op_num_threads == 1
    # ORT exposes no getter for config entries in every build; check when it does.
    if hasattr(opts, "get_session_config_entry"):
        assert opts.get_session_config_entry("session.intra_op.allow_spinning") == "0"


def test_kokoro_engine_speed_translation(monkeypatch):
    import numpy as np

    class FakeKokoro:
        def __init__(self):
            self.created_calls = []

        def create(self, text, voice, speed, lang):
            self.created_calls.append((text, voice, speed, lang))
            return np.zeros(10, dtype=np.float32), 24000

    monkeypatch.setattr(speech_reader, "build_kokoro", lambda model, voices: FakeKokoro())

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
    lock = __import__("threading").Lock()

    class MockEngine:
        def load(self):
            pass

        def synthesize(self, text, speed):
            with lock:
                synthesized_sentences.append(text)
            yield AudioChunk(np.zeros(2048, dtype=np.int16), 22050)

    r = speech_reader.SpeechReader("en_US-amy-medium")
    mock_engine = MockEngine()
    monkeypatch.setattr(r, "_get_or_create_engine", lambda spec: mock_engine)

    # Let _play_chunks stop the reader immediately when it plays the first chunk
    def mock_play(chunks):
        for samples, rate in chunks:
            r.stop()  # set the stop event
            break

    monkeypatch.setattr(r, "_play_chunks", mock_play)

    r.start("Sentence one. Sentence two. Sentence three.")
    r.wait()

    # Playback stops at the first sentence. Prefetch may have speculatively
    # synthesized exactly one sentence ahead, but never two.
    with lock:
        assert synthesized_sentences[0] == "Sentence one."
        assert "Sentence three." not in synthesized_sentences
        assert len(synthesized_sentences) <= 2


def test_prefetch_synthesizes_next_sentence_during_playback(monkeypatch):
    import threading

    from speech_reader import AudioChunk
    import numpy as np

    second_synthesized = threading.Event()

    class MockEngine:
        def load(self):
            pass

        def synthesize(self, text, speed):
            if text == "Two.":
                second_synthesized.set()
            yield AudioChunk(np.zeros(2048, dtype=np.int16), 22050)

    r = speech_reader.SpeechReader("en_US-amy-medium")
    monkeypatch.setattr(r, "_get_or_create_engine", lambda spec: MockEngine())

    overlapped = []

    def mock_play(chunks):
        for i, (samples, rate) in enumerate(chunks):
            if i == 0:
                # While the first sentence is "playing", the second must already
                # be under synthesis on the producer thread.
                overlapped.append(second_synthesized.wait(timeout=5))

    monkeypatch.setattr(r, "_play_chunks", mock_play)
    r.start("One. Two.")
    r.wait()

    assert overlapped == [True]


def test_speech_reader_on_the_fly_updates(monkeypatch):
    import threading

    from speech_reader import AudioChunk
    import numpy as np
    synthesized = []
    lock = threading.Lock()
    second_prefetched = threading.Event()

    class MockEngine:
        def __init__(self, name):
            self.name = name

        def load(self):
            pass

        def synthesize(self, text, speed):
            with lock:
                synthesized.append((self.name, text, speed))
            if text == "Sentence two.":
                second_prefetched.set()
            yield AudioChunk(np.zeros(2048, dtype=np.int16), 22050)

    r = speech_reader.SpeechReader("en_US-amy-medium")
    monkeypatch.setattr(r, "_get_or_create_engine", lambda spec: MockEngine(spec.model))

    def mock_play(chunks):
        for i, (samples, rate) in enumerate(chunks):
            if i == 0:
                # Wait until sentence two has been prefetched under the OLD
                # settings, then change them: the stale audio must be redone.
                assert second_prefetched.wait(timeout=5)
                r.set_speed(1.5)
                r.set_voice("de_DE-thorsten-high")

    monkeypatch.setattr(r, "_play_chunks", mock_play)

    r.start("Sentence one. Sentence two.")
    r.wait()

    with lock:
        calls = list(synthesized)

    # Sentence one used the initial settings; sentence two was prefetched under
    # them, discarded, and re-synthesized with the updated voice and speed.
    assert calls[0] == ("en_US-amy-medium", "Sentence one.", 1.0)
    assert calls[1] == ("en_US-amy-medium", "Sentence two.", 1.0)
    assert calls[2] == ("de_DE-thorsten-high", "Sentence two.", 1.5)
    assert len(calls) == 3
