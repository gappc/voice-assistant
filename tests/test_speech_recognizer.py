import ctypes
import hashlib
import io
import tarfile

import numpy as np
import pytest

import speech_recognizer


def _serve(tmp_path, name, data):
    """Write `data` to a file and return (file:// URL, its SHA-256)."""
    path = tmp_path / "remote" / name
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(data)
    return path.as_uri(), hashlib.sha256(data).hexdigest()


def _release_tarball(lib_bytes):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in (("pkg/README.md", b"readme"), ("pkg/libparakeet.so", lib_bytes)):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_download_rejects_checksum_mismatch_and_leaves_nothing(tmp_path):
    url, _ = _serve(tmp_path, "model.gguf", b"tampered")
    dest = tmp_path / "model.gguf"
    with pytest.raises(RuntimeError, match="checksum"):
        speech_recognizer._download(url, dest, "0" * 64)
    assert not dest.exists()
    assert not dest.with_name("model.gguf.tmp").exists()


def test_ensure_model_downloads_once(tmp_path, monkeypatch):
    url, sha = _serve(tmp_path, "m.gguf", b"weights")
    monkeypatch.setattr(speech_recognizer, "MODEL_URL", url)
    monkeypatch.setattr(speech_recognizer, "MODEL_SHA256", sha)
    path = speech_recognizer.ensure_model(tmp_path / "stt")
    assert path.read_bytes() == b"weights"

    monkeypatch.setattr(speech_recognizer, "MODEL_URL", "file:///nonexistent")
    assert speech_recognizer.ensure_model(tmp_path / "stt") == path  # cached, no refetch


def test_ensure_library_extracts_only_the_shared_library(tmp_path, monkeypatch):
    url, sha = _serve(tmp_path, "lib.tar.gz", _release_tarball(b"\x7fELF fake"))
    monkeypatch.setitem(speech_recognizer.LIBRARY_SOURCES, "vulkan", (url, sha))
    path = speech_recognizer.ensure_library("vulkan", tmp_path / "stt")
    assert path.read_bytes() == b"\x7fELF fake"
    assert sorted(p.name for p in path.parent.iterdir()) == ["libparakeet.so"]


def test_load_library_falls_back_to_cpu_build_without_vulkan_loader(tmp_path, monkeypatch):
    monkeypatch.setattr(speech_recognizer, "ensure_library", lambda v, d: tmp_path / v / "libparakeet.so")

    def fake_cdll(path):
        if "/vulkan/" in path:
            raise OSError("libvulkan.so.1: cannot open shared object file")
        return _FakeLib()

    monkeypatch.setattr(speech_recognizer.ctypes, "CDLL", fake_cdll)
    assert isinstance(speech_recognizer.load_library(tmp_path), _FakeLib)


class _FakeFn:
    def __init__(self, impl):
        self.impl = impl

    def __call__(self, *args):
        return self.impl(*args)


class _FakeLib:
    """Stands in for the ctypes library: returns real C strings so the
    recognizer's string_at/free path is exercised."""

    def __init__(self, text="Hallo Welt ", fail=False):
        self.calls = []
        self.freed = []
        self._buffers = []
        self._text, self._fail = text, fail
        self.parakeet_capi_load = _FakeFn(lambda path: 1)
        self.parakeet_capi_load_error = _FakeFn(lambda: b"")
        self.parakeet_capi_transcribe_pcm = _FakeFn(self._transcribe)
        self.parakeet_capi_last_error = _FakeFn(lambda ctx: b"bad audio")
        self.parakeet_capi_free_string = _FakeFn(self.freed.append)

    def _transcribe(self, ctx, data, n, rate, decoder):
        self.calls.append((n, rate))
        if self._fail:
            return None
        buf = ctypes.create_string_buffer(self._text.encode())
        self._buffers.append(buf)
        return ctypes.addressof(buf)


def _recognizer(lib):
    rec = speech_recognizer.SpeechRecognizer.__new__(speech_recognizer.SpeechRecognizer)
    rec._lib, rec._ctx = lib, 1
    return rec


def test_transcribe_returns_stripped_text_and_frees_it():
    lib = _FakeLib()
    rec = _recognizer(lib)
    assert rec.transcribe(np.zeros(1600, dtype=np.float64)) == "Hallo Welt"
    assert lib.calls == [(1600, 16000)]
    assert len(lib.freed) == 1


def test_transcribe_empty_audio_skips_the_library():
    lib = _FakeLib()
    assert _recognizer(lib).transcribe(np.zeros(0, dtype=np.float32)) == ""
    assert lib.calls == []


def test_transcribe_failure_raises_with_library_error():
    with pytest.raises(RuntimeError, match="bad audio"):
        _recognizer(_FakeLib(fail=True)).transcribe(np.zeros(1600, dtype=np.float32))


def test_init_warms_up_and_raises_on_load_failure(tmp_path, monkeypatch):
    lib = _FakeLib()
    monkeypatch.setattr(speech_recognizer, "load_library", lambda d: lib)
    monkeypatch.setattr(speech_recognizer, "ensure_model", lambda d: tmp_path / "m.gguf")
    speech_recognizer.SpeechRecognizer(tmp_path)
    assert lib.calls == [(16000, 16000)]

    lib.parakeet_capi_load = _FakeFn(lambda path: None)
    lib.parakeet_capi_load_error = _FakeFn(lambda: b"not a GGUF")
    with pytest.raises(RuntimeError, match="not a GGUF"):
        speech_recognizer.SpeechRecognizer(tmp_path)
