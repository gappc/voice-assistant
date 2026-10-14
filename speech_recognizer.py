"""Local speech-to-text ("dictation") via parakeet.cpp.

Runs NVIDIA Parakeet TDT 0.6B v3 (25 European languages, detected per
recording) through parakeet.cpp's prebuilt C library, loaded with ctypes.
Self-contained: knows nothing about evdev or the tray. The owning app records
audio elsewhere and hands it to transcribe().
"""

import ctypes
import hashlib
import os
import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

SAMPLERATE = 16000


def default_stt_dir():
    """Directory where the recognizer library and model are cached (XDG data dir)."""
    base = os.environ.get("XDG_DATA_HOME") or os.path.join(
        os.path.expanduser("~"), ".local", "share"
    )
    return Path(base) / "voice-assistant" / "stt"


# --- Library and model provisioning ------------------------------------

PARAKEET_VERSION = "v0.6.1"
_RELEASE_URL = "https://github.com/mudler/parakeet.cpp/releases/download/" + PARAKEET_VERSION

# The Vulkan build runs on the GPU (here the Radeon iGPU) and falls back to
# the CPU by itself when no Vulkan device is found; PARAKEET_DEVICE=cpu forces
# the CPU. The CPU build is only for systems without the Vulkan loader
# (libvulkan.so.1) at all, where the Vulkan build cannot even be loaded.
LIBRARY_SOURCES = {
    "vulkan": (
        f"{_RELEASE_URL}/parakeet-{PARAKEET_VERSION}-lib-linux-vulkan-x64.tar.gz",
        "52355c4ab9755c1e7326f6e90c365f6ede4597185917abcb307ed1efaa66b9c3",
    ),
    "cpu": (
        f"{_RELEASE_URL}/parakeet-{PARAKEET_VERSION}-lib-linux-cpu-x64.tar.gz",
        "5a2b439e83d1b6ab66c55bcfc48751ee48e53727d5b3ea33a6bc0e8626f1397c",
    ),
}

# q4_k transcribed our test recordings identically to q8_0, in ~250 MB less memory.
MODEL_FILE = "tdt-0.6b-v3-q4_k.gguf"
MODEL_URL = (
    "https://huggingface.co/mudler/parakeet-cpp-gguf/resolve/"
    f"741158ae71e64ef5c89385862c18f777d07a97a1/{MODEL_FILE}"
)
MODEL_SHA256 = "993d73feb4206dadda865ab25bd64b50c48dc4d013c3bf6126a721f28b1d5ee8"


def _download(url, dest, sha256):
    """Download `url` to `dest`, refusing a file whose SHA-256 doesn't match:
    the library is native code we load into the process."""
    print(f"[stt] downloading {url} ...")
    tmp_path = dest.with_name(dest.name + ".tmp")
    try:
        urllib.request.urlretrieve(url, tmp_path)
        digest = hashlib.sha256()
        with open(tmp_path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                digest.update(block)
        if digest.hexdigest() != sha256:
            raise RuntimeError(f"checksum mismatch for {url}")
        tmp_path.rename(dest)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def ensure_library(variant, stt_dir):
    """Return the path of libparakeet.so for `variant`, downloading if missing."""
    lib_dir = Path(stt_dir) / f"parakeet-{PARAKEET_VERSION}-{variant}"
    lib_path = lib_dir / "libparakeet.so"
    if lib_path.exists():
        return lib_path
    lib_dir.mkdir(parents=True, exist_ok=True)
    url, sha256 = LIBRARY_SOURCES[variant]
    archive = lib_dir / "release.tar.gz"
    _download(url, archive, sha256)
    try:
        with tarfile.open(archive) as tar:
            member = next(m for m in tar.getmembers() if m.name.endswith("/libparakeet.so"))
            tmp_path = lib_path.with_name(lib_path.name + ".tmp")
            tmp_path.write_bytes(tar.extractfile(member).read())
            tmp_path.rename(lib_path)
    finally:
        archive.unlink()
    return lib_path


def ensure_model(stt_dir):
    """Return the path of the Parakeet GGUF model, downloading if missing."""
    model_path = Path(stt_dir) / MODEL_FILE
    if not model_path.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        _download(MODEL_URL, model_path, MODEL_SHA256)
    return model_path


def _declare(lib):
    """Set ctypes signatures for the parts of parakeet_capi.h we call."""
    lib.parakeet_capi_load.restype = ctypes.c_void_p
    lib.parakeet_capi_load.argtypes = [ctypes.c_char_p]
    lib.parakeet_capi_load_error.restype = ctypes.c_char_p
    lib.parakeet_capi_transcribe_pcm.restype = ctypes.c_void_p
    lib.parakeet_capi_transcribe_pcm.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    lib.parakeet_capi_last_error.restype = ctypes.c_char_p
    lib.parakeet_capi_last_error.argtypes = [ctypes.c_void_p]
    lib.parakeet_capi_free_string.argtypes = [ctypes.c_void_p]
    return lib


def load_library(stt_dir):
    """Load the Vulkan build, or the CPU build if the Vulkan one can't load."""
    for variant in LIBRARY_SOURCES:
        path = ensure_library(variant, stt_dir)
        try:
            return _declare(ctypes.CDLL(str(path)))
        except OSError as exc:
            print(f"[stt] cannot load the {variant} build: {exc}", file=sys.stderr)
    raise RuntimeError("no usable parakeet.cpp library")


# --- Recognizer --------------------------------------------------------

class SpeechRecognizer:
    def __init__(self, stt_dir=None):
        stt_dir = Path(stt_dir) if stt_dir else default_stt_dir()
        self._lib = load_library(stt_dir)
        model_path = ensure_model(stt_dir)
        self._ctx = self._lib.parakeet_capi_load(str(model_path).encode())
        if not self._ctx:
            raise RuntimeError(
                f"failed to load {model_path}: {self._lib.parakeet_capi_load_error().decode()}"
            )
        # The first call sets up the GPU pipelines (~0.3 s); pay that at
        # startup rather than on the first dictation.
        self.transcribe(np.zeros(SAMPLERATE, dtype=np.float32))

    def transcribe(self, audio):
        """Transcribe mono float32 PCM at 16 kHz. Returns "" for no speech."""
        if len(audio) == 0:
            return ""
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        result = self._lib.parakeet_capi_transcribe_pcm(
            self._ctx, audio.ctypes.data, len(audio), SAMPLERATE, 0
        )
        if not result:
            error = self._lib.parakeet_capi_last_error(self._ctx).decode()
            raise RuntimeError(f"transcription failed: {error}")
        try:
            return ctypes.string_at(result).decode().strip()
        finally:
            self._lib.parakeet_capi_free_string(result)
