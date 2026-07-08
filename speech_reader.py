"""Local text-to-speech ("read aloud") via Piper and Kokoro.

Self-contained: knows nothing about evdev or the tray. The owning app fetches
text elsewhere and drives this via start()/stop()/toggle().
"""

import os
import re
import subprocess
import sys
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, NamedTuple, Protocol

import numpy as np
import sounddevice as sd
from piper import PiperVoice, SynthesisConfig


def default_voices_dir():
    """Directory where Piper/Kokoro voice models are cached (XDG data dir)."""
    base = os.environ.get("XDG_DATA_HOME") or os.path.join(
        os.path.expanduser("~"), ".local", "share"
    )
    return Path(base) / "voice-assistant" / "voices"


def ensure_voice_model(name, voices_dir):
    """Return the path to <name>.onnx, downloading it only if missing."""
    voices_dir = Path(voices_dir)
    voices_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = voices_dir / f"{name}.onnx"
    if not onnx_path.exists():
        print(f"[read] downloading voice '{name}' to {voices_dir} ...")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "piper.download_voices",
                name,
                "--data-dir",
                str(voices_dir),
            ],
            check=True,
        )
    return onnx_path


# --- Kokoro voice provisioning -----------------------------------------

KOKORO_VOICE_SOURCES = {
    "martin": {
        "repo": "Godelaune/Kokoro-82M-ONNX-German-Martin",
        "commit": "a1cba7fbf0e72fbae38f0a3a48ce0dc8e6077804",
        "model_file": "kokoro-martin.onnx",
        "voices_file": "voices-martin.npz",
    },
}


def ensure_kokoro_voice_files(model_name, voices_dir):
    """Return (model_path, voices_path) for a Kokoro voice, downloading if missing."""
    source = KOKORO_VOICE_SOURCES[model_name]
    kokoro_dir = Path(voices_dir) / "kokoro"
    kokoro_dir.mkdir(parents=True, exist_ok=True)
    model_path = kokoro_dir / source["model_file"]
    voices_path = kokoro_dir / source["voices_file"]
    commit = source["commit"]
    base_url = f"https://huggingface.co/{source['repo']}/resolve/{commit}"
    for path, filename in ((model_path, source["model_file"]),
                            (voices_path, source["voices_file"])):
        if not path.exists():
            print(f"[read] downloading {filename} for kokoro voice '{model_name}' ...")
            tmp_path = path.with_suffix(".tmp")
            try:
                urllib.request.urlretrieve(f"{base_url}/{filename}", tmp_path)
                tmp_path.rename(path)
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()
    return model_path, voices_path


# --- Sentence segmentation ---------------------------------------------

def split_sentences(text: str) -> list[str]:
    """Split text into sentences using simple regex."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


# --- Voice Specs and Catalog -------------------------------------------

@dataclass(frozen=True)
class VoiceSpec:
    engine: str  # "piper" | "kokoro"
    model: str   # piper voice name, or kokoro model identifier (e.g. "martin")
    voice_id: str | None = None  # kokoro-only: key into the voices npz (e.g. "dm_martin")
    lang: str | None = None      # kokoro-only: passed to create(text, lang=...)


VOICE_CATALOG = {
    "en_US-amy-medium":    VoiceSpec(engine="piper", model="en_US-amy-medium"),
    "en_US-ryan-medium":   VoiceSpec(engine="piper", model="en_US-ryan-medium"),
    "en_GB-alan-medium":   VoiceSpec(engine="piper", model="en_GB-alan-medium"),
    "en_GB-alba-medium":   VoiceSpec(engine="piper", model="en_GB-alba-medium"),
    "de_DE-thorsten-high": VoiceSpec(engine="piper", model="de_DE-thorsten-high"),
    "de_DE-mls-medium":    VoiceSpec(engine="piper", model="de_DE-mls-medium"),
    "kokoro-de-martin":    VoiceSpec(engine="kokoro", model="martin",
                                      voice_id="martin", lang="de"),
}

DEFAULT_VOICE = "en_US-amy-medium"


# --- TTSEngine Abstraction ---------------------------------------------

class AudioChunk(NamedTuple):
    samples: np.ndarray  # mono int16 PCM
    sample_rate: int


class TTSEngine(Protocol):
    def load(self) -> None:
        """Load model weights. Called once at startup or on voice switch."""
        ...

    def synthesize(self, text: str, speed: float) -> Iterator[AudioChunk]:
        """Synthesize `text` (already segmented to sentence granularity by the
        caller), yielding one or more audio chunks as they become available."""
        ...


class PiperEngine:
    def __init__(self, voice_name: str, voices_dir: Path):
        self.voice_name = voice_name
        self.voices_dir = voices_dir
        self.voice = None

    def load(self) -> None:
        if self.voice is None:
            onnx_path = ensure_voice_model(self.voice_name, self.voices_dir)
            print(f"[read] loading Piper voice '{self.voice_name}' ...")
            self.voice = PiperVoice.load(str(onnx_path))

    def synthesize(self, text: str, speed: float) -> Iterator[AudioChunk]:
        self.load()
        syn_config = SynthesisConfig(length_scale=speed)
        for chunk in self.voice.synthesize(text, syn_config=syn_config):
            samples = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
            yield AudioChunk(samples, chunk.sample_rate)


class KokoroEngine:
    def __init__(self, model_name: str, voice_id: str, lang: str, voices_dir: Path):
        self.model_name = model_name
        self.voice_id = voice_id
        self.lang = lang
        self.voices_dir = voices_dir
        self._kokoro = None

    def load(self) -> None:
        if self._kokoro is None:
            from kokoro_onnx import Kokoro
            model_path, voices_path = ensure_kokoro_voice_files(self.model_name, self.voices_dir)
            print(f"[read] loading Kokoro voice '{self.model_name}' ...")
            self._kokoro = Kokoro(str(model_path), str(voices_path))

    def synthesize(self, text: str, speed: float) -> Iterator[AudioChunk]:
        self.load()
        # Convert Piper's length_scale speed to Kokoro speed (reciprocal)
        kokoro_speed = 1.0 / speed
        kokoro_speed = max(0.5, min(kokoro_speed, 2.0))
        float32_samples, sample_rate = self._kokoro.create(
            text,
            voice=self.voice_id,
            speed=kokoro_speed,
            lang=self.lang,
        )
        clamped = np.clip(float32_samples, -1.0, 1.0)
        int16_samples = (clamped * 32767.0).astype(np.int16)
        yield AudioChunk(int16_samples, sample_rate)


class SpeechReader:
    """Reads text aloud with Piper or Kokoro, streaming per sentence for prompt stop."""

    BLOCK = 2048  # frames per write; bounds stop latency to a fraction of a second

    def __init__(
        self,
        voice_name,
        speed=1.0,
        output_device=None,
        voices_dir=None,
        on_state_change=None,
    ):
        self._voice_name = voice_name
        self._speed = speed
        self._output_device = output_device
        self._voices_dir = Path(voices_dir) if voices_dir else default_voices_dir()
        self._on_state_change = on_state_change
        self._reading = False
        self._stop = threading.Event()
        self._worker = None
        self._lock = threading.Lock()
        self._engines = {}
        self._active_engine = None

    # --- lifecycle -----------------------------------------------------
    def load(self):
        """Resolve/download the voice model and load it (call once at startup)."""
        spec = VOICE_CATALOG.get(self._voice_name, VOICE_CATALOG[DEFAULT_VOICE])
        self._active_engine = self._get_or_create_engine(spec)
        self._active_engine.load()

    def _get_or_create_engine(self, spec: VoiceSpec) -> TTSEngine:
        engine_key = (spec.engine, spec.model)
        if engine_key not in self._engines:
            if spec.engine == "piper":
                self._engines[engine_key] = PiperEngine(spec.model, self._voices_dir)
            elif spec.engine == "kokoro":
                self._engines[engine_key] = KokoroEngine(
                    spec.model, spec.voice_id, spec.lang, self._voices_dir
                )
            else:
                raise ValueError(f"Unknown engine: {spec.engine}")
        return self._engines[engine_key]

    @property
    def is_reading(self):
        return self._reading

    # --- control -------------------------------------------------------
    def start(self, text):
        if self._reading:
            self.stop()
        self._stop.clear()
        with self._lock:
            self._reading = True
        self._notify()
        
        # Ensure correct engine is loaded/active
        spec = VOICE_CATALOG.get(self._voice_name, VOICE_CATALOG[DEFAULT_VOICE])
        self._active_engine = self._get_or_create_engine(spec)
        
        self._worker = threading.Thread(
            target=self._read_worker, args=(text,), daemon=True
        )
        self._worker.start()

    def stop(self):
        self._stop.set()
        worker = self._worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=5)
        with self._lock:
            self._reading = False

    def toggle(self, text):
        if self._reading:
            self.stop()
        elif text:
            self.start(text)

    def wait(self):
        worker = self._worker
        if worker is not None:
            worker.join()

    def set_voice(self, name):
        self._voice_name = name
        if not self._reading:
            self.load()

    def set_speed(self, length_scale):
        self._speed = length_scale

    def set_output_device(self, index):
        self._output_device = index

    # --- internals -----------------------------------------------------
    def _notify(self):
        if self._on_state_change:
            self._on_state_change()

    def _iter_chunks(self, text):
        """Yield (int16 samples, sample_rate) per synthesized sentence."""
        sentences = split_sentences(text)
        for sentence in sentences:
            if self._stop.is_set():
                return
            spec = VOICE_CATALOG.get(self._voice_name, VOICE_CATALOG[DEFAULT_VOICE])
            self._active_engine = self._get_or_create_engine(spec)
            self._active_engine.load()
            for chunk in self._active_engine.synthesize(sentence, self._speed):
                if self._stop.is_set():
                    return
                yield chunk.samples, chunk.sample_rate

    def _play_chunks(self, chunks):
        """Write chunks to an output stream, honoring the stop flag between blocks."""
        stream = None
        try:
            for samples, sample_rate in chunks:
                if stream is None:
                    stream = sd.OutputStream(
                        samplerate=sample_rate,
                        channels=1,
                        dtype="int16",
                        device=self._output_device,
                    )
                    stream.start()
                for i in range(0, len(samples), self.BLOCK):
                    if self._stop.is_set():
                        stream.abort()
                        return
                    stream.write(samples[i : i + self.BLOCK])
            if stream is not None and not self._stop.is_set():
                stream.stop()
        finally:
            if stream is not None:
                stream.close()

    def _read_worker(self, text):
        try:
            self._play_chunks(self._iter_chunks(text))
        except Exception as exc:  # fail soft — never crash the dictation side
            print(f"[read] playback error: {exc}", file=sys.stderr)
        finally:
            with self._lock:
                self._reading = False
            self._notify()

