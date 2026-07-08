"""Local text-to-speech ("read aloud") via Piper.

Self-contained: knows nothing about evdev or the tray. The owning app fetches
text elsewhere and drives this via start()/stop()/toggle().
"""

import os
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import sounddevice as sd
from piper import PiperVoice, SynthesisConfig


def default_voices_dir():
    """Directory where Piper voice models are cached (XDG data dir)."""
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


class SpeechReader:
    """Reads text aloud with Piper, streaming per sentence for prompt stop."""

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
        self.voice = None
        self._reading = False
        self._stop = threading.Event()
        self._worker = None
        self._lock = threading.Lock()

    # --- lifecycle -----------------------------------------------------
    def load(self):
        """Resolve/download the voice model and load it (call once at startup)."""
        onnx_path = ensure_voice_model(self._voice_name, self._voices_dir)
        print(f"[read] loading voice '{self._voice_name}' ...")
        self.voice = PiperVoice.load(str(onnx_path))

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
        self.stop()
        self._voice_name = name
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
        syn_config = SynthesisConfig(length_scale=self._speed)
        for chunk in self.voice.synthesize(text, syn_config=syn_config):
            samples = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
            yield samples, chunk.sample_rate

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
