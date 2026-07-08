# Read-Aloud (Text-to-Speech) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local "read aloud" feature that speaks the highlighted selection (or clipboard) via Piper TTS, toggled by the Copilot key, integrated into the existing PySide6 tray app.

**Architecture:** A new self-contained `speech_reader.py` module owns the Piper voice, a playback worker thread, and a stop flag; it streams synthesis per sentence to a `sounddevice` output stream for fast start and prompt stop. A tiny `text_source.py` acquires the text to read (primary selection → clipboard). `voice_assistant.py` wires these in: it routes the `KEY_F23` tap to the reader, stops reading when dictation starts, and adds Output/Voice/Speed tray submenus plus a `Reading…` status.

**Tech Stack:** Python 3.10+, `piper-tts` (ohf-voice/piper1-gpl fork), `sounddevice`, `numpy`, `evdev`, `PySide6`, `wl-clipboard`; tests with `pytest`; `uv` for dependency management.

## Global Constraints

- Fully local/offline at runtime; network only for one-time voice download (matches Whisper's first-run download). — verbatim from spec
- No new system (apt) packages: `piper-tts` bundles its own espeak-ng phonemizer. — verbatim from spec
- Read-aloud failures must fail soft and never crash the dictation side. — verbatim from spec
- Trigger key: Copilot key, detected via `KEY_F23` (code 193); ignore the accompanying Meta/Shift. — verbatim from spec
- Default voice: `en_US-amy-medium`, auto-downloaded on first run. — verbatim from spec
- Voice models cached under `~/.local/share/voice-assistant/voices/` (XDG data dir). — verbatim from spec
- Speed presets map to Piper `length_scale`: Slow 1.3 / Normal 1.0 / Fast 0.8 (higher `length_scale` = slower). — verbatim from spec
- All work happens on the `read-aloud-tts` branch. Commit after every task.
- Run tests with `uv run pytest`. Run the app with `uv run voice_assistant.py`.

## File Structure

- **Create `text_source.py`** — one function `get_text_to_read()`: primary selection with clipboard fallback via `wl-paste`. No other responsibility.
- **Create `speech_reader.py`** — `default_voices_dir()`, `ensure_voice_model()`, and the `SpeechReader` class (voice loading, streaming playback, toggle/stop state machine, output-device/voice/speed setters). Knows nothing about evdev or the tray.
- **Create `tests/test_text_source.py`** — unit tests for the fallback order (mock `subprocess`).
- **Create `tests/test_speech_reader.py`** — unit tests for `ensure_voice_model` (mock `subprocess`/filesystem) and the `SpeechReader` toggle/stop/playback logic (mock the voice + output stream).
- **Modify `voice_assistant.py`** — constants, construct + load the reader, `KEY_F23` routing, stop-on-dictation, `get_output_devices`/`set_output_device`, beep device routing, tray submenus + status/icon, `--test-read` flag.
- **Modify `pyproject.toml`** — add `piper-tts` dependency and `pytest` dev group.
- **Modify `.gitignore`** — ignore a local `voices/` cache if present.
- **Modify `README.md`** — document the read-aloud key, feature, and voice storage.

---

### Task 1: Project setup — dependencies, dev tooling, gitignore

**Files:**
- Modify: `pyproject.toml`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: nothing.
- Produces: `piper` importable at runtime; `pytest` available via `uv run pytest`.

- [ ] **Step 1: Add the runtime dependency and dev group to `pyproject.toml`**

Edit the `dependencies` list to add `piper-tts`, and add a new `[dependency-groups]` table. Result:

```toml
[project]
name = "voice-assistant"
version = "0.1.0"
description = "Local voice assistant with push-to-talk and Whisper transcription"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "faster-whisper",
    "sounddevice",
    "numpy",
    "evdev",
    "PySide6-Essentials",
    "piper-tts",
]

[dependency-groups]
dev = ["pytest"]

[project.scripts]
voice-assistant = "voice_assistant:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

- [ ] **Step 2: Add voice cache to `.gitignore`**

Append these lines to `.gitignore`:

```gitignore
# Local Piper voice model cache (if ever placed in-tree)
voices/
```

- [ ] **Step 3: Sync and verify both packages resolve**

Run: `uv run python -c "import piper, pytest; print('ok')"`
Expected: `uv` installs the new deps, then prints `ok`.

- [ ] **Step 4: Verify the Piper voice-download module is runnable**

Run: `uv run python -m piper.download_voices --help`
Expected: help text listing options including `--data-dir` (confirms the download entry point exists). Exit code 0.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock .gitignore
git commit -m "build: add piper-tts dependency and pytest dev group"
```

---

### Task 2: `text_source.py` — acquire text to read

**Files:**
- Create: `text_source.py`
- Test: `tests/test_text_source.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `get_text_to_read() -> str | None` — returns the primary selection, else the clipboard, else `None` when both are empty/unavailable. Whitespace-only counts as empty.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_text_source.py`:

```python
from types import SimpleNamespace

import text_source


def test_returns_primary_selection(monkeypatch):
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(stdout="hello world")

    monkeypatch.setattr(text_source.subprocess, "run", fake_run)
    assert text_source.get_text_to_read() == "hello world"
    # Clipboard must NOT be queried when the selection has text.
    assert calls == [["wl-paste", "--primary", "--no-newline"]]


def test_falls_back_to_clipboard_when_selection_blank(monkeypatch):
    def fake_run(args, **kwargs):
        if "--primary" in args:
            return SimpleNamespace(stdout="   \n")  # whitespace only
        return SimpleNamespace(stdout="clipboard text")

    monkeypatch.setattr(text_source.subprocess, "run", fake_run)
    assert text_source.get_text_to_read() == "clipboard text"


def test_returns_none_when_both_empty(monkeypatch):
    monkeypatch.setattr(
        text_source.subprocess, "run", lambda args, **kw: SimpleNamespace(stdout="")
    )
    assert text_source.get_text_to_read() is None


def test_selection_error_falls_back_to_clipboard(monkeypatch):
    import subprocess as sp

    def fake_run(args, **kwargs):
        if "--primary" in args:
            raise sp.CalledProcessError(1, args)
        return SimpleNamespace(stdout="from clipboard")

    monkeypatch.setattr(text_source.subprocess, "run", fake_run)
    assert text_source.get_text_to_read() == "from clipboard"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_text_source.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'text_source'`.

- [ ] **Step 3: Write the implementation**

Create `text_source.py`:

```python
"""Acquire the text the user wants read aloud.

Prefers the Wayland primary selection (highlighted text); falls back to the
clipboard. Returns None when there is nothing to read.
"""

import subprocess


def _paste(args):
    """Run a wl-paste variant, returning stdout or '' on any failure."""
    try:
        result = subprocess.run(args, capture_output=True, text=True, check=True)
        return result.stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def get_text_to_read():
    """Return the primary selection, else the clipboard, else None."""
    text = _paste(["wl-paste", "--primary", "--no-newline"]).strip()
    if not text:
        text = _paste(["wl-paste", "--no-newline"]).strip()
    return text or None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_text_source.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add text_source.py tests/test_text_source.py
git commit -m "feat: add text_source.get_text_to_read (selection with clipboard fallback)"
```

---

### Task 3: Voice-model resolution and download

**Files:**
- Create: `speech_reader.py` (partial — module-level helpers only)
- Test: `tests/test_speech_reader.py` (partial)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `default_voices_dir() -> pathlib.Path` — `$XDG_DATA_HOME/voice-assistant/voices` (or `~/.local/share/...`).
  - `ensure_voice_model(name: str, voices_dir) -> pathlib.Path` — returns the path to `<name>.onnx`, downloading it via `python -m piper.download_voices` only if missing.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_speech_reader.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_speech_reader.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'speech_reader'`.

- [ ] **Step 3: Write the module helpers**

Create `speech_reader.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_speech_reader.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add speech_reader.py tests/test_speech_reader.py
git commit -m "feat: add Piper voice-model resolution and on-demand download"
```

---

### Task 4: `SpeechReader` — streaming playback + toggle/stop state machine

**Files:**
- Modify: `speech_reader.py` (add the `SpeechReader` class)
- Test: `tests/test_speech_reader.py` (add class tests)

**Interfaces:**
- Consumes: `default_voices_dir`, `ensure_voice_model` (Task 3); `PiperVoice`, `SynthesisConfig`, `sounddevice`, `numpy`.
- Produces the `SpeechReader` class:
  - `SpeechReader(voice_name, speed=1.0, output_device=None, voices_dir=None, on_state_change=None)`
  - `load() -> None` — resolve/download and load the voice into `self.voice`.
  - `start(text: str) -> None` — begin reading (stops any current read first).
  - `stop() -> None` — stop the current read; joins the worker.
  - `toggle(text: str) -> None` — stop if reading, else start if `text` is truthy.
  - `wait() -> None` — block until the current read finishes (used by `--test-read`).
  - `set_voice(name) -> None`, `set_speed(length_scale) -> None`, `set_output_device(index) -> None`.
  - `is_reading -> bool` property.
  - Class attribute `BLOCK = 2048` (frames per write, bounds stop latency).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_speech_reader.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_speech_reader.py -v`
Expected: FAIL — `AttributeError: module 'speech_reader' has no attribute 'SpeechReader'`.

- [ ] **Step 3: Implement the `SpeechReader` class**

Append to `speech_reader.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_speech_reader.py -v`
Expected: PASS (all tests in the file, including Task 3's, pass).

- [ ] **Step 5: Commit**

```bash
git add speech_reader.py tests/test_speech_reader.py
git commit -m "feat: add SpeechReader with streaming playback and toggle/stop"
```

---

### Task 5: `--test-read` flag — first audible end-to-end verification

**Files:**
- Modify: `voice_assistant.py` (imports, constants, `--test-read` in `main`, a `test_read` method)

**Interfaces:**
- Consumes: `SpeechReader` (Task 4).
- Produces: `voice_assistant.py` module-level constants `DEFAULT_VOICE`, `READ_SPEED`, `READ_KEY_CODE`, `VOICE_PRESETS`, `SPEED_PRESETS`; and CLI flag `--test-read TEXT` that speaks TEXT and exits.

- [ ] **Step 1: Add the import and constants**

In `voice_assistant.py`, add to the imports (after the existing `from evdev import ...` line):

```python
from speech_reader import SpeechReader
from text_source import get_text_to_read
```

In the `# --- Configuration ---` block, after `KEYBOARD_LAYOUT = "de" ...`, add:

```python
READ_KEY_CODE = ecodes.KEY_F23  # Copilot key emits Meta+Shift+F23; F23 is the tell
DEFAULT_VOICE = "en_US-amy-medium"
READ_SPEED = 1.0  # Piper length_scale (1.0 = normal; >1 slower, <1 faster)
VOICE_PRESETS = [
    "en_US-amy-medium",
    "en_US-ryan-medium",
    "en_GB-alan-medium",
    "en_GB-alba-medium",
]
SPEED_PRESETS = {"Slow": 1.3, "Normal": 1.0, "Fast": 0.8}
```

- [ ] **Step 2: Add the `test_read` method**

In `voice_assistant.py`, add this method to `VoiceAssistant` (next to `test_injection`):

```python
    def test_read(self, text):
        """Speak the given text with the default voice and exit (no hotkeys/tray)."""
        reader = SpeechReader(voice_name=DEFAULT_VOICE, speed=READ_SPEED)
        reader.load()
        print(f"Speaking: {text}")
        reader.start(text)
        reader.wait()
        print("Done.")
```

- [ ] **Step 3: Wire the flag in `main`**

In `main()`, add the argument (after the `--test-injection` argument):

```python
    parser.add_argument("--test-read", type=str, metavar="TEXT",
                        help="Speak the given text and exit")
```

And extend the dispatch block:

```python
    if args.test_injection:
        assistant.test_injection()
    elif args.test_read:
        assistant.test_read(args.test_read)
    else:
        assistant.run()
```

- [ ] **Step 4: Verify it speaks (manual, end-to-end)**

Run: `uv run voice_assistant.py --test-read "Hello, this is the local voice assistant reading aloud."`
Expected: on first run it downloads the Amy voice (progress printed), then you **hear** the sentence spoken, then `Done.` prints and the process exits.

- [ ] **Step 5: Commit**

```bash
git add voice_assistant.py
git commit -m "feat: add --test-read CLI flag for audible TTS verification"
```

---

### Task 6: Hotkey wiring — F23 toggle, stop-on-dictation, output-device plumbing

**Files:**
- Modify: `voice_assistant.py` (`__init__`, `run`, `listen_to_device`, `start_recording`, `play_beep`; add `on_read_key`, `get_output_devices`, `set_output_device`)
- Test: `tests/test_voice_assistant.py` (new — `get_output_devices` filtering)

**Interfaces:**
- Consumes: `SpeechReader`, `get_text_to_read`, the constants from Task 5.
- Produces:
  - `VoiceAssistant.reader: SpeechReader` and `VoiceAssistant.output_device`.
  - `on_read_key()` — routed from `KEY_F23` key-down.
  - `get_output_devices() -> list[dict]` — `[{'index', 'name'}]` for output-capable devices.
  - `set_output_device(index) -> None` — updates playback + beeps.

- [ ] **Step 1: Write the failing test for `get_output_devices`**

Create `tests/test_voice_assistant.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_voice_assistant.py -v`
Expected: FAIL — `AttributeError: 'VoiceAssistant' object has no attribute 'get_output_devices'`.

- [ ] **Step 3: Construct + load the reader in `__init__`**

In `VoiceAssistant.__init__`, after the `self.ui_updater ...` lines at the end, add:

```python
        # Read-aloud (TTS)
        self.output_device = None
        self.reader = SpeechReader(
            voice_name=DEFAULT_VOICE,
            speed=READ_SPEED,
            output_device=self.output_device,
            on_state_change=self.update_ui,
        )
```

- [ ] **Step 4: Load the voice at startup**

In `run()`, after `self.set_input_device(self.current_device)` and before the `print(f"Assistant ready! ...")` line, add:

```python
        # Load the TTS voice (downloads on first run)
        self.reader.load()
```

- [ ] **Step 5: Route the F23 key and add `on_read_key`**

In `listen_to_device`, replace the inner `if event.code == TRIGGER_KEY_CODE:` block so it also handles the read key:

```python
                if event.type == ecodes.EV_KEY:
                    if event.code == TRIGGER_KEY_CODE:
                        if event.value == 1:  # Key down
                            self.start_recording(device.path)
                        elif event.value == 0:  # Key up
                            self.stop_recording(device.path)
                    elif event.code == READ_KEY_CODE and event.value == 1:
                        self.on_read_key()
```

Add the `on_read_key` method to `VoiceAssistant`:

```python
    def on_read_key(self):
        """Copilot key tapped: toggle read-aloud of the selection/clipboard."""
        if self.reader.is_reading:
            self.reader.stop()
        else:
            text = get_text_to_read()
            if text:
                print(f"\nReading aloud ({len(text)} chars)...")
                self.reader.start(text)
            else:
                print("[read] nothing to read (empty selection and clipboard)")
                self.play_beep(frequency=300, duration=0.15)
```

- [ ] **Step 6: Stop reading when dictation starts**

At the very top of `start_recording` (before `with self.lock:`), add:

```python
        self.reader.stop()  # reading yields to dictation
```

- [ ] **Step 7: Route beeps + playback to the selected output device**

Change `play_beep`'s `sd.play(...)` call to pass the device:

```python
            sd.play(tone.astype(np.float32), SAMPLERATE, device=self.output_device)
```

Add the two device methods to `VoiceAssistant`:

```python
    def get_output_devices(self):
        """Returns a list of available output devices."""
        devices = sd.query_devices()
        return [
            {"index": i, "name": d["name"]}
            for i, d in enumerate(devices)
            if d["max_output_channels"] > 0
        ]

    def set_output_device(self, index):
        """Sets the output device for read-aloud playback and beeps."""
        self.output_device = index
        self.reader.set_output_device(index)
        print(f"Output device set to index {index}")
```

- [ ] **Step 8: Run the test to verify it passes**

Run: `uv run pytest tests/test_voice_assistant.py -v`
Expected: PASS (1 passed).

- [ ] **Step 9: Verify the full test suite still passes**

Run: `uv run pytest -v`
Expected: PASS (all tests across all three test files).

- [ ] **Step 10: Verify the hotkey end-to-end (manual)**

Run: `uv run voice_assistant.py`
Then: highlight a sentence in any window and tap the Copilot key.
Expected: you hear it read aloud; tapping again stops it; holding Right Alt (dictation) while it reads stops the speech. Terminal prints `Reading aloud (N chars)...`.

- [ ] **Step 11: Commit**

```bash
git add voice_assistant.py tests/test_voice_assistant.py
git commit -m "feat: wire Copilot key to read-aloud, stop on dictation, output device"
```

---

### Task 7: Tray integration — status, icon tint, Output/Voice/Speed submenus

**Files:**
- Modify: `voice_assistant.py` (`_do_update_ui`, `create_tray_icon`, `run_tray`; add `refresh_output_menu`, `build_voice_menu`, `build_speed_menu`, handler slots)

**Interfaces:**
- Consumes: `VoiceAssistant.reader`, `get_output_devices`/`set_output_device`, `VOICE_PRESETS`, `SPEED_PRESETS`.
- Produces: tray submenus and a `Reading…` status; no new external interface.

- [ ] **Step 1: Reflect reading in the status line**

Replace the body of `_do_update_ui` with:

```python
    def _do_update_ui(self):
        if hasattr(self, 'status_action'):
            if self.is_recording:
                status = "Recording..."
            elif self.reader.is_reading:
                status = "Reading..."
            else:
                status = "Idle"
            self.status_action.setText(f"Status: {status}")
            self.tray.setIcon(self.create_tray_icon())
```

- [ ] **Step 2: Tint the icon green while reading**

In `create_tray_icon`, replace the `color = ...` line with:

```python
        if self.is_recording:
            color = QColor(255, 80, 80)   # red: recording
        elif self.reader.is_reading:
            color = QColor(80, 220, 120)  # green: reading
        else:
            color = QColor(100, 200, 255) # blue: idle
```

- [ ] **Step 3: Add the Output, Voice, and Speed submenus in `run_tray`**

In `run_tray`, after the `self.refresh_mic_menu()` call and before the following `self.tray_menu.addSeparator()`, add:

```python
        # Output device selection menu
        self.output_menu = self.tray_menu.addMenu("Output")
        self.refresh_output_menu()

        # Voice selection menu
        self.voice_menu = self.tray_menu.addMenu("Voice")
        self.build_voice_menu()

        # Reading speed menu
        self.speed_menu = self.tray_menu.addMenu("Reading Speed")
        self.build_speed_menu()
```

- [ ] **Step 4: Add the menu builders and handler slots**

Add these methods to `VoiceAssistant` (near `refresh_mic_menu`):

```python
    def refresh_output_menu(self):
        """Populates the output-device selection submenu."""
        self.output_menu.clear()
        devices = self.get_output_devices()
        group = QActionGroup(self.output_menu)
        for d in devices:
            action = QAction(d['name'], self.output_menu, checkable=True)
            if d['index'] == self.output_device:
                action.setChecked(True)
            action.triggered.connect(
                lambda checked, idx=d['index']: self.set_output_device(idx)
            )
            self.output_menu.addAction(action)
            group.addAction(action)

    def build_voice_menu(self):
        """Populates the voice-selection submenu from the presets."""
        self.voice_menu.clear()
        group = QActionGroup(self.voice_menu)
        for name in VOICE_PRESETS:
            action = QAction(name, self.voice_menu, checkable=True)
            if name == self.reader._voice_name:
                action.setChecked(True)
            action.triggered.connect(lambda checked, n=name: self.on_select_voice(n))
            self.voice_menu.addAction(action)
            group.addAction(action)

    def build_speed_menu(self):
        """Populates the reading-speed submenu."""
        self.speed_menu.clear()
        group = QActionGroup(self.speed_menu)
        for label, scale in SPEED_PRESETS.items():
            action = QAction(label, self.speed_menu, checkable=True)
            if abs(scale - READ_SPEED) < 1e-9:
                action.setChecked(True)
            action.triggered.connect(lambda checked, s=scale: self.reader.set_speed(s))
            self.speed_menu.addAction(action)
            group.addAction(action)

    def on_select_voice(self, name):
        """Switch the Piper voice (downloads on demand)."""
        try:
            print(f"Switching voice to {name} ...")
            self.reader.set_voice(name)
        except Exception as exc:
            print(f"[read] failed to switch voice to {name}: {exc}", file=sys.stderr)
            self.tray.showMessage(
                "Voice Assistant",
                f"Could not load voice '{name}': {exc}",
                QSystemTrayIcon.Warning,
            )
```

- [ ] **Step 5: Verify the suite still passes**

Run: `uv run pytest -v`
Expected: PASS (unchanged — no test regressions).

- [ ] **Step 6: Verify the tray (manual)**

Run: `uv run voice_assistant.py`
Then click the tray icon and confirm: the menu shows **Microphone**, **Output**, **Voice**, and **Reading Speed** submenus; selecting a different Output device routes the next read there; selecting a different Voice loads it (downloading if new) and the next read uses it; Slow/Fast change the pace; the status reads `Reading...` and the icon turns green while a read is in progress.

- [ ] **Step 7: Commit**

```bash
git add voice_assistant.py
git commit -m "feat: add Output/Voice/Speed tray submenus and reading status"
```

---

### Task 8: Documentation

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: nothing.
- Produces: user-facing docs for the read-aloud feature.

- [ ] **Step 1: Add a read-aloud feature bullet**

In `README.md`, under `## Features`, add after the Push-to-Talk bullet:

```markdown
- **Read Aloud (Text-to-Speech)**: Tap the **Copilot key** (right of AltGr) to have your highlighted selection (or clipboard) read aloud with a natural local voice via [Piper](https://github.com/OHF-Voice/piper1-gpl). Tap again to stop. Fully offline after a one-time voice download.
```

- [ ] **Step 2: Add a Read Aloud usage section**

In `README.md`, after the `## Usage` section, add:

```markdown
## Read Aloud

1. Highlight any text (or copy it with Ctrl+C).
2. Tap the **Copilot key** (located to the right of AltGr).
3. The text is read aloud in the selected voice. Tap the key again to stop.

The default voice is `en_US-amy-medium`, downloaded automatically on first use and
cached under `~/.local/share/voice-assistant/voices/`. Use the tray menu to switch
the **Output** device, **Voice** (Amy / Ryan / Alan / Alba), and **Reading Speed**
(Slow / Normal / Fast). Starting dictation (Right Alt) stops any in-progress reading.

Test speech without the hotkey:
```bash
uv run voice_assistant.py --test-read "Hello from the local voice assistant."
```
```

- [ ] **Step 3: Note the read-aloud key in Troubleshooting**

In `README.md`, under `## Troubleshooting`, add:

```markdown
- **Read-aloud does nothing**: The Copilot key emits `Meta+Shift+F23`; the assistant listens for `F23`. Confirm your key sends it with `uv run key_detector.py` (tap the Copilot key, look for `KEY_F23`). If nothing is selected and the clipboard is empty, you'll hear a short low beep instead.
```

- [ ] **Step 4: Verify the docs render sensibly**

Run: `uv run python -c "print(open('README.md').read()[:200])"`
Expected: prints the top of the README without error (sanity check that the file is intact).

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: document read-aloud feature, key, voices, and --test-read"
```

---

## Self-Review

**Spec coverage:**
- Read selection/clipboard aloud → Tasks 2, 5, 6. ✓
- Piper engine, streaming per sentence → Tasks 3, 4. ✓
- Toggle on Copilot key via `KEY_F23` → Task 6. ✓
- Interruptions (read key again; dictation start) → Task 6 (`on_read_key`, `start_recording`). ✓
- Default voice auto-download, XDG cache → Tasks 1, 3. ✓
- Tray: status, icon tint, Output/Voice/Speed submenus → Task 7. ✓
- Output device drives playback + beeps → Task 6. ✓
- Config constants → Task 5. ✓
- Error handling fails soft (empty text beep, download/playback errors) → Tasks 4, 6, 7. ✓
- Testing: unit (text source, ensure_voice_model, SpeechReader logic, get_output_devices) + manual `--test-read` → Tasks 2, 3, 4, 5, 6. ✓
- Docs → Task 8. ✓
- No task needed for out-of-scope items (highlighting, pause/resume, files/PDF, SSML, per-app voices). ✓

**Placeholder scan:** No TBD/TODO/"handle edge cases"; every code step shows complete code. ✓

**Type consistency:** `SpeechReader` methods (`load`, `start`, `stop`, `toggle`, `wait`, `set_voice`, `set_speed`, `set_output_device`, `is_reading`, `_iter_chunks`, `_play_chunks`, `BLOCK`) are used consistently across Tasks 4–7. `get_text_to_read`, `get_output_devices`, `set_output_device`, `on_read_key`, `default_voices_dir`, `ensure_voice_model` names match their definitions and call sites. Constants (`READ_KEY_CODE`, `DEFAULT_VOICE`, `READ_SPEED`, `VOICE_PRESETS`, `SPEED_PRESETS`) defined in Task 5, used in Tasks 6–7. ✓
