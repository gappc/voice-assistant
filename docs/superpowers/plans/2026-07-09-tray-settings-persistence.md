# Tray Settings Persistence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tray-adjustable settings (input device, output device, voice, reading speed, paste mode) survive restarting `voice_assistant.py` instead of resetting to hardcoded defaults every launch.

**Architecture:** A new, dependency-free `tray_settings.py` module owns a `TraySettings` dataclass and `load()`/`save()` functions backed by a JSON file under `~/.config/voice-assistant/`. `voice_assistant.py` loads it once at startup (merging in the one competing CLI flag, `--device`) to seed initial state, and calls `save()` from each of the five tray-menu setters after applying a change. Devices are persisted by name, not index, because indices shift across reboots.

**Tech Stack:** Python 3.10+, stdlib `json`/`dataclasses`/`pathlib`; `pytest`.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-09-tray-settings-persistence-design.md`
- `tray_settings.py` must not import `VOICE_CATALOG`/`DEFAULT_VOICE` from `speech_reader.py` or anything from `voice_assistant.py` — keeps it circular-import-free and independently testable. Validation against `VOICE_CATALOG` happens in `voice_assistant.py`.
- Devices are stored by **name** in the settings file, never by index.
- Of the two existing CLI flags, only `--device` reaches the live tray session (`--voice` only affects the separate `--test-read` one-shot path) — it is the only setting with real CLI-vs-saved precedence to implement. `--device` wins over a saved value for that run; the file is not rewritten by a flag-only override.
- Startup (`VoiceAssistant.__init__`, and the input-device application in `run()`) must apply resolved values directly and must **never** go through the five save-triggering public setters — otherwise every launch would rewrite the file even with no user action, defeating the "CLI wins without rewriting" rule.
- Config file: `$XDG_CONFIG_HOME/voice-assistant/settings.json`, or `~/.config/voice-assistant/settings.json` if unset — mirrors `speech_reader.py`'s `default_voices_dir()` pattern for `XDG_DATA_HOME`.
- `save()` writes are atomic (temp file + rename), mirroring `ensure_kokoro_voice_files()`'s download pattern in `speech_reader.py`.
- Run tests with `.venv/bin/python -m pytest`. Baseline before any change: **24 passed**.

## File Structure

| File | Responsibility after this plan |
|---|---|
| `tray_settings.py` | `TraySettings` dataclass, `load()`/`save()`, `default_config_dir()`. Pure data, no Qt/sounddevice dependency. |
| `voice_assistant.py` | Adds `_resolve_device`/`_device_name` helpers, `_resolve_startup_settings`, `_save_settings`, and wires all five tray setters to persist. |
| `tests/test_tray_settings.py` | Round-trip, missing/corrupt-file defaults, wrong-shaped-field defaults, atomic-write safety. |
| `tests/test_voice_assistant.py` | Device resolution, startup precedence, and setters-persist tests, in addition to existing tests. |
| `README.md` | Documents that tray settings persist and where the file lives. |

---

### Task 1: `tray_settings.py` — the data layer

**Files:**
- Create: `tray_settings.py`
- Test: `tests/test_tray_settings.py`

**Interfaces:**
- Produces:
  - `default_config_dir() -> Path`
  - `TraySettings` frozen dataclass: `input_device: str | None = None`, `output_device: str | None = None`, `voice: str | None = None`, `speed: float | None = None`, `paste_with_shift: bool = True`
  - `load(path) -> TraySettings` — never raises; missing file, corrupt JSON, non-object JSON, or a wrong-shaped field all fall back to that field's default.
  - `save(settings: TraySettings, path) -> None` — atomic write; propagates I/O errors to the caller (callers decide whether to catch).

- [ ] **Step 1: Write the test file**

Create `tests/test_tray_settings.py`:

```python
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
    path.write_text(json.dumps({"speed": "fast", "paste_with_shift": "yes", "voice": 42}))
    settings = tray_settings.load(path)
    assert settings.speed is None
    assert settings.paste_with_shift is True
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_tray_settings.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tray_settings'`

- [ ] **Step 3: Implement `tray_settings.py`**

Create `tray_settings.py`:

```python
"""Persisted tray settings (voice, speed, devices, paste mode).

Deliberately independent of speech_reader.py / voice_assistant.py: no
VOICE_CATALOG or DEFAULT_VOICE import, so there is no circular import and this
module is trivially unit-testable on its own. Validation against VOICE_CATALOG
happens in voice_assistant.py, not here.
"""

import dataclasses
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path


def default_config_dir():
    """Directory where tray settings are stored (XDG config dir)."""
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(
        os.path.expanduser("~"), ".config"
    )
    return Path(base) / "voice-assistant"


@dataclass(frozen=True)
class TraySettings:
    input_device: str | None = None   # device name, not index (indices shift across reboots)
    output_device: str | None = None  # device name
    voice: str | None = None          # VOICE_CATALOG key
    speed: float | None = None
    paste_with_shift: bool = True


def load(path):
    """Best-effort load. Missing file, corrupt JSON, or a field of the wrong
    shape all fall back to that field's TraySettings() default — never raises."""
    path = Path(path)
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError:
        return TraySettings()
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[settings] failed to read {path}: {exc}", file=sys.stderr)
        return TraySettings()

    if not isinstance(raw, dict):
        print(f"[settings] {path} did not contain a JSON object; using defaults", file=sys.stderr)
        return TraySettings()

    return TraySettings(
        input_device=_str_or_default(raw.get("input_device")),
        output_device=_str_or_default(raw.get("output_device")),
        voice=_str_or_default(raw.get("voice")),
        speed=_float_or_default(raw.get("speed")),
        paste_with_shift=_bool_or_default(raw.get("paste_with_shift")),
    )


def _str_or_default(value):
    return value if isinstance(value, str) else None


def _float_or_default(value):
    if isinstance(value, bool):
        return None
    return float(value) if isinstance(value, (int, float)) else None


def _bool_or_default(value):
    return value if isinstance(value, bool) else True


def save(settings, path):
    """Atomic write: write to `<path>.tmp`, then rename over `path`, so a
    failure mid-write can't corrupt the previous settings file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    try:
        tmp_path.write_text(json.dumps(dataclasses.asdict(settings), indent=2))
        tmp_path.rename(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_tray_settings.py -v`
Expected: `8 passed`

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: `32 passed`

- [ ] **Step 6: Commit**

```bash
git add tray_settings.py tests/test_tray_settings.py
git commit -m "feat: add tray_settings.py for persisting tray state

New, dependency-free module: a TraySettings dataclass plus load()/save(),
backed by a JSON file under ~/.config/voice-assistant/. Deliberately has no
VOICE_CATALOG/DEFAULT_VOICE dependency to avoid a circular import with
voice_assistant.py and stay independently testable."
```

---

### Task 2: Device resolution — `_resolve_device`, `_device_name`, and output-device symmetry

**Why:** `set_input_device` already resolves either an index or a case-insensitive name substring against `get_input_devices()`. `set_output_device` only accepts a raw index. Persisting the output device by name (Task 4) needs the same resolution `set_input_device` already has, so this task extracts the shared lookup and widens `set_output_device` to use it — before persistence is wired in, so the refactor and the new feature are reviewable separately.

**Files:**
- Modify: `voice_assistant.py` — imports unchanged in this task; add module-level `_resolve_device`, `_device_name`; split `set_input_device` into `_apply_input_device` (does the work) + `set_input_device` (calls it — persistence added in Task 4); same split for `set_output_device`, widened from `index`-only to `device_id_or_name`; `run()` calls `_apply_input_device` instead of `set_input_device`.
- Test: `tests/test_voice_assistant.py`

**Interfaces:**
- Produces:
  - `_resolve_device(id_or_name: int | str | None, devices: list[dict]) -> int | None` — index match first, then case-insensitive name substring; `None` in, `None` out.
  - `_device_name(index: int | None, devices: list[dict]) -> str | None`
  - `VoiceAssistant._apply_input_device(self, device_id_or_name)` — unchanged behavior from today's `set_input_device`.
  - `VoiceAssistant._apply_output_device(self, device_id_or_name)` — new; resolves name-or-index like input already does.
  - `VoiceAssistant.set_output_device(self, device_id_or_name)` — widened signature (was `index` only); tray menu callers that already pass a raw index are unaffected since index-matching is tried first.

- [ ] **Step 1: Write failing tests for `_resolve_device` and `_device_name`**

Add to `tests/test_voice_assistant.py`, after `test_get_output_devices_filters_by_output_channels`:

```python
def test_resolve_device_matches_by_index():
    devices = [{"index": 0, "name": "Mic"}, {"index": 2, "name": "USB Mic"}]
    assert voice_assistant._resolve_device(2, devices) == 2
    assert voice_assistant._resolve_device("2", devices) == 2


def test_resolve_device_matches_by_name_substring():
    devices = [{"index": 0, "name": "Mic"}, {"index": 2, "name": "USB Microphone"}]
    assert voice_assistant._resolve_device("usb", devices) == 2


def test_resolve_device_returns_none_for_no_match_or_none():
    devices = [{"index": 0, "name": "Mic"}]
    assert voice_assistant._resolve_device("nonexistent", devices) is None
    assert voice_assistant._resolve_device(None, devices) is None


def test_device_name_looks_up_name_by_index():
    devices = [{"index": 0, "name": "Mic"}, {"index": 2, "name": "USB Microphone"}]
    assert voice_assistant._device_name(2, devices) == "USB Microphone"


def test_device_name_returns_none_for_none_index():
    devices = [{"index": 0, "name": "Mic"}]
    assert voice_assistant._device_name(None, devices) is None
    assert voice_assistant._device_name(99, devices) is None
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_voice_assistant.py -k "resolve_device or device_name" -v`
Expected: FAIL — `AttributeError: module 'voice_assistant' has no attribute '_resolve_device'`

- [ ] **Step 3: Implement `_resolve_device` and `_device_name`**

In `voice_assistant.py`, insert directly after the `SPEED_PRESETS` line (before `class UIUpdater`):

```python
def _resolve_device(id_or_name, devices):
    """Match `id_or_name` against `devices` ([{"index", "name"}, ...]).
    Tries an index match first, then a case-insensitive name substring match.
    Returns the matched index, or None."""
    if id_or_name is None:
        return None
    try:
        idx = int(id_or_name)
        if any(d["index"] == idx for d in devices):
            return idx
    except (ValueError, TypeError):
        pass
    for d in devices:
        if str(id_or_name).lower() in d["name"].lower():
            return d["index"]
    return None


def _device_name(index, devices):
    """Look up the name for `index` in an enumerated device list, or None."""
    if index is None:
        return None
    for d in devices:
        if d["index"] == index:
            return d["name"]
    return None
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_voice_assistant.py -k "resolve_device or device_name" -v`
Expected: `5 passed`

- [ ] **Step 5: Write a failing test for output-device name resolution**

Add to `tests/test_voice_assistant.py`:

```python
def test_set_output_device_resolves_by_name(monkeypatch):
    fake_devices = [
        {"name": "Mic Only", "max_input_channels": 2, "max_output_channels": 0},
        {"name": "Speakers", "max_input_channels": 0, "max_output_channels": 2},
    ]
    monkeypatch.setattr(voice_assistant.sd, "query_devices", lambda: fake_devices)

    va = voice_assistant.VoiceAssistant.__new__(voice_assistant.VoiceAssistant)
    va.reader = types.SimpleNamespace(set_output_device=lambda idx: None)
    va.set_output_device("speakers")

    assert va.output_device == 1
```

- [ ] **Step 6: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_voice_assistant.py::test_set_output_device_resolves_by_name -v`
Expected: FAIL — `assert 'speakers' == 1` (today's `set_output_device` assigns whatever is passed, unresolved)

- [ ] **Step 7: Refactor `set_input_device`/`set_output_device`, update `run()`**

In `voice_assistant.py`, replace `set_input_device` (today's single method that both resolves and applies):

```python
    def _apply_input_device(self, device_id_or_name):
        """Sets the input device and restarts the stream if necessary."""
        devices = self.get_input_devices()
        target_index = _resolve_device(device_id_or_name, devices)

        # Default to system default if still not found
        if target_index is None:
            if device_id_or_name:
                print(f"Warning: Device '{device_id_or_name}' not found. Using default.")
            target_index = sd.default.device[0]

        with self.lock:
            self.current_device = target_index
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()

            try:
                self.stream = sd.InputStream(
                    device=self.current_device,
                    samplerate=SAMPLERATE,
                    channels=CHANNELS,
                    callback=self.record_callback
                )
                self.stream.start()
                print(f"Input device set to: {sd.query_devices(self.current_device)['name']} (Index: {self.current_device})")
            except Exception as e:
                print(f"Error starting audio stream on device {self.current_device}: {e}")
                self.stream = None

    def set_input_device(self, device_id_or_name):
        """Tray-triggered input device change: apply it (persistence added in Task 4)."""
        self._apply_input_device(device_id_or_name)
```

Replace `set_output_device`:

```python
    def _apply_output_device(self, device_id_or_name):
        """Sets the output device for read-aloud playback and beeps."""
        target_index = _resolve_device(device_id_or_name, self.get_output_devices())
        if target_index is None and device_id_or_name is not None:
            print(f"Warning: Output device '{device_id_or_name}' not found. Using default.")
        self.output_device = target_index
        self.reader.set_output_device(target_index)
        print(f"Output device set to index {target_index}")

    def set_output_device(self, device_id_or_name):
        """Tray-triggered output device change: apply it (persistence added in Task 4)."""
        self._apply_output_device(device_id_or_name)
```

In `run()`, replace:

```python
        # Initialize the audio stream
        self.set_input_device(self.current_device)
```

with:

```python
        # Initialize the audio stream
        self._apply_input_device(self.current_device)
```

- [ ] **Step 8: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_voice_assistant.py -v`
Expected: all pass, including `test_set_output_device_resolves_by_name`

- [ ] **Step 9: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: `38 passed`

- [ ] **Step 10: Commit**

```bash
git add voice_assistant.py tests/test_voice_assistant.py
git commit -m "refactor: extract device resolution, widen set_output_device

set_input_device already resolved a name-or-index against enumerated devices;
set_output_device only accepted a raw index. Extracts the shared lookup into
_resolve_device() and widens set_output_device to use it, so persisting the
output device by name (next task) doesn't need a second resolution path."
```

---

### Task 3: Startup settings resolution

**Files:**
- Modify: `voice_assistant.py` — imports; new `_StartupSettings` NamedTuple + `_resolve_startup_settings`; `VoiceAssistant.__init__` signature and body.
- Test: `tests/test_voice_assistant.py`

**Interfaces:**
- Consumes: `tray_settings.load`, `tray_settings.default_config_dir`, `tray_settings.TraySettings` (Task 1); `_resolve_device` (Task 2); `VOICE_CATALOG`, `DEFAULT_VOICE`, `READ_SPEED` (pre-existing).
- Produces:
  - `_resolve_startup_settings(initial_device, settings_path) -> _StartupSettings(input_device, output_device, voice, speed, paste_with_shift)` — `input_device` and `output_device` may be `None` (meaning "system default", `output_device` unresolved — resolved via `_resolve_device` by the caller); `voice`, `speed`, `paste_with_shift` are always concrete, never `None`.
  - `VoiceAssistant.__init__(self, initial_device=None, settings_path=None)` — `settings_path` is new, for test overrides (mirrors `SpeechReader.__init__(..., voices_dir=None)`).

- [ ] **Step 1: Write failing tests for `_resolve_startup_settings`**

Add to `tests/test_voice_assistant.py`:

```python
def test_resolve_startup_settings_cli_device_overrides_saved(tmp_path):
    path = tmp_path / "settings.json"
    voice_assistant.tray_settings.save(
        voice_assistant.tray_settings.TraySettings(input_device="Saved Mic"), path
    )
    result = voice_assistant._resolve_startup_settings("CLI Mic", path)
    assert result.input_device == "CLI Mic"


def test_resolve_startup_settings_falls_back_to_saved_device(tmp_path):
    path = tmp_path / "settings.json"
    voice_assistant.tray_settings.save(
        voice_assistant.tray_settings.TraySettings(input_device="Saved Mic"), path
    )
    result = voice_assistant._resolve_startup_settings(None, path)
    assert result.input_device == "Saved Mic"


def test_resolve_startup_settings_unknown_voice_falls_back_to_default(tmp_path):
    path = tmp_path / "settings.json"
    voice_assistant.tray_settings.save(
        voice_assistant.tray_settings.TraySettings(voice="no-longer-exists"), path
    )
    result = voice_assistant._resolve_startup_settings(None, path)
    assert result.voice == voice_assistant.DEFAULT_VOICE


def test_resolve_startup_settings_defaults_when_nothing_saved(tmp_path):
    path = tmp_path / "settings.json"  # never written
    result = voice_assistant._resolve_startup_settings(None, path)
    assert result.input_device is None
    assert result.output_device is None
    assert result.voice == voice_assistant.DEFAULT_VOICE
    assert result.speed == voice_assistant.READ_SPEED
    assert result.paste_with_shift is True
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_voice_assistant.py -k resolve_startup_settings -v`
Expected: FAIL — `AttributeError: module 'voice_assistant' has no attribute '_resolve_startup_settings'` (and `tray_settings` isn't imported into `voice_assistant` yet either)

- [ ] **Step 3: Add imports**

In `voice_assistant.py`, replace the top of the import block:

```python
import os
import sys
import time
import subprocess
import threading
import queue
import signal
import argparse
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from PySide6.QtWidgets import QApplication, QSystemTrayIcon, QMenu
from PySide6.QtGui import QIcon, QAction, QCursor, QPixmap, QPainter, QColor, QBrush, QActionGroup
from PySide6.QtCore import QTimer, Qt, Signal, QObject, Slot
from evdev import InputDevice, categorize, ecodes, list_devices

from speech_reader import SpeechReader, VOICE_CATALOG, DEFAULT_VOICE
from text_source import get_text_to_read
```

with:

```python
import os
import sys
import time
import subprocess
import threading
import queue
import signal
import argparse
from pathlib import Path
from typing import NamedTuple

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from PySide6.QtWidgets import QApplication, QSystemTrayIcon, QMenu
from PySide6.QtGui import QIcon, QAction, QCursor, QPixmap, QPainter, QColor, QBrush, QActionGroup
from PySide6.QtCore import QTimer, Qt, Signal, QObject, Slot
from evdev import InputDevice, categorize, ecodes, list_devices

import tray_settings
from speech_reader import SpeechReader, VOICE_CATALOG, DEFAULT_VOICE
from text_source import get_text_to_read
```

- [ ] **Step 4: Implement `_StartupSettings` and `_resolve_startup_settings`**

In `voice_assistant.py`, insert directly after `_device_name` (Task 2):

```python
class _StartupSettings(NamedTuple):
    input_device: str | None
    output_device: str | None  # unresolved name; resolved via _resolve_device + get_output_devices()
    voice: str
    speed: float
    paste_with_shift: bool


def _resolve_startup_settings(initial_device, settings_path):
    """Merge a CLI-provided initial_device with saved tray settings. The CLI
    value wins when given; otherwise the saved value is used, falling back to
    the hardcoded default. Never returns a voice absent from VOICE_CATALOG."""
    settings = tray_settings.load(settings_path)
    voice = settings.voice if settings.voice in VOICE_CATALOG else DEFAULT_VOICE
    speed = settings.speed if settings.speed is not None else READ_SPEED
    input_device = initial_device if initial_device is not None else settings.input_device
    return _StartupSettings(
        input_device=input_device,
        output_device=settings.output_device,
        voice=voice,
        speed=speed,
        paste_with_shift=settings.paste_with_shift,
    )
```

- [ ] **Step 5: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_voice_assistant.py -k resolve_startup_settings -v`
Expected: `4 passed`

- [ ] **Step 6: Wire into `__init__`**

In `voice_assistant.py`, replace `VoiceAssistant.__init__`:

```python
    def __init__(self, initial_device=None):
        print(f"Loading Whisper model '{MODEL_SIZE}'...")
        self.model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
        self.is_recording = False
        self.audio_data = []
        self.transcription_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._transcription_worker, daemon=True)
        self.worker_thread.start()
        
        self.lock = threading.Lock()
        # Track active recording to prevent duplicate starts from multiple devices
        self.active_recording_device = None
        self.running = True
        self.window = None
        self.stream = None
        self.current_device = initial_device
        # Ctrl+Shift+V works in terminals and most editors; flip off for apps
        # that reserve it (e.g. LibreOffice "Paste Special").
        self.paste_with_shift = True

        # Setup signal handlers
        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)

        self.ui_updater = UIUpdater()
        self.ui_updater.update_signal.connect(self._do_update_ui)

        # Read-aloud (TTS)
        self.output_device = None
        self.reader = SpeechReader(
            voice_name=DEFAULT_VOICE,
            speed=READ_SPEED,
            output_device=self.output_device,
            on_state_change=self.update_ui,
        )
```

with:

```python
    def __init__(self, initial_device=None, settings_path=None):
        print(f"Loading Whisper model '{MODEL_SIZE}'...")
        self.model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
        self.is_recording = False
        self.audio_data = []
        self.transcription_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._transcription_worker, daemon=True)
        self.worker_thread.start()
        
        self.lock = threading.Lock()
        # Track active recording to prevent duplicate starts from multiple devices
        self.active_recording_device = None
        self.running = True
        self.window = None
        self.stream = None

        self._settings_path = (
            Path(settings_path) if settings_path else tray_settings.default_config_dir() / "settings.json"
        )
        startup = _resolve_startup_settings(initial_device, self._settings_path)
        self.current_device = startup.input_device
        # Ctrl+Shift+V works in terminals and most editors; flip off for apps
        # that reserve it (e.g. LibreOffice "Paste Special").
        self.paste_with_shift = startup.paste_with_shift

        # Setup signal handlers
        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)

        self.ui_updater = UIUpdater()
        self.ui_updater.update_signal.connect(self._do_update_ui)

        # Read-aloud (TTS)
        self.output_device = _resolve_device(startup.output_device, self.get_output_devices())
        self.reader = SpeechReader(
            voice_name=startup.voice,
            speed=startup.speed,
            output_device=self.output_device,
            on_state_change=self.update_ui,
        )
```

- [ ] **Step 7: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: `42 passed`

(No test constructs a real `VoiceAssistant()` — the existing `WhisperModel` stub is `object`, which doesn't accept `__init__` arguments, so real construction was already untestable before this change. `_resolve_startup_settings` is tested standalone instead, which is why it was extracted as a free function.)

- [ ] **Step 8: Commit**

```bash
git add voice_assistant.py tests/test_voice_assistant.py
git commit -m "feat: load saved tray settings at startup

VoiceAssistant.__init__ now seeds voice, speed, paste mode, and both devices
from ~/.config/voice-assistant/settings.json instead of hardcoded constants.
--device still overrides a saved input device for that run without
rewriting the file, since _resolve_startup_settings only reads the file and
__init__ never calls a save-triggering setter."
```

---

### Task 4: Persist on every tray change

**Files:**
- Modify: `voice_assistant.py` — new `_save_settings`; hook it into `set_input_device`, `set_output_device`, `on_select_voice`, a new `set_speed` wrapper (and `build_speed_menu`'s lambda), and `set_paste_with_shift`.
- Test: `tests/test_voice_assistant.py`

**Interfaces:**
- Consumes: `tray_settings.save`, `tray_settings.TraySettings` (Task 1); `_device_name` (Task 2); `self._settings_path` (Task 3).
- Produces: `VoiceAssistant._save_settings(self) -> None`; `VoiceAssistant.set_speed(self, scale) -> None` (new — today the speed-menu lambda calls `self.reader.set_speed` directly, with no `VoiceAssistant`-level hook to persist from).

- [ ] **Step 1: Write failing tests for `_save_settings` and the setters**

Add to `tests/test_voice_assistant.py`:

```python
class _FakeReader:
    """Minimal stand-in for SpeechReader: mutates _voice_name/_speed like the
    real thing so _save_settings sees the post-change state."""

    def __init__(self, voice_name, speed):
        self._voice_name = voice_name
        self._speed = speed

    def set_speed(self, scale):
        self._speed = scale

    def set_voice(self, name):
        self._voice_name = name


def test_save_settings_snapshots_current_state(monkeypatch, tmp_path):
    va = voice_assistant.VoiceAssistant.__new__(voice_assistant.VoiceAssistant)
    va._settings_path = tmp_path / "settings.json"
    va.current_device = 3
    va.output_device = 1
    va.paste_with_shift = False
    va.reader = _FakeReader("en_US-sarah", 1.25)
    monkeypatch.setattr(va, "get_input_devices", lambda: [{"index": 3, "name": "USB Microphone"}])
    monkeypatch.setattr(va, "get_output_devices", lambda: [{"index": 1, "name": "Speakers"}])

    va._save_settings()

    saved = voice_assistant.tray_settings.load(va._settings_path)
    assert saved == voice_assistant.tray_settings.TraySettings(
        input_device="USB Microphone",
        output_device="Speakers",
        voice="en_US-sarah",
        speed=1.25,
        paste_with_shift=False,
    )


def test_set_paste_with_shift_persists(monkeypatch, tmp_path):
    va = voice_assistant.VoiceAssistant.__new__(voice_assistant.VoiceAssistant)
    va._settings_path = tmp_path / "settings.json"
    va.current_device = None
    va.output_device = None
    va.reader = _FakeReader(voice_assistant.DEFAULT_VOICE, voice_assistant.READ_SPEED)
    monkeypatch.setattr(va, "get_input_devices", lambda: [])
    monkeypatch.setattr(va, "get_output_devices", lambda: [])

    va.set_paste_with_shift(False)

    assert va.paste_with_shift is False
    saved = voice_assistant.tray_settings.load(va._settings_path)
    assert saved.paste_with_shift is False


def test_set_speed_persists(monkeypatch, tmp_path):
    va = voice_assistant.VoiceAssistant.__new__(voice_assistant.VoiceAssistant)
    va._settings_path = tmp_path / "settings.json"
    va.current_device = None
    va.output_device = None
    va.reader = _FakeReader(voice_assistant.DEFAULT_VOICE, voice_assistant.READ_SPEED)
    monkeypatch.setattr(va, "get_input_devices", lambda: [])
    monkeypatch.setattr(va, "get_output_devices", lambda: [])

    va.set_speed(1.25)

    assert va.reader._speed == 1.25
    saved = voice_assistant.tray_settings.load(va._settings_path)
    assert saved.speed == 1.25


def test_on_select_voice_persists_on_success(monkeypatch, tmp_path):
    va = voice_assistant.VoiceAssistant.__new__(voice_assistant.VoiceAssistant)
    va._settings_path = tmp_path / "settings.json"
    va.current_device = None
    va.output_device = None
    va.reader = _FakeReader(voice_assistant.DEFAULT_VOICE, voice_assistant.READ_SPEED)
    monkeypatch.setattr(va, "get_input_devices", lambda: [])
    monkeypatch.setattr(va, "get_output_devices", lambda: [])

    va.on_select_voice("en_US-sarah")

    saved = voice_assistant.tray_settings.load(va._settings_path)
    assert saved.voice == "en_US-sarah"


def test_on_select_voice_does_not_persist_on_failure(tmp_path):
    va = voice_assistant.VoiceAssistant.__new__(voice_assistant.VoiceAssistant)
    va._settings_path = tmp_path / "settings.json"
    va.tray = types.SimpleNamespace(showMessage=lambda *a, **k: None)

    class _FailingReader:
        def set_voice(self, name):
            raise RuntimeError("boom")

    va.reader = _FailingReader()

    va.on_select_voice("bad-voice")

    assert not va._settings_path.exists()
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_voice_assistant.py -k "save_settings or persists or does_not_persist" -v`
Expected: FAIL — `AttributeError: 'VoiceAssistant' object has no attribute '_save_settings'` (and `set_speed` doesn't exist yet)

- [ ] **Step 3: Implement `_save_settings`**

In `voice_assistant.py`, insert directly above `set_input_device`:

```python
    def _save_settings(self):
        try:
            tray_settings.save(
                tray_settings.TraySettings(
                    input_device=_device_name(self.current_device, self.get_input_devices()),
                    output_device=_device_name(self.output_device, self.get_output_devices()),
                    voice=self.reader._voice_name,
                    speed=self.reader._speed,
                    paste_with_shift=self.paste_with_shift,
                ),
                self._settings_path,
            )
        except Exception as exc:
            print(f"[settings] failed to save: {exc}", file=sys.stderr)
```

- [ ] **Step 4: Hook it into the five setters**

In `voice_assistant.py`, replace `set_input_device`:

```python
    def set_input_device(self, device_id_or_name):
        """Tray-triggered input device change: apply it (persistence added in Task 4)."""
        self._apply_input_device(device_id_or_name)
```

with:

```python
    def set_input_device(self, device_id_or_name):
        """Tray-triggered input device change: apply it and persist the choice."""
        self._apply_input_device(device_id_or_name)
        self._save_settings()
```

Replace `set_output_device`:

```python
    def set_output_device(self, device_id_or_name):
        """Tray-triggered output device change: apply it (persistence added in Task 4)."""
        self._apply_output_device(device_id_or_name)
```

with:

```python
    def set_output_device(self, device_id_or_name):
        """Tray-triggered output device change: apply it and persist the choice."""
        self._apply_output_device(device_id_or_name)
        self._save_settings()
```

Replace `on_select_voice`:

```python
    def on_select_voice(self, name):
        """Switch the voice (downloads the model on demand)."""
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

with:

```python
    def on_select_voice(self, name):
        """Switch the voice (downloads the model on demand)."""
        try:
            print(f"Switching voice to {name} ...")
            self.reader.set_voice(name)
            self._save_settings()
        except Exception as exc:
            print(f"[read] failed to switch voice to {name}: {exc}", file=sys.stderr)
            self.tray.showMessage(
                "Voice Assistant",
                f"Could not load voice '{name}': {exc}",
                QSystemTrayIcon.Warning,
            )
```

Add a new `set_speed` method directly above `set_output_device`:

```python
    def set_speed(self, scale):
        """Tray-triggered speed change: apply it and persist the choice."""
        self.reader.set_speed(scale)
        self._save_settings()
```

In `build_speed_menu`, replace:

```python
            action.triggered.connect(lambda checked, s=scale: self.reader.set_speed(s))
```

with:

```python
            action.triggered.connect(lambda checked, s=scale: self.set_speed(s))
```

Replace `set_paste_with_shift`:

```python
    def set_paste_with_shift(self, enabled):
        self.paste_with_shift = enabled
        print(f"Paste mode: {'Ctrl+Shift+V' if enabled else 'Ctrl+V'}")
```

with:

```python
    def set_paste_with_shift(self, enabled):
        self.paste_with_shift = enabled
        print(f"Paste mode: {'Ctrl+Shift+V' if enabled else 'Ctrl+V'}")
        self._save_settings()
```

- [ ] **Step 5: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_voice_assistant.py -v`
Expected: all pass

- [ ] **Step 6: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: `47 passed`

- [ ] **Step 7: Commit**

```bash
git add voice_assistant.py tests/test_voice_assistant.py
git commit -m "feat: persist tray settings whenever they change

Each of the five tray-menu setters (input device, output device, voice,
speed, paste mode) now calls _save_settings() after applying its change.
set_speed is new — the speed-menu action previously called
reader.set_speed() directly with no VoiceAssistant-level hook to persist
from. Startup continues to apply values without saving (Task 3), so this is
the only path that writes the settings file."
```

---

### Task 5: Documentation and manual verification

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: the complete persistence loop from Tasks 1–4.
- Produces: no code interface. Documentation, plus a manual check for you to run (the tray/hotkey flow isn't scriptable headlessly).

- [ ] **Step 1: Add a Settings section to the README**

In `README.md`, insert a new section after `## Read Aloud` and before `## Desktop Integration`:

```markdown
## Settings

Every setting you change from the tray menu — input device, output device, voice,
reading speed, and the Ctrl+Shift+V paste-mode toggle — is remembered across
restarts. Settings are stored as JSON in `~/.config/voice-assistant/settings.json`
(respects `$XDG_CONFIG_HOME`). Delete the file to reset everything to defaults.

`--device` on the command line overrides the saved input device for that run only;
it does not overwrite the saved value.
```

- [ ] **Step 2: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: `47 passed`

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document tray settings persistence"
```

- [ ] **Step 4: Manual verification (for you to run — the tray/hotkey loop isn't scriptable headlessly)**

1. If `~/.config/voice-assistant/settings.json` exists from earlier testing, remove it so this is a clean first run.
2. `uv run voice_assistant.py`
3. From the tray menu: pick a non-default voice, a non-default speed preset, toggle the Ctrl+Shift+V paste mode off, and (if more than one is available) pick a non-default output device.
4. `cat ~/.config/voice-assistant/settings.json` — confirm it reflects your choices.
5. Quit the assistant (tray menu → "Close Voice Assistant") and relaunch with `uv run voice_assistant.py`.
6. Open the tray menu again — confirm the voice, speed, and paste-mode checkmarks match what you set in step 3, not the hardcoded defaults.
7. Relaunch once more with `uv run voice_assistant.py --device <some other mic name>` — confirm (from the startup log line `Input device set to: ...`) that the CLI-specified device wins for this run, then `cat ~/.config/voice-assistant/settings.json` again and confirm the file's `input_device` is unchanged from step 4 (the CLI override must not have been written back).

---

## Self-Review

**Spec coverage.** §2 (only `--device` has real precedence) → Task 3's `_resolve_startup_settings` and Global Constraints. §3 (module shape, decoupling) → Task 1. §4 (startup integration, `settings_path` param) → Task 3. §5 (device resolution symmetry) → Task 2. §6 (save/load precedence — startup never saves) → Tasks 2 & 3 (`run()` calls `_apply_input_device`, `__init__` never calls a public setter) and verified end-to-end in Task 5 Step 4, checklist item 7. §7 (error handling) → Task 1 (`load`/`save` failure modes) and Task 4 (`_save_settings`'s try/except). §8 (tests) → all four code tasks. §9 (out of scope) respected: no new CLI flags, no live-reload, `KEYBOARD_LAYOUT` untouched.

**Additions beyond the spec.** None — Task 2's extraction of `_resolve_device` was already called out in spec §5 as a required symmetry fix, not a new addition.

**Type consistency.** `_resolve_device(id_or_name, devices) -> int | None` is defined in Task 2 and consumed identically in Task 2 (`_apply_input_device`, `_apply_output_device`) and Task 3 (`__init__`'s output-device resolution). `_StartupSettings` fields (`input_device`, `output_device`, `voice`, `speed`, `paste_with_shift`) are produced in Task 3 and consumed by name in `__init__`'s `startup.X` accesses. `TraySettings` fields are identical across Task 1's definition, Task 3's `_resolve_startup_settings`, and Task 4's `_save_settings` — same five names throughout. `self._settings_path` is set once in Task 3's `__init__` and read in Task 4's `_save_settings`; no task before Task 3 assumes it exists.

**Test count arithmetic.** Baseline 24. Task 1 adds 8 → 32. Task 2 adds 6 (`resolve_device` ×3, `device_name` ×2, `set_output_device_resolves_by_name` ×1) → 38. Task 3 adds 4 → 42. Task 4 adds 5 → 47. Task 5 adds 0 → 47.
