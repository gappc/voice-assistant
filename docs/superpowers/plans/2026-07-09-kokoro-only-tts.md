# Kokoro-only TTS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the Piper TTS engine, its dependency, and its voice models, leaving Kokoro as the only synthesis engine.

**Architecture:** `speech_reader.py` currently holds two engine classes behind a `TTSEngine` Protocol, dispatched by a `VoiceSpec.engine` field. We delete Piper, collapse the Protocol, and reduce `VoiceSpec` to `(model, voice_id, lang)`. A `KokoroEngine` owns one ONNX session per *model file*; the voice and language become per-call arguments, because five of the six catalog voices share a single model file. Playback, the prefetch producer, and the settings-generation counter are untouched.

**Tech Stack:** Python 3.10+, `kokoro-onnx`, `onnxruntime`, `sounddevice`, `numpy`, `PySide6`, `evdev`; `pytest`; `uv` for dependency management.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-09-kokoro-only-tts-design.md`
- Kokoro's `speed` parameter is valid only in `[0.5, 2.0]`; clamp before calling `create()`.
- The model cache directory `~/.local/share/voice-assistant/voices/kokoro/` must not change. Renaming it forces a 500 MB re-download.
- `KOKORO_INTRA_OP_THREADS` (default `4`, overridable via `VOICE_KOKORO_THREADS`) and the `session.intra_op.allow_spinning = "0"` config entry in `build_kokoro()` must survive untouched. They are worth ~5x CPU.
- `sys` stays imported in `speech_reader.py` (used by `_read_worker` for `sys.stderr`). Only `subprocess` is removed.
- Run tests with `.venv/bin/python -m pytest`. Baseline before any change: **23 passed**.
- Voice IDs verified against the real fp16 model: `af_bella`, `af_sarah`, `am_michael` (`lang="en-us"`), `bm_george`, `bf_emma` (`lang="en-gb"`), `martin` (`lang="de"`).

## File Structure

| File | Responsibility after this plan |
|---|---|
| `speech_reader.py` | Kokoro provisioning, one ONNX session per model file, sentence prefetch, playback. No engine abstraction. |
| `voice_assistant.py` | Tray + hotkeys. Owns `SPEED_PRESETS` in native Kokoro units. |
| `tests/test_speech_reader.py` | Catalog, engine cache, speed clamping, prefetch/stop/live-settings concurrency. |
| `tests/test_voice_assistant.py` | Imports `voice_assistant` with hardware modules stubbed. No `piper` stub. |
| `pyproject.toml` | Declares `kokoro-onnx` and `onnxruntime`. No `piper-tts`. |
| `README.md` | Describes Kokoro, the six voices, the new default. |

---

### Task 1: Stop voices that share a model file from sharing a voice_id

**Why first:** This is a live bug, independent of Piper removal. `VOICE_CATALOG["kokoro-en-bella"]` and `["kokoro-en-sarah"]` both have `model="official"`, and `_get_or_create_engine` caches on `(spec.engine, spec.model)`. Selecting Sarah after Bella returns Bella's engine, whose `self.voice_id` is still `af_bella` — so Sarah silently speaks as Bella. Task 2 makes this worse (five voices share `official`), so it must be fixed before the catalog grows.

The fix: `KokoroEngine` owns the *model* (the expensive ONNX session, shared correctly), while `voice_id` and `lang` become arguments to `synthesize()`. `PiperEngine.synthesize` grows the same two parameters and ignores them, so the Protocol stays single-signature and no `if spec.engine ==` branch is needed in the caller.

**Files:**
- Modify: `speech_reader.py` — `TTSEngine`, `PiperEngine.synthesize`, `KokoroEngine`, `SpeechReader._get_or_create_engine`, `SpeechReader._synth_sentence`
- Test: `tests/test_speech_reader.py`

**Interfaces:**
- Consumes: `build_kokoro(model_path, voices_path)`, `ensure_kokoro_voice_files(model_name, voices_dir)` — both unchanged.
- Produces:
  - `KokoroEngine(model_name: str, voices_dir: Path)` — note: no longer takes `voice_id`/`lang`.
  - `KokoroEngine.synthesize(text: str, speed: float, voice_id: str, lang: str) -> Iterator[AudioChunk]`
  - `PiperEngine.synthesize(text: str, speed: float, voice_id=None, lang=None) -> Iterator[AudioChunk]`
  - `SpeechReader._synth_sentence(sentence, voice_name, speed) -> list[tuple[np.ndarray, int]]` — signature unchanged.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_speech_reader.py`:

```python
def test_voices_sharing_a_model_do_not_share_a_voice_id(monkeypatch):
    """kokoro-en-bella and kokoro-en-sarah share model='official'. They must
    share the ONNX session but NOT the voice_id."""
    import numpy as np

    used = []

    class FakeKokoro:
        def create(self, text, voice, speed, lang):
            used.append((voice, lang))
            return np.zeros(10, dtype=np.float32), 24000

    monkeypatch.setattr(speech_reader, "build_kokoro", lambda model, voices: FakeKokoro())
    monkeypatch.setattr(
        speech_reader, "ensure_kokoro_voice_files",
        lambda name, folder: (Path("model"), Path("voices")),
    )

    r = speech_reader.SpeechReader("kokoro-en-bella")
    r._synth_sentence("hello", "kokoro-en-bella", 1.0)
    r._synth_sentence("hello", "kokoro-en-sarah", 1.0)

    assert used == [("af_bella", "en-us"), ("af_sarah", "en-us")]
    assert len(r._engines) == 1  # one ONNX session, not two
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_speech_reader.py::test_voices_sharing_a_model_do_not_share_a_voice_id -v`

Expected: FAIL with `assert [('af_bella', 'en-us'), ('af_bella', 'en-us')] == [('af_bella', 'en-us'), ('af_sarah', 'en-us')]`

- [ ] **Step 3: Move voice_id/lang from constructor to synthesize()**

In `speech_reader.py`, replace the `TTSEngine` Protocol body:

```python
class TTSEngine(Protocol):
    def load(self) -> None:
        """Load model weights. Called once at startup or on voice switch."""
        ...

    def synthesize(
        self, text: str, speed: float, voice_id: str | None, lang: str | None
    ) -> Iterator[AudioChunk]:
        """Synthesize `text` (already segmented to sentence granularity by the
        caller), yielding one or more audio chunks as they become available."""
        ...
```

Replace `PiperEngine.synthesize` (a Piper voice is one model, so it ignores the extra arguments):

```python
    def synthesize(
        self, text: str, speed: float, voice_id: str | None = None, lang: str | None = None
    ) -> Iterator[AudioChunk]:
        self.load()
        syn_config = SynthesisConfig(length_scale=speed)
        for chunk in self.voice.synthesize(text, syn_config=syn_config):
            samples = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
            yield AudioChunk(samples, chunk.sample_rate)
```

Replace `KokoroEngine` entirely:

```python
class KokoroEngine:
    """Owns one Kokoro ONNX session. Voice and language are per-call arguments,
    because many voices are served by the same model file."""

    def __init__(self, model_name: str, voices_dir: Path):
        self.model_name = model_name
        self.voices_dir = voices_dir
        self._kokoro = None

    def load(self) -> None:
        if self._kokoro is None:
            model_path, voices_path = ensure_kokoro_voice_files(self.model_name, self.voices_dir)
            print(f"[read] loading Kokoro model '{self.model_name}' ...")
            self._kokoro = build_kokoro(model_path, voices_path)

    def synthesize(
        self, text: str, speed: float, voice_id: str | None = None, lang: str | None = None
    ) -> Iterator[AudioChunk]:
        self.load()
        # Convert Piper's length_scale speed to Kokoro speed (reciprocal)
        kokoro_speed = 1.0 / speed
        kokoro_speed = max(0.5, min(kokoro_speed, 2.0))
        float32_samples, sample_rate = self._kokoro.create(
            text,
            voice=voice_id,
            speed=kokoro_speed,
            lang=lang,
        )
        clamped = np.clip(float32_samples, -1.0, 1.0)
        int16_samples = (clamped * 32767.0).astype(np.int16)
        yield AudioChunk(int16_samples, sample_rate)
```

- [ ] **Step 4: Pass voice_id/lang through the caller**

In `speech_reader.py`, replace `SpeechReader._get_or_create_engine` (Kokoro no longer needs voice/lang at construction):

```python
    def _get_or_create_engine(self, spec: VoiceSpec) -> TTSEngine:
        engine_key = (spec.engine, spec.model)
        with self._engines_lock:  # the prefetch thread creates engines too
            if engine_key not in self._engines:
                if spec.engine == "piper":
                    self._engines[engine_key] = PiperEngine(spec.model, self._voices_dir)
                elif spec.engine == "kokoro":
                    self._engines[engine_key] = KokoroEngine(spec.model, self._voices_dir)
                else:
                    raise ValueError(f"Unknown engine: {spec.engine}")
            return self._engines[engine_key]
```

And replace `SpeechReader._synth_sentence`:

```python
    def _synth_sentence(self, sentence, voice_name, speed):
        """Synthesize one sentence to a materialized list of (samples, rate)."""
        spec = VOICE_CATALOG.get(voice_name, VOICE_CATALOG[DEFAULT_VOICE])
        engine = self._get_or_create_engine(spec)
        engine.load()
        self._active_engine = engine
        return [
            (c.samples, c.sample_rate)
            for c in engine.synthesize(sentence, speed, spec.voice_id, spec.lang)
        ]
```

- [ ] **Step 5: Update the mock engines in the three concurrency tests**

Their `synthesize` must accept the two new arguments. In `tests/test_speech_reader.py`:

In `test_speech_reader_stops_between_sentences`, change the `MockEngine.synthesize` signature:

```python
        def synthesize(self, text, speed, voice_id=None, lang=None):
            with lock:
                synthesized_sentences.append(text)
            yield AudioChunk(np.zeros(2048, dtype=np.int16), 22050)
```

In `test_prefetch_synthesizes_next_sentence_during_playback`:

```python
        def synthesize(self, text, speed, voice_id=None, lang=None):
            if text == "Two.":
                second_synthesized.set()
            yield AudioChunk(np.zeros(2048, dtype=np.int16), 22050)
```

In `test_speech_reader_on_the_fly_updates`:

```python
        def synthesize(self, text, speed, voice_id=None, lang=None):
            with lock:
                synthesized.append((self.name, text, speed))
            if text == "Sentence two.":
                second_prefetched.set()
            yield AudioChunk(np.zeros(2048, dtype=np.int16), 22050)
```

In `test_kokoro_engine_speed_translation`, the engine constructor lost two parameters and `synthesize` gained them. Replace the construction and every `synthesize` call:

```python
    engine = speech_reader.KokoroEngine(model_name="martin", voices_dir=Path("/dummy"))
    engine.load()

    # Normal speed (reciprocal of 1.0 is 1.0)
    list(engine.synthesize("test", 1.0, "martin", "de"))
    assert engine._kokoro.created_calls[-1][2] == 1.0

    # Slow speed (reciprocal of 1.3 is ~0.769)
    list(engine.synthesize("test", 1.3, "martin", "de"))
    assert abs(engine._kokoro.created_calls[-1][2] - 1.0 / 1.3) < 1e-5

    # Fast speed (reciprocal of 0.8 is 1.25)
    list(engine.synthesize("test", 0.8, "martin", "de"))
    assert engine._kokoro.created_calls[-1][2] == 1.25

    # Extreme slow (Piper length_scale 3.0 -> reciprocal 0.33 -> clamped to 0.5)
    list(engine.synthesize("test", 3.0, "martin", "de"))
    assert engine._kokoro.created_calls[-1][2] == 0.5

    # Extreme fast (Piper length_scale 0.2 -> reciprocal 5.0 -> clamped to 2.0)
    list(engine.synthesize("test", 0.2, "martin", "de"))
    assert engine._kokoro.created_calls[-1][2] == 2.0
```

- [ ] **Step 6: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: `24 passed`

- [ ] **Step 7: Run the concurrency tests repeatedly to check for flakiness**

Run: `for i in 1 2 3 4 5; do .venv/bin/python -m pytest -q -k "prefetch or on_the_fly or stops_between" 2>&1 | tail -1; done`
Expected: `3 passed, N deselected` five times.

- [ ] **Step 8: Commit**

```bash
git add speech_reader.py tests/test_speech_reader.py
git commit -m "fix: voices sharing a Kokoro model no longer share a voice_id

The engine cache keys on (engine, model), and kokoro-en-bella and
kokoro-en-sarah both have model='official'. Selecting Sarah after Bella
returned Bella's engine, whose voice_id was still af_bella, so Sarah
silently spoke as Bella.

KokoroEngine now owns the model file (the expensive ONNX session, correctly
shared); voice_id and lang are arguments to synthesize()."
```

---

### Task 2: Delete Piper and adopt the Kokoro-only catalog

**Files:**
- Modify: `speech_reader.py` — module docstring, imports, `default_voices_dir`, `ensure_voice_model` (delete), `VoiceSpec`, `VOICE_CATALOG`, `DEFAULT_VOICE`, `TTSEngine` (delete), `PiperEngine` (delete), `KokoroEngine.synthesize` signature, `SpeechReader` docstring, `_get_or_create_engine`
- Modify: `tests/test_speech_reader.py`

**Interfaces:**
- Consumes: `KokoroEngine(model_name, voices_dir)` and `KokoroEngine.synthesize(text, speed, voice_id, lang)` from Task 1.
- Produces:
  - `VoiceSpec(model: str, voice_id: str, lang: str)` — all three required, no defaults, no `engine` field.
  - `VOICE_CATALOG` keyed by `en_US-bella`, `en_US-sarah`, `en_US-michael`, `en_GB-george`, `en_GB-emma`, `de_DE-martin`.
  - `DEFAULT_VOICE = "en_US-bella"`.
  - `SpeechReader._get_or_create_engine(spec) -> KokoroEngine`, cached on `spec.model`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_speech_reader.py`:

```python
def test_catalog_is_kokoro_only():
    assert speech_reader.DEFAULT_VOICE == "en_US-bella"
    assert set(speech_reader.VOICE_CATALOG) == {
        "en_US-bella", "en_US-sarah", "en_US-michael",
        "en_GB-george", "en_GB-emma", "de_DE-martin",
    }
    assert not hasattr(speech_reader, "PiperEngine")
    assert not hasattr(speech_reader, "TTSEngine")
    assert not hasattr(speech_reader, "ensure_voice_model")

    # Every English voice rides the one shared 'official' model.
    official = [s for s in speech_reader.VOICE_CATALOG.values() if s.model == "official"]
    assert len(official) == 5
    assert {s.voice_id for s in official} == {
        "af_bella", "af_sarah", "am_michael", "bm_george", "bf_emma",
    }
    assert speech_reader.VOICE_CATALOG["en_GB-george"].lang == "en-gb"
    assert speech_reader.VOICE_CATALOG["de_DE-martin"] == speech_reader.VoiceSpec(
        "martin", "martin", "de"
    )


def test_engine_cache_is_keyed_on_model(monkeypatch):
    """Five voices, one 'official' ONNX session."""
    import numpy as np

    class FakeKokoro:
        def create(self, text, voice, speed, lang):
            return np.zeros(10, dtype=np.float32), 24000

    monkeypatch.setattr(speech_reader, "build_kokoro", lambda model, voices: FakeKokoro())
    monkeypatch.setattr(
        speech_reader, "ensure_kokoro_voice_files",
        lambda name, folder: (Path("model"), Path("voices")),
    )

    r = speech_reader.SpeechReader("en_US-bella")
    for name in ("en_US-bella", "en_US-sarah", "en_GB-george", "de_DE-martin"):
        r._synth_sentence("hello", name, 1.0)

    assert set(r._engines) == {"official", "martin"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_speech_reader.py::test_catalog_is_kokoro_only -v`
Expected: FAIL with `AssertionError: assert 'en_US-amy-medium' == 'en_US-bella'`

- [ ] **Step 3: Rewrite the head of `speech_reader.py`**

Replace lines 1–21 (module docstring through the `piper` import) with:

```python
"""Local text-to-speech ("read aloud") via Kokoro.

Self-contained: knows nothing about evdev or the tray. The owning app fetches
text elsewhere and drives this via start()/stop()/toggle().
"""

import os
import queue
import re
import sys
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, NamedTuple

import numpy as np
import onnxruntime as ort
import sounddevice as sd
```

Replace the `default_voices_dir` docstring:

```python
def default_voices_dir():
    """Directory where Kokoro voice models are cached (XDG data dir)."""
```

Delete the entire `ensure_voice_model` function (the `def ensure_voice_model` block, lines 32–50).

- [ ] **Step 4: Replace the catalog**

In `speech_reader.py`, replace the `VoiceSpec` dataclass, `VOICE_CATALOG`, and `DEFAULT_VOICE`:

```python
@dataclass(frozen=True)
class VoiceSpec:
    model: str     # which Kokoro ONNX file to load: "official" | "martin"
    voice_id: str  # key into that model's voice pack
    lang: str      # passed to Kokoro.create(lang=...)


VOICE_CATALOG = {
    "en_US-bella":   VoiceSpec("official", "af_bella",   "en-us"),
    "en_US-sarah":   VoiceSpec("official", "af_sarah",   "en-us"),
    "en_US-michael": VoiceSpec("official", "am_michael", "en-us"),
    "en_GB-george":  VoiceSpec("official", "bm_george",  "en-gb"),
    "en_GB-emma":    VoiceSpec("official", "bf_emma",    "en-gb"),
    "de_DE-martin":  VoiceSpec("martin",   "martin",     "de"),
}

DEFAULT_VOICE = "en_US-bella"
```

- [ ] **Step 5: Delete the Protocol and PiperEngine**

Replace the `# --- TTSEngine Abstraction ---` section header and delete both `TTSEngine` and `PiperEngine`, leaving only:

```python
# --- Audio chunks and the Kokoro engine --------------------------------

class AudioChunk(NamedTuple):
    samples: np.ndarray  # mono int16 PCM
    sample_rate: int
```

Then drop the now-unneeded defaults from `KokoroEngine.synthesize`, since it is the only implementation:

```python
    def synthesize(self, text: str, speed: float, voice_id: str, lang: str) -> Iterator[AudioChunk]:
        self.load()
        # Convert Piper's length_scale speed to Kokoro speed (reciprocal)
        kokoro_speed = 1.0 / speed
        kokoro_speed = max(0.5, min(kokoro_speed, 2.0))
        float32_samples, sample_rate = self._kokoro.create(
            text,
            voice=voice_id,
            speed=kokoro_speed,
            lang=lang,
        )
        clamped = np.clip(float32_samples, -1.0, 1.0)
        int16_samples = (clamped * 32767.0).astype(np.int16)
        yield AudioChunk(int16_samples, sample_rate)
```

(The reciprocal stays for now; Task 3 removes it.)

- [ ] **Step 6: Collapse the engine cache**

Replace the `SpeechReader` class docstring and `_get_or_create_engine`:

```python
class SpeechReader:
    """Reads text aloud with Kokoro, streaming per sentence for prompt stop."""
```

```python
    def _get_or_create_engine(self, spec: VoiceSpec) -> KokoroEngine:
        with self._engines_lock:  # the prefetch thread creates engines too
            if spec.model not in self._engines:
                self._engines[spec.model] = KokoroEngine(spec.model, self._voices_dir)
            return self._engines[spec.model]
```

- [ ] **Step 7: Update the tests to the new voice names**

In `tests/test_speech_reader.py`:

Delete `test_ensure_voice_model_downloads_when_missing` and `test_ensure_voice_model_skips_download_when_present` entirely.

Delete `test_voices_sharing_a_model_do_not_share_a_voice_id` from Task 1 — `test_engine_cache_is_keyed_on_model` supersedes it and the old catalog keys no longer exist.

Replace `_make_reader`:

```python
def _make_reader():
    return speech_reader.SpeechReader("en_US-bella", speed=1.0)
```

In `test_start_then_wait_resets_state_and_notifies`, replace `"en_US-amy-medium"` with `"en_US-bella"`.

In `test_speech_reader_stops_between_sentences`, `test_prefetch_synthesizes_next_sentence_during_playback`, and `test_speech_reader_on_the_fly_updates`, replace `speech_reader.SpeechReader("en_US-amy-medium")` with `speech_reader.SpeechReader("en_US-bella")`.

In `test_speech_reader_on_the_fly_updates`, the mock is constructed as `MockEngine(spec.model)`, and `spec.model` is now `"official"` / `"martin"` rather than a catalog key. Record the `voice_id` instead, which is what actually determines the voice. Replace the `MockEngine` class, the `set_voice` call, and the assertions:

```python
    class MockEngine:
        def load(self):
            pass

        def synthesize(self, text, speed, voice_id=None, lang=None):
            with lock:
                synthesized.append((voice_id, text, speed))
            if text == "Sentence two.":
                second_prefetched.set()
            yield AudioChunk(np.zeros(2048, dtype=np.int16), 22050)

    r = speech_reader.SpeechReader("en_US-bella")
    monkeypatch.setattr(r, "_get_or_create_engine", lambda spec: MockEngine())
```

```python
                r.set_speed(1.5)
                r.set_voice("de_DE-martin")
```

```python
    assert calls[0] == ("af_bella", "Sentence one.", 1.0)
    assert calls[1] == ("af_bella", "Sentence two.", 1.0)
    assert calls[2] == ("martin", "Sentence two.", 1.5)
    assert len(calls) == 3
```

In `test_kokoro_engine_speed_translation`, `test_build_kokoro_bounds_threads_and_disables_spinning`, and `test_ensure_kokoro_voice_files*`, no changes are needed.

- [ ] **Step 8: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: `23 passed` (24 from Task 1, minus 2 deleted `ensure_voice_model` tests, minus 1 superseded, plus 2 new)

- [ ] **Step 9: Confirm Piper is gone from the module**

Run: `grep -n "iper" speech_reader.py`
Expected: exactly one line — the `# Convert Piper's length_scale ...` comment, which Task 3 deletes.

- [ ] **Step 10: Commit**

```bash
git add speech_reader.py tests/test_speech_reader.py
git commit -m "refactor!: remove Piper engine, Kokoro-only voice catalog

Deletes PiperEngine, the TTSEngine Protocol, and ensure_voice_model. VoiceSpec
drops its engine field; the engine cache keys on the model file, so the five
English voices share one ONNX session.

BREAKING: voice names change. en_US-amy-medium -> en_US-bella (new default),
ryan -> en_US-michael, alan -> en_GB-george, alba -> en_GB-emma, thorsten ->
de_DE-martin. de_DE-mls-medium is dropped with no analogue: German is now a
single male voice."
```

---

### Task 3: Speed in native Kokoro units

**Files:**
- Modify: `speech_reader.py` — `KokoroEngine.synthesize`
- Modify: `voice_assistant.py:29-30` — `READ_SPEED`, `SPEED_PRESETS`
- Modify: `tests/test_speech_reader.py` — `test_kokoro_engine_speed_translation`

**Interfaces:**
- Consumes: `KokoroEngine.synthesize(text, speed, voice_id, lang)` from Task 2.
- Produces: `speed` is a plain multiplier (higher = faster), clamped to `[0.5, 2.0]`. `SPEED_PRESETS = {"Slow": 0.8, "Normal": 1.0, "Fast": 1.25}`.

- [ ] **Step 1: Replace the speed test with a pass-through-and-clamp test**

In `tests/test_speech_reader.py`, replace the whole of `test_kokoro_engine_speed_translation` with:

```python
def test_kokoro_engine_speed_is_native_and_clamped(monkeypatch):
    import numpy as np

    class FakeKokoro:
        def __init__(self):
            self.created_calls = []

        def create(self, text, voice, speed, lang):
            self.created_calls.append((text, voice, speed, lang))
            return np.zeros(10, dtype=np.float32), 24000

    monkeypatch.setattr(speech_reader, "build_kokoro", lambda model, voices: FakeKokoro())
    monkeypatch.setattr(
        speech_reader, "ensure_kokoro_voice_files",
        lambda name, folder: (Path("model"), Path("voices")),
    )

    engine = speech_reader.KokoroEngine(model_name="martin", voices_dir=Path("/dummy"))
    engine.load()

    # Higher speed = faster. Passed through untouched inside Kokoro's range.
    for given, expected in [(1.0, 1.0), (0.8, 0.8), (1.25, 1.25)]:
        list(engine.synthesize("test", given, "martin", "de"))
        assert engine._kokoro.created_calls[-1][2] == expected

    # Outside [0.5, 2.0] Kokoro raises, so clamp.
    list(engine.synthesize("test", 0.2, "martin", "de"))
    assert engine._kokoro.created_calls[-1][2] == 0.5

    list(engine.synthesize("test", 5.0, "martin", "de"))
    assert engine._kokoro.created_calls[-1][2] == 2.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_speech_reader.py::test_kokoro_engine_speed_is_native_and_clamped -v`
Expected: FAIL — with the reciprocal still in place, `speed=0.8` produces `1.25`, so `assert 1.25 == 0.8`.

- [ ] **Step 3: Delete the reciprocal**

In `speech_reader.py`, replace `KokoroEngine.synthesize`:

```python
    def synthesize(self, text: str, speed: float, voice_id: str, lang: str) -> Iterator[AudioChunk]:
        self.load()
        speed = max(0.5, min(speed, 2.0))  # Kokoro asserts this range in create()
        float32_samples, sample_rate = self._kokoro.create(
            text,
            voice=voice_id,
            speed=speed,
            lang=lang,
        )
        clamped = np.clip(float32_samples, -1.0, 1.0)
        int16_samples = (clamped * 32767.0).astype(np.int16)
        yield AudioChunk(int16_samples, sample_rate)
```

- [ ] **Step 4: Flip the tray presets**

In `voice_assistant.py`, replace lines 29–30:

```python
READ_SPEED = 1.0  # playback speed multiplier (1.0 = normal; >1 faster, <1 slower)
SPEED_PRESETS = {"Slow": 0.8, "Normal": 1.0, "Fast": 1.25}
```

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: `23 passed`

- [ ] **Step 6: Verify the direction by ear-free measurement**

Run:
```bash
.venv/bin/python -c "
import sys; sys.path.insert(0, '.')
from speech_reader import SpeechReader
r = SpeechReader('en_US-bella')
slow = r._synth_sentence('Testing playback speed.', 'en_US-bella', 0.8)
fast = r._synth_sentence('Testing playback speed.', 'en_US-bella', 1.25)
n_slow, n_fast = len(slow[0][0]), len(fast[0][0])
print(f'slow={n_slow} samples, fast={n_fast} samples')
assert n_fast < n_slow, 'Fast preset must produce shorter audio'
print('OK: higher speed produces shorter audio')
"
```
Expected: `OK: higher speed produces shorter audio`

- [ ] **Step 7: Commit**

```bash
git add speech_reader.py voice_assistant.py tests/test_speech_reader.py
git commit -m "refactor: speed presets in native Kokoro units

Speed was expressed as Piper's length_scale (higher = slower) and inverted
inside KokoroEngine. With Piper gone the inversion is vestigial. Speed is now
a plain multiplier, clamped to Kokoro's [0.5, 2.0]. Tray labels unchanged."
```

---

### Task 4: Drop the piper-tts dependency

**Files:**
- Modify: `pyproject.toml:7-15`
- Modify: `tests/test_voice_assistant.py:5-12`
- Regenerate: `uv.lock`

**Interfaces:**
- Consumes: a `speech_reader.py` with no `piper` import (Task 2).
- Produces: a venv without `piper-tts`, and `onnxruntime` as a declared direct dependency.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_voice_assistant.py`, at the end of the file:

```python
def test_piper_is_not_a_dependency():
    """piper-tts must be absent from the environment, not merely unimported."""
    import importlib.metadata as md
    import pytest

    with pytest.raises(md.PackageNotFoundError):
        md.version("piper-tts")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_voice_assistant.py::test_piper_is_not_a_dependency -v`
Expected: FAIL — `DID NOT RAISE <class 'importlib.metadata.PackageNotFoundError'>`, because `piper-tts` is still installed.

- [ ] **Step 3: Update `pyproject.toml`**

Replace the `dependencies` list:

```toml
dependencies = [
    "faster-whisper",
    "sounddevice",
    "numpy",
    "evdev",
    "PySide6-Essentials",
    "kokoro-onnx",
    "onnxruntime",
]
```

`onnxruntime` is added because `speech_reader.py` imports it directly (`import onnxruntime as ort`) for `SessionOptions`. It previously arrived transitively.

- [ ] **Step 4: Remove the piper stub from the test harness**

In `tests/test_voice_assistant.py`, replace lines 5–12:

```python
# Stub heavy/hardware modules so importing voice_assistant is cheap and safe.
for name in ["faster_whisper", "evdev"]:
    sys.modules.setdefault(name, types.ModuleType(name))

# Minimal attributes the module accesses at import time.
sys.modules["faster_whisper"].WhisperModel = object
_evdev = sys.modules["evdev"]
```

- [ ] **Step 5: Sync the environment**

Run: `uv sync`
Expected: output includes `Uninstalled 1 package` / `- piper-tts==...`

- [ ] **Step 6: Verify piper is really gone and the app still imports**

Run:
```bash
.venv/bin/python -c "import piper" 2>&1 | tail -1
```
Expected: `ModuleNotFoundError: No module named 'piper'`

Run:
```bash
.venv/bin/python -c "
import sys; sys.path.insert(0, '.')
import speech_reader
r = speech_reader.SpeechReader('de_DE-martin')
r.load()
chunks = r._synth_sentence('Guten Morgen.', 'de_DE-martin', 1.0)
print(f'OK: synthesized {len(chunks[0][0])} samples without piper installed')
"
```
Expected: `OK: synthesized <N> samples without piper installed`

This is the check that catches a hidden transitive reliance on Piper for espeak-ng data or onnxruntime.

- [ ] **Step 7: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: `24 passed`

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock tests/test_voice_assistant.py
git commit -m "build: drop piper-tts, declare onnxruntime directly

kokoro-onnx brings its own espeakng-loader, phonemizer-fork, and onnxruntime,
so removing piper-tts does not break phonemization. onnxruntime is now imported
directly by speech_reader for SessionOptions, so it is declared directly."
```

---

### Task 5: Documentation and on-disk cleanup

**Files:**
- Modify: `README.md:8`, `README.md:54-56`
- Modify: `voice_assistant.py:462`
- Delete: four Piper models and sidecars under `~/.local/share/voice-assistant/voices/`

**Interfaces:**
- Consumes: the final voice catalog from Task 2.
- Produces: no code interface. Documentation only.

- [ ] **Step 1: Update the README feature bullet**

In `README.md`, replace line 8:

```markdown
- **Read Aloud (Text-to-Speech)**: Tap the **Copilot key** (right of AltGr) to have your highlighted selection (or clipboard) read aloud with a natural local voice via [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M). Tap again to stop. Fully offline after a one-time model download.
```

- [ ] **Step 2: Update the README "Read Aloud" section**

In `README.md`, replace the paragraph at lines 54–56:

```markdown
The default voice is `en_US-bella`, downloaded automatically on first use and
cached under `~/.local/share/voice-assistant/voices/kokoro/`. Use the tray menu to
switch the **Output** device, **Voice** (Bella / Sarah / Michael / George / Emma /
Martin German), and **Reading Speed** (Slow / Normal / Fast). Starting dictation
(Right Alt) stops any in-progress reading.
```

- [ ] **Step 3: Update the docstring in `voice_assistant.py`**

Replace line 462:

```python
        """Switch the voice (downloads the model on demand)."""
```

- [ ] **Step 4: Verify no stale references remain**

Run: `grep -rniE "piper|amy|ryan|alan|alba|thorsten|mls" README.md speech_reader.py voice_assistant.py tests/`
Expected: no output.

(`docs/superpowers/` is deliberately excluded — those specs and plans are historical records.)

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: `24 passed`

- [ ] **Step 6: Commit the docs**

```bash
git add README.md voice_assistant.py
git commit -m "docs: describe Kokoro voices and the new default"
```

- [ ] **Step 7: Delete the orphaned Piper models**

These are regenerable cache files, not tracked by git. Confirm what is there before removing:

```bash
ls -la ~/.local/share/voice-assistant/voices/*.onnx*
```
Expected: `de_DE-mls-medium`, `de_DE-thorsten-high`, `en_US-amy-medium`, `en_US-ryan-medium` — four `.onnx` and four `.onnx.json` files, ~317 MB.

```bash
rm -v ~/.local/share/voice-assistant/voices/*.onnx ~/.local/share/voice-assistant/voices/*.onnx.json
ls -la ~/.local/share/voice-assistant/voices/
```
Expected: only the `kokoro/` subdirectory remains.

- [ ] **Step 8: Final end-to-end check**

Run: `.venv/bin/python voice_assistant.py --test-read "Guten Morgen, dies ist ein Test." --voice de_DE-martin`
Expected: audible German speech, then `Done.`

Run: `.venv/bin/python voice_assistant.py --test-read "Good morning, this is a test." --voice en_GB-george`
Expected: audible British English speech, then `Done.`

Run: `.venv/bin/python voice_assistant.py --test-read "Good morning, this is a test."`
Expected: audible US English speech in the default Bella voice, then `Done.`

---

## Self-Review

**Spec coverage.** §4 catalog → Task 2. §5 speed → Task 3. §6 module surface → Task 2. §7 dependencies → Task 4. §8 tests → Tasks 1–4. §9 docs and disk → Task 5. §10 verification: full suite (every task), `uv sync` without piper (Task 4 Step 6), `--test-read` by ear (Task 5 Step 8), tray switching including mid-read change (covered by `test_speech_reader_on_the_fly_updates`, updated in Task 2 Step 7). §11 out of scope respected.

**Additions beyond the spec.** Task 1 fixes a shipped bug the spec did not know about: voices sharing a model file also shared a `voice_id`, so `kokoro-en-sarah` spoke as Bella. The new catalog puts five voices on one model, which would have made this affect every English voice. It is sequenced first so no commit ever ships the worse version.

**Type consistency.** `KokoroEngine.__init__(model_name, voices_dir)` is used identically in Tasks 1, 2, and 3. `synthesize(text, speed, voice_id, lang)` is introduced with defaults in Task 1 (Protocol compatibility with `PiperEngine`), and the defaults are dropped in Task 2 Step 5 once it is the only implementation — every call site passes all four arguments from Task 1 Step 4 onward. `VoiceSpec(model, voice_id, lang)` is positional in the catalog and in the Task 2 assertion. `_engines` is keyed on `(engine, model)` tuples through Task 1 and on `spec.model` strings from Task 2 Step 6; the Task 1 test asserts `len(r._engines) == 1` (key-agnostic) and the Task 2 test asserts `set(r._engines) == {"official", "martin"}`.

**Test count arithmetic.** Baseline 23. Task 1 adds 1 → 24. Task 2 adds 2, deletes 2 `ensure_voice_model` tests and 1 superseded test → 23. Task 3 replaces 1 in place → 23. Task 4 adds 1 → 24. Task 5 adds 0 → 24.
