# Hybrid TTS Engine (Piper + Kokoro) — Design & Handoff Doc

**Status:** Investigated and largely designed via conversation; NOT yet formally approved section-by-section, NOT yet planned/implemented. Intended for handoff to another LLM/agent session (with or without repo access) to turn into an implementation plan and build.

This document is self-contained: a fresh session with no prior context should be able to read it and implement the feature without needing the original conversation.

---

## 1. Project context

**Repo:** a local, privacy-respecting Linux (Wayland) voice assistant. Push-to-talk dictation (hold Right Alt → record → local Whisper transcription → paste via `wl-copy`/`ydotool`) already works well and is out of scope here.

**Existing read-aloud feature** (already shipped, working): tap the "Copilot key" (`KEY_F23`) to toggle reading the highlighted selection (or clipboard fallback) aloud via **Piper TTS**, fully local/offline on CPU. Tap again to stop. Starting dictation interrupts reading. A PySide6 tray exposes Output device / Voice / Reading Speed submenus.

**Files this design touches:**
- `speech_reader.py` (169 lines) — the `SpeechReader` class, currently hard-wired to Piper. Full current source reproduced in §3 below.
- `voice_assistant.py` (546 lines) — wires `SpeechReader` into the evdev hotkey listener and the PySide6 tray. Relevant symbols: `READ_KEY_CODE`, `DEFAULT_VOICE`, `READ_SPEED`, `VOICE_PRESETS` (list of Piper voice names), `SPEED_PRESETS` (dict), `self.reader` (a `SpeechReader` instance), `on_read_key()`, `get_output_devices()`/`set_output_device()`, `build_voice_menu()`, `on_select_voice()`, `build_speed_menu()`.
- `text_source.py` — unrelated to this change (selection/clipboard acquisition), leave as-is.
- `pyproject.toml` — currently depends on `piper-tts`; this design adds `kokoro-onnx`.
- Voice models cache at `~/.local/share/voice-assistant/voices/` (XDG data dir), currently flat `<piper-voice-name>.onnx` files.

**Existing voice catalog (Piper only, today):**
```python
VOICE_PRESETS = [
    "en_US-amy-medium",
    "en_US-ryan-medium",
    "en_GB-alan-medium",
    "en_GB-alba-medium",
    "de_DE-thorsten-high",
    "de_DE-mls-medium",
]
SPEED_PRESETS = {"Slow": 1.3, "Normal": 1.0, "Fast": 0.8}
```

---

## 2. Problem statement

The user finds Piper's voice quality mediocre. Investigated **Kokoro-82M** (Apache-2.0, ~82M params, widely regarded as one of the best-sounding small open TTS models) as a replacement, including German fine-tunes from the community `kikiri-tts` project ("Martin" male, "Victoria" female) — the user listened to samples and preferred them clearly over Piper's German voice.

The question this design answers: **how to add Kokoro without breaking the interaction model the existing `SpeechReader` was built around** (near-instant start, near-instant stop).

---

## 3. Current `SpeechReader` (baseline — full source, for reference)

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
            [sys.executable, "-m", "piper.download_voices", name,
             "--data-dir", str(voices_dir)],
            check=True,
        )
    return onnx_path


class SpeechReader:
    """Reads text aloud with Piper, streaming per sentence for prompt stop."""

    BLOCK = 2048  # frames per write; bounds stop latency to a fraction of a second

    def __init__(self, voice_name, speed=1.0, output_device=None,
                 voices_dir=None, on_state_change=None):
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

    def load(self):
        onnx_path = ensure_voice_model(self._voice_name, self._voices_dir)
        self.voice = PiperVoice.load(str(onnx_path))

    @property
    def is_reading(self):
        return self._reading

    def start(self, text):
        if self._reading:
            self.stop()
        self._stop.clear()
        with self._lock:
            self._reading = True
        self._notify()
        self._worker = threading.Thread(target=self._read_worker, args=(text,), daemon=True)
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
                    stream = sd.OutputStream(samplerate=sample_rate, channels=1,
                                              dtype="int16", device=self._output_device)
                    stream.start()
                for i in range(0, len(samples), self.BLOCK):
                    if self._stop.is_set():
                        stream.abort()
                        return
                    stream.write(samples[i:i + self.BLOCK])
            if stream is not None and not self._stop.is_set():
                stream.stop()
        finally:
            if stream is not None:
                stream.close()

    def _read_worker(self, text):
        try:
            self._play_chunks(self._iter_chunks(text))
        except Exception as exc:
            print(f"[read] playback error: {exc}", file=sys.stderr)
        finally:
            with self._lock:
                self._reading = False
            self._notify()
```

The key property to preserve: **`_play_chunks` checks the stop flag between small (2048-frame) blocks**, so stop is near-instant regardless of how big an individual chunk is *once synthesized* — the bottleneck for responsiveness is entirely how quickly `_iter_chunks` can produce the *first* chunk, and how large each chunk is (a chunk currently mid-synthesis cannot be interrupted, only chunks not yet started can be skipped).

---

## 4. Investigation findings (all verified — source-inspected and/or benchmarked, not guessed)

### 4.1 Kokoro-onnx custom voice loading works
`kokoro_onnx.Kokoro.__init__(model_path: str, voices_path: str, ...)` (verified via `inspect.getsource`):
```python
self.sess = rt.InferenceSession(model_path, providers=providers)
self.voices: np.ndarray = np.load(voices_path)
```
No lock-in to official filenames — any `.onnx` + any file `np.load()` can read (`.npz`, or a `.bin` that's actually an npz under the hood) works. `get_voice_style(name)` does `self.voices[name]`.

### 4.2 German voices available
- **Martin** (male): ready-made ONNX export exists at community HF repos `Godelaune/Kokoro-82M-ONNX-German-Martin` and mirror `huggingFresse/Kokoro-82M-ONNX-German-Martin` — files `kokoro-martin.onnx` + `voices-martin.npz`, directly downloadable (the Docker/FastAPI wrapper in that repo is optional tooling, not required — the raw files load fine via `Kokoro(model_path=..., voices_path=...)`). Voice key inside the npz is expected to be `dm_martin` (naming convention: `d`=German, `m`=male, matching Kokoro's own `af_`/`am_`/`bf_`/`bm_` scheme).
- **Victoria** (female): only a **PyTorch** checkpoint confirmed (`kikiri-tts/kikiri-german-victoria`, used via the separate, heavier `kokoro` PyPI package's `KPipeline(lang_code="de", model_path=...)`, not `kokoro-onnx`). No confirmed ONNX export as of this investigation. **Out of scope for this design** unless someone exports her to ONNX later — pulling in full PyTorch just for one voice is a disproportionate dependency cost for a CLI tool that otherwise only needs `onnxruntime`.
- Licensing: both fine-tunes are Apache-2.0 (base Kokoro-82M license), same permissive terms as Piper's voices. ONNX weights are a static compute graph (not pickled Python objects), so there's no arbitrary-code-execution risk in downloading third-party `.onnx` files, unlike loading an untrusted `.pt`/pickle checkpoint would carry.
- Provenance caveat: `Godelaune`/`huggingFresse` are individual community HF accounts, not the canonical `kikiri-tts` org — acceptable for a personal tool, but pin a specific commit/revision if the download code wants to avoid silently picking up changes from `main`.

### 4.3 `lang` parameter is unrestricted
`Kokoro.create()`'s source shows `lang` is passed straight to `self.tokenizer.phonemize(text, lang)` with no allow-list — `lang="de"` works even though German isn't in Kokoro's "officially supported" language list; phonemization rides on espeak-ng's German backend (same underlying phonemizer Piper already uses).

### 4.4 CPU synthesis speed — real benchmark (same 65-word paragraph, same 16-core CPU machine)

| | Piper (`en_US-amy-medium`) | Kokoro (`int8` quantized) |
|---|---|---|
| Model size on disk | 63 MB | 92 MB model + 27 MB shared voices file (covers all ~54 official voices) |
| Load time | 0.70s | 0.41s |
| Synthesis time for 16.4s of audio | 0.46s | 12.44s |
| **Real-time factor (RTF)** | **0.028** (~36x faster than realtime) | **0.76** (~1.3x faster than realtime) |
| Peak RSS (whole process) | 423 MB | 637 MB |

**Kokoro is ~27x slower to synthesize than Piper on CPU.** This is the central constraint driving the design below.

### 4.5 No GPU acceleration available (on the reference machine — verify on target hardware, but the underlying software gaps are universal)
- `PiperVoice.load(use_cuda: bool = False, ...)` — NVIDIA CUDA only.
- `pip install kokoro-onnx[gpu]` → `onnxruntime-gpu` — also NVIDIA CUDA/TensorRT only; AMD ROCm support is an open, unimplemented feature request against `kokoro-onnx`.
- **ONNX Runtime has no native Vulkan execution provider at all** (open upstream GitHub issue since 2021, unresolved). A WebGPU EP exists and uses Vulkan as a backend on Linux, but neither `piper-tts` nor `kokoro-onnx` exposes that path.
- Real-world "Vulkan + Kokoro" setups found in the wild (e.g. AMD's Lemonade server, a Home Assistant community build) run Kokoro on **CPU**; Vulkan there accelerates unrelated components (`llama.cpp`, `whisper.cpp`), not Kokoro TTS itself.
- **Conclusion:** treat both engines as CPU-only for planning purposes unless the target machine has an NVIDIA GPU, in which case `use_cuda=True` (Piper) / `onnxruntime-gpu` (Kokoro) are real options worth revisiting.

### 4.6 `create_stream()` does not solve responsiveness for typical usage
Source-inspected (`inspect.getsource(Kokoro.create_stream)`): it genuinely does incremental synthesis — batches phonemes, runs each batch in a background thread via `loop.run_in_executor`, yields via an `asyncio.Queue` as each batch completes. Structurally similar to Piper's streaming.

**But** the batching is driven by `MAX_PHONEME_LENGTH = 510` (a model context-length safety limit, ≈100-130 words of English) — it is **not** a per-sentence split. Benchmarked: a 65-word test paragraph produced **exactly 1 chunk**. For the actual usage pattern of this feature (a highlighted sentence or short paragraph), `create_stream()` provides **no** practical benefit over the simpler `create()` — you wait for the entire selection to synthesize either way, and cannot interrupt mid-chunk. **Recommendation: use `create()` (not `create_stream()`) and do not attempt to lean on Kokoro's internal chunking — it exists to avoid model input-length limits, not to minimize latency, and building around it would be a fragile, version-dependent maintenance burden.**

---

## 5. Why a naive swap breaks the UX

Current interaction model (Piper):
```
Hotkey → <100ms → first sentence audible → interruptible at any point
```

Naive full swap to Kokoro, for the common case (one sentence or short paragraph):
```
Hotkey → wait several seconds (whole selection synthesizing, un-interruptible) → audio starts
```

That's a materially different, worse-feeling interaction for a feature whose whole value proposition is immediacy. Splitting the *caller's* input into sentences before handing text to the engine recovers **part** of this for multi-sentence selections (sentence 1 becomes audible in ~(sentence-1-duration × 0.76) instead of waiting for the whole paragraph) — but for a **single-sentence selection** (arguably the most common real usage — the design's own worked example is "The package will arrive tomorrow."), sentence-level splitting provides **no improvement at all**, since there's only one sentence to split. RTF 0.76 still applies directly to it. Any implementation of this design must not oversell "sentence splitting fixes Kokoro's latency" — it only helps proportionally to how many sentences are in the selection.

---

## 6. Recommended design

### 6.1 `TTSEngine` interface

Two engines behind one small interface. Keep it an **iterator-of-chunks** contract (not a single blocking return) so Piper — which can still stream sub-sentence chunks for a very long single sentence — doesn't lose any of its existing granularity, while Kokoro (which cannot chunk below whatever we hand it) simply yields exactly one chunk per call.

```python
from typing import Iterator, NamedTuple, Protocol
import numpy as np

class AudioChunk(NamedTuple):
    samples: np.ndarray  # mono int16 PCM
    sample_rate: int

class TTSEngine(Protocol):
    def load(self) -> None:
        """Load model weights. Called once at startup or on voice switch."""

    def synthesize(self, text: str, speed: float) -> Iterator[AudioChunk]:
        """Synthesize `text` (already segmented to sentence granularity by the
        caller), yielding one or more audio chunks as they become available."""
```

- `PiperEngine.synthesize(sentence, speed)` wraps the existing `self.voice.synthesize(sentence, syn_config=SynthesisConfig(length_scale=speed))` call directly — unchanged behavior, still naturally streams (a single very long sentence can still yield multiple sub-chunks from Piper's own model).
- `KokoroEngine.synthesize(sentence, speed)` calls `self._kokoro.create(sentence, voice=self._voice_id, lang=self._lang, speed=speed)` **once** (blocking — deliberately not `create_stream()`, per §4.6), wraps the single resulting array in one `AudioChunk`, and yields it.
- `KokoroEngine.speed` note: Kokoro's `speed` parameter is asserted to the range `[0.5, 2.0]` in `create()` — same semantic direction as intuition (higher = faster), the **opposite** direction from Piper's `length_scale` (higher = slower). The engine wrapper must translate between the app's speed presets and each backend's native parameter — do not pass the same raw number to both.

### 6.2 Sentence segmentation lives in `SpeechReader`, not in each engine

`SpeechReader` splits the input text into sentences once, then loops:
```python
for sentence in split_sentences(text):
    for chunk in engine.synthesize(sentence, speed):
        if stop_flag.is_set():
            return
        play(chunk)  # existing BLOCK-sized write loop from _play_chunks, unchanged
```
This is the single biggest win from this redesign: it decouples responsiveness from any given engine's internal batching heuristic (the exact trap `create_stream()` turned out to be), and gives **every** current and future engine consistent, predictable cancellation and first-audio latency behavior.

`split_sentences(text)` — a lightweight regex splitter, no new heavy dependency:
```python
import re

def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]
```
Known limitation, accepted for v1 (YAGNI — do not build an abbreviation-aware tokenizer): this mis-splits on abbreviations like "Dr. Smith" or "e.g. foo". Acceptable for a personal reading tool; worth a one-line code comment, not a fix.

**Safety note (not a bug to fix, just document):** if a "sentence" from this splitter is itself pathologically long (e.g. pasted text with no punctuation at all, which the Wayland selection/clipboard source can absolutely hand us), Kokoro's `create()` does **not** error — it transparently re-batches internally at its own `MAX_PHONEME_LENGTH` boundary and concatenates the result before returning. So the failure mode for an unsplittable giant "sentence" is graceful degradation (longer wait, still works), not a crash. No special-case code is needed for this.

### 6.3 Voice catalog replaces the flat `VOICE_PRESETS` list

```python
from dataclasses import dataclass

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
                                      voice_id="dm_martin", lang="de"),
}
DEFAULT_VOICE = "en_US-amy-medium"  # unchanged — Piper stays the default
```

**Tray UX recommendation: keep the existing flat "Voice" submenu, do not build a separate Engine/Profile selector for v1.** The other LLM's "Fast / Balanced / Natural" profile idea is reasonable future polish but adds UI surface not yet justified by a catalog of only 7 voices across 2 engines. `on_select_voice(name)` just looks up `VOICE_CATALOG[name]` and switches `SpeechReader`'s active engine + voice; the menu-building code (`build_voice_menu` in `voice_assistant.py`) iterates `VOICE_CATALOG` instead of `VOICE_PRESETS` but is otherwise unchanged.

### 6.4 `SpeechReader` changes

- Holds a dict of instantiated engines (lazy-loaded — don't load Kokoro's ~120MB of weights if the user never selects a Kokoro voice) keyed by engine name: `{"piper": PiperEngine(...), "kokoro": KokoroEngine(...)}`.
- `set_voice(name)` looks up `VOICE_CATALOG[name]`, ensures that entry's engine is loaded, and sets it as the active engine.
- `_iter_chunks(text)` (today: wraps `self.voice.synthesize`) becomes: split into sentences, then `for sentence in sentences: yield from self._active_engine.synthesize(sentence, self._speed)`.
- `_play_chunks` (the BLOCK-sized write loop with the stop-flag check) **is unchanged** — this is the part that already does the right thing and doesn't need touching.
- Speed handling: `SPEED_PRESETS` stays defined in terms of Piper's `length_scale` (existing values: Slow 1.3 / Normal 1.0 / Fast 0.8); `KokoroEngine.synthesize` internally converts `length_scale → kokoro_speed` (e.g. `kokoro_speed = 1.0 / length_scale`, clamped to Kokoro's `[0.5, 2.0]` range) so the tray's existing Slow/Normal/Fast menu keeps working unmodified for both engines.

### 6.5 Kokoro voice file provisioning

Mirror the existing Piper download pattern (`ensure_voice_model` in `speech_reader.py`) but for Kokoro's two-file format. Cache under a `kokoro/` subdirectory of the existing voices dir, since a Kokoro voice needs two files and shouldn't collide with Piper's flat `<name>.onnx` files:

```python
import urllib.request

KOKORO_VOICE_SOURCES = {
    "martin": {
        "repo": "Godelaune/Kokoro-82M-ONNX-German-Martin",
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
    base_url = f"https://huggingface.co/{source['repo']}/resolve/main"
    for path, filename in ((model_path, source["model_file"]),
                            (voices_path, source["voices_file"])):
        if not path.exists():
            print(f"[read] downloading {filename} for kokoro voice '{model_name}' ...")
            urllib.request.urlretrieve(f"{base_url}/{filename}", path)
    return model_path, voices_path
```

**Not yet decided / left for the implementer:** whether to pin a specific HF commit hash in `base_url` (safer against upstream changes) instead of tracking `main` (simpler, as sketched above). Given this is a single individual's community repo (not the canonical `kikiri-tts` org), pinning is the safer default — check the repo's actual commit history for a stable ref before hardcoding one. Also verify at implementation time that `urllib.request.urlretrieve` is an acceptable fit (no progress reporting, no resume-on-failure) versus reusing `requests`/`httpx` if either is already a transitive dependency — this sketch prioritizes zero new dependencies over download UX polish.

### 6.6 Explicitly deferred (not in this design's v1)

- **Pipelined synthesis** (synthesize sentence N+1 while sentence N plays, via a producer/consumer queue instead of the current sequential synthesize-then-play loop). Good idea, meshes well with the existing stop-flag model, but is a second nontrivial change to the playback loop — do it as a fast-follow once the engine abstraction itself is proven out, not bundled into the same change.
- **Victoria voice** (blocked on no ONNX export existing; would require the heavier PyTorch `kokoro` package for one voice — disproportionate cost).
- **Engine/Profile selector UI** (Fast/Balanced/Natural) — the flat voice list is sufficient for a 7-entry catalog; revisit if the catalog grows significantly.
- **GPU acceleration** — no viable path found on the reference hardware; revisit only if the target machine has an NVIDIA GPU (both engines already expose CUDA options).

---

## 7. Testing approach (for whoever implements this)

- **Unit:** `split_sentences()` — a handful of representative inputs (single sentence, multi-sentence, trailing/no punctuation, the empty string). `TTSEngine` conformance for both engines with mocked underlying models (assert `synthesize()` yields `AudioChunk`s with the right dtype/shape, not real audio). `SpeechReader`'s sentence-loop with a mock engine that yields multiple sentences, asserting stop-between-sentences behavior (stop flag set after sentence 1 should prevent sentence 2 from ever being synthesized).
- **Manual/audible:** the existing `--test-read TEXT` CLI flag is the right place to add an engine/voice override (e.g. `--test-read "Hallo, wie geht es dir?" --voice kokoro-de-martin`) so both engines can be verified by ear without hotkeys/tray. Confirm: (a) an existing Piper voice still works unchanged, (b) the new Kokoro Martin voice downloads on first use and produces audible, correctly-paced German speech, (c) tray voice switching between a Piper voice and the Kokoro voice works and updates `DEFAULT_VOICE`-style state correctly, (d) stopping mid-read via the hotkey still works promptly for both engines (expect it to feel instant for Piper, and to only interrupt *between* sentences — not instantly — for Kokoro, per §5's documented limitation).

---

## 8. Dependencies to add

```toml
dependencies = [
    # ...existing...
    "kokoro-onnx",
]
```
No new dependency needed for sentence splitting (stdlib `re`).

---

## 9. Summary of what to decide vs. what's settled

**Settled by this investigation (backed by benchmarks/source-inspection, treat as requirements):**
- Piper remains the default engine; Kokoro is opt-in via voice selection, not a replacement.
- Use `Kokoro.create()`, not `create_stream()` — the latter's chunking doesn't help at this feature's typical input size and would add async complexity for no benefit.
- Sentence segmentation must live in the caller (`SpeechReader`), not be delegated to either engine's internal batching.
- Only "Martin" (German, male) ships as a Kokoro voice in v1 — Victoria has no ONNX export yet.
- No GPU work — verified no viable acceleration path exists for either engine on non-NVIDIA hardware.

**Recommended defaults (reasonable calls made to keep this doc actionable, but not load-bearing — revisit if they don't fit):**
- Defer pipelined synthesis to a fast-follow rather than v1.
- Keep the tray's flat Voice submenu rather than adding an Engine/Profile selector.
- Regex-based sentence splitter, accepting abbreviation edge cases, over a heavier NLP tokenizer dependency.
- Cache Kokoro voice files under a `kokoro/` subdirectory of the existing voices cache dir.

If picking this up as an implementation plan: this design is one cohesive unit of work (new `TTSEngine` abstraction + sentence segmentation + Kokoro engine + voice catalog + tray wiring), appropriately scoped for a single plan/spec cycle — no further decomposition needed.
