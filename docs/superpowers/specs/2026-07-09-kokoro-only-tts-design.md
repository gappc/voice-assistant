# Kokoro-only TTS — Design

**Date:** 2026-07-09
**Status:** Approved, ready for implementation planning
**Supersedes:** the dual-engine parts of `2026-07-09-hybrid-tts-engine-design.md`

## 1. Goal

Remove the Piper TTS engine, its dependency, and its voice models. Kokoro becomes the
only synthesis engine.

## 2. Why now

The hybrid design landed Piper and Kokoro side by side because Kokoro was measured at
~27x Piper's synthesis cost, making Piper the safe default. Three changes since
(commit `bcea4c4`) removed that constraint:

- Bounding the ONNX Runtime session (4 intra-op threads, spinning disabled) cut
  Kokoro's CPU cost from 42.8 to 7.8 CPU-seconds per ~12s of audio.
- Switching the English model from the dynamically-quantized int8 build to fp16
  took RTF from 0.82 to 0.198.
- Prefetching one sentence ahead of playback removed the 555–860 ms of silence that
  used to fall at every sentence boundary.

Kokoro now synthesizes at RTF ~0.2 with continuous playback. Piper's remaining
advantage is raw speed nobody can hear, at the cost of a second engine, a second
model format, a second speed convention, and an abstraction layer.

## 3. What is lost

Stated plainly, because it is the only irreversible part of this change:

- **German drops from two voices to one.** `de_DE-thorsten-high` and
  `de_DE-mls-medium` go away. Kokoro's German is the `martin` fine-tune, a single
  male voice. There is no German female option: the `kikiri-german-victoria`
  fine-tune ships only `.pth` checkpoints, no ONNX.
- English loses nothing. Kokoro's official pack (already on disk, 54 voices) has
  direct analogues for all four English Piper voices.

## 4. Voice catalog

`VoiceSpec` loses its `engine` field. `model` continues to select which of the two
ONNX files to load — `martin` (fp32, 325 MB) and `official` (fp16, 177 MB) are
genuinely separate sessions, so the lazy engine cache stays.

```python
@dataclass(frozen=True)
class VoiceSpec:
    model: str       # "official" | "martin"
    voice_id: str    # key into the voice pack
    lang: str        # passed to Kokoro.create(lang=...)

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

Catalog keys drop the `kokoro-` prefix, which no longer distinguishes anything, and
adopt the `lang-name` shape the Piper keys used. No voice selection is persisted
anywhere (no `QSettings`, no config file), so renaming is safe.

Mapping from the old catalog:

| Old | New | Note |
|---|---|---|
| `en_US-amy-medium` (default) | `en_US-bella` (default) | |
| `en_US-ryan-medium` | `en_US-michael` | |
| `en_GB-alan-medium` | `en_GB-george` | |
| `en_GB-alba-medium` | `en_GB-emma` | |
| `de_DE-thorsten-high` | `de_DE-martin` | |
| `de_DE-mls-medium` | — | dropped, no analogue |
| `kokoro-en-bella` | `en_US-bella` | renamed |
| `kokoro-en-sarah` | `en_US-sarah` | renamed |
| `kokoro-de-martin` | `de_DE-martin` | renamed |

All five English voice IDs and the `en-gb` lang code were exercised against the real
fp16 model before this design was written; all synthesize correctly.

## 5. Speed

Speed becomes a plain multiplier in Kokoro's native units — higher is faster — instead
of Piper's `length_scale`, where higher was slower.

- `voice_assistant.py`: `READ_SPEED = 1.0`, comment reworded;
  `SPEED_PRESETS = {"Slow": 0.8, "Normal": 1.0, "Fast": 1.25}`.
- `speech_reader.py`: `KokoroEngine.synthesize` clamps `speed` to Kokoro's `[0.5, 2.0]`
  and passes it straight through. The reciprocal conversion is deleted.

Tray menu labels are unchanged. Direction was verified against the real model: speed
`0.8` yields 44032 samples where speed `1.25` yields 28672 for the same text.

## 6. Module surface

Deleted from `speech_reader.py`:

- `PiperEngine`
- the `TTSEngine` Protocol
- `ensure_voice_model()`
- `from piper import PiperVoice, SynthesisConfig`
- the `subprocess` import
- the `if spec.engine == ...` dispatch in `_get_or_create_engine`, which becomes a
  plain `KokoroEngine` cache keyed on `spec.model`

The Protocol existed so two implementations could vary. With one implementation it is
a layer to read through, not a layer that does work. Its original justification — an
iterator-of-chunks contract so Piper would not lose sub-sentence streaming — applied
to Piper specifically; Kokoro yields exactly one chunk per call.

Kept:

- the `AudioChunk` iterator return type on `synthesize()`. `_iter_chunks` already
  materializes it into a list for prefetch; collapsing it to a bare return would
  ripple into the prefetch code for no gain.
- the engine/reader split. `SpeechReader` should not also own model provisioning and
  ONNX session configuration.
- the prefetch producer, its bounding semaphore, and the settings-generation counter.
- the `voices/kokoro/` cache subdirectory. Flattening it would force a 500 MB
  re-download.

## 7. Dependencies

`pyproject.toml` drops `piper-tts` and adds `onnxruntime`, which `speech_reader.py`
now imports directly and should therefore declare directly. `uv.lock` regenerates.

Removing `piper-tts` does not break phonemization: `kokoro-onnx` declares its own
`espeakng-loader`, `phonemizer-fork`, and `onnxruntime`. Verified via
`importlib.metadata.requires`.

## 8. Tests

- `tests/test_voice_assistant.py` — drop the `piper` module stub and its
  `PiperVoice`/`SynthesisConfig` attributes.
- `tests/test_speech_reader.py` — drop `test_ensure_voice_model_downloads_when_missing`
  and `test_ensure_voice_model_skips_download_when_present`. Rewrite
  `test_kokoro_engine_speed_translation` as pass-through-and-clamp rather than
  reciprocal-and-clamp. Update voice names throughout.
- The three concurrency tests added in `bcea4c4` (`prefetch_synthesizes_next_sentence_during_playback`,
  `speech_reader_stops_between_sentences`, `speech_reader_on_the_fly_updates`) are
  unaffected beyond renamed voices.

## 9. Docs and on-disk cleanup

- `README.md:8` and `:54` — reword from Piper to Kokoro, new default voice.
- `voice_assistant.py:462` — docstring drops "Piper".
- `docs/superpowers/specs/2026-07-08-read-aloud-tts-design.md` and
  `2026-07-09-hybrid-tts-engine-design.md` are left untouched. They record decisions
  accurately as of the dates they were made; this document supersedes them rather
  than rewriting them.
- Delete the four Piper models and their `.json` sidecars from
  `~/.local/share/voice-assistant/voices/` (317 MB): `en_US-amy-medium`,
  `en_US-ryan-medium`, `de_DE-thorsten-high`, `de_DE-mls-medium`.

## 10. Verification

1. Full test suite green.
2. `uv sync` after the dependency change, so `piper-tts` is genuinely absent from the
   venv rather than merely unimported, then confirm `voice_assistant` imports and the
   app runs. This is the check that catches a hidden transitive reliance on Piper for
   espeak-ng data or onnxruntime.
3. `--test-read` exercised by ear on one US voice, one GB voice, and the German voice.
4. Tray voice switching and Slow/Normal/Fast still behave, including a mid-read change
   (the settings-generation path).

## 11. Out of scope

- A German female voice. Would require converting `kikiri-german-victoria` from
  `.pth` to ONNX; revisit separately if wanted.
- Exposing more of Kokoro's 54 voices. The six-entry flat menu is sufficient; grouping
  by language would be needed beyond that.
- Piper's own unconfigured ONNX session (all hardware threads, spinning on) — moot
  once Piper is gone.
