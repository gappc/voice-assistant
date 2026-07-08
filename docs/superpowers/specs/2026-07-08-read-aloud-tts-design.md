# Read-Aloud (Text-to-Speech) — Design

**Date:** 2026-07-08
**Status:** Approved (pending implementation plan)

## Summary

Add local text-to-speech ("read aloud") as the mirror image of the existing
push-to-talk dictation feature. The user highlights (or copies) text anywhere in
the OS, taps a dedicated key, and hears it read aloud in a natural neural voice.
Everything runs locally and offline, matching the project's existing privacy
stance (local Whisper, no network at runtime).

## Goals

- Read the current selection/clipboard aloud on a single key tap (toggle).
- Fast time-to-first-word and near-instant stop, even for long text.
- Fully local/offline after a one-time voice-model download.
- Integrate cleanly with the existing PySide6 tray app without bloating
  `voice_assistant.py`.

## Non-Goals (YAGNI for v1)

- Word/sentence highlighting synced to speech.
- Pause/resume (only start/stop).
- Reading files/PDFs directly, or SSML markup.
- Per-application voices.

## User Experience

1. Highlight text anywhere, or copy it with Ctrl+C.
2. Tap the **Copilot key** (to the right of AltGr).
3. Hear the text read in the "Amy" (en_US, medium) voice.
4. Tap the Copilot key again to stop early; otherwise it stops when the text
   ends.
5. If dictation is started (Right Alt) while reading, reading stops immediately
   so the two never talk over each other.

## Key Decisions (from brainstorming)

| Decision | Choice |
| --- | --- |
| Feature | Read selection/clipboard aloud (TTS reader) |
| Text source | Highlighted selection, fall back to clipboard |
| Engine | Piper (`piper-tts`, `ohf-voice/piper1-gpl` fork), local/offline |
| Start/stop | Toggle on one key |
| Trigger key | Copilot key, detected via `KEY_F23` (code 193) |
| Default voice | `en_US-amy-medium` (auto-downloaded on first run) |
| Interruptions | Read key again (toggle-off), OR starting dictation |

### Why detect `KEY_F23`

The Copilot key does not emit a single keycode. Pressing it emits a three-key
chord all at once — `Meta + Shift + F23` — then releases all three
near-instantly (a momentary tap, no sustained "held" state). This is why it is
unusable for hold-to-talk but ideal for a toggle. `F23` is the distinctive part
of the chord and is effectively never pressed by anything else, so watching for
`KEY_F23` **press** (`value == 1`) reliably means "the Copilot key was tapped."
The Meta/Shift parts are ignored.

**Caveat:** because the key also emits Super+Shift, the Wayland compositor still
sees those (evdev listening is passive and cannot swallow them). On most systems
`Super+Shift+F23` is unbound and harmless. If a stray window/overview action ever
fires on tap, that is the cause and would be addressed separately (e.g. rebinding
to a plain single key).

## Architecture

### New module: `speech_reader.py`

`voice_assistant.py` is already ~400 lines and carries recording, transcription,
injection, and tray responsibilities. Rather than grow it further, read-aloud
lives in a self-contained `SpeechReader` class in a new module. It knows nothing
about evdev or the tray.

**Owns:**
- The loaded Piper voice (`PiperVoice`).
- A single playback worker thread.
- A `threading.Event` stop flag.
- A `sounddevice.OutputStream` (opened on the selected output device).
- The current voice name, playback speed, and selected output device index.

**Interface (small and testable):**
- `load()` — load the Piper voice model. Called once at startup, like Whisper.
- `toggle(text)` — if currently reading, stop; otherwise start reading `text`.
- `stop()` — stop any in-progress reading (used by dictation start and shutdown).
- `set_voice(name)` — switch voice, downloading the model on demand if missing.
- `set_speed(length_scale)` — adjust playback speed.
- `set_output_device(index)` — choose the audio output device for playback.
- `is_reading` — bool property, for the tray status/icon.

**Depends on:** `piper`, `sounddevice`, `numpy` (only `piper` is a new
dependency).

### Wiring in `voice_assistant.py`

- Construct a `SpeechReader` alongside the Whisper model; call `load()` at
  startup.
- In `listen_to_device`, add handling: on `KEY_F23` key-down, call
  `reader.toggle(get_text_to_read())`.
- On `KEY_RIGHTALT` key-down (existing dictation start), also call
  `reader.stop()` first.
- Extend the tray: status line, icon tint, and Output/Voice/Speed submenus.
- Route beeps to the selected output device too (shared setting).

### Text acquisition helper: `get_text_to_read()`

1. Run `wl-paste --primary --no-newline` → the highlighted (primary) selection.
2. If empty or it errors → run `wl-paste --no-newline` → the clipboard.
3. If still empty → play a subtle low "nothing to read" beep and do nothing.

Returns the text string, or `None` when there is nothing to read.

## Playback: streaming for responsiveness

The worker iterates `voice.synthesize(text)`, which yields audio **per sentence**
(`chunk.audio_int16_bytes`, `chunk.sample_rate`, `chunk.sample_width`,
`chunk.sample_channels`). Each chunk's `int16` audio is converted to a NumPy
array and written to an `sd.OutputStream` opened on the selected output device.

Between chunks, the worker checks the stop `Event`. On stop it `abort()`s the
stream and stops synthesizing. Consequences:

- **Fast start:** only the first sentence must synthesize before audio begins.
- **Near-instant stop:** stopping does not wait for the whole text to be
  synthesized; we never synthesize the full document up front.

`SynthesisConfig(length_scale=<speed>)` controls speed; the default is `1.0`.

## Interruption / State Machine

A single lock guards a simple `is_reading` state:

- **Copilot key (F23) down** → `toggle(get_text_to_read())`
  (start if idle; stop if already reading — a second tap is always "stop", never
  "read the new selection").
- **Right Alt down** (dictation start) → `reader.stop()` before recording begins.
- **Natural end of text** → worker clears state and refreshes the tray.

## Tray Integration

- **Status line:** shows `Reading…` in addition to the existing
  `Recording…` / `Idle`.
- **Icon tint:** red = recording, green = reading, blue = idle.
- **Microphone submenu:** unchanged (input device selection).
- **Output submenu (new):** mirrors the Microphone submenu. Built from
  `sd.query_devices()` filtered by `max_output_channels > 0`. Selecting a device
  sets the output for **both** read-aloud playback and the beeps. Current device
  checkmarked.
- **Voice submenu:** Amy / Ryan / Alan / Alba. Selecting a not-yet-downloaded
  voice fetches it on the fly, then switches. Current voice checkmarked.
- **Speed submenu:** Slow / Normal / Fast → `length_scale` 1.3 / 1.0 / 0.8.

## Configuration, Storage, Dependencies

- New constants near the top of `voice_assistant.py`:
  - `READ_KEY_CODE = ecodes.KEY_F23`
  - `DEFAULT_VOICE = "en_US-amy-medium"`
  - `READ_SPEED = 1.0`
- Voice models cached under `~/.local/share/voice-assistant/voices/` (XDG data
  dir), auto-downloaded on first run and when a new voice is selected, via
  `python -m piper.download_voices <name> --data-dir <dir>`.
- `pyproject.toml`: add `piper-tts`. It bundles its own espeak-ng phonemizer, so
  **no new system (apt) packages** are required.
- `.gitignore`: ignore the local voices cache if it ever lands in-tree.
- README: document the Copilot/read-aloud key, the new feature, and voice
  storage.

## Error Handling

All read-aloud failures fail soft and never crash the dictation side:

- **Empty text** (no selection and no clipboard) → subtle low beep, no-op.
- **Missing / failed voice download** → clear console message + tray tooltip;
  reading is skipped.
- **Playback device error** → caught, logged; state reset to idle.

## Testing

- **Unit (logic, no hardware):**
  - `get_text_to_read()` fallback order — mock `subprocess` to simulate
    selection present / selection empty / both empty.
  - `SpeechReader` toggle/stop state transitions — mock the voice and the output
    stream; assert `is_reading` transitions and that `stop()` aborts.
- **Manual end-to-end:** a `--test-read "some text"` CLI flag (mirrors
  `--test-injection`) that loads the voice and speaks a sample with no
  hotkeys/tray. This is the primary way to verify audible output.

## Open Risks

- Piper voice download requires network on first run (documented; matches
  Whisper's first-run model download).
- Compositor-level Super+Shift handling of the Copilot key (see caveat above).
- `onnxruntime` (pulled in by `piper-tts`) increases install size; acceptable
  given the project already ships Whisper.
