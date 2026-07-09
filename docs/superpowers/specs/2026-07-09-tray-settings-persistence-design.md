# Tray Settings Persistence — Design

**Date:** 2026-07-09
**Status:** Approved, ready for implementation planning

## 1. Goal

Tray-adjustable settings reset to hardcoded defaults every time `voice_assistant.py`
restarts. Persist them across restarts: input device, output device, voice, reading
speed, and the Ctrl+Shift+V paste-mode toggle.

## 2. Current state

Five settings are adjustable from the tray menu, and none survive a restart:

| Setting | Live attribute | Setter | Hardcoded default |
|---|---|---|---|
| Input (mic) device | `VoiceAssistant.current_device` | `set_input_device(id_or_name)` | `initial_device` (from `--device`, else `None` → system default) |
| Output device | `VoiceAssistant.output_device` | `set_output_device(index)` | `None` → system default |
| Voice | `SpeechReader._voice_name` | `on_select_voice(name)` → `reader.set_voice(name)` | `DEFAULT_VOICE` |
| Reading speed | `SpeechReader._speed` | `reader.set_speed(scale)` (called directly from the speed-menu lambda — no `VoiceAssistant`-level wrapper exists yet) | `READ_SPEED` |
| Paste mode | `VoiceAssistant.paste_with_shift` | `set_paste_with_shift(enabled)` | `True` |

`--voice` and `--device` are the only CLI flags that touch any of this, and only
`--device` reaches the live tray session: `--voice` is wired solely into the
`--test-read` one-shot path, which builds a throwaway `SpeechReader` and never
touches `self.reader`. So the "CLI vs. saved" precedence question from brainstorming
only has a real answer to give for `--device`; the other four settings have no
competing CLI flag today, and none is being added by this change.

## 3. New module: `tray_settings.py`

A plain-Python, Qt-free module — mirrors how `speech_reader.py` is already a
self-contained module with its own test file, not entangled with the tray/GUI code.

```python
def default_config_dir() -> Path:
    """$XDG_CONFIG_HOME/voice-assistant, or ~/.config/voice-assistant."""

@dataclass(frozen=True)
class TraySettings:
    input_device: str | None = None
    output_device: str | None = None
    voice: str | None = None
    speed: float | None = None
    paste_with_shift: bool = True

def load(path: Path) -> TraySettings:
    """Best-effort load. Missing file, corrupt JSON, or a field of the wrong
    shape all fall back to that field's TraySettings() default — never raises."""

def save(settings: TraySettings, path: Path) -> None:
    """Atomic write: write to `<path>.tmp`, then rename over `path`, the same
    pattern ensure_kokoro_voice_files() already uses for downloads."""
```

`TraySettings` deliberately has no dependency on `speech_reader.py` or
`voice_assistant.py` constants (`DEFAULT_VOICE`, `READ_SPEED`). Its defaults are
generic "unset" values (`None`); the caller decides what an unset field means.
This keeps the module trivially unit-testable in isolation and avoids a circular
import (`voice_assistant.py` will import `tray_settings`).

Devices are stored **by name**, not index — indices shift across reboots and when
USB devices are plugged/unplugged in a different order.

Example file, `~/.config/voice-assistant/settings.json`:

```json
{
  "input_device": "USB Microphone",
  "output_device": "Built-in Audio Analog Stereo",
  "voice": "en_US-sarah",
  "speed": 1.25,
  "paste_with_shift": false
}
```

## 4. Integration in `voice_assistant.py`

**Startup (`main()` / `VoiceAssistant.__init__` / `run()`):**

1. `settings = tray_settings.load(settings_path)` — once, at the top of `main()`.
2. Input device: `effective_device = args.device if args.device is not None else settings.input_device`. This is the one setting with a competing CLI flag, so it gets explicit precedence logic; passed to `VoiceAssistant(initial_device=effective_device, ...)` exactly as `args.device` is today.
3. Voice, speed, output device, paste mode: no CLI competitor, so `VoiceAssistant.__init__` uses `settings.X` directly, falling back to today's hardcoded constant when the field is `None` (e.g. `settings.voice if settings.voice in VOICE_CATALOG else DEFAULT_VOICE`).
4. `VoiceAssistant.__init__` gains a `settings_path=None` parameter, defaulting to `tray_settings.default_config_dir() / "settings.json"` — the same optional-override-for-testability shape `SpeechReader.__init__(..., voices_dir=None)` already uses.
5. Output device is resolved from a saved *name* to a live *index* using the same `_resolve_device` helper `set_input_device` uses (see §5) — called **directly**, not through `set_output_device`. Startup never goes through the public, save-triggering setters for any of the five settings (see §6): doing so would rewrite the file on every launch even with no user action.

**Ongoing (tray menu actions):** each of the five setters, after applying the
change, calls a new `VoiceAssistant._save_settings()` helper that snapshots current
live state into a `TraySettings` and calls `tray_settings.save(...)`.

## 5. Symmetry fix: output device gains name-or-index resolution

`set_input_device(device_id_or_name)` already resolves either an index or a
case-insensitive name substring against `get_input_devices()`. `set_output_device(index)`
takes only a raw index — there's no equivalent for output today. Loading a saved
output device *name* needs exactly the resolution `set_input_device` already has, so
this design extracts the shared lookup into one helper:

```python
def _resolve_device(id_or_name: str | int | None, devices: list[dict]) -> int | None:
    """Match by index first, then by case-insensitive name substring.
    Returns None if id_or_name is None or nothing matches."""
```

used by both `set_input_device` (unchanged behavior) and the new
`set_output_device(device_id_or_name)` (widened from `index`-only). This is a small,
targeted refactor motivated directly by this feature — without it, output-device name
resolution would be duplicated rather than shared. `refresh_output_menu`'s existing
callers pass a raw index today, which still works unchanged since indices are matched
first.

## 6. Save/load precedence, spelled out

- **Startup applies effective values directly** (constructor assignment /
  `_apply_input_device`-style internal call), bypassing the save-triggering public
  setters. This is what makes "`--device` overrides the saved value without
  rewriting the file" true: the value that wins is simply the one used to
  initialize state, and initialization never persists.
- **Tray menu actions call the public setters**, which apply the change *and* persist
  it. A user turning a knob in the tray is the only thing that writes the file.
- Net effect: launching with `--device foo` when `bar` is saved runs this session on
  `foo` and leaves `bar` in the file untouched. If the user then changes the input
  device from the tray, whatever they pick becomes the new saved value.

## 7. Error handling

- Missing settings file → `TraySettings()` all-defaults, silent (expected on first
  run).
- Corrupt/unparseable JSON → caught broadly in `load()`, one `stderr` warning, falls
  back to all-defaults — matches the "fail soft" style already used in
  `_read_worker`'s playback-error handling.
- Saved voice no longer in `VOICE_CATALOG` (e.g. a future catalog rename, as just
  happened moving off Piper) → falls back to `DEFAULT_VOICE`. This check happens in
  `voice_assistant.py` when computing the effective voice (§4.3), not inside
  `tray_settings.load()` — `tray_settings.py` has no `VOICE_CATALOG` to check
  against by design (§3). Mirrors the defensive
  `VOICE_CATALOG.get(name, VOICE_CATALOG[DEFAULT_VOICE])` pattern already used
  elsewhere.
- Saved device name no longer present (unplugged) → falls back to system default and
  prints the same `Warning: Device '...' not found. Using default.` message
  `set_input_device` already emits for an unresolvable CLI `--device` value.
- `save()` failing (e.g. read-only filesystem) → caught and logged to `stderr`;
  never crashes the tray action that triggered it.

## 8. Tests

- New `tests/test_tray_settings.py`: round-trip save/load; missing file → defaults;
  corrupt file → defaults; unknown-field-shape (e.g. `speed` as a string) →
  that field's default; atomic write (a failure mid-`save()` leaves the previous
  file intact, not a half-written one).
- `tests/test_voice_assistant.py`: `--device` overrides a saved `input_device`
  without rewriting the file; a setter call (e.g. `set_paste_with_shift`) triggers
  exactly one `save()` with the expected snapshot; `_resolve_device` matches by
  index and by name substring, and returns `None` for no match.

## 9. Out of scope

- No live-reload if the file is hand-edited while the app is running.
- No new CLI flags (`--speed`, `--output-device`, `--paste-with-shift`) — `--voice`
  and `--device` are the only two that exist today, and only `--device` gets
  precedence logic (§2, §6). Adding the others is a separate, optional follow-up.
- Persisting anything outside the tray menu (e.g. `KEYBOARD_LAYOUT`, a source-level
  constant with no UI).
