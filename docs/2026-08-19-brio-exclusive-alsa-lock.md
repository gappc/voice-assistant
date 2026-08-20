# Voice assistant locks the BRIO mic away from the whole system

**Status:** fixed 2026-08-20 in `voice_assistant.py` (see *Fix* below)
**Found:** 2026-08-19, after the assistant had been running ~5h

## Symptom

The Logitech BRIO microphone was unusable in every other application (Slack,
Chromium, GNOME Settings). The webcam enumerated fine at the USB and ALSA layer
(`lsusb`, `arecord -l` → `card 3: BRIO`), but no PipeWire/PulseAudio source
existed for it, so nothing could select it.

## Root cause

`~/.config/voice-assistant/settings.json` had:

```json
"input_device": "Logitech BRIO: USB Audio (hw:3,0)"
```

That is the **raw ALSA** device. Opening `hw:3,0` through PortAudio claims the
sound card *exclusively*, so PipeWire cannot open it:

```
spa.alsa: 'front:3': capture open failed: Device or resource busy
pw.node: (alsa_input...BRIO...analog-stereo-132) suspended -> error
wireplumber: Failed to create alsa_input...BRIO...analog-stereo
```

The card's profile stayed `input:analog-stereo` claiming 1 source, but the node
never left `error`, so `pactl`/`wpctl` listed no BRIO source at all.

Killing the process released `/dev/snd/pcmC3D0c`, PipeWire immediately reclaimed
the card, and the source came back as `RUNNING`. Restarting PipeWire alone does
**not** help — it just retries the busy device.

## Why it lasted 5 hours, not just while recording

`_apply_input_device()` opens `sd.InputStream(...)` and calls `.start()` once at
startup, then keeps it open for the entire process lifetime. The exclusive lock
is therefore held continuously, whether or not the user is recording.

## Second, latent bug: the saved name no longer resolves

Device visibility is **mutually exclusive** — whichever side owns card 3 is the
side that shows up in `sd.query_devices()`:

| Card owned by | Visible entry | Absent |
|---|---|---|
| the assistant (raw) | `Logitech BRIO: USB Audio (hw:3,0)` | PipeWire node |
| PipeWire | `alsa_input.usb-046d_Logitech_BRIO_A4E3920F-03.analog-stereo` | `hw:3,0` |

So now that PipeWire holds the card, the saved string `"...(hw:3,0)"` matches
nothing. `_resolve_device()` returns `None`, and `_apply_input_device()` falls
back to `sd.default.device[0]` — currently the **DisplayLink dock**, not the
BRIO. The user gets the wrong microphone with only a `Warning: ... not found.
Using default.` on stdout.

Related fragility: `_resolve_device()` also accepts an integer index, but
indices shift as devices appear and disappear (the dock moved between
enumerations during this investigation). Indices are not safe to persist.

## Corrections to the analysis above (verified 2026-08-20)

Two details were wrong or imprecise:

- **The fallback was not the dock.** `sd.default.device[0]` is the `default`
  ALSA PCM, which follows the *system* default source — `pactl
  get-default-source` was the BRIO itself. Still wrong in kind (it does not
  pin a device), but it did not land on the DisplayLink dock.
- **Visibility is not "whoever owns the card".** PortAudio probes each ALSA
  device *by opening it*, once, during `Pa_Initialize`. A busy device is
  dropped from that list, and the list is never re-probed afterwards.
  Verified: holding `hw:4,0` open in-process left `sd.query_devices()`
  byte-identical.

  Two consequences: resolution only breaks when the card is busy **at the
  moment the process starts** (e.g. restarting the assistant during a call),
  and the tray device list is frozen at startup — hotplugged microphones never
  appear, unplugged ones linger.

## A trap in fix 1, found while implementing

Opening the *PipeWire node* while the card is locked **succeeds** — no
exception — and then delivers no frames at all; a 1.5 s blocking read hung
indefinitely. Switching the saved name to the node without also opening lazily
would have traded "silently records from the wrong microphone" for "records
nothing, forever, with no error". Hence the watchdog below.

## Direction for the fix

1. **Persist the PipeWire node name**, not the raw ALSA one:
   `alsa_input.usb-046d_Logitech_BRIO_A4E3920F-03.analog-stereo`
   (`pipewire` / `default` also work, but they follow the *system default*
   source, which is the dock right now — they will not pin the BRIO.)
2. **Stop offering raw `hw:N,M` devices** in the tray picker, or deprioritise
   them, so a user cannot pick a device that locks the card system-wide.
3. **Consider opening the stream lazily** — only while actually recording —
   so the assistant never holds a capture device idle for hours.
4. **Surface resolution failures in the UI**, not just stdout; silently falling
   back to a different microphone is hard to notice.
5. Migrate the existing `settings.json` value on load so current users are not
   left pointing at a stale `hw:` name.

## Reproducing / verifying

```bash
# who holds the capture device
fuser -v /dev/snd/pcmC3D0c
cat /proc/asound/card3/pcm0c/sub0/status

# is the BRIO exposed to the rest of the system?
pactl list sources short | grep -i brio

# what the app itself can see
.venv/bin/python3 -c "import sounddevice as sd; print(sd.query_devices())"
```

Healthy state: `fuser` shows `pipewire`, and `pactl` lists
`alsa_input.usb-046d_Logitech_BRIO_A4E3920F-03.analog-stereo`.

## Fix (2026-08-20)

All five directions were implemented in `voice_assistant.py`:

1. **Lazy open** — `_open_input_stream()` / `_close_input_stream()` are now
   called from `start_recording()` / `stop_recording()`. `_apply_input_device()`
   only *resolves* a device; it reopens a stream solely when one was already
   open (i.e. mid-recording). The card is held for the duration of a dictation,
   not the process.
2. **First-frame watchdog** — `record_callback` sets `_first_frame`; a
   `threading.Timer` of `FIRST_FRAME_TIMEOUT` (1.5 s) fires
   `_on_first_frame_timeout()`, which notifies the user if the stream opened
   but no audio arrived. This is what catches the trap described above.
3. **Migration** — `_migrate_raw_alsa_name()` maps a saved `hw:N,M` name onto
   the shared node for the same card by normalising the card label
   (`Logitech BRIO` → `logitech_brio`) and matching it against non-raw device
   names. Applied in `__init__`, so old settings files stop locking the card.
4. **Raw devices deprioritised** — `refresh_mic_menu()` lists shared devices
   first and moves `hw:N,M` entries into a "Raw ALSA (locks the card)" submenu.
5. **Failures surface in the UI** — `UIUpdater.notify_signal` carries
   `_notify()` calls to a tray warning from any thread; unresolved devices,
   open failures and silent streams all raise one instead of a stdout line.

### Verified

- Migration on the real settings file:
  `Logitech BRIO: USB Audio (hw:3,0)` → `alsa_input.usb-046d_..._-03.analog-stereo`.
- Full app startup: `Input device set to: alsa_input.usb-046d_..., Index: 18`,
  and `fuser /dev/snd/pcmC3D0c` reports **free** while the assistant sits idle.
- While recording, `fuser` shows `pipewire` (shared), not an exclusive claim,
  and 1.2 s of real audio was captured (peak 0.057).
- **Simultaneous access works**: `pw-record` captured 2 s of BRIO audio *while*
  the assistant was recording from the same microphone. Both got audio.
- `pytest tests/` — 59 passed.

### Also fixed: a fallback no longer overwrites the saved preference

`_save_settings()` used to persist `_device_name(self.current_device, ...)` —
the device that actually resolved. After a fallback that is the *default*
device, so any later tray change (speed, voice, paste mode) quietly wrote it
over the user's real choice. The same applied to the output device.

`preferred_input_name` / `preferred_output_name` now hold the names to persist,
separately from `current_device` / `output_device`, which hold whatever
resolved. `_apply_input_device()` and `_apply_output_device()` update the
preference on a successful match and **keep the request** when the device is
absent — an unplugged microphone is no longer a lost setting.

A useful side effect: the migration in `__init__` seeds the preference, so the
first tray change rewrites `settings.json` with the shared node name. Old files
self-heal instead of carrying the `hw:` string forever.

Verified against the real settings file: `Logitech BRIO: USB Audio (hw:3,0)` →
`alsa_input.usb-046d_..._-03.analog-stereo` after one tray change; and with the
microphone made absent, an unrelated tray change left that value intact while
dictation still fell back to a working device.

### Follow-ups, also fixed (2026-08-20)

**Hotplug.** `_reload_devices()` tears PortAudio down and back up
(`sd._terminate()` / `sd._initialize()`) — the only way to refresh a list that
is otherwise built once, inside `Pa_Initialize`. It runs from
`_on_menu_about_to_show()`, wired to the tray menu's `aboutToShow`, so the
picker rescans every time it opens. Because `Pa_Terminate` invalidates open
streams, the rescan is a no-op while one is open.

Indices are only meaningful within a single enumeration, so the rescan
re-resolves through `preferred_input_name` / `preferred_output_name` rather
than trusting the old integers. Measured: a rescan costs ~22 ms, and adding a
null sink shifted the BRIO from index 18 to 19 while it still resolved
correctly by name.

**SIGTERM.** Python runs signal handlers only between bytecodes, and no
bytecode executes while Qt owns the loop — so `handle_signal` never fired and
the process could only be `SIGKILL`ed. A `QTimer` firing every 200 ms into an
empty slot gives the interpreter the chance to run them. `stop()` no longer
calls `sys.exit()` either: it ends `app.exec()` via `app.quit()` and lets
`run()` do the cleanup and own process exit. (`sys.exit()` is still used when
the signal arrives before the tray exists — nothing else would exit for us.)

Verified: `kill -TERM` on the running assistant now exits in under a second,
printing `Received signal 15, shutting down...`.
