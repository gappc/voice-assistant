import sys
import types

# Stub heavy/hardware modules so importing voice_assistant is cheap and safe.
for name in ["faster_whisper", "evdev"]:
    sys.modules.setdefault(name, types.ModuleType(name))

# Minimal attributes the module accesses at import time.
sys.modules["faster_whisper"].WhisperModel = object
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




def test_piper_is_not_a_dependency():
    """piper-tts must be absent from the environment, not merely unimported."""
    import importlib.metadata as md
    import pytest

    with pytest.raises(md.PackageNotFoundError):
        md.version("piper-tts")


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
    va.preferred_input_name = "USB Microphone"
    va.preferred_output_name = "Speakers"
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
    va.preferred_input_name = None
    va.preferred_output_name = None
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
    va.preferred_input_name = None
    va.preferred_output_name = None
    va.paste_with_shift = True
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
    va.preferred_input_name = None
    va.preferred_output_name = None
    va.paste_with_shift = True
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


# --- Raw-ALSA detection and migration (2026-08-19 exclusive-lock bug) ---


def test_is_raw_alsa_device_detects_hw_suffix():
    assert voice_assistant._is_raw_alsa_device("Logitech BRIO: USB Audio (hw:3,0)")
    assert voice_assistant._is_raw_alsa_device("HD-Audio Generic: ALC245 Alt Analog (hw:1,2)")


def test_is_raw_alsa_device_false_for_shared_devices():
    assert not voice_assistant._is_raw_alsa_device(
        "alsa_input.usb-046d_Logitech_BRIO_A4E3920F-03.analog-stereo"
    )
    assert not voice_assistant._is_raw_alsa_device("pipewire")
    assert not voice_assistant._is_raw_alsa_device(None)


def test_migrate_raw_alsa_name_maps_to_shared_node():
    """A saved hw: name must migrate to the PipeWire node for the same card,
    otherwise opening it locks the card away from the rest of the system."""
    devices = [
        {"index": 1, "name": "HD-Audio Generic: ALC245 Analog (hw:1,0)"},
        {"index": 6, "name": "pipewire"},
        {"index": 17, "name": "alsa_input.usb-046d_Logitech_BRIO_A4E3920F-03.analog-stereo"},
    ]
    assert (
        voice_assistant._migrate_raw_alsa_name("Logitech BRIO: USB Audio (hw:3,0)", devices)
        == "alsa_input.usb-046d_Logitech_BRIO_A4E3920F-03.analog-stereo"
    )


def test_migrate_raw_alsa_name_leaves_shared_names_untouched():
    devices = [{"index": 17, "name": "alsa_input.usb-046d_Logitech_BRIO_A4E3920F-03.analog-stereo"}]
    assert voice_assistant._migrate_raw_alsa_name("pipewire", devices) == "pipewire"


def test_migrate_raw_alsa_name_keeps_original_when_no_equivalent():
    devices = [{"index": 1, "name": "HD-Audio Generic: ALC245 Analog (hw:1,0)"}]
    saved = "Logitech BRIO: USB Audio (hw:3,0)"
    assert voice_assistant._migrate_raw_alsa_name(saved, devices) == saved


# --- Lazy stream open: the card must not be held while idle ---


class _FakeStream:
    """Records open/start/stop/close so tests can assert the card is only
    held while recording."""

    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.started = False
        self.closed = False
        _FakeStream.instances.append(self)

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def close(self):
        self.closed = True


def _bare_assistant(monkeypatch, devices):
    va = voice_assistant.VoiceAssistant.__new__(voice_assistant.VoiceAssistant)
    va.lock = voice_assistant.threading.Lock()
    va.stream = None
    va.is_recording = False
    va.audio_data = []
    va.current_device = None
    va.output_device = None
    va.active_recording_device = None
    va.notifications = []
    va._first_frame = voice_assistant.threading.Event()
    va._watchdog = None
    monkeypatch.setattr(va, "get_input_devices", lambda: devices)
    monkeypatch.setattr(va, "_notify", lambda title, msg: va.notifications.append((title, msg)))
    monkeypatch.setattr(va, "update_ui", lambda: None)
    monkeypatch.setattr(va, "play_beep", lambda **kw: None)
    va.reader = types.SimpleNamespace(stop=lambda: None, is_reading=False,
                                     set_output_device=lambda idx: None)
    _FakeStream.instances = []
    monkeypatch.setattr(voice_assistant.sd, "InputStream", _FakeStream)
    return va


def test_apply_input_device_does_not_open_a_stream(monkeypatch):
    """Selecting a device must only record the choice — holding the card open
    while idle locks it away from every other application."""
    devices = [{"index": 17, "name": "alsa_input.usb-046d_Logitech_BRIO.analog-stereo"}]
    va = _bare_assistant(monkeypatch, devices)

    va._apply_input_device("BRIO")

    assert va.current_device == 17
    assert _FakeStream.instances == []
    assert va.stream is None


def test_apply_input_device_notifies_when_device_not_found(monkeypatch):
    """A silent stdout warning is invisible; the user must be told in the UI."""
    va = _bare_assistant(monkeypatch, [{"index": 5, "name": "Some Other Mic"}])
    monkeypatch.setattr(voice_assistant.sd, "default", types.SimpleNamespace(device=[5, 5]))

    va._apply_input_device("Logitech BRIO: USB Audio (hw:3,0)")

    assert va.notifications, "expected a user-visible notification"
    assert "BRIO" in va.notifications[0][1]


def test_start_recording_opens_stream_and_stop_closes_it(monkeypatch):
    devices = [{"index": 17, "name": "alsa_input.usb-046d_Logitech_BRIO.analog-stereo"}]
    va = _bare_assistant(monkeypatch, devices)
    va.current_device = 17
    va.transcription_queue = voice_assistant.queue.Queue()

    va.start_recording("/dev/input/event0")
    assert len(_FakeStream.instances) == 1
    stream = _FakeStream.instances[0]
    assert stream.started and stream.kwargs["device"] == 17

    va.stop_recording("/dev/input/event0")
    assert stream.closed
    assert va.stream is None


def test_record_callback_marks_first_frame_seen(monkeypatch):
    va = _bare_assistant(monkeypatch, [])
    va.is_recording = True

    va.record_callback(np_zeros(), 4, None, None)

    assert va._first_frame.is_set()


def test_first_frame_timeout_warns_when_no_audio_arrives(monkeypatch):
    """Opening a PipeWire node whose card is locked succeeds but delivers no
    frames — the stream just hangs. That must surface, not fail silently."""
    va = _bare_assistant(monkeypatch, [])
    va.is_recording = True
    va._first_frame.clear()

    va._on_first_frame_timeout()

    assert va.notifications, "expected a user-visible notification"


def test_first_frame_timeout_silent_when_audio_arrived(monkeypatch):
    va = _bare_assistant(monkeypatch, [])
    va.is_recording = True
    va._first_frame.set()

    va._on_first_frame_timeout()

    assert va.notifications == []


def np_zeros():
    import numpy
    return numpy.zeros((4, 1), dtype="float32")


class _FakeMenu:
    """Stands in for QMenu: records actions and nested submenus."""

    def __init__(self, title=None):
        self.title = title
        self.actions = []
        self.submenus = []
        self.separators = 0

    def clear(self):
        self.actions = []
        self.submenus = []

    def addAction(self, action):
        self.actions.append(action)

    def addSeparator(self):
        self.separators += 1

    def addMenu(self, title):
        menu = _FakeMenu(title)
        self.submenus.append(menu)
        return menu


def test_refresh_mic_menu_hides_raw_alsa_devices_in_submenu(monkeypatch):
    """Raw hw: entries must not sit alongside shared ones — picking one locks
    the card away from the whole system."""
    devices = [
        {"index": 1, "name": "HD-Audio Generic: ALC245 Analog (hw:1,0)"},
        {"index": 17, "name": "alsa_input.usb-046d_Logitech_BRIO.analog-stereo"},
        {"index": 6, "name": "pipewire"},
    ]
    va = voice_assistant.VoiceAssistant.__new__(voice_assistant.VoiceAssistant)
    va.current_device = 17
    monkeypatch.setattr(va, "get_input_devices", lambda: devices)
    monkeypatch.setattr(voice_assistant, "QAction", lambda name, menu, checkable=False: types.SimpleNamespace(
        name=name, setChecked=lambda v: None,
        triggered=types.SimpleNamespace(connect=lambda fn: None)))
    monkeypatch.setattr(voice_assistant, "QActionGroup", lambda menu: types.SimpleNamespace(
        addAction=lambda a: None))
    va.mic_menu = _FakeMenu()

    va.refresh_mic_menu()

    top = [a.name for a in va.mic_menu.actions]
    assert top == ["alsa_input.usb-046d_Logitech_BRIO.analog-stereo", "pipewire"]
    assert len(va.mic_menu.submenus) == 1
    assert [a.name for a in va.mic_menu.submenus[0].actions] == [
        "HD-Audio Generic: ALC245 Analog (hw:1,0)"
    ]


# --- A failed resolution must not erase the user's saved device ---


def _persisting_assistant(monkeypatch, tmp_path, in_devices, out_devices=()):
    va = _bare_assistant(monkeypatch, in_devices)
    va._settings_path = tmp_path / "settings.json"
    va.paste_with_shift = True
    va.reader = _FakeReader(voice_assistant.DEFAULT_VOICE, voice_assistant.READ_SPEED)
    va.reader.stop = lambda: None
    va.reader.set_output_device = lambda idx: None
    va.preferred_input_name = None
    va.preferred_output_name = None
    monkeypatch.setattr(va, "get_output_devices", lambda: list(out_devices))
    return va


def test_absent_microphone_does_not_overwrite_saved_choice(monkeypatch, tmp_path):
    """A mic that is merely unplugged right now must survive: otherwise the
    next unrelated tray change persists the fallback over the real choice."""
    va = _persisting_assistant(monkeypatch, tmp_path, [{"index": 5, "name": "Some Other Mic"}])
    monkeypatch.setattr(voice_assistant.sd, "default", types.SimpleNamespace(device=[5, 5]))
    wanted = "alsa_input.usb-046d_Logitech_BRIO_A4E3920F-03.analog-stereo"

    va._apply_input_device(wanted)
    va._save_settings()

    assert va.current_device == 5, "should still fall back so dictation works"
    assert voice_assistant.tray_settings.load(va._settings_path).input_device == wanted


def test_resolved_microphone_persists_its_canonical_name(monkeypatch, tmp_path):
    devices = [{"index": 17, "name": "alsa_input.usb-046d_Logitech_BRIO.analog-stereo"}]
    va = _persisting_assistant(monkeypatch, tmp_path, devices)

    va._apply_input_device("brio")  # substring match, as the tray/CLI may pass
    va._save_settings()

    saved = voice_assistant.tray_settings.load(va._settings_path)
    assert saved.input_device == "alsa_input.usb-046d_Logitech_BRIO.analog-stereo"


def test_absent_speaker_does_not_overwrite_saved_choice(monkeypatch, tmp_path):
    va = _persisting_assistant(monkeypatch, tmp_path, [], out_devices=[])

    va._apply_output_device("Fancy USB DAC")
    va._save_settings()

    assert voice_assistant.tray_settings.load(va._settings_path).output_device == "Fancy USB DAC"


def test_set_output_device_resolves_by_name(monkeypatch, tmp_path):
    fake_devices = [
        {"name": "Mic Only", "max_input_channels": 2, "max_output_channels": 0},
        {"name": "Speakers", "max_input_channels": 0, "max_output_channels": 2},
    ]
    monkeypatch.setattr(voice_assistant.sd, "query_devices", lambda: fake_devices)

    va = voice_assistant.VoiceAssistant.__new__(voice_assistant.VoiceAssistant)
    va._settings_path = tmp_path / "settings.json"
    va.current_device = None
    va.preferred_input_name = None
    va.preferred_output_name = None
    va.paste_with_shift = True
    va.reader = _FakeReader(voice_assistant.DEFAULT_VOICE, voice_assistant.READ_SPEED)
    va.reader.set_output_device = lambda idx: None
    va.set_output_device("speakers")

    assert va.output_device == 1
    assert voice_assistant.tray_settings.load(va._settings_path).output_device == "Speakers"


# --- Hotplug: PortAudio's device list is built once at Pa_Initialize ---


def test_reload_devices_reinitializes_portaudio_when_idle(monkeypatch):
    """Without a re-init, a microphone plugged in after startup stays
    invisible to the tray picker for the life of the process."""
    calls = []
    monkeypatch.setattr(voice_assistant.sd, "_terminate", lambda: calls.append("terminate"))
    monkeypatch.setattr(voice_assistant.sd, "_initialize", lambda: calls.append("initialize"))
    va = _bare_assistant(monkeypatch, [{"index": 0, "name": "Mic"}])
    va.preferred_input_name = None
    va.preferred_output_name = None
    monkeypatch.setattr(va, "get_output_devices", lambda: [])
    monkeypatch.setattr(voice_assistant.sd, "default", types.SimpleNamespace(device=[0, 0]))

    va._reload_devices()

    assert calls == ["terminate", "initialize"]


def test_reload_devices_skipped_while_stream_is_open(monkeypatch):
    """Pa_Terminate invalidates open streams — never re-init mid-recording."""
    calls = []
    monkeypatch.setattr(voice_assistant.sd, "_terminate", lambda: calls.append("terminate"))
    monkeypatch.setattr(voice_assistant.sd, "_initialize", lambda: calls.append("initialize"))
    va = _bare_assistant(monkeypatch, [])
    va.preferred_input_name = None
    va.preferred_output_name = None
    va.stream = _FakeStream()

    va._reload_devices()

    assert calls == []


def test_reload_devices_reresolves_indices_from_preferred_names(monkeypatch):
    """Indices shift when devices come and go; the saved name is the stable
    handle, so a re-enumeration must re-resolve through it."""
    monkeypatch.setattr(voice_assistant.sd, "_terminate", lambda: None)
    monkeypatch.setattr(voice_assistant.sd, "_initialize", lambda: None)
    va = _bare_assistant(monkeypatch, [
        {"index": 0, "name": "Newly Plugged Webcam"},
        {"index": 4, "name": "alsa_input.usb-046d_Logitech_BRIO.analog-stereo"},
    ])
    va.current_device = 1  # stale index from the previous enumeration
    va.output_device = 9   # stale
    va.preferred_input_name = "alsa_input.usb-046d_Logitech_BRIO.analog-stereo"
    va.preferred_output_name = "Speakers"
    monkeypatch.setattr(va, "get_output_devices", lambda: [{"index": 2, "name": "Speakers"}])

    va._reload_devices()

    assert va.current_device == 4
    assert va.output_device == 2


def test_reload_devices_falls_back_when_preferred_device_vanished(monkeypatch):
    monkeypatch.setattr(voice_assistant.sd, "_terminate", lambda: None)
    monkeypatch.setattr(voice_assistant.sd, "_initialize", lambda: None)
    monkeypatch.setattr(voice_assistant.sd, "default", types.SimpleNamespace(device=[3, 3]))
    va = _bare_assistant(monkeypatch, [{"index": 3, "name": "Built-in Mic"}])
    va.current_device = 7
    va.preferred_input_name = "Unplugged USB Mic"
    va.preferred_output_name = None
    monkeypatch.setattr(va, "get_output_devices", lambda: [])

    va._reload_devices()

    assert va.current_device == 3
    assert va.preferred_input_name == "Unplugged USB Mic", "preference must survive"


def test_menu_about_to_show_rescans_and_rebuilds(monkeypatch):
    va = voice_assistant.VoiceAssistant.__new__(voice_assistant.VoiceAssistant)
    seen = []
    monkeypatch.setattr(va, "_reload_devices", lambda: seen.append("reload"))
    monkeypatch.setattr(va, "refresh_mic_menu", lambda: seen.append("mic"))
    monkeypatch.setattr(va, "refresh_output_menu", lambda: seen.append("output"))

    va._on_menu_about_to_show()

    assert seen == ["reload", "mic", "output"]


# --- SIGTERM must stop the app while the Qt event loop is running ---


def test_stop_does_not_exit_the_process(monkeypatch):
    """stop() runs inside a Qt slot or a deferred signal handler; sys.exit()
    there does not unwind the event loop. run() owns process exit."""
    va = voice_assistant.VoiceAssistant.__new__(voice_assistant.VoiceAssistant)
    va.running = True
    va.transcription_queue = voice_assistant.queue.Queue()
    quits = []
    va.app = types.SimpleNamespace(quit=lambda: quits.append("quit"))

    va.stop()  # must not raise SystemExit

    assert va.running is False
    assert quits == ["quit"]
    assert va.transcription_queue.get_nowait() is None, "worker must be woken to exit"


def test_reload_devices_skipped_while_reading_aloud(monkeypatch):
    """Pa_Terminate aborts *output* streams too: a rescan during read-aloud
    truncates the speech mid-sentence (measured: a 3.0 s tone cut at 0.45 s)."""
    calls = []
    monkeypatch.setattr(voice_assistant.sd, "_terminate", lambda: calls.append("terminate"))
    monkeypatch.setattr(voice_assistant.sd, "_initialize", lambda: calls.append("initialize"))
    va = _bare_assistant(monkeypatch, [])
    va.preferred_input_name = None
    va.preferred_output_name = None
    va.reader = types.SimpleNamespace(is_reading=True, stop=lambda: None,
                                      set_output_device=lambda idx: None)

    va._reload_devices()

    assert calls == []
