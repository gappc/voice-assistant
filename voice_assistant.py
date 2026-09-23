import os
import sys
import time
import subprocess
import threading
import queue
import signal
import argparse
import re
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

# --- Configuration ---
MODEL_SIZE = "base"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
CHANNELS = 1
SAMPLERATE = 16000
TRIGGER_KEY_CODE = ecodes.KEY_RIGHTALT  # Scan code 100
KEYBOARD_LAYOUT = "de" # Set to "de" for German, "us" for US
READ_KEY_CODE = ecodes.KEY_F23  # Copilot key emits Meta+Shift+F23; F23 is the tell
READ_SPEED = 1.0  # playback speed multiplier (1.0 = normal; >1 faster, <1 slower)
SPEED_PRESETS = {"Slow": 0.8, "Normal": 1.0, "Fast": 1.25}
# Dictation language: saved code -> tray label. "auto" lets Whisper detect it
# per recording, which is unreliable on short phrases with the base model.
STT_LANGUAGES = {"en": "English", "de": "German", "auto": "Auto-detect"}
DEFAULT_STT_LANGUAGE = "en"
# Audio conditioning before transcription. The laptop's built-in mic delivers
# speech ~0.01 deep on a ~0.25 DC offset, which Whisper's VAD drops entirely.
TARGET_PEAK = 0.5  # boost quiet recordings to this peak
MAX_GAIN = 10.0    # never boost more than this, so near-silence stays quiet
# A device can open successfully and still never deliver audio (e.g. a PipeWire
# node whose card another process holds). Warn if nothing arrives by then.
FIRST_FRAME_TIMEOUT = 1.5
# Backstop for a key-up that never arrives (unplugged keyboard, lost event).
# Without it the stream stays open and audio_data grows ~62 KB/s forever.
MAX_RECORDING_SECONDS = 300


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


_RAW_ALSA_RE = re.compile(r"\(hw:\d+,\d+\)")


def _is_raw_alsa_device(name):
    """True for PortAudio names like ``Logitech BRIO: USB Audio (hw:3,0)``.

    Opening one of these claims the ALSA card *exclusively*, so PipeWire (and
    therefore every other application) can no longer use the microphone."""
    return bool(name) and bool(_RAW_ALSA_RE.search(str(name)))


def _normalize_device_token(text):
    """Lowercase and collapse punctuation to ``_`` so ``Logitech BRIO`` matches
    ``alsa_input.usb-046d_Logitech_BRIO_....analog-stereo``."""
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


def _migrate_raw_alsa_name(saved_name, devices):
    """Map a persisted raw ``hw:N,M`` device name onto the equivalent shared
    device, so old settings files stop locking the card. Returns `saved_name`
    unchanged when it is not raw, or when no shared equivalent is present."""
    if not isinstance(saved_name, str) or not _is_raw_alsa_device(saved_name):
        return saved_name
    # "Logitech BRIO: USB Audio (hw:3,0)" -> "logitech_brio"
    label = _normalize_device_token(saved_name.split(":")[0])
    if not label:
        return saved_name
    for d in devices:
        if _is_raw_alsa_device(d["name"]):
            continue
        if label in _normalize_device_token(d["name"]):
            print(f"[settings] migrating input device '{saved_name}' -> '{d['name']}'")
            return d["name"]
    return saved_name



def _condition_audio(audio):
    """Remove DC offset and boost quiet audio toward TARGET_PEAK (never
    attenuate, never above MAX_GAIN). Returns (audio, peak, gain), where
    peak is measured after offset removal."""
    centered = audio - audio.mean()
    peak = float(np.abs(centered).max()) if centered.size else 0.0
    gain = MAX_GAIN if peak == 0 else min(max(TARGET_PEAK / peak, 1.0), MAX_GAIN)
    return (centered * gain).astype(np.float32), peak, gain


def _start_daemon_timer(seconds, callback):
    """Fire `callback` once after `seconds`, without holding up interpreter exit."""
    timer = threading.Timer(seconds, callback)
    timer.daemon = True
    timer.start()
    return timer


class _StartupSettings(NamedTuple):
    input_device: str | None
    output_device: str | None  # unresolved name; resolved via _resolve_device + get_output_devices()
    voice: str
    speed: float
    paste_with_shift: bool
    stt_language: str


def _resolve_startup_settings(initial_device, settings_path):
    """Merge a CLI-provided initial_device with saved tray settings. The CLI
    value wins when given; otherwise the saved value is used, falling back to
    the hardcoded default. Never returns a voice absent from VOICE_CATALOG
    or a dictation language absent from STT_LANGUAGES."""
    settings = tray_settings.load(settings_path)
    voice = settings.voice if settings.voice in VOICE_CATALOG else DEFAULT_VOICE
    speed = settings.speed if settings.speed is not None else READ_SPEED
    stt_language = (
        settings.stt_language if settings.stt_language in STT_LANGUAGES else DEFAULT_STT_LANGUAGE
    )
    input_device = initial_device if initial_device is not None else settings.input_device
    return _StartupSettings(
        input_device=input_device,
        output_device=settings.output_device,
        voice=voice,
        speed=speed,
        paste_with_shift=settings.paste_with_shift,
        stt_language=stt_language,
    )


# Helper for thread-safe UI updates
class UIUpdater(QObject):
    update_signal = Signal()
    notify_signal = Signal(str, str)

class VoiceAssistant:
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
        self.window = None
        self.stream = None
        self._first_frame = threading.Event()
        self._watchdog = None
        self._max_duration = None

        self._settings_path = (
            Path(settings_path) if settings_path else tray_settings.default_config_dir() / "settings.json"
        )
        startup = _resolve_startup_settings(initial_device, self._settings_path)
        # A saved raw hw: name would re-lock the card; map it to the shared node.
        self.current_device = _migrate_raw_alsa_name(
            startup.input_device, self.get_input_devices()
        )
        # Names to persist. Kept apart from current_device/output_device, which
        # hold whatever actually resolved — possibly a fallback.
        self.preferred_input_name = (
            self.current_device if isinstance(self.current_device, str) else None
        )
        self.preferred_output_name = startup.output_device
        # Ctrl+Shift+V works in terminals and most editors; flip off for apps
        # that reserve it (e.g. LibreOffice "Paste Special").
        self.paste_with_shift = startup.paste_with_shift
        self.stt_language = startup.stt_language

        # Setup signal handlers
        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)

        self.ui_updater = UIUpdater()
        self.ui_updater.update_signal.connect(self._do_update_ui)
        self.ui_updater.notify_signal.connect(self._do_notify)

        # Read-aloud (TTS)
        self.output_device = _resolve_device(startup.output_device, self.get_output_devices())
        self.reader = SpeechReader(
            voice_name=startup.voice,
            speed=startup.speed,
            output_device=self.output_device,
            on_state_change=self.update_ui,
        )

    def record_callback(self, indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        self._first_frame.set()
        if self.is_recording:
            self.audio_data.append(indata.copy())

    def play_beep(self, frequency=440, duration=0.1):
        """Plays a short beep using sounddevice."""
        try:
            t = np.linspace(0, duration, int(SAMPLERATE * duration), False)
            tone = np.sin(frequency * t * 2 * np.pi)
            # Ensure tone is float32 for sounddevice
            sd.play(tone.astype(np.float32), SAMPLERATE, device=self.output_device)
        except Exception as e:
            print(f"Warning: Could not play beep: {e}")

    def start_recording(self, device_path):
        self.reader.stop()  # reading yields to dictation
        with self.lock:
            if not self.is_recording:
                self.active_recording_device = device_path
                self.audio_data = []
                # Open the card only now, and only for as long as we record.
                if not self._open_input_stream():
                    self.active_recording_device = None
                    return
                self.is_recording = True
                self.update_ui()
                print(f"\nRecording started (Device: {device_path})...")
                self.play_beep(frequency=600) # High beep for start
                self._watchdog = _start_daemon_timer(
                    FIRST_FRAME_TIMEOUT, self._on_first_frame_timeout
                )
                self._max_duration = _start_daemon_timer(
                    MAX_RECORDING_SECONDS, self._on_recording_timeout
                )

    def stop_recording(self, device_path):
        """Ends the recording started by `device_path`. Key-ups from other
        keyboards are ignored so they cannot cut someone else's dictation."""
        with self.lock:
            if self.is_recording and self.active_recording_device == device_path:
                self._finish_recording()

    def _finish_recording(self):
        """Releases the microphone and queues the audio for transcription.
        Caller must hold self.lock."""
        self.is_recording = False
        self.active_recording_device = None
        for timer in (self._watchdog, self._max_duration):
            if timer is not None:
                timer.cancel()
        self._watchdog = self._max_duration = None
        self._close_input_stream()
        self.update_ui()
        print("Recording stopped. Processing...")
        self.play_beep(frequency=400) # Lower beep for stop
        if self.audio_data:
            audio = np.concatenate(self.audio_data, axis=0).flatten()
            self.transcription_queue.put(audio)
        self.audio_data = []

    def _on_recording_timeout(self):
        """A recording this long means the key-up was lost. Release the mic
        rather than hold it — and keep the audio, it is probably wanted."""
        with self.lock:
            if not self.is_recording:
                return
            print(f"Recording passed {MAX_RECORDING_SECONDS}s; releasing the "
                  "microphone.", file=sys.stderr)
            self._finish_recording()
        self._notify(
            "Recording ended automatically",
            f"Dictation ran past {MAX_RECORDING_SECONDS}s. The trigger key was "
            "probably released without the app seeing it.",
        )

    def _transcription_worker(self):
        while True:
            audio = self.transcription_queue.get()
            if audio is None:
                break
            try:
                text = self.transcribe(audio)
                # Small delay to ensure the OS has processed the Alt key release
                time.sleep(0.3) 
                self.inject_text(text)
            except Exception as e:
                print(f"Error in transcription worker: {e}")
            self.transcription_queue.task_done()

    def transcribe(self, audio):
        # None lets Whisper detect the language; Silero VAD improves accuracy
        language = None if self.stt_language == "auto" else self.stt_language
        audio, peak, gain = _condition_audio(audio)
        print(f"Audio: {len(audio) / SAMPLERATE:.1f}s, peak {peak:.4f}, gain {gain:.1f}x")
        segments, info = self.model.transcribe(audio, beam_size=5, language=language, vad_filter=True)
        text = " ".join([segment.text for segment in segments]).strip()
        if not text:
            print("No speech recognized.")
        return text

    def inject_text(self, text):
        if not text:
            return
        
        print(f"Injecting: {text}")
        try:
            env = os.environ.copy()
            uid = os.getuid()
            if "YDOTOOL_SOCKET" not in env:
                env["YDOTOOL_SOCKET"] = f"/run/user/{uid}/.ydotool_socket"
            
            # 1. Use wl-copy to put text in clipboard
            print("  - Copying to clipboard...")
            subprocess.run(["wl-copy", text], check=True)
            
            # Small delay to ensure the OS has registered the clipboard change
            time.sleep(0.1)
            
            # 2. Use ydotool to trigger paste. Codes: 29=LCTRL, 42=LSHIFT, 47=V.
            if self.paste_with_shift:
                print("  - Triggering Ctrl+Shift+V...")
                paste_keys = ["29:1", "42:1", "47:1", "47:0", "42:0", "29:0"]
            else:
                print("  - Triggering Ctrl+V...")
                paste_keys = ["29:1", "47:1", "47:0", "29:0"]
            subprocess.run(["ydotool", "key", *paste_keys], check=True, env=env)
            print("  - Done.")
            
        except subprocess.CalledProcessError as e:
            # Fallback to typing if wl-copy or ydotool fails
            print(f"Clipboard injection failed, falling back to typing: {e}")
            subprocess.run(["ydotool", "type", "--key-delay", "1", text], check=True, env=env)

    def get_input_devices(self):
        """Returns a list of available input devices."""
        devices = sd.query_devices()
        input_devices = []
        for i, d in enumerate(devices):
            if d['max_input_channels'] > 0:
                input_devices.append({'index': i, 'name': d['name']})
        return input_devices

    def _apply_input_device(self, device_id_or_name):
        """Selects the input device. Does **not** open it: the stream is opened
        only while recording, so an idle assistant never holds the card."""
        devices = self.get_input_devices()
        target_index = _resolve_device(device_id_or_name, devices)

        # Default to system default if still not found
        if target_index is None:
            if device_id_or_name:
                print(f"Warning: Device '{device_id_or_name}' not found. Using default.")
                self._notify(
                    "Microphone not found",
                    f"'{device_id_or_name}' is unavailable. Falling back to the "
                    "system default microphone.",
                )
                # Keep the request as the stored preference: the device may be
                # merely unplugged, and persisting the fallback would silently
                # discard the user's real choice.
                if isinstance(device_id_or_name, str):
                    self.preferred_input_name = device_id_or_name
            target_index = sd.default.device[0]
        else:
            self.preferred_input_name = _device_name(target_index, devices)

        with self.lock:
            was_open = self.stream is not None
            if was_open:
                self._close_input_stream()
            self.current_device = target_index
            name = _device_name(target_index, devices) or target_index
            print(f"Input device set to: {name} (Index: {target_index})")
            # Only reopen mid-recording; otherwise stay closed until needed.
            if was_open:
                self._open_input_stream()

    def _open_input_stream(self):
        """Opens and starts the capture stream on the current device.

        Caller must hold ``self.lock``. Returns True on success."""
        self._first_frame.clear()
        try:
            self.stream = sd.InputStream(
                device=self.current_device,
                samplerate=SAMPLERATE,
                channels=CHANNELS,
                callback=self.record_callback,
            )
            self.stream.start()
            return True
        except Exception as e:
            print(f"Error starting audio stream on device {self.current_device}: {e}")
            self.stream = None
            self._notify("Microphone unavailable", f"Could not open the microphone: {e}")
            return False

    def _close_input_stream(self):
        """Closes the capture stream, releasing the card back to the system."""
        stream, self.stream = self.stream, None
        if stream is None:
            return
        try:
            stream.stop()
            stream.close()
        except Exception as e:
            print(f"Warning: error closing audio stream: {e}", file=sys.stderr)

    def _reload_devices(self):
        """Re-enumerate audio devices so hotplugged hardware shows up.

        PortAudio builds its device list once, inside Pa_Initialize, and never
        re-probes it — a microphone plugged in after startup stays invisible for
        the life of the process. Tearing PortAudio down and back up is the only
        way to refresh that list, and it invalidates every open stream — capture
        *and* playback — so this is a no-op while any audio is in flight."""
        with self.lock:
            if self.stream is not None or self.reader.is_reading:
                return
            try:
                sd._terminate()
                sd._initialize()
            except Exception as e:
                print(f"Warning: could not rescan audio devices: {e}", file=sys.stderr)
                return

            # Indices are only meaningful within one enumeration, so re-resolve
            # through the stored names rather than trusting the old integers.
            resolved = _resolve_device(self.preferred_input_name, self.get_input_devices())
            self.current_device = (
                resolved if resolved is not None else sd.default.device[0]
            )
            self.output_device = _resolve_device(
                self.preferred_output_name, self.get_output_devices()
            )
            self.reader.set_output_device(self.output_device)

    def _on_menu_about_to_show(self):
        """Rescan and rebuild the device submenus each time the tray menu opens."""
        self._reload_devices()
        self.refresh_mic_menu()
        self.refresh_output_menu()

    def _on_first_frame_timeout(self):
        """Opening a device can succeed and still deliver nothing — a PipeWire
        node whose card is locked by another process accepts the stream, then
        blocks forever. Warn instead of recording silence."""
        if self.is_recording and not self._first_frame.is_set():
            print("Warning: no audio frames arrived; the device may be in use.",
                  file=sys.stderr)
            self._notify(
                "Microphone delivered no audio",
                "The selected microphone opened but sent no audio. Another "
                "application may be holding it.",
            )

    def _notify(self, title, message):
        """Shows a tray notification from any thread."""
        self.ui_updater.notify_signal.emit(title, message)

    def _do_notify(self, title, message):
        if hasattr(self, "tray"):
            self.tray.showMessage(title, message, QSystemTrayIcon.Warning)

    def _save_settings(self):
        try:
            tray_settings.save(
                tray_settings.TraySettings(
                    input_device=self.preferred_input_name,
                    output_device=self.preferred_output_name,
                    voice=self.reader._voice_name,
                    speed=self.reader._speed,
                    paste_with_shift=self.paste_with_shift,
                    stt_language=self.stt_language,
                ),
                self._settings_path,
            )
        except Exception as exc:
            print(f"[settings] failed to save: {exc}", file=sys.stderr)

    def set_input_device(self, device_id_or_name):
        """Tray-triggered input device change: apply it and persist the choice."""
        self._apply_input_device(device_id_or_name)
        self._save_settings()

    def find_keyboards(self):
        keyboards = []
        for path in list_devices():
            dev = InputDevice(path)
            name = dev.name.lower()
            # Ignore virtual/noise devices
            if any(x in name for x in ["ydotool", "virtual", "video", "button"]):
                continue
            
            if ecodes.EV_KEY in dev.capabilities():
                if ecodes.KEY_A in dev.capabilities()[ecodes.EV_KEY]:
                    print(f"Detected keyboard: {dev.name} ({dev.path})")
                    keyboards.append(dev)
        return keyboards

    def listen_to_device(self, device):
        print(f"Listening to {device.name}...")
        try:
            for event in device.read_loop():
                if event.type == ecodes.EV_KEY:
                    if event.code == TRIGGER_KEY_CODE:
                        if event.value == 1: # Key down
                            self.start_recording(device.path)
                        elif event.value == 0: # Key up
                            self.stop_recording(device.path)
                    elif event.code == READ_KEY_CODE and event.value == 1:
                        self.on_read_key()
        except Exception as e:
            print(f"Device error or disconnected: {device.name} - {e}")
        finally:
            # If the keyboard died mid-press the key-up will never arrive, so
            # release anything this device is still holding.
            self.stop_recording(device.path)

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

    def get_output_devices(self):
        """Returns a list of available output devices."""
        devices = sd.query_devices()
        return [
            {"index": i, "name": d["name"]}
            for i, d in enumerate(devices)
            if d["max_output_channels"] > 0
        ]

    def _apply_output_device(self, device_id_or_name):
        """Sets the output device for read-aloud playback and beeps."""
        devices = self.get_output_devices()
        target_index = _resolve_device(device_id_or_name, devices)
        if target_index is None:
            if device_id_or_name is not None:
                print(f"Warning: Output device '{device_id_or_name}' not found. Using default.")
                self._notify(
                    "Speaker not found",
                    f"'{device_id_or_name}' is unavailable. Falling back to the "
                    "system default output.",
                )
                # As above: an absent speaker must not erase the preference.
                if isinstance(device_id_or_name, str):
                    self.preferred_output_name = device_id_or_name
        else:
            self.preferred_output_name = _device_name(target_index, devices)
        self.output_device = target_index
        self.reader.set_output_device(target_index)
        print(f"Output device set to index {target_index}")

    def set_speed(self, scale):
        """Tray-triggered speed change: apply it and persist the choice."""
        self.reader.set_speed(scale)
        self._save_settings()

    def set_stt_language(self, code):
        """Tray-triggered dictation language change: apply it and persist the choice."""
        self.stt_language = code
        self._save_settings()

    def set_output_device(self, device_id_or_name):
        """Tray-triggered output device change: apply it and persist the choice."""
        self._apply_output_device(device_id_or_name)
        self._save_settings()

    def ensure_ydotoold(self):
        """Ensures ydotoold is running and sets the socket environment variable."""
        uid = os.getuid()
        os.environ["YDOTOOL_SOCKET"] = f"/run/user/{uid}/.ydotool_socket"
        try:
            # Check if ydotoold is running
            subprocess.run(["pgrep", "ydotoold"], check=True, capture_output=True)
        except subprocess.CalledProcessError:
            print("Starting ydotoold...")
            subprocess.Popen(["ydotoold"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(1)

    def handle_signal(self, signum, frame):
        # Restore the default handlers: stop() no longer exits the process, so
        # a second Ctrl+C must still be able to force-quit if shutdown hangs.
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        print(f"\nReceived signal {signum}, shutting down...")
        self.stop()

    def stop(self):
        print("Stopping Voice Assistant...")
        # Wake up transcription worker to exit
        self.transcription_queue.put(None)
        if hasattr(self, 'app'):
            # Ends app.exec(); run() then does the cleanup and exits. Calling
            # sys.exit() here instead would raise inside the event loop.
            self.app.quit()
        else:
            # Signalled before the tray came up: nothing else will exit for us.
            sys.exit(0)

    def update_ui(self):
        self.ui_updater.update_signal.emit()

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

    def create_tray_icon(self):
        """Creates a custom microphone icon using QPainter."""
        pixmap = QPixmap(64, 64)
        pixmap.fill(Qt.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        if self.is_recording:
            color = QColor(255, 80, 80)   # red: recording
        elif self.reader.is_reading:
            color = QColor(80, 220, 120)  # green: reading
        else:
            color = QColor(100, 200, 255) # blue: idle
        
        # Draw a simple microphone shape
        painter.setBrush(QBrush(color))
        painter.setPen(Qt.NoPen)
        # Mic head
        painter.drawRoundedRect(22, 10, 20, 30, 10, 10)
        # Mic stand
        painter.setBrush(QBrush(QColor(150, 150, 150)))
        painter.drawRect(31, 40, 2, 10)
        # Mic base
        painter.drawRect(22, 50, 20, 3)
        
        painter.end()
        return QIcon(pixmap)

    def on_tray_activated(self, reason):
        if reason == QSystemTrayIcon.Trigger: # Left click
            self.tray.contextMenu().popup(QCursor.pos())

    def run_tray(self):
        """Starts the PySide6 application and tray icon."""
        self.app = QApplication(sys.argv)
        self.app.setQuitOnLastWindowClosed(False)
        
        self.tray = QSystemTrayIcon(self.create_tray_icon(), parent=self.app)
        
        self.tray_menu = QMenu()
        self.status_action = self.tray_menu.addAction("Status: Idle")
        self.status_action.setEnabled(False)
        
        self.tray_menu.addSeparator()
        
        # Microphone selection menu
        self.mic_menu = self.tray_menu.addMenu("Microphone")
        self.refresh_mic_menu()

        # Output device selection menu
        self.output_menu = self.tray_menu.addMenu("Output")
        self.refresh_output_menu()

        # Voice selection menu
        self.voice_menu = self.tray_menu.addMenu("Voice")
        self.build_voice_menu()

        # Reading speed menu
        self.speed_menu = self.tray_menu.addMenu("Reading Speed")
        self.build_speed_menu()

        # Dictation (speech-to-text) language menu
        self.stt_language_menu = self.tray_menu.addMenu("Dictation Language")
        self.build_stt_language_menu()

        self.tray_menu.addSeparator()

        paste_action = self.tray_menu.addAction("Use Ctrl+Shift+V (terminal-compatible)")
        paste_action.setCheckable(True)
        paste_action.setChecked(self.paste_with_shift)
        paste_action.toggled.connect(self.set_paste_with_shift)

        self.tray_menu.addSeparator()
        quit_action = self.tray_menu.addAction("Close Voice Assistant")
        quit_action.triggered.connect(self.stop)
        
        # Rescan devices whenever the menu opens (hotplug).
        self.tray_menu.aboutToShow.connect(self._on_menu_about_to_show)

        # Python runs signal handlers only between bytecodes, and no bytecode
        # executes while Qt owns the loop — without this pump SIGTERM is never
        # delivered and the app can only be killed.
        self._signal_pump = QTimer(self.app)
        self._signal_pump.timeout.connect(lambda: None)
        self._signal_pump.start(200)

        self.tray.setContextMenu(self.tray_menu)
        self.tray.activated.connect(self.on_tray_activated)
        self.tray.show()
        
        print("Tray icon started (PySide6).")
        return self.app.exec()

    def set_paste_with_shift(self, enabled):
        self.paste_with_shift = enabled
        print(f"Paste mode: {'Ctrl+Shift+V' if enabled else 'Ctrl+V'}")
        self._save_settings()

    def refresh_mic_menu(self):
        """Populates the microphone selection submenu.

        Raw ALSA devices (``hw:N,M``) are pushed into a clearly-labelled
        submenu: opening one claims the sound card exclusively, which hides the
        microphone from every other application on the system."""
        self.mic_menu.clear()
        devices = self.get_input_devices()
        group = QActionGroup(self.mic_menu)

        shared = [d for d in devices if not _is_raw_alsa_device(d["name"])]
        raw = [d for d in devices if _is_raw_alsa_device(d["name"])]

        def add_to(menu, entries):
            for d in entries:
                action = QAction(d["name"], menu, checkable=True)
                if d["index"] == self.current_device:
                    action.setChecked(True)
                # Default argument captures the index for this iteration.
                action.triggered.connect(
                    lambda checked, idx=d["index"]: self.set_input_device(idx)
                )
                menu.addAction(action)
                group.addAction(action)

        add_to(self.mic_menu, shared)
        if raw:
            self.mic_menu.addSeparator()
            # Kept as an attribute so Qt does not garbage-collect the submenu.
            self._raw_mic_menu = self.mic_menu.addMenu("Raw ALSA (locks the card)")
            add_to(self._raw_mic_menu, raw)

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
        for name in VOICE_CATALOG.keys():
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
            if abs(scale - self.reader._speed) < 1e-9:
                action.setChecked(True)
            action.triggered.connect(lambda checked, s=scale: self.set_speed(s))
            self.speed_menu.addAction(action)
            group.addAction(action)

    def build_stt_language_menu(self):
        """Populates the dictation-language submenu."""
        self.stt_language_menu.clear()
        group = QActionGroup(self.stt_language_menu)
        for code, label in STT_LANGUAGES.items():
            action = QAction(label, self.stt_language_menu, checkable=True)
            if code == self.stt_language:
                action.setChecked(True)
            action.triggered.connect(lambda checked, c=code: self.set_stt_language(c))
            self.stt_language_menu.addAction(action)
            group.addAction(action)

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

    def run(self):
        self.ensure_ydotoold()
        keyboards = self.find_keyboards()
        if not keyboards:
            print("No keyboard devices found! Check permissions or /dev/input permissions.")
            return

        # Resolve the input device (does not open it: see _open_input_stream)
        self._apply_input_device(self.current_device)

        # Load the TTS voice (downloads on first run)
        self.reader.load()

        print(f"Assistant ready! Hold Right Alt to record (with beeps).")
        
        for kb in keyboards:
            threading.Thread(target=self.listen_to_device, args=(kb,), daemon=True).start()
        
        # Start PySide6 Tray (blocks until quit)
        exit_code = self.run_tray()
        
        # Cleanup
        self._close_input_stream()
        sys.exit(exit_code)

    def test_read(self, text, voice=None):
        """Speak the given text with the default voice and exit (no hotkeys/tray)."""
        voice = voice or DEFAULT_VOICE
        reader = SpeechReader(voice_name=voice, speed=READ_SPEED)
        reader.load()
        print(f"Speaking: {text}")
        reader.start(text)
        reader.wait()
        print("Done.")

    def test_injection(self):
        """Tests the injection logic with a dummy message."""
        self.ensure_ydotoold()
        print("Test injection starting in 3 seconds... Switch to your editor!")
        time.sleep(3)
        test_text = "This is a test transcription from the local voice assistant."
        self.inject_text(test_text)
        print("Test injection sequence complete.")

def main():
    parser = argparse.ArgumentParser(description="Local voice assistant with push-to-talk.")
    parser.add_argument("--device", type=str, help="Preferred input device name or index")
    parser.add_argument("--test-injection", action="store_true", help="Test text injection and exit")
    parser.add_argument("--test-read", type=str, metavar="TEXT",
                        help="Speak the given text and exit")
    parser.add_argument("--voice", type=str, default=DEFAULT_VOICE,
                        help="Voice to use for --test-read")
    args = parser.parse_args()

    assistant = VoiceAssistant(initial_device=args.device)

    if args.test_injection:
        assistant.test_injection()
    elif args.test_read:
        assistant.test_read(args.test_read, voice=args.voice)
    else:
        assistant.run()

if __name__ == "__main__":
    main()
