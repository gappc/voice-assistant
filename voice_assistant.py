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

from speech_reader import SpeechReader
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
DEFAULT_VOICE = "en_US-amy-medium"
READ_SPEED = 1.0  # Piper length_scale (1.0 = normal; >1 slower, <1 faster)
VOICE_PRESETS = [
    "en_US-amy-medium",
    "en_US-ryan-medium",
    "en_GB-alan-medium",
    "en_GB-alba-medium",
    "de_DE-thorsten-high",
    "de_DE-mls-medium",
]
SPEED_PRESETS = {"Slow": 1.3, "Normal": 1.0, "Fast": 0.8}

# Helper for thread-safe UI updates
class UIUpdater(QObject):
    update_signal = Signal()

class VoiceAssistant:
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

    def record_callback(self, indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
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
                self.is_recording = True
                self.update_ui()
                print(f"\nRecording started (Device: {device_path})...")
                self.play_beep(frequency=600) # High beep for start

    def stop_recording(self, device_path):
        with self.lock:
            if self.is_recording and self.active_recording_device == device_path:
                self.is_recording = False
                self.active_recording_device = None
                self.update_ui()
                print("Recording stopped. Processing...")
                self.play_beep(frequency=400) # Lower beep for stop
                if self.audio_data:
                    audio = np.concatenate(self.audio_data, axis=0).flatten()
                    self.transcription_queue.put(audio)
                self.audio_data = []

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
        # Force English as requested, enable Silero VAD for better accuracy
        segments, info = self.model.transcribe(audio, beam_size=5, language="en", vad_filter=True)
        text = " ".join([segment.text for segment in segments]).strip()
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

    def set_input_device(self, device_id_or_name):
        """Sets the input device and restarts the stream if necessary."""
        devices = self.get_input_devices()
        target_index = None

        # Try to find by index first if it's an int or string digit
        try:
            idx = int(device_id_or_name)
            if any(d['index'] == idx for d in devices):
                target_index = idx
        except (ValueError, TypeError):
            pass

        # Try to find by name if not found by index
        if target_index is None:
            for d in devices:
                if device_id_or_name and device_id_or_name.lower() in d['name'].lower():
                    target_index = d['index']
                    break
        
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
        except (OSError, Exception) as e:
            print(f"Device error or disconnected: {device.name} - {e}")

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

    def set_output_device(self, index):
        """Sets the output device for read-aloud playback and beeps."""
        self.output_device = index
        self.reader.set_output_device(index)
        print(f"Output device set to index {index}")

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
        # Disable future signals to avoid double execution
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        print(f"\nReceived signal {signum}, shutting down...")
        self.stop()

    def stop(self):
        print("Stopping Voice Assistant...")
        self.running = False
        # Wake up transcription worker to exit
        self.transcription_queue.put(None)
        # Small delay to allow threads to exit
        time.sleep(0.5)
        if hasattr(self, 'app'):
            self.app.quit()
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

        self.tray_menu.addSeparator()

        paste_action = self.tray_menu.addAction("Use Ctrl+Shift+V (terminal-compatible)")
        paste_action.setCheckable(True)
        paste_action.setChecked(self.paste_with_shift)
        paste_action.toggled.connect(self.set_paste_with_shift)

        self.tray_menu.addSeparator()
        quit_action = self.tray_menu.addAction("Close Voice Assistant")
        quit_action.triggered.connect(self.stop)
        
        self.tray.setContextMenu(self.tray_menu)
        self.tray.activated.connect(self.on_tray_activated)
        self.tray.show()
        
        print("Tray icon started (PySide6).")
        return self.app.exec()

    def set_paste_with_shift(self, enabled):
        self.paste_with_shift = enabled
        print(f"Paste mode: {'Ctrl+Shift+V' if enabled else 'Ctrl+V'}")

    def refresh_mic_menu(self):
        """Populates the microphone selection submenu."""
        self.mic_menu.clear()
        devices = self.get_input_devices()
        group = QActionGroup(self.mic_menu)
        
        for d in devices:
            action = QAction(d['name'], self.mic_menu, checkable=True)
            if d['index'] == self.current_device:
                action.setChecked(True)
            
            # Use a lambda with default argument to capture the current device index
            action.triggered.connect(lambda checked, idx=d['index']: self.set_input_device(idx))
            self.mic_menu.addAction(action)
            group.addAction(action)

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

    def run(self):
        self.ensure_ydotoold()
        keyboards = self.find_keyboards()
        if not keyboards:
            print("No keyboard devices found! Check permissions or /dev/input permissions.")
            return

        # Initialize the audio stream
        self.set_input_device(self.current_device)

        # Load the TTS voice (downloads on first run)
        self.reader.load()

        print(f"Assistant ready! Hold Right Alt to record (with beeps).")
        
        for kb in keyboards:
            threading.Thread(target=self.listen_to_device, args=(kb,), daemon=True).start()
        
        # Start PySide6 Tray (blocks until quit)
        exit_code = self.run_tray()
        
        # Cleanup
        if self.stream:
            self.stream.stop()
            self.stream.close()
        sys.exit(exit_code)

    def test_read(self, text):
        """Speak the given text with the default voice and exit (no hotkeys/tray)."""
        reader = SpeechReader(voice_name=DEFAULT_VOICE, speed=READ_SPEED)
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
    args = parser.parse_args()

    assistant = VoiceAssistant(initial_device=args.device)

    if args.test_injection:
        assistant.test_injection()
    elif args.test_read:
        assistant.test_read(args.test_read)
    else:
        assistant.run()

if __name__ == "__main__":
    main()
