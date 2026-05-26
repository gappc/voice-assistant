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

# --- Configuration ---
MODEL_SIZE = "base"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
CHANNELS = 1
SAMPLERATE = 16000
TRIGGER_KEY_CODE = ecodes.KEY_RIGHTALT  # Scan code 100
KEYBOARD_LAYOUT = "de" # Set to "de" for German, "us" for US

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
            sd.play(tone.astype(np.float32), SAMPLERATE)
        except Exception as e:
            print(f"Warning: Could not play beep: {e}")

    def start_recording(self, device_path):
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
        except (OSError, Exception) as e:
            print(f"Device error or disconnected: {device.name} - {e}")

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
            status = "Recording..." if self.is_recording else "Idle"
            self.status_action.setText(f"Status: {status}")
            # Update the icon as well
            self.tray.setIcon(self.create_tray_icon())

    def create_tray_icon(self):
        """Creates a custom microphone icon using QPainter."""
        pixmap = QPixmap(64, 64)
        pixmap.fill(Qt.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Color: Red for recording, Blue for idle
        color = QColor(255, 80, 80) if self.is_recording else QColor(100, 200, 255)
        
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

    def run(self):
        self.ensure_ydotoold()
        keyboards = self.find_keyboards()
        if not keyboards:
            print("No keyboard devices found! Check permissions or /dev/input permissions.")
            return

        # Initialize the audio stream
        self.set_input_device(self.current_device)
        
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
    args = parser.parse_args()

    assistant = VoiceAssistant(initial_device=args.device)
    
    if args.test_injection:
        assistant.test_injection()
    else:
        assistant.run()

if __name__ == "__main__":
    main()
