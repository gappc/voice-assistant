import os
import sys
import time
import subprocess
import threading
import queue
import signal
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from PySide6.QtWidgets import QApplication, QSystemTrayIcon, QMenu
from PySide6.QtGui import QIcon, QAction, QCursor, QPixmap, QPainter, QColor, QBrush
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
    def __init__(self):
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
        # Append a space for convenience if needed, or keep as is
        print(f"Injecting: {text}")
        try:
            env = os.environ.copy()
            if "YDOTOOL_SOCKET" not in env:
                env["YDOTOOL_SOCKET"] = "/tmp/.ydotool_socket"
            
            # 1. Use wl-copy to put text in clipboard
            subprocess.run(["wl-copy", text], check=True)
            
            # 2. Use ydotool to trigger Ctrl+V (Paste)
            # This is layout-independent and many versions of ydotool support it directly
            subprocess.run(["ydotool", "key", "ctrl+v"], check=True, env=env)
            
        except subprocess.CalledProcessError as e:
            # Fallback to typing if wl-copy fails, though less reliable for layout
            print(f"Clipboard injection failed, falling back to typing: {e}")
            subprocess.run(["ydotool", "type", "--key-delay", "1", text], check=True, env=env)

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
        os.environ["YDOTOOL_SOCKET"] = "/tmp/.ydotool_socket"
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
        quit_action = self.tray_menu.addAction("close voice assistant")
        quit_action.triggered.connect(self.stop)
        
        self.tray.setContextMenu(self.tray_menu)
        self.tray.activated.connect(self.on_tray_activated)
        self.tray.show()
        
        print("Tray icon started (PySide6).")
        sys.exit(self.app.exec())

    def run(self):
        self.ensure_ydotoold()
        keyboards = self.find_keyboards()
        if not keyboards:
            print("No keyboard devices found! Check permissions or /dev/input permissions.")
            return

        print(f"Assistant ready! Hold Right Alt to record (with beeps).")
        
        # Start input stream and keyboard threads
        with sd.InputStream(samplerate=SAMPLERATE, channels=CHANNELS, callback=self.record_callback):
            for kb in keyboards:
                threading.Thread(target=self.listen_to_device, args=(kb,), daemon=True).start()
            
            # Start PySide6 Tray (blocks until quit)
            self.run_tray()

def main():
    assistant = VoiceAssistant()
    assistant.run()

if __name__ == "__main__":
    main()
