import os
import sys
import time
import subprocess
import threading
import queue
import glob
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from evdev import InputDevice, categorize, ecodes, list_devices

# --- Configuration ---
MODEL_SIZE = "base"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"
CHANNELS = 1
SAMPLERATE = 16000
TRIGGER_KEY_CODE = ecodes.KEY_RIGHTALT  # Scan code 100
KEYBOARD_LAYOUT = "de" # Set to "de" for German, "us" for US

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

    def record_callback(self, indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        if self.is_recording:
            self.audio_data.append(indata.copy())

    def start_recording(self, device_path):
        with self.lock:
            if not self.is_recording:
                self.active_recording_device = device_path
                self.audio_data = []
                self.is_recording = True
                print(f"\nRecording started (Device: {device_path})...")

    def stop_recording(self, device_path):
        with self.lock:
            if self.is_recording and self.active_recording_device == device_path:
                self.is_recording = False
                self.active_recording_device = None
                print("Recording stopped. Processing...")
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
        # Force English as requested
        segments, info = self.model.transcribe(audio, beam_size=5, language="en")
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

    def run(self):
        keyboards = self.find_keyboards()
        if not keyboards:
            print("No keyboard devices found! Check permissions or /dev/input permissions.")
            return

        print(f"Assistant ready! Hold Right Alt to record.")
        
        # Start the sound stream
        with sd.InputStream(samplerate=SAMPLERATE, channels=CHANNELS, callback=self.record_callback):
            # Listen to all keyboards in parallel
            threads = []
            for kb in keyboards:
                t = threading.Thread(target=self.listen_to_device, args=(kb,), daemon=True)
                t.start()
                threads.append(t)
            
            # Keep main thread alive
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nExiting...")

if __name__ == "__main__":
    assistant = VoiceAssistant()
    assistant.run()
