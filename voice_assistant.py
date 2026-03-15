import os
import sys
import time
import subprocess
import threading
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

# --- Configuration ---
MODEL_SIZE = "base"  # "base" or "small" for better speed on CPU
DEVICE = "cpu"       # faster-whisper is highly optimized for CPU
COMPUTE_TYPE = "int8" # Higher speed, less memory
CHANNELS = 1
SAMPLERATE = 16000   # Whisper expects 16kHz
SILENCE_THRESHOLD = 0.01
SILENCE_DURATION = 1.5 # Seconds of silence before stopping

class VoiceAssistant:
    def __init__(self):
        print(f"Loading Whisper model '{MODEL_SIZE}'...")
        self.model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
        self.is_recording = False
        self.audio_data = []

    def record_callback(self, indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        if self.is_recording:
            self.audio_data.append(indata.copy())

    def start_recording(self):
        self.audio_data = []
        self.is_recording = True
        print("Recording started...")

    def stop_recording(self):
        self.is_recording = False
        print("Recording stopped.")
        if not self.audio_data:
            return None
        
        # Concatenate and normalize
        audio = np.concatenate(self.audio_data, axis=0).flatten()
        return audio

    def transcribe(self, audio):
        print("Transcribing...")
        segments, info = self.model.transcribe(audio, beam_size=5)
        text = " ".join([segment.text for segment in segments]).strip()
        return text

    def inject_text(self, text):
        if not text:
            return
        print(f"Injecting: {text}")
        try:
            # ydotool type requires ydotoold to be running
            env = os.environ.copy()
            env["YDOTOOL_SOCKET"] = "/tmp/.ydotool_socket"
            subprocess.run(["ydotool", "type", text], check=True, env=env)
        except subprocess.CalledProcessError as e:
            print(f"Error injecting text with ydotool: {e}")
            print("Ensure ydotoold is running and you have permissions for /dev/uinput.")

    def run_once(self, duration=None):
        """Records for a fixed duration or until stopped manually via logic (to be added)"""
        with sd.InputStream(samplerate=SAMPLERATE, channels=CHANNELS, callback=self.record_callback):
            self.start_recording()
            if duration:
                time.sleep(duration)
            else:
                input("Press Enter to stop recording...")
            audio = self.stop_recording()
            
        if audio is not None:
            text = self.transcribe(audio)
            self.inject_text(text)

if __name__ == "__main__":
    assistant = VoiceAssistant()
    # For initial testing, we'll run once and wait for Enter
    assistant.run_once()
