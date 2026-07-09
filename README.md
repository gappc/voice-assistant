# Local Voice Assistant

A fast, privacy-respecting, and reliable local voice-to-text assistant designed specifically for Linux (Wayland). It records your voice while you hold a global hotkey, transcribes it locally using Whisper, and injects the text perfectly anywhere your cursor is, regardless of your keyboard layout.

## Features

- **Push-to-Talk**: Hold **Right Alt** anywhere in the OS to record, release to transcribe and type. 
- **Read Aloud (Text-to-Speech)**: Tap the **Copilot key** (right of AltGr) to have your highlighted selection (or clipboard) read aloud with a natural local voice via [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M). Tap again to stop. Fully offline after a one-time model download.
- **Wayland-Native Global Hotkeys**: Uses `evdev` to capture key events directly from the hardware, bypassing Wayland security limitations that block traditional listeners like `pynput`.
- **Layout-Independent Text Injection**: Uses `wl-copy` and `ydotool` (Ctrl+Shift+V simulation by default) to paste the transcribed text. This ensures 100% accuracy for special characters and prevents mixed-up letters on non-US layouts (like Z/Y on German QWERTZ). The default Ctrl+Shift+V works in terminals as well as editors and IDEs; a tray menu toggle is provided to switch to plain Ctrl+V for apps that reserve Ctrl+Shift+V (e.g. LibreOffice "Paste Special").
- **Audio Feedback**: Plays subtle beeps indicating when recording starts and stops—no need to look at a terminal or status bar.
- **Voice Activity Detection (VAD)**: Built-in Silero VAD filtering ensures cleaner, more accurate transcriptions by trimming silent audio.
- **Background Service & Desktop Integration**: Includes a simple `.desktop` file for easy launching from your app menu or startup applications.

## Requirements

Ensure the following system dependencies are installed:
- `ydotool` (for keyboard simulation)
- `wl-clipboard` (for Wayland clipboard access)
- Your user must be in the `input` group for `evdev` access (`sudo usermod -aG input $USER`).

## Installation

This project uses `uv` for seamless, lightning-fast Python dependency management.

1. **Install uv**: If you haven't already:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. **Setup and Run**: 
   ```bash
   uv run voice_assistant.py
   ```
   `uv` will automatically create an isolated virtual environment and fetch all python dependencies (`faster-whisper`, `sounddevice`, `evdev`, etc.).

## Usage

1. **Run the assistant:**
   ```bash
   uv run voice_assistant.py
   ```
2. You'll see "Assistant ready!" in the terminal. It runs entirely in the background.
3. Place your cursor anywhere you want to type.
4. Hold the **Right Alt** (AltGr) key. You will hear a high beep.
5. Speak your text.
6. Release the key. You will hear a lower beep, and your transcribed text will be instantly pasted.

## Read Aloud

1. Highlight any text (or copy it with Ctrl+C).
2. Tap the **Copilot key** (located to the right of AltGr).
3. The text is read aloud in the selected voice. Tap the key again to stop.

The default voice is `en_US-bella`, downloaded automatically on first use and
cached under `~/.local/share/voice-assistant/voices/kokoro/`. Use the tray menu to
switch the **Output** device, **Voice** (Bella / Sarah / Michael / George / Emma /
Martin German), and **Reading Speed** (Slow / Normal / Fast). Starting dictation
(Right Alt) stops any in-progress reading.

Test speech without the hotkey:
```bash
uv run voice_assistant.py --test-read "Hello from the local voice assistant."
```

## Settings

Every setting you change from the tray menu — input device, output device, voice,
reading speed, and the Ctrl+Shift+V paste-mode toggle — is remembered across
restarts. Settings are stored as JSON in `~/.config/voice-assistant/settings.json`
(respects `$XDG_CONFIG_HOME`). Delete the file to reset everything to defaults.

`--device` on the command line overrides the saved input device for that run only;
it does not overwrite the saved value.

## Desktop Integration

To add the assistant to your app launcher so you can start it without opening a terminal:

```bash
mkdir -p ~/.local/share/applications
sed "s|/PATH/TO/PROJECT|$(pwd)|g" voice-assistant.desktop > ~/.local/share/applications/voice-assistant.desktop
```

You can then search for "Voice Assistant" in your application menu, or add it to your "Startup Applications" to have it always ready.

## Troubleshooting

- **No Key Detection**: If the script boots up but holding Right Alt does nothing, make sure your user is part of the `input` group. (You may need to log out and log back in after joining).
- **Read-aloud does nothing**: The Copilot key emits `Meta+Shift+F23`; the assistant listens for `F23`. Confirm your key sends it with `uv run key_detector.py` (tap the Copilot key, look for `KEY_F23`). If nothing is selected and the clipboard is empty, you'll hear a short low beep instead.
- **Text isn't pasting**:
    - The assistant uses `wl-copy` and `ydotool` to simulate a Ctrl+Shift+V paste by default (works in terminals and most editors). Ensure both are installed. If your target app does not accept Ctrl+Shift+V (e.g. LibreOffice), toggle off "Use Ctrl+Shift+V (terminal-compatible)" in the tray menu to fall back to plain Ctrl+V.
    - **Test Injection**: Run `uv run voice_assistant.py --test-injection` to verify if the script can type into a focused window without recording.
    - **Environment Check**: Run `uv run tools/debug_input.py` to check for `ydotool` socket issues and list your input devices.
    - **Permissions**: Ensure your user is in the `input` group and has access to `/dev/uinput`.
- **Wrong Key Detected**: If your keyboard isn't being picked up (e.g., you suspected `/dev/input/event16`), the script automatically finds all keyboards. You can verify which one is being used in the terminal output at startup.
- **Microphone issues**: Check your default input device in your GNOME/desktop sound settings.

---

> [!NOTE]
> This entire project was **vibe coded**. ✨
