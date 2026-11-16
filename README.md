# Local Voice Assistant

A fast, privacy-respecting, and reliable local voice-to-text assistant designed specifically for Linux (Wayland). It records your voice while you hold a global hotkey, transcribes it locally using [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3), and injects the text perfectly anywhere your cursor is, regardless of your keyboard layout.

## Features

- **Push-to-Talk**: Hold **Right Alt** anywhere in the OS to record, release to transcribe and type. 
- **Multilingual Dictation**: English, German and 23 other European languages, detected per recording; there is no language to pick.
- **Read Aloud (Text-to-Speech)**: Tap the **Copilot key** (right of AltGr) to have your highlighted selection (or clipboard) read aloud with a natural local voice via [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M). Tap again to stop. Fully offline after a one-time model download.
- **Wayland-Native Global Hotkeys without the `input` group**: A tiny root-installed system service (`hotkey_helper/voice-hotkeyd.py`) reads the keyboards and passes on only Right Alt and F23 over a Unix socket. Your user, and so every program you run, never gets to read keystrokes.
- **Layout-Independent Text Injection**: Uses `wl-copy` and the RemoteDesktop portal (Ctrl+Shift+V by default) to paste the transcribed text. This ensures 100% accuracy for special characters and prevents mixed-up letters on non-US layouts (like Z/Y on German QWERTZ). The default Ctrl+Shift+V works in terminals as well as editors and IDEs; a tray menu toggle is provided to switch to plain Ctrl+V for apps that reserve Ctrl+Shift+V (e.g. LibreOffice "Paste Special").
- **Audio Feedback**: Plays subtle beeps indicating when recording starts and stops—no need to look at a terminal or status bar.
- **GPU Transcription**: Runs on the GPU through Vulkan (including AMD/Intel integrated graphics) via [parakeet.cpp](https://github.com/mudler/parakeet.cpp), taking ~0.1 s per phrase with almost no CPU use, and falls back to the CPU when no Vulkan device is found. Silence and key clicks produce no text.
- **Background Service & Desktop Integration**: Includes a simple `.desktop` file for easy launching from your app menu or startup applications.

## Requirements

Ensure the following system dependencies are installed:
- `wl-clipboard` (for Wayland clipboard access)
- GNOME or KDE: the paste goes through the xdg-desktop-portal RemoteDesktop
  interface, which wlroots desktops (Sway, Hyprland) do not offer.
- The hotkey helper, installed once as root (below). Your user does **not**
  need to be in the `input` group, and should not be.

### Hotkey helper

Reading keyboards needs the `input` group, and anything in that group can read
every keystroke on the machine, in every session: passwords and PINs typed
elsewhere included. So only a small system service holds it. It runs as a
throwaway user with no network and no writable files, passes on exactly Right
Alt and F23 (press and release, nothing else, not even timing), and only while
your session is the active one. It uses the Python standard library only, so it
loads no code your user can change.

```bash
sudo install -D -m644 -o root -g root hotkey_helper/voice-hotkeyd.py /usr/local/lib/voice-hotkey/voice-hotkeyd.py
sudo install -m644 hotkey_helper/voice-hotkey.socket hotkey_helper/voice-hotkey.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now voice-hotkey.socket
```

`voice-hotkey.socket` names the user who may connect (`SocketUser=chris`):
change it for another account. Check it works: run
`python3 -c "import socket; s=socket.socket(socket.AF_UNIX); s.connect('/run/voice-hotkey.sock'); print(s.recv(64))"`
and press Right Alt; it prints `b'talk down event…'`.

### Keyboard access for pasting

On first start GNOME asks once whether the Voice Assistant may control the
keyboard ("remote interaction"). Allow it; the answer is remembered in
`~/.config/voice-assistant/portal_restore_token`. Delete that file to be asked
again. Unlike `ydotool`, this needs no access to `/dev/uinput`, and keys reach
only your own session.

The keyboard session is opened only for each paste, so GNOME's remote-control
indicator flashes briefly while text is pasted. If it shows up while you are
not dictating, some other program is using the portal.

Changes to `hotkey_helper/` take effect only after installing them again as
root. Why it is built this way, and what risk remains:
[docs/2026-11-16-hotkeys-without-input-group.md](docs/2026-11-16-hotkeys-without-input-group.md).

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
   `uv` will automatically create an isolated virtual environment and fetch all python dependencies (`sounddevice`, `jeepney`, etc.).

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

## Speech Recognition

On first start the assistant downloads the parakeet.cpp library (~60 MB) and the
quantized Parakeet model (~675 MB), checks their SHA-256 checksums, and caches them
under `~/.local/share/voice-assistant/stt/`. The model stays loaded while the
assistant runs: about 1 GB of memory, held by the GPU on integrated graphics.

The Vulkan build picks the first GPU it finds. To force the CPU instead (about
0.2 s per phrase, briefly using all cores), start with `PARAKEET_DEVICE=cpu`.

`tools/stt_bench.py` compares recognizers on your own recordings; see its docstring.
Why Parakeet replaced Whisper, with measurements: [docs/2026-10-14-speech-recognition-parakeet.md](docs/2026-10-14-speech-recognition-parakeet.md).

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

- **No Key Detection**: If holding Right Alt does nothing, the app shows a "Hotkeys unavailable" notice when it cannot reach the helper. Check `systemctl status voice-hotkey.socket voice-hotkey.service` and `journalctl -u voice-hotkey`. Hotkeys are off while another user's session is in front.
- **Read-aloud does nothing**: The Copilot key emits `Meta+Shift+F23`; the assistant listens for `F23`. Confirm your key sends it with `sudo .venv/bin/python key_detector.py` (tap the Copilot key, look for `KEY_F23`). If nothing is selected and the clipboard is empty, you'll hear a short low beep instead.
- **Recordings produce no text**: Each recording logs `Audio: <length>, peak <level>, gain <boost>`. Recordings are centred (DC offset removed) and quiet ones are boosted up to 10×, which rescues built-in laptop mics that deliver very quiet audio on a DC offset. A peak well below ~0.01 while you were speaking means the mic is barely picking you up: pick another device in the tray, or raise the capture/boost level in `alsamixer` (F4).
- **Text isn't pasting**:
    - The assistant uses `wl-copy` and the RemoteDesktop portal to press Ctrl+Shift+V by default (works in terminals and most editors). If you declined GNOME's keyboard prompt, the text is left on the clipboard and a notice says so: delete `~/.config/voice-assistant/portal_restore_token` and restart to be asked again. If your target app does not accept Ctrl+Shift+V (e.g. LibreOffice), toggle off "Use Ctrl+Shift+V (terminal-compatible)" in the tray menu to fall back to plain Ctrl+V.
    - **Test Injection**: Run `uv run voice_assistant.py --test-injection` to verify if the script can type into a focused window without recording.
- **Wrong Key Detected**: The helper picks up every keyboard, including ones plugged in later (within ~2 s). `journalctl -u voice-hotkey` lists each one it listens to.
- **Microphone issues**: Check your default input device in your GNOME/desktop sound settings.

---

> [!NOTE]
> This entire project was **vibe coded**. ✨
