# Local Voice Assistant

A fast, privacy-respecting, and reliable local voice-to-text assistant designed specifically for Linux (Wayland). It records your voice while you hold a global hotkey, transcribes it locally using [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3), and injects the text perfectly anywhere your cursor is, regardless of your keyboard layout.

## Features

- **Push-to-Talk**: Hold **Right Alt** anywhere in the OS to record, release to transcribe and type. 
- **Multilingual Dictation**: English, German and 23 other European languages, detected per recording; there is no language to pick.
- **Read Aloud (Text-to-Speech)**: Tap the **Copilot key** (right of AltGr), or on keyboards without one tap **Right Ctrl** on its own, to have your highlighted selection (or clipboard) read aloud with a natural local voice via [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M). Tap again to stop. Fully offline after a one-time model download.
- **Wayland-Native Global Hotkeys without the `input` group**: A tiny root-installed system service (`hotkey_helper/voice-hotkeyd.py`) reads the keyboards and passes on only Right Alt, F23 and lone Right Ctrl taps over a Unix socket. Your user, and so every program you run, never gets to read keystrokes.
- **Layout-Independent Text Injection**: Uses the RemoteDesktop portal to put the transcribed text on the clipboard and paste it (Ctrl+Shift+V by default), without the focus-grabbing helper window of `wl-copy` that makes the screen flicker. This ensures 100% accuracy for special characters and prevents mixed-up letters on non-US layouts (like Z/Y on German QWERTZ). The default Ctrl+Shift+V works in terminals as well as editors and IDEs; a tray menu toggle is provided to switch to plain Ctrl+V for apps that reserve Ctrl+Shift+V (e.g. LibreOffice "Paste Special"). Whatever was on the clipboard before is put back once the app has fetched the text (tray toggle "Restore clipboard after pasting"). GNOME keeps only one format when the portal session closes, so rich text (HTML) comes back as plain text; plain text and images come back unchanged. An empty clipboard cannot be restored through the portal, so the dictation stays on it.
- **Correction Rules**: Words the recognizer keeps getting wrong (names, jargon) are fixed by your own regex rules; add one from the tray right after a wrong dictation. See [Correcting Words](#correcting-words-the-recognizer-gets-wrong).
- **Audio Feedback**: Plays subtle beeps indicating when recording starts and stops—no need to look at a terminal or status bar.
- **GPU Transcription**: Runs on the GPU through Vulkan (including AMD/Intel integrated graphics) via [parakeet.cpp](https://github.com/mudler/parakeet.cpp), taking ~0.1 s per phrase with almost no CPU use, and falls back to the CPU when no Vulkan device is found. Silence and key clicks produce no text.
- **Background Service & Desktop Integration**: Includes a simple `.desktop` file for easy launching from your app menu or startup applications.

## Requirements

Ensure the following system dependencies are installed:
- `wl-clipboard` (read-aloud reads the selection with `wl-paste`)
- GNOME or KDE: the paste goes through the xdg-desktop-portal RemoteDesktop
  interface, which wlroots desktops (Sway, Hyprland) do not offer.
- The hotkey helper, installed once as root (below). Your user does **not**
  need to be in the `input` group, and should not be.

### Hotkey helper

Reading keyboards needs the `input` group, and anything in that group can read
every keystroke on the machine, in every session: passwords and PINs typed
elsewhere included. So only a small system service holds it. It runs as a
throwaway user with no network and no writable files, passes on exactly Right
Alt, F23 and lone Right Ctrl taps (nothing else, not even timing), and only while
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

## Correcting Words the Recognizer Gets Wrong

Names and jargon that Parakeet keeps mishearing (a project name heard as "Link",
"merge" heard as "Merch") are fixed by rules in
`~/.config/voice-assistant/corrections.toml`, applied to every dictation before it
is pasted. The file is reloaded when it changes; an invalid file is reported in the
log and the previous rules stay active.

```toml
[[rule]]
replace = ["Link", "linken"]          # misrecognized words: whole words, any case
with = "Lynq"                         # what they should be
followed_by = ["API", "Projekt"]      # optional: only right before one of these words
preceded_by = ["das"]                 # optional: only right after one of these words
examples = [["Die alte Link API.", "Die alte Lynq API."]]   # input -> expected
keep = ["Gib mir einen Link."]                              # must stay unchanged
```

For the rare case plain words can't express, `regex = '...'` (Python, case-insensitive)
replaces `replace`, and `with` becomes an `re.sub` template.

**Quick fix:** right after a wrong dictation, pick **Fix Last Dictation…** in the
tray menu. Either correct the sentence (the rule is derived from your edit) or fill
in the rule fields (the sentence follows). The dialog shows which earlier dictations
(from the journal) the rule would also change; add words to "Only if followed/preceded
by" until only the right ones remain, then save. The rule is appended with your
sentence as its example.

If the new rule has the same result and context words as an existing one, the
dialog offers to add the words as variants of that rule instead, which keeps the
file short. It warns when a word to replace is an ordinary word (found in the
installed hunspell/`/usr/share/dict` word lists) and has no context words. Only
English lists are usually installed; `sudo apt install hunspell-de-de` adds German.

**Correction Rules…** in the tray lists every rule with how often it fired on past
dictations and when last, flags ordinary words and rules that never fired, shows
a rule's examples and matches, and can delete a rule or open the file in your
editor. The numbers come from the journal, so they include dictations from before
the rule existed.

**Command line** (`uv run text_corrections.py …`):

- `check`: verify every example against all rules (a new rule may break an old one)
- `try "text"`: show what the rules make of a sentence
- `history [regex]`: apply the rules to all past dictations, showing what would change
- `add "wrong" "right"`: derive and append a rule, like the tray dialog

Each dictation logs the raw text as `Recognized:` and the pasted text as `Injecting:`.

Small local LLMs (including text-diffusion models) were tried for this and rejected:
slower (0.5 s to 80 s per sentence) and they silently drop or swap words.

## Read Aloud

1. Highlight any text (or copy it with Ctrl+C).
2. Tap the **Copilot key** (located to the right of AltGr), or tap **Right Ctrl** on its own on a keyboard without one.
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
reading speed, the Ctrl+Shift+V paste-mode toggle and the clipboard-restore
toggle — is remembered across
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
- **Read-aloud does nothing**: The Copilot key emits `Meta+Shift+F23`; the assistant listens for `F23`. Confirm your key sends it with `sudo .venv/bin/python key_detector.py` (tap the Copilot key, look for `KEY_F23`). Right Ctrl counts only when tapped alone and released within half a second; as part of a shortcut (Right Ctrl+C) it does nothing. If nothing is selected and the clipboard is empty, you'll hear a short low beep instead.
- **Recordings produce no text**: Each recording logs `Audio: <length>, peak <level>, gain <boost>`. Recordings are centred (DC offset removed) and quiet ones are boosted up to 10×, which rescues built-in laptop mics that deliver very quiet audio on a DC offset. A peak well below ~0.01 while you were speaking means the mic is barely picking you up: pick another device in the tray, or raise the capture/boost level in `alsamixer` (F4).
- **Text isn't pasting**:
    - The assistant pastes through the RemoteDesktop portal, which needs keyboard and clipboard access, and presses Ctrl+Shift+V by default (works in terminals and most editors). If you declined GNOME's prompt, nothing is pasted and a notice says so: delete `~/.config/voice-assistant/portal_restore_token` and restart to be asked again. If your target app does not accept Ctrl+Shift+V (e.g. LibreOffice), toggle off "Use Ctrl+Shift+V (terminal-compatible)" in the tray menu to fall back to plain Ctrl+V.
    - **Test Injection**: Run `uv run voice_assistant.py --test-injection` to verify if the script can type into a focused window without recording.
- **Wrong Key Detected**: The helper picks up every keyboard, including ones plugged in later (within ~2 s). `journalctl -u voice-hotkey` lists each one it listens to.
- **Microphone issues**: Check your default input device in your GNOME/desktop sound settings.

---

> [!NOTE]
> This entire project was **vibe coded**. ✨
