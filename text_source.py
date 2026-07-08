"""Acquire the text the user wants read aloud.

Prefers the Wayland primary selection (highlighted text); falls back to the
clipboard. Returns None when there is nothing to read.
"""

import subprocess


def _paste(args):
    """Run a wl-paste variant, returning stdout or '' on any failure."""
    try:
        result = subprocess.run(args, capture_output=True, text=True, check=True)
        return result.stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def get_text_to_read():
    """Return the primary selection, else the clipboard, else None."""
    text = _paste(["wl-paste", "--primary", "--no-newline"]).strip()
    if not text:
        text = _paste(["wl-paste", "--no-newline"]).strip()
    return text or None
