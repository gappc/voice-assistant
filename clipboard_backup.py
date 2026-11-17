"""Save the clipboard before a dictation paste and put it back afterwards.

`wl-copy` offers one MIME type per call, so only the most useful type is
kept: an image if there is one, otherwise plain text, otherwise whatever
comes first. Rich formats (e.g. the HTML a browser offers next to the plain
text) are lost on restore; the content itself is not.
"""

import subprocess
import sys
from typing import NamedTuple

TIMEOUT = 2  # seconds; an unresponsive clipboard owner must not stall dictation
MAX_BYTES = 50 * 1024 * 1024

# Offered by X11 (XWayland) apps; they describe the selection, not content.
_PSEUDO_TYPES = {"TARGETS", "TIMESTAMP", "MULTIPLE", "SAVE_TARGETS", "DELETE"}
_TEXT_TYPES = ["text/plain;charset=utf-8", "text/plain", "UTF8_STRING", "STRING", "TEXT"]


class Snapshot(NamedTuple):
    mime: str
    data: bytes


# The clipboard held nothing; restoring it means clearing it.
EMPTY = Snapshot("", b"")


def save():
    """Return a Snapshot, EMPTY, or None when the clipboard could not be read
    (then it must be left alone rather than cleared)."""
    try:
        listed = subprocess.run(
            ["wl-paste", "--list-types"], capture_output=True, check=True, timeout=TIMEOUT
        )
    except subprocess.CalledProcessError:
        return EMPTY  # wl-paste exits 1 with "Nothing is copied"
    except (OSError, subprocess.TimeoutExpired) as e:
        print(f"[clipboard] could not list types: {e}", file=sys.stderr)
        return None

    types = [t for t in listed.stdout.decode(errors="replace").splitlines() if t]
    mime = _pick_type(types)
    if mime is None:
        return EMPTY
    try:
        content = subprocess.run(
            ["wl-paste", "--no-newline", "--type", mime],
            capture_output=True, check=True, timeout=TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError) as e:
        print(f"[clipboard] could not read {mime}: {e}", file=sys.stderr)
        return None
    if len(content.stdout) > MAX_BYTES:
        print(f"[clipboard] {mime} too large to back up", file=sys.stderr)
        return None
    return Snapshot(mime, content.stdout)


def restore(snapshot):
    """Put `snapshot` back on the clipboard. Returns False on failure."""
    if snapshot is EMPTY:
        args, data = ["wl-copy", "--clear"], None
    else:
        args, data = ["wl-copy", "--type", snapshot.mime], snapshot.data
    try:
        # wl-copy forks to serve the content and returns right away.
        subprocess.run(args, input=data, check=True, timeout=TIMEOUT)
        return True
    except (OSError, subprocess.SubprocessError) as e:
        print(f"[clipboard] restore failed: {e}", file=sys.stderr)
        return False


def _pick_type(types):
    types = [t for t in types if t not in _PSEUDO_TYPES]
    images = [t for t in types if t.startswith("image/")]
    if images:
        return "image/png" if "image/png" in images else images[0]
    for t in _TEXT_TYPES:
        if t in types:
            return t
    return types[0] if types else None
