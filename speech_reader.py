"""Local text-to-speech ("read aloud") via Piper.

Self-contained: knows nothing about evdev or the tray. The owning app fetches
text elsewhere and drives this via start()/stop()/toggle().
"""

import os
import subprocess
import sys
from pathlib import Path


def default_voices_dir():
    """Directory where Piper voice models are cached (XDG data dir)."""
    base = os.environ.get("XDG_DATA_HOME") or os.path.join(
        os.path.expanduser("~"), ".local", "share"
    )
    return Path(base) / "voice-assistant" / "voices"


def ensure_voice_model(name, voices_dir):
    """Return the path to <name>.onnx, downloading it only if missing."""
    voices_dir = Path(voices_dir)
    voices_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = voices_dir / f"{name}.onnx"
    if not onnx_path.exists():
        print(f"[read] downloading voice '{name}' to {voices_dir} ...")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "piper.download_voices",
                name,
                "--data-dir",
                str(voices_dir),
            ],
            check=True,
        )
    return onnx_path
