"""Persisted tray settings (voice, speed, devices, paste mode).

Deliberately independent of speech_reader.py / voice_assistant.py: no
VOICE_CATALOG or DEFAULT_VOICE import, so there is no circular import and this
module is trivially unit-testable on its own. Validation against VOICE_CATALOG
happens in voice_assistant.py, not here.
"""

import dataclasses
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path


def default_config_dir():
    """Directory where tray settings are stored (XDG config dir)."""
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(
        os.path.expanduser("~"), ".config"
    )
    return Path(base) / "voice-assistant"


@dataclass(frozen=True)
class TraySettings:
    input_device: str | None = None   # device name, not index (indices shift across reboots)
    output_device: str | None = None  # device name
    voice: str | None = None          # VOICE_CATALOG key
    speed: float | None = None
    paste_with_shift: bool = True


def load(path):
    """Best-effort load. Missing file, corrupt JSON, or a field of the wrong
    shape all fall back to that field's TraySettings() default — never raises."""
    path = Path(path)
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError:
        return TraySettings()
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[settings] failed to read {path}: {exc}", file=sys.stderr)
        return TraySettings()

    if not isinstance(raw, dict):
        print(f"[settings] {path} did not contain a JSON object; using defaults", file=sys.stderr)
        return TraySettings()

    return TraySettings(
        input_device=_str_or_default(raw.get("input_device")),
        output_device=_str_or_default(raw.get("output_device")),
        voice=_str_or_default(raw.get("voice")),
        speed=_float_or_default(raw.get("speed")),
        paste_with_shift=_bool_or_default(raw.get("paste_with_shift")),
    )


def _str_or_default(value):
    return value if isinstance(value, str) else None


def _float_or_default(value):
    if isinstance(value, bool):
        return None
    return float(value) if isinstance(value, (int, float)) else None


def _bool_or_default(value):
    return value if isinstance(value, bool) else True


def save(settings, path):
    """Atomic write: write to `<path>.tmp`, then rename over `path`, so a
    failure mid-write can't corrupt the previous settings file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    try:
        tmp_path.write_text(json.dumps(dataclasses.asdict(settings), indent=2))
        tmp_path.rename(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
