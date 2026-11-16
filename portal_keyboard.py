"""Press keys in the focused window through the RemoteDesktop portal.

Replaces ydotool, which needs write access to /dev/uinput: with that, any
program the user runs can press keys silently and the kernel cannot tell them
from the real keyboard, in every session on the machine. The portal goes
through the compositor instead. GNOME asks once ("Allow remote interaction?")
and remembers the answer through a restore token, so later starts are silent.
Injected keys reach only this user's own session.

Keycodes are Linux evdev codes (29 = left Ctrl, 42 = left Shift, 47 = V); the
compositor maps them through the current layout, like a real keyboard.
"""

import os
import secrets
import threading
import time
from pathlib import Path

from jeepney import DBusAddress, MatchRule, message_bus, new_method_call
from jeepney.io.blocking import Proxy, open_dbus_connection
from jeepney.wrappers import DBusErrorResponse

PORTAL = DBusAddress(
    "/org/freedesktop/portal/desktop",
    bus_name="org.freedesktop.portal.Desktop",
    interface="org.freedesktop.portal.RemoteDesktop",
)
DEVICE_KEYBOARD = 1
PERSIST_UNTIL_REVOKED = 2
KEY_PRESSED, KEY_RELEASED = 1, 0
KEY_RETURN_SYM = 0xFF0D
# Pause between key events: GNOME drops a press and release that arrive in
# the same instant on some apps.
KEY_GAP = 0.01


class PortalError(RuntimeError):
    """The portal refused or failed; `cancelled` if the user said no."""

    def __init__(self, message, cancelled=False):
        super().__init__(message)
        self.cancelled = cancelled


def keysym_for(char):
    """X keysym for one character: Latin-1 maps directly, the rest of
    Unicode lives at 0x01000000 + codepoint."""
    if char == "\n":
        return KEY_RETURN_SYM
    cp = ord(char)
    if 0x20 <= cp <= 0x7E or 0xA0 <= cp <= 0xFF:
        return cp
    return 0x01000000 | cp


class PortalKeyboard:
    """One RemoteDesktop session, opened on first use and kept for the life
    of the process. Thread-safe: every call holds the lock."""

    def __init__(self, token_path: Path):
        self._token_path = Path(token_path)
        self._lock = threading.Lock()
        self._conn = None
        self._session = None

    # --- public API ---

    def connect(self):
        """Open the session now (shows GNOME's prompt the first time)."""
        with self._lock:
            self._ensure_session()

    def press_combo(self, keycodes):
        """Press `keycodes` in order, then release them in reverse."""
        events = [(k, KEY_PRESSED) for k in keycodes]
        events += [(k, KEY_RELEASED) for k in reversed(keycodes)]
        self._send(lambda: [self._keycode(k, s) for k, s in events])

    def type_text(self, text):
        """Type `text` character by character via keysyms (slow; fallback)."""
        def go():
            for ch in text:
                sym = keysym_for(ch)
                self._keysym(sym, KEY_PRESSED)
                self._keysym(sym, KEY_RELEASED)
        self._send(go)

    def close(self):
        with self._lock:
            self._drop()

    # --- internals ---

    def _send(self, action):
        with self._lock:
            self._ensure_session()
            try:
                action()
            except (DBusErrorResponse, OSError):
                # The user stopped sharing in GNOME, or the portal restarted:
                # open a fresh session (silently, via the token) and retry once.
                self._drop()
                self._ensure_session()
                action()

    def _keycode(self, keycode, state):
        self._call("NotifyKeyboardKeycode", "oa{sv}iu", (self._session, {}, keycode, state))
        time.sleep(KEY_GAP)

    def _keysym(self, keysym, state):
        self._call("NotifyKeyboardKeysym", "oa{sv}iu", (self._session, {}, keysym, state))
        time.sleep(KEY_GAP)

    def _call(self, method, signature, body):
        reply = self._conn.send_and_get_reply(
            new_method_call(PORTAL, method, signature, body), timeout=5
        )
        if reply.header.message_type.name == "error":
            raise DBusErrorResponse(reply)
        return reply.body

    def _drop(self):
        if self._conn is not None:
            try:
                self._conn.close()
            except OSError:
                pass
        self._conn = self._session = None

    def _ensure_session(self):
        if self._session is not None:
            return
        self._conn = open_dbus_connection(bus="SESSION")
        try:
            results = self._request("CreateSession", "a{sv}", lambda opts: (
                {**opts, "session_handle_token": ("s", "va" + secrets.token_hex(6))},
            ))
            session = results["session_handle"][1]

            options = {
                "types": ("u", DEVICE_KEYBOARD),
                "persist_mode": ("u", PERSIST_UNTIL_REVOKED),
            }
            token = self._load_token()
            if token:
                options["restore_token"] = ("s", token)
            self._request("SelectDevices", "oa{sv}", lambda opts: (session, {**opts, **options}))

            # Start is where GNOME may ask the user: no timeout.
            results = self._request("Start", "osa{sv}", lambda opts: (session, "", opts),
                                    timeout=None)
            if not results.get("devices", ("u", 0))[1] & DEVICE_KEYBOARD:
                raise PortalError("the portal granted no keyboard")
            new_token = results.get("restore_token")
            if new_token:
                self._save_token(new_token[1])
            self._session = session
        except BaseException:
            self._drop()
            raise

    def _request(self, method, signature, make_body, timeout=30):
        """Call a portal method that answers through a Request object's
        Response signal; return the results dict."""
        handle_token = "va" + secrets.token_hex(6)
        sender = self._conn.unique_name.lstrip(":").replace(".", "_")
        request_path = f"/org/freedesktop/portal/desktop/request/{sender}/{handle_token}"
        rule = MatchRule(
            type="signal",
            interface="org.freedesktop.portal.Request",
            member="Response",
            path=request_path,
        )
        Proxy(message_bus, self._conn).AddMatch(rule)
        with self._conn.filter(rule) as responses:
            body = make_body({"handle_token": ("s", handle_token)})
            reply = self._conn.send_and_get_reply(
                new_method_call(PORTAL, method, signature, body), timeout=10
            )
            if reply.header.message_type.name == "error":
                raise PortalError(f"{method}: {reply.body}")
            signal = self._conn.recv_until_filtered(responses, timeout=timeout)
        code, results = signal.body
        if code != 0:
            raise PortalError(f"{method} was refused (code {code})", cancelled=code == 1)
        return results

    def _load_token(self):
        try:
            return self._token_path.read_text().strip() or None
        except OSError:
            return None

    def _save_token(self, token):
        self._token_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self._token_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(token)
