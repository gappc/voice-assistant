"""Paste text into the focused window through the RemoteDesktop portal.

Replaces ydotool, which needs write access to /dev/uinput: with that, any
program the user runs can press keys silently and the kernel cannot tell them
from the real keyboard, in every session on the machine. The portal goes
through the compositor instead. GNOME asks once ("Allow remote interaction?")
and remembers the answer through a restore token, so later starts are silent.
Injected keys reach only this user's own session.

The session is opened for each paste and closed right after, so GNOME's
"remote control" indicator shows only while keys are actually being sent: if
it appears while you are not dictating, something else is using the portal.

The portal's Clipboard interface reads and sets the clipboard inside the same
session. Unlike wl-copy/wl-paste it needs no focus-grabbing helper window, so
nothing flickers. GNOME keeps one format of the clipboard when the session
closes, so the restored clipboard survives as text or as an image; rich
formats like HTML do not.

Keycodes are Linux evdev codes (29 = left Ctrl, 42 = left Shift, 47 = V); the
compositor maps them through the current layout, like a real keyboard.
"""

import os
import secrets
import select
import sys
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
CLIPBOARD = DBusAddress(
    PORTAL.object_path,
    bus_name=PORTAL.bus_name,
    interface="org.freedesktop.portal.Clipboard",
)
DEVICE_KEYBOARD = 1
PERSIST_UNTIL_REVOKED = 2
KEY_PRESSED, KEY_RELEASED = 1, 0
# Pause between key events: GNOME drops a press and release that arrive in
# the same instant on some apps.
KEY_GAP = 0.01

TEXT = "text"  # sniff_mime() result for text, offered under all TEXT_TYPES
TEXT_TYPES = ["text/plain;charset=utf-8", "text/plain", "UTF8_STRING", "STRING", "TEXT"]
_IMAGE_MAGIC = [
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF8", "image/gif"),
]
READ_TIMEOUT = 2.0      # an unresponsive clipboard owner must not stall dictation
MAX_BYTES = 50 * 1024 * 1024
# GNOME copies a new selection at once (measured: ~2 ms); wait at most this.
GNOME_COPY_TIMEOUT = 0.5
# After Ctrl+V, wait this long for the app to fetch the text...
PASTE_TIMEOUT = 1.0
# ...and after its last fetch this long for further ones (some apps ask for
# several types) before the previous clipboard goes back.
PASTE_SETTLE = 0.15


def sniff_mime(data):
    """MIME type of clipboard bytes: the portal returns the content whatever
    type is asked for, so the type is told from the bytes themselves."""
    if not data:
        return None
    for magic, mime in _IMAGE_MAGIC:
        if data.startswith(magic):
            return mime
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    return TEXT


class PortalError(RuntimeError):
    """The portal refused or failed; `cancelled` if the user said no."""

    def __init__(self, message, cancelled=False):
        super().__init__(message)
        self.cancelled = cancelled


class ClipboardNotGranted(PortalError):
    """The session started, but without clipboard access."""


class PortalKeyboard:
    """Opens a RemoteDesktop session per action and closes it afterwards.
    Thread-safe: every call holds the lock."""

    def __init__(self, token_path: Path):
        self._token_path = Path(token_path)
        self._lock = threading.Lock()
        self._conn = None
        self._session = None
        self._transfers = None

    # --- public API ---

    def connect(self):
        """Open and close a session now, so GNOME's one-time prompt appears
        at startup rather than at the first paste."""
        with self._lock:
            try:
                self._ensure_session()
            finally:
                self._drop()

    def paste(self, text, keycodes, restore=True):
        """Put `text` on the clipboard, press `keycodes` (the paste shortcut)
        and, if `restore`, put the previous clipboard back afterwards."""
        payload = {mime: text.encode() for mime in TEXT_TYPES}
        backup = []  # read once: a retry must not back up our own text

        def go():
            if restore and not backup:
                backup.append(self._backup())
            self._offer(payload)
            self._press(keycodes)
            # The app fetches the text only after it sees the keys; putting
            # the old clipboard back sooner would paste that instead.
            self._serve(payload, PASTE_TIMEOUT, settle=PASTE_SETTLE)
            if backup and backup[0] is not None:
                mime, data = backup[0]
                types = TEXT_TYPES if mime == TEXT else [mime]
                self._offer({t: data for t in types})
        self._send(go)

    def close(self):
        with self._lock:
            self._drop()

    # --- internals ---

    def _send(self, action):
        with self._lock:
            try:
                self._ensure_session()
                try:
                    action()
                except (DBusErrorResponse, OSError):
                    # The portal restarted or dropped the session under us:
                    # open a fresh one (silently, via the token), retry once.
                    self._drop()
                    self._ensure_session()
                    action()
            finally:
                self._drop()

    def _press(self, keycodes):
        """Press `keycodes` in order, then release them in reverse."""
        for k in keycodes:
            self._keycode(k, KEY_PRESSED)
        for k in reversed(keycodes):
            self._keycode(k, KEY_RELEASED)

    def _backup(self):
        """(mime, bytes) of the current clipboard, or None if it is empty or
        unreadable. Images may answer a text request with nothing."""
        for mime in ("text/plain;charset=utf-8", "image/png"):
            try:
                data = self._read(mime)
            except (OSError, TimeoutError, DBusErrorResponse) as e:
                print(f"[clipboard] backup failed: {e}", file=sys.stderr)
                return None
            kind = sniff_mime(data)
            if kind is not None:
                return kind, data
            if data:
                print("[clipboard] unknown content, not backed up", file=sys.stderr)
                return None
        return None

    def _offer(self, payload):
        """Take the clipboard with `payload` ({mime: bytes}) and serve GNOME's
        own copy, which is what remains once the session closes."""
        self._set_selection(list(payload))
        self._serve(payload, GNOME_COPY_TIMEOUT)

    def _serve(self, payload, timeout, settle=0.0):
        """Answer transfer requests; return after the first one plus `settle`
        seconds without another, or after `timeout` without any."""
        deadline = time.monotonic() + timeout
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                return
            request = self._next_transfer(left)
            if request is None:
                return
            mime, serial = request
            self._write(serial, payload.get(mime))
            deadline = time.monotonic() + settle

    # --- D-Bus I/O (replaced in tests) ---

    def _keycode(self, keycode, state):
        self._call("NotifyKeyboardKeycode", "oa{sv}iu", (self._session, {}, keycode, state))
        time.sleep(KEY_GAP)

    def _set_selection(self, mime_types):
        self._call("SetSelection", "oa{sv}", (self._session, {"mime_types": ("as", mime_types)}),
                   CLIPBOARD)

    def _read(self, mime):
        (fd,) = self._call("SelectionRead", "os", (self._session, mime), CLIPBOARD)
        deadline = time.monotonic() + READ_TIMEOUT
        chunks, size = [], 0
        with fd.to_file("rb") as f:
            while True:
                left = deadline - time.monotonic()
                if left <= 0 or not select.select([f], [], [], left)[0]:
                    raise TimeoutError(f"reading {mime} timed out")
                chunk = f.read1(65536)
                if not chunk:
                    return b"".join(chunks)
                size += len(chunk)
                if size > MAX_BYTES:
                    raise OSError(f"{mime} larger than {MAX_BYTES} bytes")
                chunks.append(chunk)

    def _next_transfer(self, timeout):
        """(mime, serial) of the next paste request for this session, or None."""
        deadline = time.monotonic() + timeout
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                return None
            try:
                msg = self._conn.recv_until_filtered(self._transfers.queue, timeout=left)
            except TimeoutError:
                return None
            session, mime, serial = msg.body
            if session == self._session:
                return mime, serial

    def _write(self, serial, data):
        """Answer one transfer; `data` None means the type is not offered."""
        if data is None:
            self._call("SelectionWriteDone", "oub", (self._session, serial, False), CLIPBOARD)
            return
        (fd,) = self._call("SelectionWrite", "ou", (self._session, serial), CLIPBOARD)
        ok = True
        try:
            with fd.to_file("wb") as f:
                f.write(data)
        except OSError:
            ok = False  # the reader went away; nothing to retry
        self._call("SelectionWriteDone", "oub", (self._session, serial, ok), CLIPBOARD)

    def _call(self, method, signature, body, address=PORTAL):
        reply = self._conn.send_and_get_reply(
            new_method_call(address, method, signature, body), timeout=5
        )
        if reply.header.message_type.name == "error":
            raise DBusErrorResponse(reply)
        return reply.body

    def _drop(self):
        """Close the session (ends GNOME's indicator) and the connection."""
        if self._transfers is not None:
            self._transfers.close()
            self._transfers = None
        if self._conn is not None:
            if self._session is not None:
                try:
                    self._conn.send_and_get_reply(new_method_call(
                        DBusAddress(self._session, bus_name=PORTAL.bus_name,
                                    interface="org.freedesktop.portal.Session"),
                        "Close",
                    ), timeout=2)
                except Exception:
                    pass  # closing the connection ends the session anyway
            try:
                self._conn.close()
            except OSError:
                pass
        self._conn = self._session = None

    def _ensure_session(self):
        if self._session is not None:
            return
        token = self._load_token()
        try:
            self._open_session(token)
        except ClipboardNotGranted:
            if not token:
                raise
            # A token saved before clipboard access was requested restores
            # the old keyboard-only grant without asking. Drop it so GNOME
            # asks once for both.
            print("[portal] saved grant lacks clipboard access; asking again", file=sys.stderr)
            self._forget_token()
            self._open_session(None)

    def _open_session(self, token):
        # enable_fds: the clipboard calls pass data through file descriptors.
        self._conn = open_dbus_connection(bus="SESSION", enable_fds=True)
        try:
            results = self._request("CreateSession", "a{sv}", lambda opts: (
                {**opts, "session_handle_token": ("s", "va" + secrets.token_hex(6))},
            ))
            session = results["session_handle"][1]

            options = {
                "types": ("u", DEVICE_KEYBOARD),
                "persist_mode": ("u", PERSIST_UNTIL_REVOKED),
            }
            if token:
                options["restore_token"] = ("s", token)
            self._request("SelectDevices", "oa{sv}", lambda opts: (session, {**opts, **options}))
            # Must come before Start; GNOME asks for it in the same prompt.
            self._call("RequestClipboard", "oa{sv}", (session, {}), CLIPBOARD)

            # Start is where GNOME may ask the user: no timeout.
            results = self._request("Start", "osa{sv}", lambda opts: (session, "", opts),
                                    timeout=None)
            if not results.get("devices", ("u", 0))[1] & DEVICE_KEYBOARD:
                raise PortalError("the portal granted no keyboard")
            if not results.get("clipboard_enabled", ("b", False))[1]:
                raise ClipboardNotGranted("the portal granted no clipboard access")
            new_token = results.get("restore_token")
            if new_token:
                self._save_token(new_token[1])
            rule = MatchRule(type="signal", interface=CLIPBOARD.interface,
                             member="SelectionTransfer")
            Proxy(message_bus, self._conn).AddMatch(rule)
            self._transfers = self._conn.filter(rule, bufsize=64)
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

    def _forget_token(self):
        try:
            self._token_path.unlink()
        except FileNotFoundError:
            pass

    def _save_token(self, token):
        self._token_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self._token_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(token)
