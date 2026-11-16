#!/usr/bin/python3
"""Hotkey helper for the voice assistant.

Reads the keyboards and passes on exactly four events over a Unix socket:

    talk down <dev>    talk up <dev>     (Right Alt, hold to dictate)
    read down <dev>    read up <dev>     (F23, the Copilot key)

<dev> is the kernel event node ("event3"), so the app can tell keyboards apart.
Every other key is read and dropped: nothing about it, not even its timing,
leaves this process. That lets the desktop user drop out of the `input` group,
which otherwise lets any program they run read every keystroke on the machine.

Down events are passed on only while the socket owner's session is the active
one on seat0, so presses in another user's session stay private. An up event
is passed on whenever the matching down was, so a session switch mid-hold
cannot leave a recording running.

Standard library only, so it never loads code the desktop user can change.

Run by systemd (voice-hotkey.socket + voice-hotkey.service), which hands over
the listening socket as fd 3. For a manual test:

    sudo python3 voice-hotkeyd.py --socket /tmp/hotkey.sock --allow-uid 1000
"""

import argparse
import errno
import fcntl
import os
import select
import socket
import stat
import struct
import sys
import time

KEY_A = 30
KEY_RIGHTALT = 100
KEY_F23 = 193
EV_KEY = 1

# Linux keycode -> name on the wire. Nothing outside this map is ever sent.
FORWARDED_KEYS = {KEY_RIGHTALT: "talk", KEY_F23: "read"}

# struct input_event on 64-bit: struct timeval (two longs), u16 type, u16 code, s32 value.
EVENT_FORMAT = "llHHi"
EVENT_SIZE = struct.calcsize(EVENT_FORMAT)

INPUT_DIR = "/dev/input"
SEAT_FILE = "/run/systemd/seats/seat0"
RESCAN_SECONDS = 2.0
SD_LISTEN_FDS_START = 3

# Virtual devices are other programs typing, not the person at the keyboard.
IGNORED_NAME_PARTS = ("ydotool", "virtual", "video", "button")


def _ioc_read(nr, size):
    """_IOC(_IOC_READ, 'E', nr, size) from <asm-generic/ioctl.h>."""
    return (2 << 30) | (size << 16) | (ord("E") << 8) | nr


def device_name(fd):
    buf = bytearray(256)
    fcntl.ioctl(fd, _ioc_read(0x06, len(buf)), buf)  # EVIOCGNAME
    return buf.split(b"\0", 1)[0].decode("utf-8", "replace")


def has_key(fd, keycode):
    buf = bytearray(96)  # KEY_MAX is 0x2ff: 768 bits
    fcntl.ioctl(fd, _ioc_read(0x20 + EV_KEY, len(buf)), buf)  # EVIOCGBIT(EV_KEY)
    return bool(buf[keycode // 8] & (1 << (keycode % 8)))


def is_keyboard(fd):
    """A keyboard has letter keys; mice, power buttons and webcams don't."""
    name = device_name(fd).lower()
    if any(part in name for part in IGNORED_NAME_PARTS):
        return False
    return has_key(fd, KEY_A)


def active_uid(seat_file=SEAT_FILE):
    """UID of the session in front of the screen on seat0, or None.

    Reads logind's state file. Anything unexpected counts as "nobody", so a
    format change fails closed: hotkeys stop rather than leak."""
    try:
        with open(seat_file) as f:
            for line in f:
                key, _, value = line.strip().partition("=")
                if key == "ACTIVE_UID":
                    return int(value)
    except (OSError, ValueError):
        pass
    return None


class Forwarder:
    """Decides which key events become messages. Pure logic, no I/O."""

    def __init__(self, allowed_uid, active_uid_fn=active_uid):
        self.allowed_uid = allowed_uid
        self._active_uid = active_uid_fn
        self._held = set()  # (dev, keyname) whose down was passed on

    def on_event(self, dev, code, value):
        """Message line for a key event, or None to drop it."""
        name = FORWARDED_KEYS.get(code)
        if name is None:
            return None
        if value == 1:
            if self._active_uid() != self.allowed_uid:
                return None
            self._held.add((dev, name))
            return f"{name} down {dev}\n"
        if value == 0 and (dev, name) in self._held:
            self._held.discard((dev, name))
            return f"{name} up {dev}\n"
        return None  # autorepeat (value 2), or an up we never announced

    def on_device_lost(self, dev):
        """A keyboard unplugged mid-hold never sends its key-up: send it now."""
        lines = []
        for held_dev, name in sorted(self._held):
            if held_dev == dev:
                lines.append(f"{name} up {dev}\n")
        self._held = {h for h in self._held if h[0] != dev}
        return lines


def log(msg):
    print(msg, file=sys.stderr, flush=True)


class Helper:
    def __init__(self, listener, forwarder):
        self.listener = listener
        self.forwarder = forwarder
        self.devices = {}  # fd -> dev name ("event3")
        self.clients = set()
        self.skipped = set()  # nodes checked and found not to be keyboards

    def rescan(self):
        try:
            nodes = {n for n in os.listdir(INPUT_DIR) if n.startswith("event")}
        except OSError as e:
            log(f"cannot list {INPUT_DIR}: {e}")
            return
        self.skipped &= nodes  # a node number freed by unplugging may be reused
        known = set(self.devices.values())
        for node in sorted(nodes - known - self.skipped):
            self._try_open(node)

    def _try_open(self, node):
        path = os.path.join(INPUT_DIR, node)
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
        except OSError as e:
            if e.errno not in (errno.ENOENT, errno.ENODEV):
                log(f"cannot open {path}: {e}")
            self.skipped.add(node)
            return
        try:
            keyboard = is_keyboard(fd)
            name = device_name(fd)
        except OSError:
            keyboard = False
        if not keyboard:
            os.close(fd)
            self.skipped.add(node)
            return
        self.devices[fd] = node
        log(f"listening to {node} ({name})")

    def _drop_device(self, fd):
        node = self.devices.pop(fd)
        os.close(fd)
        log(f"lost {node}")
        for line in self.forwarder.on_device_lost(node):
            self.broadcast(line)

    def read_device(self, fd):
        node = self.devices[fd]
        try:
            data = os.read(fd, EVENT_SIZE * 64)
        except BlockingIOError:
            return
        except OSError:
            self._drop_device(fd)
            return
        if not data:
            self._drop_device(fd)
            return
        for off in range(0, len(data) - EVENT_SIZE + 1, EVENT_SIZE):
            _, _, etype, code, value = struct.unpack_from(EVENT_FORMAT, data, off)
            if etype != EV_KEY:
                continue
            line = self.forwarder.on_event(node, code, value)
            if line:
                self.broadcast(line)

    def broadcast(self, line):
        payload = line.encode()
        for client in list(self.clients):
            try:
                client.send(payload, socket.MSG_DONTWAIT | socket.MSG_NOSIGNAL)
            except OSError:
                # Gone, or not reading: a stuck client must not stall the rest.
                self._drop_client(client)

    def _drop_client(self, client):
        self.clients.discard(client)
        client.close()

    def accept(self):
        try:
            client, _ = self.listener.accept()
        except OSError:
            return
        client.setblocking(False)
        self.clients.add(client)

    def client_readable(self, client):
        # Clients have nothing to say; readable means closed (or junk to discard).
        try:
            data = client.recv(4096)
        except BlockingIOError:
            return
        except OSError:
            data = b""
        if not data:
            self._drop_client(client)

    def run(self):
        self.rescan()
        next_scan = time.monotonic() + RESCAN_SECONDS
        while True:
            timeout = max(0.0, next_scan - time.monotonic())
            rlist = [self.listener, *self.clients, *self.devices]
            readable, _, _ = select.select(rlist, [], [], timeout)
            for item in readable:
                if item is self.listener:
                    self.accept()
                elif isinstance(item, socket.socket):
                    if item in self.clients:
                        self.client_readable(item)
                elif item in self.devices:
                    self.read_device(item)
            if time.monotonic() >= next_scan:
                self.rescan()
                next_scan = time.monotonic() + RESCAN_SECONDS


def systemd_listener():
    """The socket systemd passed in, or None when not socket-activated."""
    if os.environ.get("LISTEN_PID") != str(os.getpid()):
        return None
    if int(os.environ.get("LISTEN_FDS", "0")) != 1:
        raise SystemExit("expected exactly one socket from systemd")
    return socket.socket(fileno=SD_LISTEN_FDS_START)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--socket", help="create and listen on this path (manual testing)")
    parser.add_argument("--allow-uid", type=int,
                        help="whose active session enables hotkeys "
                             "(default: owner of the socket file)")
    args = parser.parse_args(argv)

    listener = systemd_listener()
    if listener is None:
        if not args.socket:
            raise SystemExit("not started by systemd; pass --socket for a manual run")
        if os.path.exists(args.socket):
            os.unlink(args.socket)
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(args.socket)
        listener.listen(4)
    path = listener.getsockname()

    allowed_uid = args.allow_uid
    if allowed_uid is None:
        # systemd created the socket with SocketUser=<desktop user>: the owner
        # of the file is exactly who may connect, so enable hotkeys for them.
        st = os.stat(path)
        if not stat.S_ISSOCK(st.st_mode):
            raise SystemExit(f"{path} is not a socket")
        allowed_uid = st.st_uid
    log(f"serving {path} for uid {allowed_uid}")

    Helper(listener, Forwarder(allowed_uid)).run()


if __name__ == "__main__":
    main()
