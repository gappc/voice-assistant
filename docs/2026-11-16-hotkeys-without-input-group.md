# Hotkeys and pasting without the `input` group

**Status:** done 2026-11-16 (`hotkey_helper/`, `portal_keyboard.py`)

## Problem

The assistant used to read `/dev/input/event*` with evdev to see Right Alt and
F23, and pasted with `ydotool`, which writes to `/dev/uinput`. Both need the
desktop user in the `input` group, which hands every program that user runs
two powers over the whole machine, in every session:

- **read** every keystroke on every keyboard, including passwords and PINs
  typed in another user's session;
- **inject** keystrokes the kernel cannot tell from the real keyboard, into
  whatever window has focus, again in any session.

That defeats keeping a separate admin account out of reach of malware running
as the desktop user. The goal: the desktop user leaves `input`, and dictation
keeps working, including hold-to-talk.

## Decisions

**Hotkeys: a root-installed helper, not a GNOME shortcut.** A GNOME custom
shortcut would have needed no helper at all, but GNOME fires shortcuts on press
only and cannot bind a lone modifier, so hold-to-talk on Right Alt would become
press-to-start, press-to-stop on some other key. Hold-to-talk was worth a small
service. `voice-hotkeyd`:

- passes on only Right Alt and F23, press and release, as
  `talk|read down|up eventN`. No other key, no timestamps, no counters, so the
  timing of other keys cannot be reconstructed;
- passes on a press only while the socket owner's session is active on seat0
  (`ACTIVE_UID` in `/run/systemd/seats/seat0`), and fails closed if that file
  is missing or unreadable. A release is always passed on when its press was,
  so switching sessions mid-hold cannot leave a recording running;
- uses the Python standard library only and runs with `python3 -I` from
  `/usr/local/lib`, so it loads no code the desktop user can change. Keeping it
  in the repo and running it from there would let that user rewrite it;
- is socket-activated: systemd creates `/run/voice-hotkey.sock` owned by the
  desktop user, mode 0600, and the service runs as a `DynamicUser` with
  `SupplementaryGroups=input`, no network and a read-only system;
- rescans `/dev/input` every 2 s for keyboards plugged in later, and sends the
  missing key-up when a keyboard disappears mid-hold.

**Pasting: RemoteDesktop portal, not ydotool.** Granting the user write access
to `/dev/uinput` alone (an ACL instead of the group) would have stopped the
reading but not the injecting. The xdg-desktop-portal RemoteDesktop keyboard
goes through the compositor instead: GNOME asks once, the answer is kept as a
restore token, and keys reach only the user's own session. The paste is still
`wl-copy` plus Ctrl+(Shift+)V, so text is layout-independent as before.

**Portal session per paste.** GNOME shows a remote-control indicator for as
long as a RemoteDesktop session is open. The session is therefore opened for
each paste and closed right after (restored silently from the token, ~0.03 s).
The indicator only flashes during a paste, so seeing it at any other time means
something else is using the portal.

## Remaining risk

- Programs running as the desktop user can still act inside that user's own
  graphical session: GNOME Shell runs as that user and loads extensions from
  `~/.local/share/gnome-shell/extensions`, and the portal restore token sits in
  the user's config directory. An admin terminal opened as a window inside
  this session is therefore not protected. A separate login session is.
- Right Alt is AltGr on German layouts. Inside the user's own session the
  helper passes on its presses, so the timing of `@`, `€`, `{` etc. typed there
  reaches the desktop user. In other sessions it passes on nothing.

## Maintenance

- Never read `/dev/input` or use `/dev/uinput` in the app itself again; the
  point is that the desktop user does not need either.
- Changes to `hotkey_helper/` take effect only after re-installing as root (see
  README, *Hotkey helper*).
- The helper and the portal need GNOME or KDE; wlroots desktops (Sway,
  Hyprland) have no RemoteDesktop portal.
