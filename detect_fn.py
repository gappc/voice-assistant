import evdev
from evdev import ecodes, InputDevice, list_devices

def find_all_devices():
    return [InputDevice(path) for path in list_devices()]

def listen_to_all_devices():
    devices_list = find_all_devices()
    if not devices_list:
        print("No devices found. Try running with sudo.")
        return

    print("Listening to ALL devices:")
    for dev in devices_list:
        print(f" - {dev.name} ({dev.path})")
    
    print("\nPress the 'Fn' key now. (Press Ctrl+C to stop)")
    
    try:
        from select import select
        devices = {dev.fd: dev for dev in devices_list}
        while True:
            r, w, x = select(devices, [], [])
            for fd in r:
                for event in devices[fd].read():
                    if event.type == ecodes.EV_KEY:
                        key_event = evdev.categorize(event)
                        # Filter for key down (1) or key hold (2)
                        if key_event.keystate in (1, 2):
                             print(f"Key event: {key_event.keycode} (Code: {key_event.scancode}) on {devices[fd].name}")
                    elif event.type == ecodes.EV_MSC and event.code == ecodes.MSC_SCAN:
                        print(f"Raw scan code: {event.value} on {devices[fd].name}")
    except KeyboardInterrupt:
        print("\nStopped.")
    except PermissionError:
        print("\nPermission denied. Please run with sudo: 'sudo python3 detect_fn.py'")

if __name__ == "__main__":
    listen_to_all_devices()
