from evdev import InputDevice, list_devices, ecodes
import threading

def listen_to_device(device):
    print(f"Monitoring {device.name} ({device.path})...")
    try:
        for event in device.read_loop():
            if event.type == ecodes.EV_KEY:
                # event.value 1 is press, 0 is release
                state = "PRESSED" if event.value == 1 else "RELEASED"
                if event.value in [0, 1]:
                    print(f"Key: {ecodes.KEY[event.code]} | Code: {event.code} | State: {state}")
    except Exception as e:
        print(f"Lost device {device.name}: {e}")

def main():
    devices = [InputDevice(path) for path in list_devices()]
    keyboards = []
    for dev in devices:
        if ecodes.EV_KEY in dev.capabilities() and ecodes.KEY_A in dev.capabilities()[ecodes.EV_KEY]:
            keyboards.append(dev)

    if not keyboards:
        print("No keyboards found. Try running with sudo if permissions are an issue.")
        return

    print("--- Key Detector ---")
    print("Press any key to see its code. Press Ctrl+C to exit.")
    
    threads = []
    for kb in keyboards:
        t = threading.Thread(target=listen_to_device, args=(kb,), daemon=True)
        t.start()
        threads.append(t)

    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print("\nExiting...")

if __name__ == "__main__":
    main()
