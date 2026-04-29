import evdev
from evdev import ecodes, InputDevice, list_devices
import os
import subprocess
import time

def list_all_input_devices():
    print("\n--- Input Devices ---")
    devices = [InputDevice(path) for path in list_devices()]
    for dev in devices:
        print(f"Device: {dev.name}")
        print(f"  Path: {dev.path}")
        print(f"  Phys: {dev.phys}")
        print(f"  Bus:  {dev.info.bustype}")
        capabilities = dev.capabilities(verbose=True)
        if ecodes.EV_KEY in capabilities:
            has_a = any("KEY_A" in str(k) for k in capabilities[ecodes.EV_KEY])
            print(f"  Keyboard-like: {'Yes' if has_a else 'No'}")
        print("-" * 20)

def test_injection_environment():
    print("\n--- Injection Environment ---")
    uid = os.getuid()
    socket_path = f"/run/user/{uid}/.ydotool_socket"
    print(f"UID: {uid}")
    print(f"Expected Socket: {socket_path}")
    
    if os.path.exists(socket_path):
        print(f"Socket exists: Yes")
    else:
        print(f"Socket exists: NO (ydotoold might not be running or in wrong place)")

    try:
        subprocess.run(["which", "ydotool"], check=True, capture_output=True)
        print("ydotool installed: Yes")
    except:
        print("ydotool installed: NO")

    try:
        subprocess.run(["which", "wl-copy"], check=True, capture_output=True)
        print("wl-copy installed: Yes")
    except:
        print("wl-copy installed: NO")

def monitor_ydotool_events():
    print("\n--- Monitoring ydotool virtual device ---")
    print("Searching for 'ydotool virtual device'...")
    ydotool_dev = None
    for path in list_devices():
        dev = InputDevice(path)
        if "ydotool" in dev.name.lower():
            ydotool_dev = dev
            break
    
    if not ydotool_dev:
        print("Could not find ydotool virtual device. Is ydotoold running?")
        return

    print(f"Listening to {ydotool_dev.name} ({ydotool_dev.path})...")
    print("I will now trigger a 'ydotool key' command. You should see events below.")
    
    env = os.environ.copy()
    env["YDOTOOL_SOCKET"] = f"/run/user/{os.getuid()}/.ydotool_socket"
    
    # Run ydotool in a separate process
    subprocess.Popen(["ydotool", "key", "29:1", "29:0"], env=env) # Send Ctrl down/up

    try:
        # Read events for a short time
        for event in ydotool_dev.read_loop():
            if event.type == ecodes.EV_KEY:
                key_event = evdev.categorize(event)
                print(f"Event: {key_event.keycode} ({key_event.keystate})")
                if key_event.keystate == 0: # Stop after release
                    break
    except Exception as e:
        print(f"Error reading events: {e}")

if __name__ == "__main__":
    list_all_input_devices()
    test_injection_environment()
    try:
        monitor_ydotool_events()
    except PermissionError:
        print("\nPermission denied to read input devices. Try running with 'sudo'.")
