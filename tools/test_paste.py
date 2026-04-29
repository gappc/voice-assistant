import subprocess
import os
import time

env = os.environ.copy()
env["YDOTOOL_SOCKET"] = "/run/user/1000/.ydotool_socket"

text = "Pasted from clipboard!"
print(f"Putting '{text}' into clipboard...")
subprocess.run(["wl-copy", text], check=True)

print("Testing ydotool key ctrl+v in 3 seconds... Switch to an input field!")
time.sleep(3)
try:
    # Try the ctrl+v syntax
    subprocess.run(["ydotool", "key", "ctrl+v"], check=True, env=env)
    print("Sent ctrl+v")
except Exception as e:
    print(f"Failed: {e}")
