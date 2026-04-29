import subprocess
import os
import time

env = os.environ.copy()
env["YDOTOOL_SOCKET"] = "/run/user/1000/.ydotool_socket"

print("Testing ydotool type in 3 seconds... Switch to an input field!")
time.sleep(3)
try:
    subprocess.run(["ydotool", "type", "Hello from ydotool"], check=True, env=env)
    print("Success")
except Exception as e:
    print(f"Failed: {e}")
