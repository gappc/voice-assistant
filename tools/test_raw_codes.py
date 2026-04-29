import subprocess
import os
import time

env = os.environ.copy()
env["YDOTOOL_SOCKET"] = "/run/user/1000/.ydotool_socket"

text = "Pasted with raw codes!"
print(f"Putting '{text}' into clipboard...")
subprocess.run(["wl-copy", text], check=True)

print("Testing ydotool key 29:1 47:1 47:0 29:0 in 3 seconds...")
time.sleep(3)
try:
    # 29 is KEY_LEFTCTRL, 47 is KEY_V
    subprocess.run(["ydotool", "key", "29:1", "47:1", "47:0", "29:0"], check=True, env=env)
    print("Sent raw codes")
except Exception as e:
    print(f"Failed: {e}")
