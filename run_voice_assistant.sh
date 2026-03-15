#!/bin/bash
# Wrapper script for the voice assistant
# This script starts recording when run, and stops on the second run? 
# No, let's make it a simple "record for 5 seconds" or "record until Enter" for now.
# Actually, the python script currently waits for Enter. 

# Ensure ydotoold is running
if ! pgrep ydotoold > /dev/null; then
    YDOTOOL_SOCKET=/tmp/.ydotool_socket ydotoold &
    sleep 1
fi

export YDOTOOL_SOCKET=/tmp/.ydotool_socket
# Run the voice assistant in the foreground or background? 
# The python script now runs a persistent listener.
python3 /home/chris/projects/ml/voice/voice_assistant.py
