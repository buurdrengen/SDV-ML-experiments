#!/usr/bin/env python3
"""
Stage 1 plumbing:
- Capture a region of the screen (fast) using mss
- Optionally preview it with OpenCV
- Send a short keypress sequence to the active window

Usage:
  1) Start Stardew in windowed mode.
  2) Alt-tab so Stardew is the focused window.
  3) Run:
     python scripts/s1_capture_and_keys.py
"""

import time
import numpy as np
import cv2
import mss
import pyautogui

# Safety: prevents runaway automation if something goes wrong
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.02

# --------- Configure capture region ---------
# Start with "full screen" capture, then narrow it later for speed/stability.
# If you have multiple monitors, you can change monitor index below.
MONITOR_INDEX = 1  # 1 = primary monitor in mss
CAPTURE_FULL_MONITOR = False

# If CAPTURE_FULL_MONITOR = False, set a fixed region (top-left x,y + width,height)
REGION = {"left": 320, "top": 212, "width": 1280, "height": 720}

# --------- Configure input test ---------
# WASD movement is typical; Stardew also supports arrow keys depending on settings.
MOVE_KEY = "d"     # move right
MOVE_SECONDS = 0.7 # how long to hold movement


def main():
    print("Stage 1: pixels in, keys out")
    print("IMPORTANT: Click/alt-tab into Stardew so it has focus.")
    print("PyAutoGUI failsafe is ON: move mouse to top-left corner to abort.")
    time.sleep(2.0)

    with mss.mss() as sct:
        if CAPTURE_FULL_MONITOR:
            monitor = sct.monitors[MONITOR_INDEX]
            bbox = {
                "left": monitor["left"],
                "top": monitor["top"],
                "width": monitor["width"],
                "height": monitor["height"],
            }
        else:
            bbox = REGION

        print(f"Capturing region: {bbox}")

        # Grab one frame to confirm capture works
        raw = sct.grab(bbox)  # BGRA
        frame = np.array(raw, dtype=np.uint8)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        frame_small = cv2.resize(frame_bgr, (320,180), interpolation=cv2.INTER_AREA)

        # Show preview window (press q to close)
        cv2.imshow("capture_preview (press ESC)", frame_small)
        print("Preview opened. Press 'ESC' in the preview window to continue.")
        while True:
            if cv2.waitKey(10) & 0xFF == 27: #ESC
                break
        cv2.destroyAllWindows()

    # Send an input test (to the currently focused window!)
    print("Click Stardew now to focus it. Sending input in 2 seconds...")
    time.sleep(2.0)
    print(f"Sending input: hold '{MOVE_KEY}' for {MOVE_SECONDS:.2f}s")
    pyautogui.keyDown(MOVE_KEY)
    time.sleep(MOVE_SECONDS)
    pyautogui.keyUp(MOVE_KEY)

    print("Done. If Stardew was focused, your character should have moved right briefly.")


if __name__ == "__main__":
    main()