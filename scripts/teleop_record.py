#!/usr/bin/env python3
"""
Teleop recorder (Stage 1):
- Pixels-only observation (screen capture -> downscale)
- Human controls Stardew with keyboard
- Records frames + currently-held keys at a fixed rate

How to use:
1) Open Stardew (windowed) and make sure it's focused.
2) Run this script.
3) When prompted, click Stardew.
4) Play normally using keyboard.
5) Press ESC to stop recording.

Notes:
- No preview window is shown (avoids stealing focus).
- Data is saved under data/teleop/<timestamp>/.
"""

import time
import json
import threading
from pathlib import Path

import numpy as np
import cv2
import mss
from pynput import keyboard


# ---- Your fixed capture region (already set in your project scripts) ----
REGION = {"left": 320, "top": 212, "width": 1280, "height": 720}

# ---- Model/input resolution ----
OUT_W, OUT_H = 320, 180

# ---- Recording settings ----
HZ = 10          # 10 Hz is a nice default for teleop (try 5 or 10)
N_STEPS_MAX = 10_000  # safety cap (~1000s at 10 Hz)
JPEG_QUALITY = 90

# ---- Keys to record (customize to your bindings) ----
# We'll record these as a multi-hot vector each timestep.
# Typical Stardew keyboard: WASD/arrow movement + tools/actions.
KEYMAP = [
    ("up",    {"w", "up"}),
    ("down",  {"s", "down"}),
    ("left",  {"a", "left"}),
    ("right", {"d", "right"}),

    # Common actions (adjust if your bindings differ)
    ("use_tool", {"mouse_left", "c"}),   # many players use mouse; 'c' is just an example fallback
    ("interact", {"x", "e", "enter"}),   # depends on bindings
    ("menu", {"esc", "tab"}),            # menus
    ("run", {"left_shift", "right_shift"}),
]

# If you do not use mouse yet, you can remove mouse_left from KEYMAP.
# (This script is keyboard-only; mouse buttons won't be captured unless you add a mouse listener.)


def _key_to_name(key) -> str | None:
    """Convert pynput key to a simple string name."""
    # Character keys
    try:
        if key.char is not None:
            return key.char.lower()
    except Exception:
        pass

    # Special keys
    if key == keyboard.Key.up:
        return "up"
    if key == keyboard.Key.down:
        return "down"
    if key == keyboard.Key.left:
        return "left"
    if key == keyboard.Key.right:
        return "right"
    if key == keyboard.Key.esc:
        return "esc"
    if key == keyboard.Key.enter:
        return "enter"
    if key == keyboard.Key.tab:
        return "tab"
    if key == keyboard.Key.shift:
        return "left_shift"  # pynput doesn't always distinguish; good enough for now
    if key == keyboard.Key.shift_r:
        return "right_shift"

    # You can add more mappings here if needed
    return None


def main():
    print("Teleop recorder")
    print(" - Focus Stardew and play normally.")
    print(" - Press ESC to stop recording.")
    print("Starting in 3 seconds... click Stardew now.")
    time.sleep(3.0)

    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path("data") / "teleop" / run_id
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Shared state for key holds
    held = set()
    held_lock = threading.Lock()
    stop_flag = {"stop": False}

    def on_press(key):
        name = _key_to_name(key)
        if name is None:
            return

        if name == "esc":
            stop_flag["stop"] = True
            return False  # stops the listener

        with held_lock:
            held.add(name)

    def on_release(key):
        name = _key_to_name(key)
        if name is None:
            return
        with held_lock:
            held.discard(name)

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    meta = {
        "region": REGION,
        "out_size": [OUT_W, OUT_H],
        "hz": HZ,
        "keymap": [{"name": name, "aliases": sorted(list(aliases))} for name, aliases in KEYMAP],
        "start_time_unix": time.time(),
        "steps": [],
    }

    dt = 1.0 / HZ

    with mss.mss() as sct:
        for t in range(N_STEPS_MAX):
            if stop_flag["stop"]:
                break

            step_start = time.time()

            # Capture
            raw = sct.grab(REGION)  # BGRA
            frame = np.array(raw, dtype=np.uint8)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            obs = cv2.resize(frame_bgr, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)

            # Snapshot currently held keys
            with held_lock:
                held_now = set(held)

            # Convert held keys -> multi-hot action vector based on KEYMAP
            action_vec = []
            for _, aliases in KEYMAP:
                action_vec.append(1 if any(a in held_now for a in aliases) else 0)

            # Save frame
            frame_path = frames_dir / f"{t:05d}.jpg"
            cv2.imwrite(str(frame_path), obs, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])

            # Log
            meta["steps"].append({
                "t": t,
                "time_unix": time.time(),
                "held_keys": sorted(list(held_now)),
                "action": action_vec,
                "frame": f"frames/{t:05d}.jpg",
            })

            # Timing
            elapsed = time.time() - step_start
            time.sleep(max(0.0, dt - elapsed))

    # Ensure listener stops
    try:
        listener.stop()
    except Exception:
        pass

    with open(out_dir / "rollout.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved teleop rollout to: {out_dir}")
    print("Done.")


if __name__ == "__main__":
    main()