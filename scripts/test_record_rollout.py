#!/usr/bin/env python3
import time
import json
from pathlib import Path

import numpy as np
import cv2
import mss
import pyautogui

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.02

# ---- Capture region (your Stardew window) ----
REGION = {"left": 320, "top": 212, "width": 1280, "height": 720}

# ---- Model/input resolution ----
OUT_W, OUT_H = 320, 180

# ---- Rollout settings ----
HZ = 5                 # decisions per second
N_STEPS = 50           # 10 seconds at 5 Hz
SHOW_PREVIEW = False    # set False if you want it headless

# ---- Simple discrete actions ----
# You can extend this later.
ACTIONS = [
    ("noop", None),
    ("right", "d"),
    ("left", "a"),
    ("up", "w"),
    ("down", "s"),
]

def press_key(key: str, duration: float):
    pyautogui.keyDown(key)
    time.sleep(duration)
    pyautogui.keyUp(key)

def main():
    print("Stage 1 recorder: will record frames + actions.")
    print("PyAutoGUI failsafe ON (mouse to top-left to abort).")
    print("Click Stardew to focus it. Starting in 3 seconds...")
    time.sleep(3.0)

    # Output directory
    run_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path("data") / "rollouts" / run_id
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "region": REGION,
        "out_size": [OUT_W, OUT_H],
        "hz": HZ,
        "n_steps": N_STEPS,
        "actions": [name for name, _ in ACTIONS],
        "start_time_unix": time.time(),
    }

    dt = 1.0 / HZ

    with mss.mss() as sct:
        for t in range(N_STEPS):
            step_start = time.time()

            # Capture
            raw = sct.grab(REGION)  # BGRA
            frame = np.array(raw, dtype=np.uint8)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            obs = cv2.resize(frame_bgr, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)

            # Choose an action (for now: deterministic demo pattern)
            # e.g., move right for 10 steps, then noop
            if t < 10:
                a_idx = 1  # "right"
            else:
                a_idx = 0  # "noop"

            a_name, a_key = ACTIONS[a_idx]

            # Save frame (jpg is smaller/faster than png)
            frame_path = frames_dir / f"{t:05d}.jpg"
            cv2.imwrite(str(frame_path), obs, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

            # Log step
            meta_step = {
                "t": t,
                "action_index": a_idx,
                "action_name": a_name,
                "time_unix": time.time(),
                "frame": f"frames/{t:05d}.jpg",
            }
            meta.setdefault("steps", []).append(meta_step)

            # Preview
            if SHOW_PREVIEW:
                cv2.imshow("obs_320x180 (ESC to quit)", obs)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    print("ESC pressed. Stopping.")
                    break

            # Execute action as a short press within the timestep
            if a_key is not None:
                # Hold key for half the timestep; tune later.
                press_key(a_key, duration=dt * 0.5)

            # Keep timing roughly consistent
            elapsed = time.time() - step_start
            sleep_for = max(0.0, dt - elapsed)
            time.sleep(sleep_for)

    cv2.destroyAllWindows()

    # Save metadata
    with open(out_dir / "rollout.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved rollout to: {out_dir}")

if __name__ == "__main__":
    main()