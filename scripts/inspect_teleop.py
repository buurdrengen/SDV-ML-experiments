#!/usr/bin/env python3
import json
from pathlib import Path
from collections import Counter

import cv2

# Change this to your latest rollout folder if needed
# e.g. data/teleop/20260225_142233
ROLLOUT_DIR = None  # auto-pick newest if None


def newest_rollout_dir(base="data/teleop") -> Path:
    base = Path(base)
    runs = sorted([p for p in base.glob("*") if p.is_dir()])
    if not runs:
        raise FileNotFoundError(f"No rollouts found under {base}")
    return runs[-1]


def main():
    roll_dir = newest_rollout_dir() if ROLLOUT_DIR is None else Path(ROLLOUT_DIR)
    meta_path = roll_dir / "rollout.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    steps = meta["steps"]
    keymap = meta["keymap"]
    names = [k["name"] for k in keymap]

    print(f"Rollout: {roll_dir}")
    print(f"Frames: {len(steps)}")
    print(f"Hz: {meta['hz']}")
    print(f"Action dims: {len(names)} -> {names}")

    # Count how often each action bit is ON
    on_counts = Counter()
    combos = Counter()
    for s in steps:
        a = s["action"]
        for i, v in enumerate(a):
            if v == 1:
                on_counts[names[i]] += 1
        combos[tuple(a)] += 1

    print("\nAction ON counts:")
    for n in names:
        print(f"  {n:12s}: {on_counts[n]} ({on_counts[n]/len(steps):.1%})")

    print(f"\nUnique action combos: {len(combos)} (top 10)")
    for combo, c in combos.most_common(10):
        pretty = [names[i] for i,v in enumerate(combo) if v==1]
        print(f"  {c:4d}  {pretty if pretty else ['noop']}")

    # Replay a quick preview
    print("\nReplaying (press ESC to quit)...")
    for s in steps:
        img = cv2.imread(str(roll_dir / s["frame"]))
        if img is None:
            continue
        # overlay active actions
        active = [names[i] for i,v in enumerate(s["action"]) if v==1]
        txt = ", ".join(active) if active else "noop"
        cv2.putText(img, txt, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
        cv2.imshow("teleop_replay", img)
        if (cv2.waitKey(int(1000/meta["hz"])) & 0xFF) == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()