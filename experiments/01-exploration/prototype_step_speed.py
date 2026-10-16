"""Prototype for #4: how many lockstep Steps per second can the Bridge do, rendered vs fast mode?

Start the game through SMAPI with the Bridge installed, then run:
    uv run python experiments/01-exploration/prototype_step_speed.py
"""

import argparse
import collections
import random
import time

from sdv_env.bridge_client import DEFAULT_PORT, BridgeClient

DIRECTIONS = ["up", "down", "left", "right"]
EPISODE_MINUTES = (26 - 6) * 60  # a full Episode runs 6:00 to 2:00


def to_minutes(time_of_day: int) -> int:
    return time_of_day // 100 * 60 + time_of_day % 100


def benchmark(client: BridgeClient, save: str, fast: bool, steps: int, rng: random.Random) -> dict:
    client.request("mode", fast=False)  # load in render mode so it can be watched; then switch
    start = client.request("load", save=save)
    client.request("mode", fast=fast)

    outcomes = collections.Counter()
    ticks = 0
    locations = {start["location"]}
    began = time.perf_counter()
    for _ in range(steps):
        reply = client.request("move", dir=rng.choice(DIRECTIONS))
        outcomes[reply["outcome"]] += 1
        ticks += reply["ticks"]
        locations.add(reply["location"])
    elapsed = time.perf_counter() - began
    end = client.request("state")

    game_minutes = to_minutes(end["time"]) - to_minutes(start["time"])
    return {
        "mode": "fast" if fast else "rendered",
        "steps": steps,
        "seconds": round(elapsed, 2),
        "steps_per_sec": round(steps / elapsed, 1),
        "updates_per_sec": round((end["updates"] - start["updates"]) / elapsed, 1),
        "ticks_per_step": round(ticks / steps, 2),
        "game_minutes": game_minutes,
        "est_steps_per_episode": round(steps * EPISODE_MINUTES / game_minutes) if game_minutes else None,
        "outcomes": dict(outcomes),
        "locations": sorted(locations),
        "start_time": start["time"],
        "end_time": end["time"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", default="exploration_451834694")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--rendered-steps", type=int, default=300)
    parser.add_argument("--fast-steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with BridgeClient(port=args.port, timeout=300) as client:
        for fast, steps in [(False, args.rendered_steps), (True, args.fast_steps)]:
            print(benchmark(client, args.save, fast, steps, random.Random(args.seed)), flush=True)


if __name__ == "__main__":
    main()
