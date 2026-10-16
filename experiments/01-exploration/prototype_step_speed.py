"""Prototype for #4: how many lockstep Steps per second can the Bridge do, rendered vs fast mode?

Start the game through SMAPI with the Bridge installed, then run:
    uv run python experiments/01-exploration/prototype_step_speed.py
"""

import argparse
import collections
import random
import time
from pathlib import Path

from sdv_env.bridge_client import Action, BridgeClient

MOVES = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
STARTING_SAVE = Path(__file__).parent / "starting-save/exploration_451834694"
EPISODE_MINUTES = (26 - 6) * 60  # a full Episode runs 6:00 to 2:00


def to_minutes(time_of_day: int) -> int:
    return time_of_day // 100 * 60 + time_of_day % 100


def benchmark(client: BridgeClient, save: Path, fast: bool, steps: int, rng: random.Random) -> dict:
    client.set_fast(False)  # load in render mode so it can be watched; then switch
    start = client.reset(save)
    client.set_fast(fast)

    outcomes: collections.Counter[str] = collections.Counter()
    updates = 0
    locations = {start.location}
    end = start
    began = time.perf_counter()
    for _ in range(steps):
        result = client.step(rng.choice(MOVES))
        outcomes[result.outcome.value] += 1
        updates += result.updates
        locations.add(result.state.location)
        end = result.state
    elapsed = time.perf_counter() - began

    game_minutes = to_minutes(end.time) - to_minutes(start.time)
    return {
        "mode": "fast" if fast else "rendered",
        "steps": steps,
        "seconds": round(elapsed, 2),
        "steps_per_sec": round(steps / elapsed, 1),
        "updates_per_sec": round((end.total_updates - start.total_updates) / elapsed, 1),
        "updates_per_step": round(updates / steps, 2),
        "game_minutes": game_minutes,
        "est_steps_per_episode": round(steps * EPISODE_MINUTES / game_minutes) if game_minutes else None,
        "outcomes": dict(outcomes),
        "locations": sorted(locations),
        "start_time": start.time,
        "end_time": end.time,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", type=Path, default=STARTING_SAVE)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--rendered-steps", type=int, default=300)
    parser.add_argument("--fast-steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with BridgeClient(port=args.port, timeout=300) as client:
        for fast, steps in [(False, args.rendered_steps), (True, args.fast_steps)]:
            print(benchmark(client, args.save, fast, steps, random.Random(args.seed)), flush=True)


if __name__ == "__main__":
    main()
