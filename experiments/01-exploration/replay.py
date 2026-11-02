"""Replay logged Episodes in the game and check each one reproduces its logged trajectory of Tiles.

Start the game through SMAPI with the Bridge installed, start recording the screen (OBS or ffmpeg) if you want
footage, then run:
    uv run python experiments/01-exploration/replay.py runs/01-ppo-1m/episodes/episode_00200.json
Replays draw at normal speed (60 updates per second): a 5000-Step Episode takes 8-17 minutes, depending on how
many of its moves were blocked. --fast skips drawing, to check many Episodes for determinism quickly:
    uv run python experiments/01-exploration/replay.py --fast runs/01-ppo-1m/episodes/*.json
Exits with status 1 if any replay diverged from its log.
"""

import argparse
import sys
from pathlib import Path

from sdv_env.bridge_client import BridgeClient
from sdv_env.env import STARTING_SAVE
from sdv_env.episode_log import read_episode
from sdv_env.replay import first_divergence, replay

STARTING_SAVES = STARTING_SAVE.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logs", type=Path, nargs="+", help="Episode logs (episode_NNNNN.json)")
    parser.add_argument("--fast", action="store_true", help="don't draw; for determinism checks")
    args = parser.parse_args()

    diverged = 0
    with BridgeClient(timeout=300) as client:
        client.set_fast(args.fast)
        for path in args.logs:
            log = read_episode(path)
            tiles = replay(client, log, STARTING_SAVES / log.starting_save)
            step = first_divergence(log.tiles, tiles)
            if step is None:
                print(f"{path}: matches, {len(log.actions)} Steps")
                continue
            diverged += 1
            logged = log.tiles[step] if step < len(log.tiles) else "the end of the log"
            replayed = tiles[step] if step < len(tiles) else "the end of the Episode"
            print(f"{path}: diverges after {step} Steps: logged {logged}, replayed {replayed}")
    print(f"{len(args.logs) - diverged} of {len(args.logs)} replays reproduced their logged trajectory")
    sys.exit(1 if diverged else 0)


if __name__ == "__main__":
    main()
