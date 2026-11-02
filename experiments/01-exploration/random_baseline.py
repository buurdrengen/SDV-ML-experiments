"""The Random baseline for Experiment 01: Episodes of uniformly random actions, logged like a trained agent's.

Start the game through SMAPI with the Bridge installed, then run:
    uv run python experiments/01-exploration/random_baseline.py
Writes TensorBoard metrics and one Episode log per Episode to runs/<run name>/, then prints a summary table.
"""

import argparse
import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from sdv_env.env import SdvEnv
from sdv_env.episode_log import EpisodeRecorder, read_episode, summarise

RUNS = Path(__file__).parents[2] / "runs"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0, help="seeds the random actions")
    parser.add_argument("--run-name", default=f"01-random-baseline-{time.strftime('%Y%m%d-%H%M%S')}")
    args = parser.parse_args()

    run = RUNS / args.run_name
    if run.exists():
        parser.error(f"{run} already exists; pick another --run-name")
    with SummaryWriter(run) as writer:
        env = EpisodeRecorder(SdvEnv(), run / "episodes", writer)
        env.action_space.seed(args.seed)
        for episode in range(args.episodes):
            _, info = env.reset()
            done = False
            while not done:
                *_, terminated, truncated, info = env.step(env.action_space.sample())
                done = terminated or truncated
            print(f"Episode {episode + 1}/{args.episodes}: {info['unique_tiles']} Tiles, {info['unique_locations']} Locations")

    logs = [read_episode(path) for path in sorted((run / "episodes").glob("episode_*.json"))]
    print(f"\n{len(logs)} Episodes, logged to {run}\n\n| Metric | Mean ± std |\n|---|---|")
    for tag, stat in summarise(logs).items():
        print(f"| `{tag}` | {stat.mean:.2f} ± {stat.std:.2f} |")


if __name__ == "__main__":
    main()
