"""Heatmap of the Tiles visited across many Episodes: one panel per Location, each Tile shaded by the share of
Episodes that visited it, on a log scale. Panels are plain grids around the visited Tiles (x right, y down, as in
SDV); unvisited Tiles are grey.

    uv run python experiments/01-exploration/heatmap.py runs/01-ppo-1m           # --last 10 --output ...
    uv run python experiments/01-exploration/heatmap.py runs/01-random-baseline
Writes <run>/heatmap.png unless --output says otherwise.
"""

import argparse
import math
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from sdv_env.episode_log import EpisodeLog, read_episode


def visit_counts(logs: list[EpisodeLog]) -> dict[str, Counter[tuple[int, int]]]:
    """For each Location, how many Episodes visited each of its (x, y) Tiles."""
    counts: dict[str, Counter[tuple[int, int]]] = {}
    for log in logs:
        for tile in set(log.tiles):
            counts.setdefault(tile.location, Counter())[(tile.x, tile.y)] += 1
    return counts


def render(logs: list[EpisodeLog], path: Path, title: str, max_locations: int = 12) -> None:
    """Draws the `max_locations` Locations most Episodes reached into the image `path`."""
    counts = visit_counts(logs)
    reached = Counter(location for log in logs for location in {tile.location for tile in log.tiles})
    locations = [location for location, _ in reached.most_common(max_locations)]

    columns = min(4, len(locations))
    rows = math.ceil(len(locations) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4.5 * columns, 4.2 * rows), squeeze=False, layout="constrained")
    cmap = plt.get_cmap("viridis").with_extremes(bad="#e6e6e6")
    # A log scale, so Tiles only one Episode reached stand apart from those a few did.
    norm = LogNorm(vmin=1 / len(logs), vmax=1)
    image = None
    for ax, location in zip(axes.flat, locations):
        tiles = counts[location]
        xs, ys = [x for x, _ in tiles], [y for _, y in tiles]
        x0, y0 = min(xs) - 1, min(ys) - 1
        grid = np.full((max(ys) - y0 + 2, max(xs) - x0 + 2), np.nan)
        for (x, y), episodes in tiles.items():
            grid[y - y0, x - x0] = episodes / len(logs)
        extent = (x0 - 0.5, x0 + grid.shape[1] - 0.5, y0 + grid.shape[0] - 0.5, y0 - 0.5)
        image = ax.imshow(grid, cmap=cmap, norm=norm, extent=extent, interpolation="nearest")
        ax.set_title(f"{location}: {reached[location]}/{len(logs)} Episodes, {len(tiles)} Tiles", fontsize=10)
        ax.tick_params(labelsize=7)
    for ax in axes.flat[len(locations):]:
        ax.set_axis_off()
    ticks = [share for share in (0.02, 0.05, 0.1, 0.2, 0.5, 1.0) if share >= norm.vmin]
    colorbar = fig.colorbar(image, ax=axes, shrink=0.6, label="Share of Episodes that visited the Tile")
    colorbar.set_ticks(ticks, labels=[f"{share:.0%}" for share in ticks])
    colorbar.minorticks_off()
    fig.suptitle(title)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="a run folder with an episodes/ folder of Episode logs")
    parser.add_argument("--last", type=int, help="only the last N logged Episodes")
    parser.add_argument("--max-locations", type=int, default=12)
    parser.add_argument("--output", type=Path, help="defaults to <run>/heatmap.png")
    args = parser.parse_args()

    paths = sorted((args.run / "episodes").glob("episode_*.json"))[-args.last if args.last else 0:]
    if not paths:
        parser.error(f"no Episode logs in {args.run / 'episodes'}")
    output = args.output or args.run / "heatmap.png"
    render([read_episode(path) for path in paths], output, f"Tiles visited: {args.run.name}, {len(paths)} Episodes", args.max_locations)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
