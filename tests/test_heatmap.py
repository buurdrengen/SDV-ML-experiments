"""The visited-Tiles heatmap (experiments/01-exploration/heatmap.py)."""

import importlib.util
from pathlib import Path

from sdv_env.env import Tile
from sdv_env.episode_log import Ending, EpisodeLog

HEATMAP = Path(__file__).parents[1] / "experiments/01-exploration/heatmap.py"
_spec = importlib.util.spec_from_file_location("heatmap", HEATMAP)
heatmap = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(heatmap)


def log_through(*tiles):
    return EpisodeLog(
        starting_save="save", actions=["noop"] * (len(tiles) - 1), tiles=[Tile(*tile) for tile in tiles],
        reward=0.0, unique_tiles=0, unique_locations=0, ending=Ending.TERMINATED,
    )


def test_each_tile_counts_the_episodes_that_visited_it_once_however_often_they_did():
    logs = [
        log_through(("FarmHouse", 9, 9), ("FarmHouse", 8, 9), ("FarmHouse", 9, 9), ("Farm", 64, 15)),
        log_through(("FarmHouse", 9, 9), ("FarmHouse", 9, 9), ("FarmHouse", 9, 10)),
    ]

    counts = heatmap.visit_counts(logs)

    assert counts == {
        "FarmHouse": {(9, 9): 2, (8, 9): 1, (9, 10): 1},
        "Farm": {(64, 15): 1},
    }


def test_render_writes_a_png(tmp_path):
    logs = [log_through(("FarmHouse", 9, 9), ("Farm", 64, 15)), log_through(("FarmHouse", 9, 9), ("FarmHouse", 8, 9))]

    heatmap.render(logs, tmp_path / "heatmap.png", "two Episodes")

    assert (tmp_path / "heatmap.png").read_bytes().startswith(b"\x89PNG")
