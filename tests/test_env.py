"""SdvEnv against Bridge replies recorded from the real game; no game needed."""

import copy
import re
from pathlib import Path

import numpy as np
import pytest

from recorded_bridge import RECORDING, RecordedBridge, make_env, recorded_actions
from sdv_env.bridge_client import Direction
from sdv_env.env import VIEW_SIZE, SdvEnv
from sdv_env.locations import location_id

VIEW, VISITED = 0, 3  # first and last View channel
CENTRE = 7


def test_reset_observes_the_start_location_time_facing_and_only_the_start_tile_as_visited():
    obs, info = make_env().reset()

    assert obs["location"] == location_id("FarmHouse")
    assert obs["time"] == pytest.approx([0.0])
    assert obs["facing"] == list(Direction).index(Direction.RIGHT)  # the Starting save faces right
    visited = obs["view"][VISITED]
    assert visited[CENTRE, CENTRE] == 1
    assert visited.sum() == 1
    assert info == {"tile": ("FarmHouse", 9, 9), "unique_tiles": 1, "unique_locations": 1}


def test_new_tiles_pay_the_tile_reward_and_a_new_location_adds_the_location_bonus():
    env = make_env(tile_reward=0.01, location_bonus=1.0)
    env.reset()

    rewards = [env.step(action)[1] for action in recorded_actions(15)]

    assert rewards == pytest.approx([
        0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,  # 8 new FarmHouse Tiles
        1.01,  # out the door: a new Location and its first Tile
        0.01,  # one Tile further onto the Farm
        0.0, 0.0, 0.0,  # back onto the door Tile, idle, bump into the door
        0.0,  # back inside, onto a FarmHouse Tile visited before
        0.0,  # out again, onto the visited Farm Tile
    ])


def test_info_counts_unique_tiles_and_locations_visited_this_episode():
    env = make_env()
    env.reset()

    infos = [env.step(action)[4] for action in recorded_actions(15)]

    assert infos[-1] == {"tile": ("Farm", 64, 15), "unique_tiles": 11, "unique_locations": 2}


def visited_tiles(obs) -> set[tuple[int, int]]:
    """Visited Tiles in the View, as (dx, dy) from the player."""
    rows, cols = np.nonzero(obs["view"][VISITED])
    return {(int(col) - CENTRE, int(row) - CENTRE) for row, col in zip(rows, cols)}


def test_the_visited_channel_marks_visited_tiles_of_the_current_location_around_the_player():
    env = make_env()
    env.reset()
    actions = recorded_actions(15)

    obs = env.step(actions[0])[0]  # one Tile left of the start
    assert visited_tiles(obs) == {(0, 0), (1, 0)}

    for action in actions[1:9]:
        obs = env.step(action)[0]  # out onto the Farm
    assert visited_tiles(obs) == {(0, 0)}

    for action in actions[9:14]:
        obs = env.step(action)[0]  # back inside at (3, 11), below the walk to the door
    assert visited_tiles(obs) == {(0, 0), (0, -1)} | {(dx, -2) for dx in range(0, 7)}


def test_the_episode_is_truncated_at_the_step_limit():
    env = make_env(step_limit=3)
    env.reset()

    ends = [env.step(action)[2:4] for action in recorded_actions(3)]

    assert ends == [(False, False), (False, False), (False, True)]


def test_the_episode_terminates_when_the_bridge_ends_it():
    recording = copy.deepcopy(RECORDING)
    recording["steps"][1]["reply"].update(time=2600, episode_end="two_am")
    env = SdvEnv(client=RecordedBridge(recording), starting_save=Path("unused"), step_limit=2)
    env.reset()

    ends = [env.step(action)[2:4] for action in recorded_actions(2)]

    assert ends == [(False, False), (True, False)]


def test_the_env_passes_gymnasiums_env_checker():
    from gymnasium.utils.env_checker import check_env

    check_env(SdvEnv(client=RecordedBridge(any_action=True), starting_save=Path("unused")), skip_render_check=True)


def test_view_size_matches_the_bridge_source():
    source = (Path(__file__).parent.parent / "bridge" / "View.cs").read_text()
    match = re.search(r"const int Radius = (\d+);", source)
    assert match, "View.Radius not found"
    assert VIEW_SIZE == 2 * int(match.group(1)) + 1
