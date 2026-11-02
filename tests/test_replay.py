"""Replaying an Episode log against recorded Bridge replies; the live determinism check is replay.py --fast."""

from pathlib import Path

from recorded_bridge import RecordedBridge, make_env, recorded_actions
from sdv_env.env import Tile
from sdv_env.episode_log import EpisodeRecorder, read_episode
from sdv_env.replay import first_divergence, replay


def recorded_log(tmp_path):
    env = EpisodeRecorder(make_env(step_limit=15), tmp_path)
    env.reset()
    for action in recorded_actions(15):
        env.step(action)
    return read_episode(tmp_path / "episode_00000.json")


def test_replaying_an_episode_log_reproduces_its_tiles(tmp_path):
    log = recorded_log(tmp_path)

    tiles = replay(RecordedBridge(), log, starting_save=Path("unused"))

    assert tiles == log.tiles
    assert first_divergence(log.tiles, tiles) is None


def test_first_divergence_is_the_first_differing_tile_or_where_the_shorter_trajectory_ends():
    start, farm, town = Tile("FarmHouse", 9, 9), Tile("Farm", 64, 15), Tile("Town", 0, 54)

    assert first_divergence([start, farm, farm], [start, farm, town]) == 2
    assert first_divergence([start, farm, farm], [start, farm]) == 2
    assert first_divergence([start], [start, farm]) == 1
    assert first_divergence([start, farm], [start, farm]) is None


def test_a_replay_stops_when_the_episode_ends_before_the_actions_run_out(tmp_path):
    log = recorded_log(tmp_path)
    recording = RecordedBridge().recording
    steps = [dict(step) for step in recording["steps"]]
    steps[4] = {**steps[4], "reply": {**steps[4]["reply"], "episode_end": "passed_out"}}

    tiles = replay(RecordedBridge({**recording, "steps": steps}), log, starting_save=Path("unused"))

    assert tiles == log.tiles[:6]  # the start, then Steps 0-4
    assert first_divergence(log.tiles, tiles) == 6
