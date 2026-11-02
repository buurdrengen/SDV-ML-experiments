"""EpisodeRecorder over the recorded Episode: the Episode log #9 replays, and the metrics TensorBoard gets."""

import pytest

from recorded_bridge import make_env, recorded_actions
from sdv_env.env import Tile
from sdv_env.episode_log import Ending, EpisodeLog, EpisodeRecorder, Summary, read_episode, summarise


class FakeWriter:
    """Collects what would go to TensorBoard."""

    def __init__(self):
        self.scalars = []

    def add_scalar(self, tag, scalar_value, global_step=None):
        self.scalars.append((tag, scalar_value, global_step))


def play_recorded_episode(tmp_path, writer=None):
    env = EpisodeRecorder(make_env(step_limit=15), tmp_path, writer)
    env.reset()
    for action in recorded_actions(15):
        env.step(action)
    return env


def test_an_episode_log_holds_the_actions_and_the_tile_after_each_step(tmp_path):
    play_recorded_episode(tmp_path)

    log = read_episode(tmp_path / "episode_00000.json")

    assert log.actions[:9] == ["left"] * 6 + ["down"] * 3
    assert log.actions[9:] == ["down", "up", "noop", "up", "interact", "down"]
    assert log.tiles[0] == ("FarmHouse", 9, 9)  # the start, before any Step
    assert log.tiles[9] == ("Farm", 64, 15)  # out the FarmHouse door
    assert log.tiles[14] == ("FarmHouse", 3, 11)  # back in through it
    assert len(log.tiles) == len(log.actions) + 1


def test_an_episode_log_records_the_totals_and_how_the_episode_ended(tmp_path):
    play_recorded_episode(tmp_path)

    log = read_episode(tmp_path / "episode_00000.json")

    assert log.reward == pytest.approx(1.10)  # 10 new Tiles at 0.01, one new Location at 1.0
    assert (log.unique_tiles, log.unique_locations) == (11, 2)
    assert log.ending is Ending.TRUNCATED
    assert log.starting_save == "unused"


def test_each_episode_logs_its_metrics_against_the_total_steps_so_far(tmp_path):
    writer = FakeWriter()
    env = play_recorded_episode(tmp_path, writer)
    env.reset()
    for action in recorded_actions(15):
        env.step(action)

    assert sorted(writer.scalars) == pytest.approx(sorted([
        ("episode/reward", 1.10, 15), ("episode/unique_tiles", 11, 15),
        ("episode/unique_locations", 2, 15), ("episode/length", 15, 15),
        ("episode/reward", 1.10, 30), ("episode/unique_tiles", 11, 30),
        ("episode/unique_locations", 2, 30), ("episode/length", 15, 30),
    ]))
    assert sorted(path.name for path in tmp_path.iterdir()) == ["episode_00000.json", "episode_00001.json"]


def log_with(reward, unique_tiles, unique_locations, steps):
    return EpisodeLog(
        starting_save="save", actions=["noop"] * steps, tiles=[Tile("FarmHouse", 9, 9)] * (steps + 1),
        reward=reward, unique_tiles=unique_tiles, unique_locations=unique_locations, ending=Ending.TERMINATED,
    )


def test_summarise_gives_the_mean_and_sample_standard_deviation_of_each_metric():
    logs = [log_with(1.0, 10, 1, 4800), log_with(2.0, 20, 2, 4850), log_with(6.0, 30, 6, 4900)]

    summary = summarise(logs)

    assert {tag: (stat.mean, stat.std) for tag, stat in summary.items()} == {
        "episode/reward": (3.0, pytest.approx(2.6457513)),
        "episode/unique_tiles": (20.0, 10.0),
        "episode/unique_locations": (3.0, pytest.approx(2.6457513)),
        "episode/length": (4850.0, 50.0),
    }
