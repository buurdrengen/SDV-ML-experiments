"""Episode logs and metrics shared by every agent in an Experiment, so runs compare like for like.

An Episode log is one JSON file per Episode: the actions taken and the Tile after each Step. That is enough to
replay the Episode from the Starting save and to check the replay ends up on the same Tiles.
"""

import json
import statistics
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

import gymnasium as gym

from sdv_env.env import ACTIONS, SdvEnv, Tile

REWARD = "episode/reward"
UNIQUE_TILES = "episode/unique_tiles"
UNIQUE_LOCATIONS = "episode/unique_locations"
LENGTH = "episode/length"
"""TensorBoard tags, logged once per Episode against the total number of Steps so far."""


class Ending(StrEnum):
    """How an Episode stopped, in Gymnasium's terms (the Bridge's `EpisodeEnd` says why it terminated)."""

    TERMINATED = "terminated"
    """2:00 AM or passing out."""
    TRUNCATED = "truncated"
    """The step limit."""


@dataclass(frozen=True)
class EpisodeLog:
    starting_save: str
    """Folder name of the Starting save the Episode was reset to."""
    actions: list[str]
    tiles: list[Tile]
    """The start Tile, then the Tile after each Step."""
    reward: float
    unique_tiles: int
    unique_locations: int
    ending: Ending


def metrics(log: EpisodeLog) -> dict[str, float]:
    """The Episode's value for each TensorBoard tag."""
    return {REWARD: log.reward, UNIQUE_TILES: log.unique_tiles, UNIQUE_LOCATIONS: log.unique_locations, LENGTH: len(log.actions)}


@dataclass(frozen=True)
class Summary:
    mean: float
    std: float
    """Sample standard deviation across Episodes."""


def summarise(logs: list[EpisodeLog]) -> dict[str, Summary]:
    """Mean and standard deviation of each metric across Episodes; needs at least two."""
    if len(logs) < 2:
        raise ValueError(f"a standard deviation needs at least two Episodes, got {len(logs)}")
    per_episode = [metrics(log) for log in logs]
    summary = {}
    for tag in per_episode[0]:
        values = [episode[tag] for episode in per_episode]
        summary[tag] = Summary(statistics.mean(values), statistics.stdev(values))
    return summary


def read_episode(path: Path) -> EpisodeLog:
    """Loads an Episode log written by `EpisodeRecorder`."""
    data = json.loads(path.read_text())
    return EpisodeLog(**{**data, "tiles": [Tile(*tile) for tile in data["tiles"]], "ending": Ending(data["ending"])})


class ScalarWriter(Protocol):
    """The part of TensorBoard's `SummaryWriter` the recorder uses."""

    def add_scalar(self, tag: str, scalar_value: float, global_step: int | None = None) -> None: ...


class EpisodeRecorder(gym.Wrapper):
    """Writes an Episode log for every `log_every`-th Episode into `directory` (numbered by Episode, starting with
    the first) and, given a `writer`, every Episode's metrics."""

    def __init__(self, env: SdvEnv, directory: Path, writer: ScalarWriter | None = None, log_every: int = 1):
        super().__init__(env)
        self.starting_save = env.starting_save.name
        self.directory = directory
        self.writer = writer
        self.log_every = log_every
        self.episodes = 0
        self.total_steps = 0
        self._actions: list[str] = []
        self._tiles: list[Tile] = []
        self._reward = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._actions, self._tiles, self._reward = [], [info["tile"]], 0.0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.total_steps += 1
        self._actions.append(ACTIONS[action].value)
        self._tiles.append(info["tile"])
        self._reward += float(reward)
        if terminated or truncated:
            self._finish(info, Ending.TERMINATED if terminated else Ending.TRUNCATED)
        return obs, reward, terminated, truncated, info

    def _finish(self, info: dict, ending: Ending) -> None:
        log = EpisodeLog(
            starting_save=self.starting_save,
            actions=self._actions, tiles=self._tiles, reward=self._reward,
            unique_tiles=info["unique_tiles"], unique_locations=info["unique_locations"], ending=ending,
        )
        if self.episodes % self.log_every == 0:
            self.directory.mkdir(parents=True, exist_ok=True)
            (self.directory / f"episode_{self.episodes:05d}.json").write_text(json.dumps(asdict(log)) + "\n")
        self.episodes += 1
        if self.writer is not None:
            for tag, value in metrics(log).items():
                self.writer.add_scalar(tag, value, self.total_steps)
