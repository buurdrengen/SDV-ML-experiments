"""The Gymnasium environment for SDV: observations built from the Bridge's Game state, and the Exploration reward."""

from pathlib import Path
from typing import NamedTuple, Protocol

import gymnasium as gym
import numpy as np

from sdv_env.bridge_client import Action, BridgeClient, Direction, GameState, StepResult, TileFlag
from sdv_env.locations import LOCATION_COUNT, location_id

VIEW_SIZE = 15
"""The View is VIEW_SIZE x VIEW_SIZE Tiles centred on the player; the Bridge sends the same size."""

VIEW_CHANNELS = (TileFlag.WALKABLE, TileFlag.EXIT, TileFlag.OCCUPIED)
"""The View's channels in order, from the Bridge's Tile flags; a last channel marks Tiles visited this Episode."""

VIEW_SHAPE = (len(VIEW_CHANNELS) + 1, VIEW_SIZE, VIEW_SIZE)

START_TIME, END_TIME = 600, 2600
"""6:00 and 2:00 AM, as SDV writes the time of day."""

STARTING_SAVE = Path(__file__).parents[1] / "experiments/01-exploration/starting-save/exploration_451834694"


ACTIONS = tuple(Action)
"""The action space: index i is ACTIONS[i]."""


class Tile(NamedTuple):
    """A Tile: (Location, x, y)."""

    location: str
    x: int
    y: int


class BridgeLike(Protocol):
    """What the env needs from a Bridge connection; `BridgeClient`, or recorded replies in tests."""

    def reset(self, save: Path) -> GameState: ...

    def step(self, action: Action) -> StepResult: ...


class SdvEnv(gym.Env):
    """One Episode runs from the Starting save until 2:00 AM, the player passing out, or `step_limit` Steps.

    Actions are `Action` members by index. The Exploration reward pays `tile_reward` for each (Location, x, y)
    first visited this Episode and `location_bonus` on top for each Location first entered this Episode.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        client: BridgeLike | None = None,
        starting_save: Path = STARTING_SAVE,
        step_limit: int = 5000,
        tile_reward: float = 0.01,
        location_bonus: float = 1.0,
    ):
        if client is None:
            client = BridgeClient()
            client.set_fast(True)
        self.client = client
        self.starting_save = starting_save
        self.step_limit = step_limit
        self.tile_reward = tile_reward
        self.location_bonus = location_bonus

        self.action_space = gym.spaces.Discrete(len(ACTIONS))
        self.observation_space = gym.spaces.Dict({
            "view": gym.spaces.Box(0, 1, VIEW_SHAPE, np.uint8),
            "location": gym.spaces.Discrete(LOCATION_COUNT),
            "time": gym.spaces.Box(0.0, 1.0, (1,), np.float32),
            "facing": gym.spaces.Discrete(len(Direction)),
        })

        self._visited: set[Tile] = set()
        self._tile: Tile | None = None
        self._locations: set[str] = set()
        self._steps = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        state = self.client.reset(self.starting_save)
        self._visited.clear()
        self._locations.clear()
        self._steps = 0
        self._visit(state)
        return self._observe(state), self._info()

    def step(self, action: int):
        result = self.client.step(ACTIONS[action])
        self._steps += 1
        reward = self._visit(result.state)
        terminated = result.state.episode_end is not None
        truncated = not terminated and self._steps >= self.step_limit
        return self._observe(result.state), reward, terminated, truncated, self._info()

    def _visit(self, state: GameState) -> float:
        """Marks the player's Tile and Location as visited and returns the Exploration reward for them."""
        reward = 0.0
        self._tile = Tile(state.location, state.x, state.y)
        if self._tile not in self._visited:
            self._visited.add(self._tile)
            reward += self.tile_reward
        if state.location not in self._locations:
            self._locations.add(state.location)
            reward += self.location_bonus
        return reward

    def _observe(self, state: GameState) -> dict:
        view = np.zeros(VIEW_SHAPE, np.uint8)
        radius = VIEW_SIZE // 2
        for row in range(VIEW_SIZE):
            for col in range(VIEW_SIZE):
                flags = state.view[row][col]
                for channel, flag in enumerate(VIEW_CHANNELS):
                    view[channel, row, col] = flag in flags
                tile = Tile(state.location, state.x - radius + col, state.y - radius + row)
                view[len(VIEW_CHANNELS), row, col] = tile in self._visited
        return {
            "view": view,
            "location": location_id(state.location),
            "time": np.array([normalised_time(state.time)], np.float32),
            "facing": list(Direction).index(state.facing),
        }

    def _info(self) -> dict:
        return {"tile": self._tile, "unique_tiles": len(self._visited), "unique_locations": len(self._locations)}


def normalised_time(time: int) -> float:
    """0 at 6:00, 1 at 2:00 AM, linear in game minutes (SDV writes 13:50 as 1350)."""
    elapsed = _minutes(time) - _minutes(START_TIME)
    return min(max(elapsed / (_minutes(END_TIME) - _minutes(START_TIME)), 0.0), 1.0)


def _minutes(time: int) -> int:
    return time // 100 * 60 + time % 100
