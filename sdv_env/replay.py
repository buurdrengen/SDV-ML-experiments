"""Replaying an Episode log: reset to its Starting save, play its actions back and record the Tiles the player reaches.

Comparing those Tiles with the log's is the determinism check; experiments/01-exploration/README.md has its results.
"""

from pathlib import Path

from sdv_env.bridge_client import Action
from sdv_env.env import BridgeLike, Tile
from sdv_env.episode_log import EpisodeLog


def replay(client: BridgeLike, log: EpisodeLog, starting_save: Path) -> list[Tile]:
    """Plays `log`'s actions from `starting_save`; returns the start Tile, then the Tile after each Step, stopping early
    if the Episode ends before the actions run out."""
    state = client.reset(starting_save)
    tiles = [Tile(state.location, state.x, state.y)]
    for action in log.actions:
        state = client.step(Action(action)).state
        tiles.append(Tile(state.location, state.x, state.y))
        if state.episode_end is not None:
            break
    return tiles


def first_divergence(expected: list[Tile], replayed: list[Tile]) -> int | None:
    """Index of the first Tile where the trajectories differ (or where the shorter one ends); None if they match."""
    for index, (want, got) in enumerate(zip(expected, replayed)):
        if want != got:
            return index
    if len(expected) != len(replayed):
        return min(len(expected), len(replayed))
    return None
