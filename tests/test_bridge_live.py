"""Smoke test against the real game: start SDV through SMAPI with the Bridge installed, then run
    uv run pytest tests/test_bridge_live.py
Skipped when no Bridge is listening. Takes control of the game window's loop and leaves it in fast mode.
"""

import dataclasses
import random
import socket
from pathlib import Path

import pytest

from sdv_env.bridge_client import Action, BridgeClient, GameState, Outcome, bridge_port

STARTING_SAVE = Path(__file__).parents[1] / "experiments/01-exploration/starting-save/exploration_451834694"


def bridge_is_listening() -> bool:
    try:
        socket.create_connection(("127.0.0.1", bridge_port()), timeout=0.5).close()
        return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not bridge_is_listening(), reason="no Bridge listening (start the game)")


@pytest.fixture(scope="module")
def client():
    with BridgeClient(timeout=300) as client:
        client.set_fast(True)
        yield client


def without_update_count(state: GameState) -> GameState:
    return dataclasses.replace(state, total_updates=0)


def test_random_steps_change_location_close_every_menu_and_reset_to_the_identical_start(client):
    rng = random.Random(0)
    start = client.reset(STARTING_SAVE)
    locations = {start.location}

    for _ in range(500):
        # Lean towards the FarmHouse exit at the bottom, so the walk is sure to change Location.
        result = client.step(rng.choices(list(Action), weights=[1, 3, 1, 1, 1, 1])[0])
        assert result.state.menu is None
        locations.add(result.state.location)

    assert len(locations) >= 2
    assert without_update_count(client.reset(STARTING_SAVE)) == without_update_count(start)


def test_interacting_with_a_town_door_enters_the_building(client):
    client.reset(STARTING_SAVE)
    # Stand below Pierre's door (Town 43,56) after it opens at 9:00.
    client.request("teleport", location="Town", x=43, y=57, facing="up", time=900)

    result = client.step(Action.INTERACT)

    assert result.outcome is Outcome.WARPED
    assert result.state.location == "SeedShop"
    assert result.state.menu is None
