"""Smoke test against the real game: start SDV through SMAPI with the Bridge installed, then run
    uv run pytest tests/test_bridge_live.py
Skipped when no Bridge is listening. Takes control of the game window's loop and leaves it in fast mode.
"""

import dataclasses
import random
import socket

import pytest

from sdv_env.bridge_client import Action, BridgeClient, GameState, Outcome, TileFlag, bridge_port
from sdv_env.env import STARTING_SAVE



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
    below_door = GameState.from_reply(client.request("teleport", location="Town", x=43, y=57, facing="up", time=900))
    assert TileFlag.EXIT in below_door.view[6][7]

    result = client.step(Action.INTERACT)

    assert result.outcome is Outcome.WARPED
    assert result.state.location == "SeedShop"
    assert result.state.menu is None


def test_walking_out_of_the_farmhouse_lands_below_its_door_and_interacting_with_the_door_goes_back(client):
    client.reset(STARTING_SAVE)
    for action in [Action.LEFT] * 6 + [Action.DOWN] * 2:
        client.step(action)

    out = client.step(Action.DOWN)
    assert out.outcome is Outcome.WARPED
    assert out.state.location == "Farm"
    # The Farm-side door is the exit Tile right above the player.
    assert TileFlag.EXIT in out.state.view[6][7]

    assert client.step(Action.UP).outcome is Outcome.BLOCKED
    back = client.step(Action.INTERACT)
    assert back.outcome is Outcome.WARPED
    assert back.state.location == "FarmHouse"


def test_the_live_env_passes_gymnasiums_env_checker(client):
    from gymnasium.utils.env_checker import check_env

    from sdv_env.env import SdvEnv

    check_env(SdvEnv(client=client, starting_save=STARTING_SAVE), skip_render_check=True)
