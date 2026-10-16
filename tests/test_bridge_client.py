import json
import socket
import threading
from pathlib import Path

import pytest

from sdv_env.bridge_client import (
    Action,
    BridgeClient,
    BridgeError,
    Direction,
    EpisodeEnd,
    GameState,
    Outcome,
)


class FakeBridge:
    """A TCP server speaking the Bridge's newline-delimited JSON protocol."""

    def __init__(self, handler):
        self.handler = handler
        self.requests = []
        self._server = socket.create_server(("127.0.0.1", 0))
        self.port = self._server.getsockname()[1]
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self):
        conn, _ = self._server.accept()
        with conn, conn.makefile("rwb") as stream:
            for line in stream:
                request = json.loads(line)
                self.requests.append(request)
                reply = self.handler(request)
                if reply is None:
                    return
                stream.write(reply if isinstance(reply, bytes) else json.dumps(reply).encode() + b"\n")
                stream.flush()

    def close(self):
        self._server.close()


@pytest.fixture
def bridge_with():
    bridges = []

    def make(handler):
        bridge = FakeBridge(handler)
        bridges.append(bridge)
        return bridge

    yield make
    for bridge in bridges:
        bridge.close()


def test_move_sends_command_and_returns_new_tile(bridge_with):
    bridge = bridge_with(lambda req: {"ok": True, "location": "FarmHouse", "x": 4, "y": 9})

    with BridgeClient(port=bridge.port) as client:
        reply = client.request("move", dir="up")

    assert bridge.requests == [{"cmd": "move", "dir": "up"}]
    assert reply == {"ok": True, "location": "FarmHouse", "x": 4, "y": 9}


def test_bridge_refusal_raises_with_its_message(bridge_with):
    bridge = bridge_with(lambda req: {"ok": False, "error": "no save loaded"})

    with BridgeClient(port=bridge.port) as client:
        with pytest.raises(BridgeError, match="no save loaded"):
            client.request("move", dir="left")


def test_bridge_closing_the_connection_raises(bridge_with):
    bridge = bridge_with(lambda req: None)

    with BridgeClient(port=bridge.port) as client:
        with pytest.raises(BridgeError, match="closed the connection"):
            client.request("move", dir="down")


START = {
    "ok": True,
    "total_updates": 120,
    "time": 600,
    "day": 1,
    "location": "FarmHouse",
    "x": 3,
    "y": 9,
    "facing": "down",
    "menu": None,
    "episode_end": None,
}


def test_reset_sends_the_absolute_save_path_and_returns_the_start_state(bridge_with, tmp_path, monkeypatch):
    bridge = bridge_with(lambda req: START)
    monkeypatch.chdir(tmp_path)

    with BridgeClient(port=bridge.port) as client:
        state = client.reset(Path("saves/exploration_1"))

    assert bridge.requests == [{"cmd": "reset", "save": str(tmp_path / "saves" / "exploration_1")}]
    assert state == GameState(
        total_updates=120, time=600, day=1, location="FarmHouse", x=3, y=9,
        facing=Direction.DOWN, menu=None, episode_end=None,
    )


def test_step_sends_the_action_and_returns_outcome_updates_and_new_state(bridge_with):
    bridge = bridge_with(lambda req: {**START, "x": 4, "outcome": "moved", "updates": 12})

    with BridgeClient(port=bridge.port) as client:
        result = client.step(Action.RIGHT)

    assert bridge.requests == [{"cmd": "step", "action": "right"}]
    assert result.outcome is Outcome.MOVED
    assert result.updates == 12
    assert (result.state.location, result.state.x, result.state.y) == ("FarmHouse", 4, 9)


def test_step_reports_the_end_of_the_episode(bridge_with):
    bridge = bridge_with(
        lambda req: {**START, "time": 2600, "outcome": "stayed", "updates": 3, "episode_end": "two_am"}
    )

    with BridgeClient(port=bridge.port) as client:
        result = client.step(Action.NOOP)

    assert result.state.episode_end is EpisodeEnd.TWO_AM


def test_client_connects_to_the_port_in_SDV_BRIDGE_PORT_like_the_bridge(bridge_with, monkeypatch):
    bridge = bridge_with(lambda req: START)
    monkeypatch.setenv("SDV_BRIDGE_PORT", str(bridge.port))

    with BridgeClient() as client:
        client.set_fast(True)

    assert bridge.requests == [{"cmd": "mode", "fast": True}]
