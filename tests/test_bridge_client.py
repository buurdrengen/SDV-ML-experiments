import json
import socket
import threading

import pytest

from sdv_env.bridge_client import BridgeClient, BridgeError


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
