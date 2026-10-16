"""Client for the Bridge: newline-delimited JSON over a localhost TCP socket, one reply per request."""

import json
import socket

DEFAULT_PORT = 47800


class BridgeError(RuntimeError):
    """The Bridge refused or failed a request."""


class BridgeClient:
    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_PORT, timeout: float = 60.0):
        self._socket = socket.create_connection((host, port), timeout=timeout)
        self._stream = self._socket.makefile("rwb")

    def request(self, cmd: str, **args) -> dict:
        self._stream.write(json.dumps({"cmd": cmd, **args}).encode() + b"\n")
        self._stream.flush()
        line = self._stream.readline()
        if not line:
            raise BridgeError("the Bridge closed the connection (did the game exit?)")
        reply = json.loads(line)
        if not reply.get("ok"):
            raise BridgeError(reply.get("error", "unknown Bridge error"))
        return reply

    def close(self) -> None:
        self._stream.close()
        self._socket.close()

    def __enter__(self) -> "BridgeClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
