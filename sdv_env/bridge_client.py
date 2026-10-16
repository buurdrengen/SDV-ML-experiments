"""Client for the Bridge: newline-delimited JSON over a localhost TCP socket, one reply per request.

The protocol is described in `bridge/README.md`.
"""

import json
import os
import socket
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

DEFAULT_PORT = 47800
"""Mirrors `BridgeServer.DefaultPort`; both sides honour the `SDV_BRIDGE_PORT` environment variable."""


def bridge_port() -> int:
    return int(os.environ.get("SDV_BRIDGE_PORT", DEFAULT_PORT))


class BridgeError(RuntimeError):
    """The Bridge refused or failed a request."""


class Direction(StrEnum):
    """Which way the player faces."""

    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


class Action(StrEnum):
    """The agent's Discrete(6) action space; the env maps action indices onto these in this order."""

    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    INTERACT = "interact"
    NOOP = "noop"


class Outcome(StrEnum):
    """Whether a Step changed the player's Tile or Location."""

    MOVED = "moved"
    BLOCKED = "blocked"
    WARPED = "warped"
    STAYED = "stayed"


class EpisodeEnd(StrEnum):
    """Why the Bridge ended the Episode."""

    TWO_AM = "two_am"
    PASSED_OUT = "passed_out"


@dataclass(frozen=True)
class GameState:
    """The Game state the Bridge reports after every request."""

    total_updates: int
    time: int
    day: int
    location: str
    x: int
    y: int
    facing: Direction
    menu: str | None
    episode_end: EpisodeEnd | None

    @classmethod
    def from_reply(cls, reply: dict) -> "GameState":
        return cls(
            total_updates=reply["total_updates"],
            time=reply["time"],
            day=reply["day"],
            location=reply["location"],
            x=reply["x"],
            y=reply["y"],
            facing=Direction(reply["facing"]),
            menu=reply["menu"],
            episode_end=EpisodeEnd(reply["episode_end"]) if reply["episode_end"] else None,
        )


@dataclass(frozen=True)
class StepResult:
    outcome: Outcome
    updates: int
    """Game updates the Step took."""
    state: GameState


class BridgeClient:
    def __init__(self, host: str = "127.0.0.1", port: int | None = None, timeout: float = 60.0):
        self._socket = socket.create_connection((host, port or bridge_port()), timeout=timeout)
        self._stream = self._socket.makefile("rwb")

    def set_fast(self, fast: bool) -> None:
        """Fast mode runs updates back to back without drawing; otherwise the game draws at 60 FPS."""
        self.request("mode", fast=fast)

    def reset(self, save: Path) -> GameState:
        """Copy the Starting save folder `save` into SDV's saves folder, load it and return the start state."""
        return GameState.from_reply(self.request("reset", save=str(save.absolute())))

    def step(self, action: Action) -> StepResult:
        reply = self.request("step", action=action.value)
        return StepResult(Outcome(reply["outcome"]), reply["updates"], GameState.from_reply(reply))

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
