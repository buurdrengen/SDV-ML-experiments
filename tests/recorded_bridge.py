"""Bridge replies recorded from the real game (see tests/recordings/record.py), replayed so tests need no game."""

import json
from pathlib import Path

from sdv_env.bridge_client import Action, GameState, Outcome, StepResult
from sdv_env.env import ACTIONS, SdvEnv

RECORDING = json.loads((Path(__file__).parent / "recordings/farmhouse_to_farm.json").read_text())


class RecordedBridge:
    """Replays the recorded Episode. Checks the env sends the recorded actions in order, unless `any_action`."""

    def __init__(self, recording=RECORDING, any_action=False):
        self.recording = recording
        self.any_action = any_action
        self.next_step = 0

    def reset(self, save: Path) -> GameState:
        self.next_step = 0
        return GameState.from_reply(self.recording["reset"])

    def step(self, action: Action) -> StepResult:
        steps = self.recording["steps"]
        recorded = steps[self.next_step % len(steps)]
        assert self.any_action or action.value == recorded["action"], f"step {self.next_step}"
        self.next_step += 1
        reply = recorded["reply"]
        return StepResult(Outcome(reply["outcome"]), reply["updates"], GameState.from_reply(reply))


def recorded_actions(count: int) -> list[int]:
    return [ACTIONS.index(Action(step["action"])) for step in RECORDING["steps"][:count]]


def make_env(**kwargs) -> SdvEnv:
    return SdvEnv(client=RecordedBridge(), starting_save=Path("unused"), **kwargs)
