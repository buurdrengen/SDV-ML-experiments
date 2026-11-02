"""Record Bridge replies for the offline env tests. Needs the game running with the Bridge:
    uv run python tests/recordings/record.py
"""

import json
from pathlib import Path

from sdv_env.bridge_client import BridgeClient
from sdv_env.env import STARTING_SAVE

# Walk out of the FarmHouse onto the Farm (64, 15), step down and back onto the visited Tile, bump into the door,
# go back inside by interacting with it (onto the visited Tile 3, 11), and walk out again.
ACTIONS = ["left"] * 6 + ["down"] * 3 + ["down", "up", "noop", "up", "interact", "down"]


def record(path: Path) -> None:
    with BridgeClient(timeout=300) as client:
        client.set_fast(True)
        replies = {"reset": client.request("reset", save=str(STARTING_SAVE.absolute())), "steps": []}
        for action in ACTIONS:
            replies["steps"].append({"action": action, "reply": client.request("step", action=action)})
    path.write_text(json.dumps(replies, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    record(Path(__file__).with_name("farmhouse_to_farm.json"))
