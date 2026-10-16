# Bridge

SMAPI mod that lets an agent drive SDV in lockstep over a localhost socket (see `docs/adr/0001-smapi-bridge-lockstep-game-state.md`).

## Build and install

Requires SMAPI and the .NET SDK (8 works; the mod targets .NET 6 like the game).

```bash
dotnet build bridge
```

`Pathoschild.Stardew.ModBuildConfig` copies the built mod into the game's `Mods/` folder. Restart the game to pick up a new build.

## How it runs the game

- Until an agent sends its first request, the game runs normally.
- After that the Bridge owns MonoGame's frame loop: the game only updates while a request needs it, always with a fixed 1/60 s timestep, so game logic sees normal speed however fast it really runs.
- **Rendered mode** runs one update per frame at 60 FPS and draws. **Fast mode** runs updates back to back and never draws (the window freezes on its last frame).
- The game is kept "focused" so it neither pauses nor ignores input when the window is in the background. Real keyboard input still reaches it, so don't type into the window.
- Menus, dialogue boxes and skippable cutscenes are closed after every update. Questions are answered "No" (or the last option), so walking onto the bed doesn't end the day.
- If the agent disconnects mid-request, the Bridge drops that request within 100 ms.

## Protocol

Newline-delimited JSON on `127.0.0.1:47800` (override with `SDV_BRIDGE_PORT`; the Python client reads the same variable). Exactly one reply per request. Every reply has `ok`; failures carry only `error`.

### Requests

| Request | Effect | Extra reply fields |
|---|---|---|
| `{"cmd": "reset", "save": "<absolute path to a save folder>"}` | Copy the save folder (e.g. the repo's Starting save) into SDV's `Saves/` folder, overwriting SDV's copy, return to the title screen if needed, load it and wait until the player can act | `updates` |
| `{"cmd": "step", "action": "<action>"}` | Carry out one Step (below). Refused until `reset`, and again once the Episode has ended | `outcome`, `updates` |
| `{"cmd": "mode", "fast": true}` | Switch between fast and rendered mode | |
| `{"cmd": "state"}` | Nothing; just report the Game state | |
| `{"cmd": "teleport", "location": "Town", "x": 43, "y": 57, "facing": "up", "time": 900}` | **Testing aid, not for agents.** Warp the player (optionally setting the clock) and wait until they can act | |

### Game state (every successful reply)

| Field | Meaning |
|---|---|
| `total_updates` | Game updates the Bridge has run since it took over the loop |
| `time`, `day` | `Game1.timeOfDay` (600 = 6:00, 2600 = 2:00 AM) and day of the month |
| `location`, `x`, `y` | Current Location and Tile; `null` before the first `reset` |
| `facing` | `up`, `down`, `left` or `right` |
| `menu` | Type name of an open menu (dialogue boxes included), otherwise `null`. After a Step it is `null`, unless the Step hit its 3,600-update cap (an unskippable cutscene; the Bridge logs a warning) or ended the Episode |
| `episode_end` | `null` while the Episode runs, otherwise `two_am` or `passed_out` |

### Steps

`action` is one of the Discrete(6) actions:

- `up`, `down`, `left`, `right`: walk one Tile. The move ends at the next Tile's centre (`moved`) or when the player stops making progress for 4 updates (`blocked`). That includes being frozen, but not a warp or fade in progress. SDV nudges the player around corners, so a `moved` Step re-centres the player on the target Tile if they fit there, otherwise on whichever Tile they ended on; `x`, `y` always report the real Tile.
- `interact`: act on the Tile the player faces, like the action button (doors, NPCs, objects), then idle.
- `noop`: idle.

`interact` and `noop` last 12 updates, about as long as a move, so every Step advances the clock by a similar amount.

After acting, every Step keeps updating until the player is settled: no warp, fade, menu, dialogue or cutscene, and the player can move. The fade-in after a warp therefore belongs to the Step that caused it. `outcome` is then `warped` if the Location changed, otherwise the move's result, or `stayed` for `interact` and `noop`. `updates` is the number of game updates the Step took.

A Step ends immediately, with `episode_end` set, when the clock reaches 2:00 AM or the player passes out. The Bridge stops updating there, so the game never starts the next day or saves.
