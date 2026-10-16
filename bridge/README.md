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
- The game is kept "focused" so it neither pauses nor ignores input when the window is in the background.
- Menus, dialogue and skippable cutscenes are closed as they appear. Questions are answered "No" (or the last option).

## Protocol (prototype)

Newline-delimited JSON on `127.0.0.1:47800` (override with `SDV_BRIDGE_PORT`), one reply per request. Every reply has `ok`; failures carry `error`. Successful replies include `updates`, `time`, `day`, and once a save is loaded `location`, `x`, `y`.

| Request | Effect | Extra reply fields |
|---|---|---|
| `{"cmd": "state"}` | Nothing | |
| `{"cmd": "mode", "fast": true}` | Switch fast/rendered mode | |
| `{"cmd": "load", "save": "<save folder>"}` | Return to title if needed, load the save from SDV's `Saves/` folder, wait until the player can move | `ticks` |
| `{"cmd": "move", "dir": "up\|down\|left\|right"}` | Walk one Tile | `outcome` (`moved`, `blocked`, `warped`, `timeout`), `ticks`; `debug` on timeout |

The full reset/step protocol is #5.
