# SDV-ML-experiments

Deep learning and reinforcement learning experiments that train agents to play Stardew Valley, inspired by Peter Whidden's Pokémon Red RL project.

## Language

**SDV**:
Stardew Valley, the farming-sim video game the agents play.
_Avoid_: Synthetic Data Vault, Software-Defined Vehicle

**Bridge**:
The SMAPI mod that runs inside SDV, sends game state to the agent and applies the agent's actions.
_Avoid_: Plugin, connector, wrapper

**Game state**:
Structured facts read from inside SDV (location, position, tiles, inventory, time, energy), as opposed to screen pixels.
_Avoid_: RAM, memory, observation (the observation is built *from* game state)

## World

**Location**:
One of SDV's named maps (Farm, FarmHouse, Town, Beach…), connected to others by warps.
_Avoid_: Map, area, scene, level

**Tile**:
A single grid cell within a Location, identified by (Location, x, y).
_Avoid_: Cell, square, position

## Training

**Experiment**:
A numbered, self-contained study with one objective (e.g. 01 Exploration), reusing the shared environment.
_Avoid_: Project, task, run

**Starting save**:
The hand-made, versioned SDV save every Episode of an Experiment is reset to.
_Avoid_: Checkpoint (that means saved network weights), snapshot

**View**:
The square of Tiles centred on the agent, with per-Tile features, that forms the spatial part of its observation.
_Avoid_: Window, crop, viewport

**Random baseline**:
An agent choosing uniformly random actions, used as the bar a trained agent must clearly beat.
_Avoid_: Control, dummy agent

**Episode**:
One playthrough from the fixed starting save (Spring 1, 6:00, FarmHouse) until 2:00 AM, the character passing out, or the step limit.
_Avoid_: Run, game, day

**Step**:
One agent action: move one Tile in a direction, interact, or do nothing. The game only advances during a Step.
_Avoid_: Tick, frame, turn

**Exploration reward**:
Reward for visiting a Tile for the first time in the Episode, plus a larger bonus for entering a Location for the first time in the Episode.
_Avoid_: Novelty reward, curiosity
