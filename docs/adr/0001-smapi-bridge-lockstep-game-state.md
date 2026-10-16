# Drive the real game through a lockstep SMAPI Bridge, observing game state

Stardew Valley has no emulator, so we cannot copy the Pokémon RL setup (PyBoy: save states, uncapped speed, dozens of parallel instances). We control the real game through a SMAPI mod (the Bridge) that exposes game state and accepts actions over a socket, and the game advances only when the agent sends an action (lockstep). Agents observe structured game state rather than screen pixels, because without emulator-level throughput we cannot afford the sample cost of learning from pixels.

## Considered Options

- **Screen capture + simulated input**: no C# needed, but real-time only, fragile, one window per instance, and no clean resets.
- **A Python re-implementation of Stardew**: fast and parallel, but it would not be Stardew, and fidelity would become its own project.
- **Free-running game sampled by Python**: what the agent experiences would depend on training speed, making runs non-reproducible.

## Consequences

- The project owns a C# SMAPI mod alongside the Python code.
- Throughput is bounded by how fast one game instance can tick and by how many instances fit in RAM. Running several instances at once is unproven (saves live in a shared user folder).
- Pixel-based agents remain possible later by having the Bridge also send frames.
