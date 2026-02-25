# SDV-ML-experiments
Stardew valley agent and experiments 

## Core setup assumptions 
- Game runs in windowed mode at a fixed resolution and UI scale.
- Agent acts at a fixed decision tick (e.g., 5--10 Hz to be determined).
- Agent actions are keypress-level (keyboard/controller style), mouse optional later.
- Observation for the learning policy is always pixels (screen capture).
- API/mod can be used for reward and evaluation (and optionally as a teacher), not as the agent's observation.
