# Experiment 01: Exploration

Train an agent to explore SDV, rewarded for every new Tile and every new Location it reaches in an Episode. The spec is issue #1; the vocabulary is in `GLOSSARY.md`.

## Running

Start SDV through SMAPI with the Bridge installed (see `bridge/README.md`), then:

```bash
uv run python experiments/01-exploration/random_baseline.py   # --episodes 50 --seed 0 --run-name ...
uv run python experiments/01-exploration/train_ppo.py         # --config ppo.toml --run-name ...
uv run tensorboard --logdir runs
```

Each run writes to `runs/<run name>/` (not versioned):

- TensorBoard metrics, logged once per Episode against the total number of Steps so far. The tags are in `sdv_env/episode_log.py`, shared with PPO training: `episode/reward`, `episode/unique_tiles`, `episode/unique_locations` and `episode/length` (Steps).
- `episodes/episode_NNNNN.json`, one Episode log per Episode (PPO training keeps every 10th, `episode_log_every`): the Starting save, the action of every Step, the (Location, x, y) Tile at the start and after every Step, the Episode reward, unique Tiles and Locations, and its `ending`: `terminated` (2:00 AM or passing out) or `truncated` (the step limit). Read one with `sdv_env.episode_log.read_episode`.

## Random baseline

50 Episodes of uniformly random actions (`random_baseline.py --episodes 50 --seed 0`, run 2026-11-02), with the default Exploration reward (0.01 per new Tile, 1.0 per new Location) and step limit (5000):

| Metric | Mean ± std |
|---|---|
| Episode reward | 3.81 ± 1.44 |
| Unique Tiles | 234.2 ± 82.8 |
| Unique Locations | 2.48 ± 0.71 |
| Episode length (Steps) | 4578 ± 65 |

- Every Episode ended by termination, at 2:00 AM or passing out, before the step limit.
- Unique Tiles ranged from 55 to 451. The FarmHouse and the Farm account for most of them.
- 2 Episodes never left the FarmHouse, and 25 reached only the FarmHouse and the Farm. The Locations reached beyond those were the BusStop (23 Episodes), Backwoods (2) and Tunnel (1). No Episode reached Town.

## PPO

`train_ppo.py` is CleanRL's PPO in one file. It trains against one game instance. Settings live in `ppo.toml`: the four from the spec (learning rate 2.5e-4, γ 0.99, GAE λ 0.95, clip 0.2) plus CleanRL's other defaults. The rollout shape is the exception. CleanRL collects 4 envs × 128 Steps; with a single game, this collects 2048 Steps per update in 8 minibatches of 256. Episodes ended by the step limit are not bootstrapped, as in CleanRL.

The network has 3 conv layers over the View (4×15×15). An MLP takes a 16-wide Location embedding, the time and the one-hot facing. These join in a 256-wide shared body with separate policy and value heads. Runs write `config.json`, the losses (`losses/*`, `charts/SPS`) and a checkpoint every 25 updates (`checkpoints/agent_<Steps>.pt`) on top of the shared Episode metrics.

### First run: 1M Steps

`train_ppo.py --run-name 01-ppo-1m`, run 2026-11-02 with the default `ppo.toml`: 999,424 Steps (488 updates) and 201 Episodes in 55 minutes, ~300 Steps/s on an RTX 5060. The table compares the last 50 training Episodes (the last ~250k Steps, stochastic policy) with the 50 Random baseline Episodes:

| Metric | Random baseline | PPO, last 50 Episodes |
|---|---|---|
| Episode reward | 3.81 ± 1.44 | 11.88 ± 4.13 |
| Unique Tiles | 234.2 ± 82.8 | 801.2 ± 351.0 |
| Unique Locations | 2.48 ± 0.71 | 4.88 ± 1.04 |
| Episode length (Steps) | 4578 ± 65 | 4993 ± 36 |

![Episode metrics in TensorBoard, PPO (light blue) against the Random baseline (dark)](images/tb-episode.png)

- PPO clearly beats the Random baseline: about 3.4× the unique Tiles and twice the unique Locations. Every one of its last 50 Episodes reached at least 4 Locations, and the best reached 7. Unique Tiles ranged from 332 to 2236.
- Within ~100k Steps the agent learns to leave the FarmHouse. By ~300k it reliably goes Farm → BusStop → Town. The logged Episodes also reach the Mountain, Backwoods, Tunnel, Beach and Tent (Linus's). It enters no shop or house in Town; those doors need `interact`.
- The step limit now ends almost every Episode: 176 of 201, and 48 of the last 50. Unlike the Random baseline, the trained agent seldom reaches 2:00 AM within 5000 Steps. It walks into walls a lot. In the last logged Episode (episode 200), 4,066 of 4,885 moves were blocked, against 1,433 in episode 150 and 739 of 3,034 in a Random Episode. Blocked moves probably pass little game time. The step limit, not the day, now bounds an Episode. Once the agent has stopped finding new Tiles, nothing in the Exploration reward pushes it off a wall.
- Entropy fell from 1.79 (uniform over 6 actions) to ~0.5, and explained variance settled around 0.85–0.95 late in the run.

![PPO losses and Steps per second](images/tb-losses.png)
