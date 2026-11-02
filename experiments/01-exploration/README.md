# Experiment 01: Exploration

Train an agent to explore SDV, rewarded for every new Tile and every new Location it reaches in an Episode. The spec is issue #1; the vocabulary is in `GLOSSARY.md`.

## Running

Start SDV through SMAPI with the Bridge installed (see `bridge/README.md`), then:

```bash
uv run python experiments/01-exploration/random_baseline.py   # --episodes 50 --seed 0 --run-name ...
uv run tensorboard --logdir runs
```

Each run writes to `runs/<run name>/` (not versioned):

- TensorBoard metrics, logged once per Episode against the total number of Steps so far. The tags are in `sdv_env/episode_log.py`, shared with PPO training: `episode/reward`, `episode/unique_tiles`, `episode/unique_locations` and `episode/length` (Steps).
- `episodes/episode_NNNNN.json`, one Episode log per Episode: the Starting save, the action of every Step, the (Location, x, y) Tile at the start and after every Step, the Episode reward, unique Tiles and Locations, and its `ending`: `terminated` (2:00 AM or passing out) or `truncated` (the step limit). Read one with `sdv_env.episode_log.read_episode`.

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
