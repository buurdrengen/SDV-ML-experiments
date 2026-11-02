"""PPO training for Experiment 01 (experiments/01-exploration/train_ppo.py), against recorded Bridge replies."""

import importlib.util
import dataclasses
import math
from pathlib import Path

import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from recorded_bridge import RecordedBridge, make_env

TRAIN_PPO = Path(__file__).parents[1] / "experiments/01-exploration/train_ppo.py"
_spec = importlib.util.spec_from_file_location("train_ppo", TRAIN_PPO)
train_ppo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_ppo)


def test_gae_discounts_future_rewards_and_stops_at_an_episode_end():
    # gamma = lambda = 0.5 keeps the arithmetic by hand:
    # Step 2: 2 + 0.5 * 4 (bootstrapped) - 0 = 4
    # Step 1 ends its Episode, so nothing after it counts: 0 - 1 = -1
    # Step 0: delta 1 + 0.5 * 1 - 0.5 = 1, plus 0.25 * -1 from Step 1 = 0.75
    advantages, returns = train_ppo.compute_gae(
        rewards=torch.tensor([1.0, 0.0, 2.0]),
        values=torch.tensor([0.5, 1.0, 0.0]),
        episode_ends=torch.tensor([False, True, False]),
        next_value=torch.tensor(4.0),
        gamma=0.5,
        gae_lambda=0.5,
    )

    assert advantages.tolist() == pytest.approx([0.75, -1.0, 4.0])
    assert returns.tolist() == pytest.approx([1.25, 0.0, 4.0])


def sampled_observations(count: int) -> dict[str, torch.Tensor]:
    space = make_env().observation_space
    space.seed(0)
    return train_ppo.batch_observations([space.sample() for _ in range(count)], torch.device("cpu"))


def test_an_untrained_agent_scores_a_batch_of_observations_with_a_near_uniform_policy():
    agent = train_ppo.Agent()

    action, log_prob, entropy, value = agent.get_action_and_value(sampled_observations(5), torch.tensor([0, 1, 2, 3, 5]))

    assert action.tolist() == [0, 1, 2, 3, 5]
    assert log_prob.tolist() == pytest.approx([-math.log(6)] * 5, abs=0.05)  # policy head starts near uniform
    assert entropy.tolist() == pytest.approx([math.log(6)] * 5, abs=0.01)
    assert value.shape == (5,)
    assert agent.get_value(sampled_observations(5)).tolist() == pytest.approx(value.tolist())


def test_the_agent_samples_valid_actions_when_none_are_given():
    env = make_env()
    agent = train_ppo.Agent()
    obs, _ = env.reset()

    action, *_ = agent.get_action_and_value(train_ppo.batch_observations([obs], torch.device("cpu")))

    assert action.shape == (1,)
    assert env.action_space.contains(int(action))


def test_the_shipped_config_uses_cleanrl_ppo_defaults_and_the_experiment_settings():
    config = train_ppo.load_config(train_ppo.DEFAULT_CONFIG)

    assert (config.learning_rate, config.gamma, config.gae_lambda, config.clip_coef) == (2.5e-4, 0.99, 0.95, 0.2)
    assert (config.step_limit, config.tile_reward, config.location_bonus) == (5000, 0.01, 1.0)
    assert config.total_steps == 1_000_000


def test_a_config_with_an_unknown_key_is_rejected(tmp_path):
    path = tmp_path / "typo.toml"
    path.write_text("learning_rat = 1e-3\n")

    with pytest.raises(TypeError, match="learning_rat"):
        train_ppo.load_config(path)


def test_a_short_training_run_writes_checkpoints_episode_logs_and_tensorboard_metrics(tmp_path):
    # 3 updates of 16 Steps over 15-Step Episodes: 3 complete Episodes.
    config = dataclasses.replace(
        train_ppo.load_config(train_ppo.DEFAULT_CONFIG),
        total_steps=48, rollout_steps=16, num_minibatches=2, update_epochs=2, device="cpu",
        step_limit=15, checkpoint_every=2, episode_log_every=1,
    )

    train_ppo.train(config, tmp_path, client=RecordedBridge(any_action=True))

    checkpoints = sorted(path.name for path in (tmp_path / "checkpoints").iterdir())
    assert checkpoints == ["agent_00000032.pt", "agent_00000048.pt"]  # every 2nd update, and the last
    agent = train_ppo.Agent()
    agent.load_state_dict(torch.load(tmp_path / "checkpoints/agent_00000048.pt"))
    assert sorted(path.name for path in (tmp_path / "episodes").iterdir()) == [
        "episode_00000.json", "episode_00001.json", "episode_00002.json",
    ]
    events = EventAccumulator(str(tmp_path))
    events.Reload()
    assert [event.step for event in events.Scalars("episode/unique_tiles")] == [15, 30, 45]
    assert [event.step for event in events.Scalars("losses/value_loss")] == [16, 32, 48]
