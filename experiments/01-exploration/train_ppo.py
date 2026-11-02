"""PPO for Experiment 01, CleanRL style: one file, one game instance.

Start the game through SMAPI with the Bridge installed, then run:
    uv run python experiments/01-exploration/train_ppo.py   # --config ppo.toml --run-name ...
Writes TensorBoard metrics, network checkpoints and the Episode log of every N-th Episode to runs/<run name>/.
"""

import argparse
import json
import random
import time
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

from sdv_env.bridge_client import Direction
from sdv_env.env import ACTIONS, VIEW_SHAPE, BridgeLike, SdvEnv
from sdv_env.episode_log import EpisodeRecorder
from sdv_env.locations import LOCATION_COUNT

DEFAULT_CONFIG = Path(__file__).with_name("ppo.toml")
RUNS = Path(__file__).parents[2] / "runs"
LOCATION_EMBEDDING = 16


@dataclass(frozen=True)
class Config:
    """Training settings, read from a TOML file such as `ppo.toml`, which documents them."""

    total_steps: int
    seed: int
    device: str
    learning_rate: float
    anneal_lr: bool
    gamma: float
    gae_lambda: float
    clip_coef: float
    clip_vloss: bool
    norm_adv: bool
    ent_coef: float
    vf_coef: float
    max_grad_norm: float
    rollout_steps: int
    num_minibatches: int
    update_epochs: int
    step_limit: int
    tile_reward: float
    location_bonus: float
    checkpoint_every: int
    episode_log_every: int


def load_config(path: Path) -> Config:
    """Reads a Config; unknown or missing keys raise TypeError."""
    with path.open("rb") as file:
        return Config(**tomllib.load(file))


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    episode_ends: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalised advantage estimates and returns for one rollout.

    `episode_ends[t]` is whether Step t ended its Episode; `next_value` is the value of the observation after the
    rollout's last Step, which bootstraps it unless that Step ended its Episode.
    """
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros(())
    for t in reversed(range(len(rewards))):
        not_end = 1.0 - episode_ends[t].float()
        following_value = next_value if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * following_value * not_end - values[t]
        last_advantage = delta + gamma * gae_lambda * not_end * last_advantage
        advantages[t] = last_advantage
    return advantages, advantages + values


def batch_observations(observations: list[dict], device: torch.device) -> dict[str, torch.Tensor]:
    """Stacks env observations into one batch of tensors on `device`."""
    return {
        "view": torch.as_tensor(np.stack([obs["view"] for obs in observations]), dtype=torch.float32, device=device),
        "location": torch.as_tensor([int(obs["location"]) for obs in observations], device=device),
        "time": torch.as_tensor(np.stack([obs["time"] for obs in observations]), dtype=torch.float32, device=device),
        "facing": torch.as_tensor([int(obs["facing"]) for obs in observations], device=device),
    }


def layer_init[L: nn.Linear | nn.Conv2d](layer: L, std: float = np.sqrt(2), bias: float = 0.0) -> L:
    """CleanRL's initialisation: orthogonal weights scaled by `std`, constant bias."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class Agent(nn.Module):
    """Conv layers over the View joined with an MLP over the Location embedding, time and facing, then one shared
    body with separate policy and value heads."""

    def __init__(self):
        super().__init__()
        channels, height, width = VIEW_SHAPE
        self.facing_count = len(Direction)
        self.view = nn.Sequential(
            layer_init(nn.Conv2d(channels, 32, 3, padding=1)), nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 3, stride=2)), nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3)), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            view_features = self.view(torch.zeros(1, channels, height, width)).shape[1]
        self.location = nn.Embedding(LOCATION_COUNT, LOCATION_EMBEDDING)
        self.scalars = nn.Sequential(layer_init(nn.Linear(LOCATION_EMBEDDING + 1 + self.facing_count, 64)), nn.ReLU())
        self.body = nn.Sequential(layer_init(nn.Linear(view_features + 64, 256)), nn.ReLU())
        self.policy = layer_init(nn.Linear(256, len(ACTIONS)), std=0.01)
        self.value = layer_init(nn.Linear(256, 1), std=1.0)

    def _features(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        facing = nn.functional.one_hot(obs["facing"], self.facing_count).float()
        scalars = self.scalars(torch.cat([self.location(obs["location"]), obs["time"], facing], dim=1))
        return self.body(torch.cat([self.view(obs["view"]), scalars], dim=1))

    def get_value(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """The value of each observation in the batch."""
        return self.value(self._features(obs)).squeeze(1)

    def get_action_and_value(self, obs: dict[str, torch.Tensor], action: torch.Tensor | None = None):
        """Samples an action unless one is given; returns it with its log-probability, the policy's entropy and
        the value."""
        features = self._features(obs)
        distribution = Categorical(logits=self.policy(features))
        if action is None:
            action = distribution.sample()
        return action, distribution.log_prob(action), distribution.entropy(), self.value(features).squeeze(1)


def train(config: Config, run: Path, client: BridgeLike | None = None) -> Agent:
    """Trains an Agent for `config.total_steps` Steps (rounded down to whole rollouts), logging into `run`.

    `client` is the Bridge connection; by default a `BridgeClient` to the running game. Episodes ended by the step
    limit are not bootstrapped, as in CleanRL.
    """
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    checkpoints = run / "checkpoints"
    checkpoints.mkdir(exist_ok=True)

    writer = SummaryWriter(run)
    env = EpisodeRecorder(
        SdvEnv(client, step_limit=config.step_limit, tile_reward=config.tile_reward,
               location_bonus=config.location_bonus),
        run / "episodes", writer, log_every=config.episode_log_every,
    )
    agent = Agent().to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=config.learning_rate, eps=1e-5)

    rollout_steps = config.rollout_steps
    minibatch_size = rollout_steps // config.num_minibatches
    updates = config.total_steps // rollout_steps
    # The rollout is collected on the CPU, one Step at a time, and moved to `device` for the update.
    observations: list[dict] = [{}] * rollout_steps
    actions = torch.zeros(rollout_steps, dtype=torch.long)
    log_probs = torch.zeros(rollout_steps)
    rewards = torch.zeros(rollout_steps)
    episode_ends = torch.zeros(rollout_steps, dtype=torch.bool)
    values = torch.zeros(rollout_steps)

    global_step = 0
    start = time.time()
    next_obs, _ = env.reset(seed=config.seed)
    for update in range(1, updates + 1):
        if config.anneal_lr:
            optimizer.param_groups[0]["lr"] = (1.0 - (update - 1.0) / updates) * config.learning_rate

        for step in range(rollout_steps):
            global_step += 1
            observations[step] = next_obs
            with torch.no_grad():
                action, log_prob, _, value = agent.get_action_and_value(batch_observations([next_obs], device))
            action_index, log_probs[step], values[step] = torch.stack([action.float(), log_prob, value], 1)[0].tolist()
            actions[step] = int(action_index)
            next_obs, reward, terminated, truncated, _ = env.step(int(action_index))
            rewards[step] = float(reward)
            episode_ends[step] = terminated or truncated
            if terminated or truncated:
                next_obs, _ = env.reset()

        with torch.no_grad():
            next_value = agent.get_value(batch_observations([next_obs], device))[0].cpu()
        advantages, returns = compute_gae(rewards, values, episode_ends, next_value, config.gamma, config.gae_lambda)

        batch = batch_observations(observations, device)
        b_actions, b_log_probs, b_values, b_advantages, b_returns = (
            tensor.to(device) for tensor in (actions, log_probs, values, advantages, returns))
        indices = np.arange(rollout_steps)
        clip_fractions = []
        losses: dict[str, float] = {}
        for _ in range(config.update_epochs):
            np.random.shuffle(indices)
            for begin in range(0, rollout_steps, minibatch_size):
                mb = indices[begin:begin + minibatch_size]
                _, new_log_prob, entropy, new_value = agent.get_action_and_value(
                    {key: tensor[mb] for key, tensor in batch.items()}, b_actions[mb])
                log_ratio = new_log_prob - b_log_probs[mb]
                ratio = log_ratio.exp()
                with torch.no_grad():
                    old_approx_kl = (-log_ratio).mean()
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    clip_fractions.append(((ratio - 1.0).abs() > config.clip_coef).float().mean().item())

                mb_advantages = b_advantages[mb]
                if config.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                policy_loss = torch.max(
                    -mb_advantages * ratio,
                    -mb_advantages * torch.clamp(ratio, 1 - config.clip_coef, 1 + config.clip_coef),
                ).mean()

                if config.clip_vloss:
                    unclipped = (new_value - b_returns[mb]) ** 2
                    clipped_value = b_values[mb] + torch.clamp(new_value - b_values[mb], -config.clip_coef, config.clip_coef)
                    clipped = (clipped_value - b_returns[mb]) ** 2
                    value_loss = 0.5 * torch.max(unclipped, clipped).mean()
                else:
                    value_loss = 0.5 * ((new_value - b_returns[mb]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = policy_loss - config.ent_coef * entropy_loss + config.vf_coef * value_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
                optimizer.step()
                losses = {
                    "value_loss": value_loss.item(), "policy_loss": policy_loss.item(),
                    "entropy": entropy_loss.item(), "old_approx_kl": old_approx_kl.item(),
                    "approx_kl": approx_kl.item(),
                }

        variance = returns.var()
        explained_variance = float("nan") if variance == 0 else (1 - (returns - values).var() / variance).item()
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        for name, value in losses.items():  # from the last minibatch, as CleanRL logs them
            writer.add_scalar(f"losses/{name}", value, global_step)
        writer.add_scalar("losses/clipfrac", float(np.mean(clip_fractions)), global_step)
        writer.add_scalar("losses/explained_variance", explained_variance, global_step)
        steps_per_second = global_step / (time.time() - start)
        writer.add_scalar("charts/SPS", steps_per_second, global_step)

        if update % config.checkpoint_every == 0 or update == updates:
            torch.save(agent.state_dict(), checkpoints / f"agent_{global_step:08d}.pt")
            print(f"Step {global_step}/{updates * rollout_steps}: {env.episodes} Episodes, "
                  f"{steps_per_second:.0f} Steps/s")

    writer.close()
    return agent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-name", default=f"01-ppo-{time.strftime('%Y%m%d-%H%M%S')}")
    args = parser.parse_args()

    run = RUNS / args.run_name
    if run.exists():
        parser.error(f"{run} already exists; pick another --run-name")
    train(load_config(args.config), run)


if __name__ == "__main__":
    main()
