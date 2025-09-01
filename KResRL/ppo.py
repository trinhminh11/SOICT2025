import os
import sys
from dataclasses import asdict, dataclass

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from KResRL.policy.ac_policy import PolicyOptions, NodeLevelActorCriticPolicy
from KResRL.environment.env import KRes


class RewardCallback(BaseCallback):
    def __init__(self, verbose=1):
        super().__init__(verbose)

        self.rewards = np.zeros((100, ), dtype=np.float32)

    def _on_step(self) -> bool:
        for reward in self.locals.get("rewards", []):
            self.rewards = np.roll(self.rewards, -1)
            self.rewards[-1] = reward

            print(
                f"Step: {self.num_timesteps:<10} Reward: {reward:>8.2f} Max Reward: {self.rewards.max():>8.2f} Mean Reward: {self.rewards.mean():>8.2f} Min Reward: {self.rewards.min():>8.2f}",
            )
        return True


@dataclass
class EnvOptions:
    n_envs: int
    n_drones: int
    k: int
    size: int
    alpha: float = 0.1


@dataclass
class RLOptions:
    verbose: int = 1
    learning_rate: float = 3e-4

    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95


@dataclass
class TrainOptions:
    total_timesteps: int = 1_000_000
    callback: BaseCallback = RewardCallback(1)
    log_interval: int = 100
    tb_log_name: str = "PPO"
    reset_num_timesteps: bool = True
    progress_bar: bool = False


def train(
    env_options: EnvOptions,
    policy_options: PolicyOptions,
    rl_options: RLOptions,
    train_options,
):
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    def make_env():
        return KRes(
            n_drones=env_options.n_drones,
            k=env_options.k,
            size=env_options.size,
            alpha=env_options.alpha,
        )

    env = make_vec_env(make_env, n_envs=env_options.n_envs)

    policy_kwargs = asdict(policy_options)
    policy_kwargs.pop("policy_cls", None)

    # Create PPO model with custom GCN policy
    model = PPO(
        policy=NodeLevelActorCriticPolicy,
        env=env,
        policy_kwargs=policy_kwargs,
        **asdict(rl_options),
    )

    print("Node-level policy created successfully for Stable Baselines3!")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Test the policy
    obs = env.reset()
    print(f"Observation shape: {obs.shape}")

    action, _ = model.predict(obs, deterministic=True)
    print(f"Predicted action: {action}")

    # Test a short training run
    print("\nTesting short training run...")
    model.learn(**asdict(train_options))
    print("Training completed successfully!")

    return model
