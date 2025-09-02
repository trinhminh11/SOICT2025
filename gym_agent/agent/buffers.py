import random
from abc import ABC, abstractmethod
from typing import NamedTuple, Optional

import gymnasium.spaces as spaces
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
# from tensordict import TensorDict

from .. import utils

# import psutil

# MAX_MEM_AVAILABLE = psutil.virtual_memory().available


class ReplayBufferSamples(NamedTuple):
    observations: Tensor | dict[str, Tensor]
    actions: Tensor | dict[str, Tensor]
    rewards: Tensor | dict[str, Tensor]
    next_observations: Tensor
    terminals: Tensor


class RolloutBufferSamples(NamedTuple):
    observations: Tensor | dict[str, Tensor]
    actions: Tensor | dict[str, Tensor]
    rewards: Tensor | dict[str, Tensor]
    values: Tensor
    log_prob: Tensor
    advantages: Tensor
    returns: Tensor


class BaseBuffer(ABC):
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        reward_space: Optional[spaces.Space] = None,
        device: torch.device | str = "auto",
        n_envs: int = 1,
        seed: Optional[int] = None,
    ):
        if n_envs <= 0:
            raise ValueError("The number of environments must be greater than 0.")

        self.buffer_size = buffer_size
        self.obs_shape = utils.get_shape(observation_space)
        self.action_shape = utils.get_shape(action_space)
        self.reward_shape = utils.get_shape(reward_space) if reward_space else None
        self.n_envs = n_envs
        self.device = utils.get_device(device)

        self.mem_cntr = 0
        self.full = False

        self.seed = random.seed(seed)

        self.register_base_memory()

    def register_base_memory(self) -> None:
        self.observations = self.mem_register(self.obs_shape)
        self.actions = self.mem_register(self.action_shape)
        self.rewards = self.mem_register(self.reward_shape)

        # self.rewards = np.zeros([buffer_size, n_envs], dtype=np.float32)

    def get_mem(self, mem_name: str) -> dict | tuple[int, ...]:
        return self.__getattribute__(mem_name)

    def add2mem(self, mem_name: str, idx: int, mem_data: dict | tuple[int, ...], mem_shape: Optional[dict | tuple[int, ...]] = None):
        if isinstance(mem_shape, dict):
            for key in mem_data.keys():
                self.__getattribute__(mem_name)[key][idx] = mem_data[key]
        else:
            self.__getattribute__(mem_name)[idx] = mem_data



    def mem_register(self, shape: Optional[dict | tuple[int, ...]] = None):
        if shape is None:
            return np.zeros([self.buffer_size, self.n_envs], dtype=np.float32)
        elif isinstance(shape, dict):
            return  {
                key: np.zeros([self.buffer_size, self.n_envs, *item_shape], dtype=np.float32)
                for key, item_shape in shape.items()
            }
        else:
            return np.zeros(
                [self.buffer_size, self.n_envs, *shape], dtype=np.float32
            )

    def to(self, device: torch.device | str) -> None:
        self.device = utils.get_device(device)

    def __len__(self):
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.mem_cntr

    @abstractmethod
    def add(self, *args, **kwargs) -> None:
        """
        Add elements to the buffer.
        """
        raise NotImplementedError()

    @abstractmethod
    def sample(self):
        """
        Sample elements from the buffer.
        """
        raise NotImplementedError()

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.mem_cntr = 0
        self.full = False


class ReplayBuffer(BaseBuffer):
    observations: NDArray | dict[str, NDArray]
    actions: NDArray | dict[str, NDArray]
    rewards: NDArray | dict[str, NDArray]
    terminals: NDArray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device="auto",
        n_envs: int = 1,
        seed: int = None,
    ):
        super().__init__(
            buffer_size, observation_space, action_space, device, n_envs, seed
        )

        self.terminals = self.mem_register()
    


    def add(
        self,
        observation: NDArray | dict[str, NDArray],
        action: NDArray | dict[str, NDArray],
        reward: NDArray,
        next_observation: NDArray | dict[str, NDArray],
        terminal: NDArray,
    ) -> None:
        """Add a new experience to memory."""
        idx = self.mem_cntr

        if isinstance(self.obs_shape, dict):
            for key in observation.keys():
                self.observations[key][idx] = observation[key]
                self.observations[key][(idx + 1) % self.buffer_size] = next_observation[
                    key
                ]
        else:
            self.observations[idx] = observation
            self.observations[(idx + 1) % self.buffer_size] = next_observation

        if isinstance(self.action_shape, dict):
            for key in action.keys():
                self.actions[key][idx] = action[key]
        else:
            self.actions[idx] = action

        if isinstance(self.reward_shape, dict):
            for key in reward.keys():
                self.rewards[key][idx] = reward[key]
        else:
            self.rewards[idx] = reward

        self.terminals[idx] = terminal

        self.mem_cntr += 1

        if self.mem_cntr == self.buffer_size:
            self.full = True
            self.mem_cntr = 0

    def sample(self, batch_size: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Randomly sample a batch of experiences from memory."""

        if self.full:
            batch = (
                np.random.randint(1, self.buffer_size, size=batch_size) + self.mem_cntr
            ) % self.buffer_size
        else:
            batch = np.random.randint(0, self.mem_cntr, size=batch_size)

        return self._get_sample(batch)

    def _get_sample(
        self, batch: NDArray[np.uint32]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        env_ind = np.random.randint(0, self.n_envs, size=len(batch))

        if isinstance(self.obs_shape, dict):
            observations = {
                key: obs[batch, env_ind] for key, obs in self.observations.items()
            }
            next_observations = {
                key: obs[(batch + 1) % self.buffer_size, env_ind]
                for key, obs in self.observations.items()
            }
        else:
            observations = self.observations[batch, env_ind]
            next_observations = self.observations[
                (batch + 1) % self.buffer_size, env_ind
            ]

        if isinstance(self.action_shape, dict):
            actions = {key: act[batch, env_ind] for key, act in self.actions.items()}
        else:
            actions = self.actions[batch, env_ind]

        if isinstance(self.reward_shape, dict):
            rewards = {key: rew[batch, env_ind] for key, rew in self.rewards.items()}
        else:
            rewards = self.rewards[batch, env_ind]

        terminals = self.terminals[batch, env_ind]

        return ReplayBufferSamples(
            observations=utils.to_torch(observations, device=self.device),
            actions=utils.to_torch(actions, device=self.device),
            rewards=utils.to_torch(rewards, device=self.device),
            next_observations=utils.to_torch(next_observations, device=self.device),
            terminals=utils.to_torch(terminals, device=self.device),
        )


class RolloutBuffer(BaseBuffer):
    observations: NDArray | dict[str, NDArray]
    actions: NDArray | dict[str, NDArray]
    rewards: NDArray | dict[str, NDArray]

    advantages: NDArray
    returns: NDArray
    log_probs: NDArray
    values: NDArray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        gamma=0.99,
        gae_lambda=0.95,
        device="auto",
        n_envs: int = 1,
        seed: int = None,
    ):
        super().__init__(
            buffer_size, observation_space, action_space, device, n_envs, seed
        )

        self.log_probs = self.mem_register()
        self.values = self.mem_register()
        

        self.advantages = self.mem_register()
        self.returns = self.mem_register()

        self.dies = np.zeros((n_envs, ), dtype=np.bool_)
        self.end_mem_pos = np.zeros((n_envs, ), dtype=np.uint32)

        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.processed = False

    def add(
        self,
        observation: NDArray | dict[str, NDArray],
        action: NDArray | dict[str, NDArray],
        reward: NDArray | dict[str, NDArray],
        value: NDArray,
        log_prob: NDArray,
        terminal: NDArray[np.bool_],
    ) -> None:
        if self.processed:
            raise ValueError(
                "Cannot add new experiences to the buffer after processing the buffer."
            )

        idx = self.mem_cntr

        if isinstance(self.obs_shape, dict):
            for key in observation.keys():
                self.observations[key][idx, ~self.dies] = observation[key][~self.dies]
        else:
            self.observations[idx, ~self.dies] = observation[~self.dies]

        if isinstance(self.action_shape, dict):
            for key in action.keys():
                self.actions[key][idx, ~self.dies] = action[key][~self.dies]
        else:
            self.actions[idx, ~self.dies] = action[~self.dies]

        if isinstance(self.reward_shape, dict):
            for key in reward.keys():
                self.rewards[key][idx, ~self.dies] = reward[key][~self.dies]
        else:
            self.rewards[idx, ~self.dies] = reward[~self.dies]

        self.values[idx, ~self.dies] = value[~self.dies]
        self.log_probs[idx, ~self.dies] = log_prob[~self.dies]

        self.mem_cntr += 1
        self.end_mem_pos += ~self.dies

        self.dies = self.dies | terminal

        if self.mem_cntr == self.buffer_size:
            raise ValueError("Rollout buffer is full. Please reset the buffer.")

    def reset(self) -> None:
        super().reset()
        self.dies = np.zeros((self.n_envs,), dtype=np.bool_)
        self.end_mem_pos = np.zeros((self.n_envs,), dtype=np.uint32)
        self.processed = False

        if isinstance(self.obs_shape, dict):
            self.observations = {
                key: np.zeros([self.buffer_size, self.n_envs, *shape], dtype=np.float32)
                for key, shape in self.obs_shape.items()
            }
        else:
            self.observations = np.zeros(
                [self.buffer_size, self.n_envs, *self.obs_shape], dtype=np.float32
            )

        if isinstance(self.action_shape, dict):
            self.actions = {
                key: np.zeros([self.buffer_size, self.n_envs, *shape], dtype=np.float32)
                for key, shape in self.action_shape.items()
            }
        else:
            self.actions = np.zeros(
                [self.buffer_size, self.n_envs, *self.action_shape], dtype=np.float32
            )
        
        if isinstance(self.reward_shape, dict):
            self.rewards = {
                key: np.zeros([self.buffer_size, self.n_envs, *shape], dtype=np.float32)
                for key, shape in self.reward_shape.items()
            }
        else:
            self.rewards = np.zeros([self.buffer_size, self.n_envs, *self.reward_shape], dtype=np.float32)

        self.values = np.zeros([self.buffer_size, self.n_envs], dtype=np.float32)
        self.log_probs = np.zeros([self.buffer_size, self.n_envs], dtype=np.float32)
        self.advantages = np.zeros([self.buffer_size, self.n_envs], dtype=np.float32)
        self.returns = np.zeros([self.buffer_size, self.n_envs], dtype=np.float32)

    def calc_advantages_and_returns(
        self, last_values: NDArray[np.float32], last_terminals: NDArray[np.bool_]
    ) -> None:
        if self.processed:
            raise ValueError(
                "Cannot calculate advantages and returns after processing the buffer."
            )

        def _calc_advantages_and_returns(env):
            if self.gae_lambda == 1:
                # No GAE, returns are just discounted rewards
                T = self.end_mem_pos[env]
                advantages = np.zeros(T, dtype=np.float32)
                returns = np.zeros(T, dtype=np.float32)
                next_return = 0
                for t in reversed(range(T)):
                    next_return = self.rewards[t][env] + self.gamma * next_return
                    returns[t] = next_return

                advantages = returns - self.values[:T, env]

                return advantages, returns

            T = self.end_mem_pos[env]
            advantages = np.zeros(T, dtype=np.float32)
            last_gae_lambda = 0

            for t in reversed(range(T)):
                if t == T - 1:
                    next_non_terminal = 1.0 - last_terminals[env]
                    next_values = last_values[env]
                else:
                    next_non_terminal = 1.0
                    next_values = self.values[t + 1][env]

                delta = (
                    self.rewards[t][env]
                    + next_non_terminal * self.gamma * next_values
                    - self.values[t][env]
                )
                last_gae_lambda = (
                    delta
                    + next_non_terminal * self.gamma * self.gae_lambda * last_gae_lambda
                )

                advantages[t] = last_gae_lambda

            returns = advantages + self.values[:T, env]

            return advantages, returns

        for env in range(self.n_envs):
            advantages, returns = _calc_advantages_and_returns(env)
            self.advantages[: self.end_mem_pos[env], env] = advantages
            self.returns[: self.end_mem_pos[env], env] = returns

    def process_mem(self) -> NDArray:
        if self.processed:
            raise ValueError("Cannot process the buffer again.")

        def swap_and_flatten(arr: NDArray) -> NDArray:
            """
            Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
            to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
            to [n_envs * n_steps, ...] (which maintain the order)
            """
            if isinstance(arr, dict):
                ret_arr = {}
                for key in arr:
                    ret_arr[key] = np.zeros(
                        (sum(self.end_mem_pos), *arr[key].shape[2:]),
                        dtype=arr[key].dtype,
                    )

                    ret_arr[key][: self.end_mem_pos[0]] = arr[key][
                        : self.end_mem_pos[0], 0
                    ]
                    for i in range(1, self.n_envs):
                        ret_arr[self.end_mem_pos[i - 1] : self.end_mem_pos[i]] = arr[
                            key
                        ][: self.end_mem_pos[i], i]

            else:
                ret_arr = np.zeros(
                    (sum(self.end_mem_pos), *arr.shape[2:]), dtype=arr.dtype
                )

                ret_arr[: self.end_mem_pos[0]] = arr[: self.end_mem_pos[0], 0]
                for i in range(1, self.n_envs):
                    ret_arr[self.end_mem_pos[i - 1] : self.end_mem_pos[i]] = arr[
                        : self.end_mem_pos[i], i
                    ]

            return ret_arr

        self.observations = swap_and_flatten(self.observations)
        self.actions = swap_and_flatten(self.actions)
        self.rewards = swap_and_flatten(self.rewards)
        self.values = swap_and_flatten(self.values)
        self.log_probs = swap_and_flatten(self.log_probs)
        self.advantages = swap_and_flatten(self.advantages)
        self.returns = swap_and_flatten(self.returns)

        self.processed = True

    def get(self, batch_size: int = None):
        if not self.processed:
            self.process_mem()

        mem_size = sum(self.end_mem_pos)

        if batch_size is None or batch_size is False:
            yield self._get_sample(np.arange(mem_size))

        else:
            batch_size = mem_size

            indices = np.random.permutation(mem_size)

            for start_idx in range(0, mem_size, batch_size):
                yield self._get_sample(indices[start_idx : start_idx + batch_size])

    def sample(self, batch_size: int) -> RolloutBufferSamples:
        if not self.processed:
            self.process_mem()

        batch = np.random.randint(0, sum(self.end_mem_pos), size=batch_size)

        return self._get_sample(batch)

    def _get_sample(self, batch: NDArray[np.uint32]) -> RolloutBufferSamples:
        if isinstance(self.obs_shape, dict):
            observations = {
                key: self.observations[key][batch] for key in self.obs_shape.keys()
            }
        else:
            observations = self.observations[batch]

        if isinstance(self.action_shape, dict):
            actions = {
                key: self.actions[key][batch] for key in self.action_shape.keys()
            }
        else:
            actions = self.actions[batch]

        return RolloutBufferSamples(
            observations=utils.to_torch(observations, device=self.device),
            actions=utils.to_torch(actions, device=self.device),
            rewards=utils.to_torch(self.rewards[batch], device=self.device),
            values=utils.to_torch(self.values[batch], device=self.device),
            log_prob=utils.to_torch(self.log_probs[batch], device=self.device),
            advantages=utils.to_torch(self.advantages[batch], device=self.device),
            returns=utils.to_torch(self.returns[batch], device=self.device),
        )
