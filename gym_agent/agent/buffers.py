import random
from abc import ABC, abstractmethod
from typing import Any, Literal, Optional
from dataclasses import dataclass

import gymnasium.spaces as spaces
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor
# from tensordict import TensorDict

from gym_agent import utils

# import psutil

# MAX_MEM_AVAILABLE = psutil.virtual_memory().available


@dataclass(frozen=True)
class BaseBufferSamples:
    observations: Tensor | dict[str, Tensor]
    actions: Tensor | dict[str, Tensor]
    rewards: Tensor
    agent_rewards: Optional[Tensor]

@dataclass(frozen=True)
class ReplayBufferSamples(BaseBufferSamples):
    next_observations: Tensor | dict[str, Tensor]
    terminals: Tensor

@dataclass(frozen=True)
class RolloutBufferSamples(BaseBufferSamples):
    log_prob: Tensor
    values: Tensor
    advantages: Tensor
    returns: Tensor


    agent_log_prob: Optional[Tensor]
    agent_values: Optional[Tensor]
    agent_advantages: Optional[Tensor]
    agent_returns: Optional[Tensor]


class BaseBuffer(ABC):
    observations: NDArray | dict[str, NDArray]
    actions: NDArray | dict[str, NDArray]
    rewards: NDArray

    agent_rewards: Optional[NDArray]

    def __init__(
        self,
        /,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: torch.device | str,
        n_envs: int,
        seed: Optional[int],
        n_agents: Optional[int],
        marl_agent_prefix_key: Optional[str],
    ):
        if n_envs <= 0:
            raise ValueError("The number of environments must be greater than 0.")

        self.buffer_size = buffer_size
        self.obs_shape = utils.get_shape(observation_space)
        self.action_shape = utils.get_shape(action_space)
        self.n_envs = n_envs
        self.device = utils.get_device(device)

        self.seed = random.seed(seed)

        self.marl = n_agents is not None

        self.n_agents = n_agents
        if marl_agent_prefix_key is None:
            marl_agent_prefix_key = "agent_"
        self.marl_agent_prefix_key = marl_agent_prefix_key

        self.reset()


    def register_base_memory(self) -> None:
        self.observations = self.mem_register(self.obs_shape)
        self.actions = self.mem_register(self.action_shape)
        self.rewards = self.mem_register()

        if self.marl:
            self.agent_rewards = self.mem_register((self.n_agents,))
        else:
            self.agent_rewards = None

    def get_mem(self, mem_name: str) -> dict | tuple[int, ...]:
        return self.__getattribute__(mem_name)

    def add2mem(
        self,
        mem_name: str,
        idx: int,
        mem_data: dict | tuple[int, ...],
        mem_shape: Optional[dict | tuple[int, ...]] = None,
    ):
        if isinstance(mem_shape, dict):
            for key in mem_data.keys():
                self.__getattribute__(mem_name)[key][idx] = mem_data[key]
        else:
            self.__getattribute__(mem_name)[idx] = mem_data

    def mem_register(self, shape: Optional[dict[str, tuple[int, ...]] | tuple[int, ...]] = None, dtype: np.dtype = np.float32) -> NDArray | dict[str, NDArray]:
        if shape is None:
            return np.zeros([self.buffer_size, self.n_envs], dtype=dtype)
        elif isinstance(shape, dict):
            return {
                key: np.zeros(
                    [self.buffer_size, self.n_envs, *item_shape], dtype=dtype
                )
                for key, item_shape in shape.items()
            }
        else:
            return np.zeros([self.buffer_size, self.n_envs, *shape], dtype=dtype)

    def to(self, device: torch.device | str) -> None:
        self.device = utils.get_device(device)
    
    def to_torch(self, x: NDArray | dict[str, NDArray]) -> Tensor | dict[str, Tensor]:      
        return utils.to_torch(x, device=self.device) if x is not None else None

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

        self.register_base_memory()


class ReplayBuffer(BaseBuffer):
    terminals: NDArray

    def __init__(
        self,
        /,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device="auto",
        n_envs: int = 1,
        seed: Optional[int] = None,
        n_agents: Optional[int] = None,
        marl_agent_prefix_key: Optional[str] = None,
    ):
        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            n_envs=n_envs,
            seed=seed,
            n_agents=n_agents,
            marl_agent_prefix_key=marl_agent_prefix_key,
        )

        self.terminals = self.mem_register(dtype=np.bool_)

    def add(
        self,
        observation: NDArray | dict[str, NDArray],
        action: NDArray | dict[str, NDArray],
        reward: NDArray,
        next_observation: NDArray | dict[str, NDArray],
        terminal: NDArray,
        info: dict[str, Any],
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

        self.rewards[idx] = reward
        if self.marl:
            self.agent_rewards[idx] = info[self.marl_agent_prefix_key + "rewards"]

        self.terminals[idx] = terminal

        self.mem_cntr += 1

        if self.mem_cntr == self.buffer_size:
            self.full = True
            self.mem_cntr = 0

    def sample(self, batch_size: int) -> ReplayBufferSamples:
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

        rewards = self.rewards[batch, env_ind]

        if self.marl:
            agent_rewards = self.agent_rewards[batch, env_ind]
        else:
            agent_rewards = None

        terminals = self.terminals[batch, env_ind]

        # return ReplayBufferSamples.from_numpy(
        #     observations=observations,
        #     actions=actions,
        #     rewards=rewards,
        #     agent_rewards=agent_rewards,
        #     next_observations=next_observations,
        #     terminals=terminals,
        #     device=self.device,
        # )

        return ReplayBufferSamples(
            observations=self.to_torch(observations),
            actions=self.to_torch(actions),
            rewards=self.to_torch(rewards),
            agent_rewards=self.to_torch(agent_rewards),
            next_observations=self.to_torch(next_observations),
            terminals=self.to_torch(terminals),
        )


class RolloutBuffer(BaseBuffer):
    # group 1
    log_probs: NDArray
    values: NDArray
    advantages: NDArray
    returns: NDArray

    # group 2
    agent_log_probs: Optional[NDArray]
    agent_values: Optional[NDArray]
    agent_advantages: Optional[NDArray]
    agent_returns: Optional[NDArray]

    # at least one of those 2 group must exist

    def __init__(
        self,
        /,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        gamma=0.99,
        gae_lambda=0.95,
        device="auto",
        n_envs: int = 1,
        seed: Optional[int] = None,
        n_agents: Optional[int] = None,
        marl_agent_prefix_key: Optional[str] = None,
        values_type: Literal["global", "individual", "hybrid"] = "hybrid",
    ):
        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            n_envs=n_envs,
            seed=seed,
            n_agents=n_agents,
            marl_agent_prefix_key=marl_agent_prefix_key,
        )

        self.values_type = values_type
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.reset()

    def add(
        self,
        observation: NDArray | dict[str, NDArray],
        action: NDArray | dict[str, NDArray],
        reward: NDArray,
        value: NDArray,
        log_prob: NDArray,
        terminal: NDArray[np.bool_],
        info: dict[str, Any],
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

        self.rewards[idx, ~self.dies] = reward[~self.dies]
        self.values[idx, ~self.dies] = value[~self.dies]
        self.log_probs[idx, ~self.dies] = log_prob[~self.dies]

        if self.values_type == "hybrid":
            self.agent_rewards[idx, ~self.dies] = info[self.marl_agent_prefix_key + "rewards"][~self.dies]
            self.agent_values[idx, ~self.dies] = info[self.marl_agent_prefix_key + "values"][~self.dies]
            self.agent_log_probs[idx, ~self.dies] = info[self.marl_agent_prefix_key + "log_probs"][~self.dies]

        self.end_mem_pos += ~self.dies

        self.dies = self.dies | terminal

        self.mem_cntr += 1
        if self.mem_cntr == self.buffer_size:
            raise ValueError("Rollout buffer is full. Please reset the buffer.")

    def reset(self) -> None:
        super().reset()
        self.processed = False
        
        self.agent_log_probs = None
        self.agent_values = None
        self.agent_advantages = None
        self.agent_returns = None

        if self.values_type == "global" or self.values_type == "hybrid":
            self.log_probs = self.mem_register()
            self.values = self.mem_register()
            self.advantages = self.mem_register()
            self.returns = self.mem_register()

        if self.values_type == "individual":
            self.log_probs = self.mem_register((self.n_agents,))
            self.values = self.mem_register((self.n_agents,))
            self.advantages = self.mem_register((self.n_agents,))
            self.returns = self.mem_register((self.n_agents,))
        
        if self.values_type == "hybrid":
            self.agent_log_probs = self.mem_register((self.n_agents,))
            self.agent_values = self.mem_register((self.n_agents,))
            self.agent_advantages = self.mem_register((self.n_agents,))
            self.agent_returns = self.mem_register((self.n_agents,))

        self.dies = np.zeros((self.n_envs,), dtype=np.bool_)
        self.end_mem_pos = np.zeros((self.n_envs,), dtype=np.uint32)

    def normal_calc_advantages_and_returns(
        self, env_idx: int, last_values: NDArray[np.float32], last_terminals: NDArray[np.bool_]
    ):
        T = self.end_mem_pos[env_idx]
        advantages = np.zeros(T, dtype=np.float32)

        if self.gae_lambda == 1:
            # No GAE, returns are just discounted rewards
            returns = np.zeros(T, dtype=np.float32)
            next_return = 0
            for t in reversed(range(T)):
                next_return = self.rewards[t][env_idx] + self.gamma * next_return
                returns[t] = next_return

            advantages = returns - self.values[:T, env_idx]

            return advantages, returns

        last_gae_lambda = 0

        for t in reversed(range(T)):
            if t == T - 1:
                next_non_terminal = 1.0 - last_terminals
                next_values = last_values
            else:
                next_non_terminal = 1.0
                next_values = self.values[t + 1][env_idx]

            delta = (
                self.rewards[t][env_idx]
                + next_non_terminal * self.gamma * next_values
                - self.values[t][env_idx]
            )
            last_gae_lambda = (
                delta
                + next_non_terminal * self.gamma * self.gae_lambda * last_gae_lambda
            )

            advantages[t] = last_gae_lambda

        returns = advantages + self.values[:T, env_idx]

        return advantages, returns

    
    def agent_calc_advantages_and_returns(
        self, env_idx: int, agent_idx: int, last_values: NDArray[np.float32], last_terminals: NDArray[np.bool_]
    ):
        T = self.end_mem_pos[env_idx]
        advantages = np.zeros(T, dtype=np.float32)

        if self.gae_lambda == 1:
            # No GAE, returns are just discounted rewards
            returns = np.zeros(T, dtype=np.float32)
            next_return = 0
            for t in reversed(range(T)):
                next_return = self.agent_rewards[t, env_idx, agent_idx] + self.gamma * next_return
                returns[t] = next_return

            advantages = returns - self.agent_values[:T, env_idx, agent_idx]

            return advantages, returns

        last_gae_lambda = 0

        for t in reversed(range(T)):
            if t == T - 1:
                next_non_terminal = 1.0 - last_terminals
                next_values = last_values
            else:
                next_non_terminal = 1.0
                next_values = self.agent_values[t + 1, env_idx, agent_idx]

            delta = (
                self.agent_rewards[t, env_idx, agent_idx]
                + next_non_terminal * self.gamma * next_values
                - self.agent_values[t, env_idx, agent_idx]
            )
            last_gae_lambda = (
                delta
                + next_non_terminal * self.gamma * self.gae_lambda * last_gae_lambda
            )

            advantages[t] = last_gae_lambda

        returns = advantages + self.agent_values[:T, env_idx, agent_idx]

        return advantages, returns

    def calc_advantages_and_returns(
        self, last_values: NDArray[np.float32], last_terminals: NDArray[np.bool_], info: dict[str, Any]
    ) -> None:
        if self.processed:
            raise ValueError(
                "Cannot calculate advantages and returns after processing the buffer."
            )

        for env_idx in range(self.n_envs):
            if self.values_type == "global" or self.values_type == "hybrid":
                advantages, returns = self.normal_calc_advantages_and_returns(env_idx, last_values[env_idx], last_terminals[env_idx])
                self.advantages[: self.end_mem_pos[env_idx], env_idx] = advantages
                self.returns[: self.end_mem_pos[env_idx], env_idx] = returns

            if self.values_type == "individual":
                for agent_idx in range(self.n_agents):
                    advantages, returns = self.agent_calc_advantages_and_returns(env_idx, agent_idx, last_values[env_idx][agent_idx], last_terminals[env_idx][agent_idx])
                    self.advantages[: self.end_mem_pos[env_idx], env_idx, agent_idx] = advantages
                    self.returns[: self.end_mem_pos[env_idx], env_idx, agent_idx] = returns

            if self.values_type == "hybrid":
                for agent_idx in range(self.n_agents):
                    advantages, returns = self.agent_calc_advantages_and_returns(env_idx, agent_idx, info[self.marl_agent_prefix_key + "values"][env_idx][agent_idx], last_terminals[env_idx][agent_idx])
                    self.agent_advantages[: self.end_mem_pos[env_idx], env_idx, agent_idx] = advantages
                    self.agent_returns[: self.end_mem_pos[env_idx], env_idx, agent_idx] = returns


    def process_mem(self) -> NDArray:
        if self.processed:
            raise ValueError("Cannot process the buffer again.")

        def array_swap_and_flatten(arr: NDArray) -> NDArray:
            ret_arr = np.zeros((sum(self.end_mem_pos), *arr.shape[2:]), dtype=arr.dtype)

            ret_arr[: self.end_mem_pos[0]] = arr[: self.end_mem_pos[0], 0]

            for env in range(1, self.n_envs):
                ret_arr[self.end_mem_pos[env - 1] : self.end_mem_pos[env]] = arr[
                    : self.end_mem_pos[env], env
                ]
            return ret_arr

        def swap_and_flatten(
            arr: NDArray | dict[str, NDArray],
        ) -> NDArray | dict[str, NDArray]:
            """
            Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
            to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
            to [n_envs * n_steps, ...] (which maintain the order)
            """
            return (
                {key: array_swap_and_flatten(arr[key]) for key in arr} if isinstance(arr, dict)
                else array_swap_and_flatten(arr)
            )

        self.observations = swap_and_flatten(self.observations)
        self.actions = swap_and_flatten(self.actions)
        self.rewards = swap_and_flatten(self.rewards)
        self.values = swap_and_flatten(self.values)
        self.log_probs = swap_and_flatten(self.log_probs)
        self.advantages = swap_and_flatten(self.advantages)
        self.returns = swap_and_flatten(self.returns)


        # handling marl 
        self.agent_returns = swap_and_flatten(self.agent_rewards) if self.marl else None
        if self.values_type == "hybrid":
            self.agent_log_probs = swap_and_flatten(self.agent_log_probs)
            self.agent_values = swap_and_flatten(self.agent_values)
            self.agent_advantages = swap_and_flatten(self.agent_advantages)
            self.agent_returns = swap_and_flatten(self.agent_returns)

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

        if self.marl:
            agent_rewards = self.agent_rewards[batch]
        else:
            agent_rewards = None

        if self.values_type == "hybrid":
            agent_values = self.agent_values[batch]
            agent_log_probs = self.agent_log_probs[batch]
            agent_advantages = self.agent_advantages[batch]
            agent_returns = self.agent_returns[batch]
        else:
            agent_values = None
            agent_log_probs = None
            agent_advantages = None
            agent_returns = None

        return RolloutBufferSamples(
            observations=self.to_torch(observations),
            actions=self.to_torch(actions),
            rewards=self.to_torch(self.rewards[batch]),
            agent_rewards=self.to_torch(agent_rewards),

            values=self.to_torch(self.values[batch]),
            log_prob=self.to_torch(self.log_probs[batch]),
            advantages=self.to_torch(self.advantages[batch]),
            returns=self.to_torch(self.returns[batch]),

            agent_values=self.to_torch(agent_values),
            agent_log_prob=self.to_torch(agent_log_probs),
            agent_advantages=self.to_torch(agent_advantages),
            agent_returns=self.to_torch(agent_returns),
        )
