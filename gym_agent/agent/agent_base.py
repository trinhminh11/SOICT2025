# Standard library imports
import random
import os
from abc import ABC, abstractmethod
from typing import Type, TypeVar, Callable, Any, Generator
import typing

# Third-party imports
import gymnasium as gym
import numpy as np
from numpy._typing import _ShapeLike
from numpy.typing import NDArray
from tqdm import tqdm
import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim

from .buffers import ReplayBuffer, RolloutBuffer, BaseBuffer, RolloutBufferSamples

from .agent_callbacks import Callbacks

from .. import utils
from .. import core

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class AgentBase(ABC):
    memory: BaseBuffer
    envs: gym.vector.AsyncVectorEnv | gym.vector.SyncVectorEnv

    def __init__(
            self,
            policy: nn.Module,
            env: core.EnvWithTransform | gym.Env | str,
            name = None,
            device: str = 'auto',
            seed = None,
        ):
        if name == None:
            name = self.__class__.__name__
        
        if not isinstance(name, str):
            raise ValueError("name must be a string")

        self.name = name
        
        utils.check_for_nested_spaces(env.observation_space)
        utils.check_for_nested_spaces(env.action_space)
        
        if not isinstance(policy, nn.Module):
            raise ValueError("policy must be an instance of torch.nn.Module")
        
        if isinstance(env, gym.vector.VectorEnv):
            self.envs = env
        elif isinstance(env, str):
            self.envs = core.make_vec(env, 1)
        elif isinstance(env, core.EnvWithTransform):
            self.envs = gym.vector.AsyncVectorEnv([lambda: env])
        elif isinstance(env, gym.Env):
            self.envs = gym.vector.AsyncVectorEnv([lambda: core.EnvWithTransform(env)])
        
        self.dummy_env = self.envs.env_fns[0]()
        
        self.n_envs = self.envs.num_envs

        self.policy = policy

        self.device = utils.get_device(device)
        self.seed = seed

        self.memory = None

        self.episode = 0
        self.total_timesteps = 0
        self.scores = []
        self.mean_scores = []

        self._optimizers = []

        self.save_kwargs = []

        self.to(self.device)
    
    @property
    def info(self):
        return {
            'scores': self.scores,
            'mean_scores': self.mean_scores,
            'episode': self.episode,
            'total_timesteps': self.total_timesteps
        }
    
    def plot_scores(self, filename = None):
        if filename is not None and not isinstance(filename, str):
            raise ValueError("filename must be a string")
        
        import matplotlib.pyplot as plt

        plt.plot(self.scores, label='scores')
        plt.plot(self.mean_scores, label='mean_scores')
        plt.title(self.name)

        plt.legend()
        
        if filename:
            plt.savefig(filename)
        else:
            plt.show()

    
    def apply(self, fn: Callable[[nn.Module], None]):
        self.policy.apply(fn)
    
    def to(self, device):
        self.device = device
        if self.memory:
            self.memory.to(device)
        self.policy.to(device)
        
        return self
    
    def add_optimizer(self, name: str, value: optim.Optimizer):
        if not isinstance(value, optim.Optimizer):
            raise ValueError("value must be an instance of torch.optim.Optimizer")
        if not isinstance(name, str):
            raise ValueError("name must be a string")
        
        name = 'optimizer_' + name

        if hasattr(self, name):
            raise ValueError(f"{name} already exists in the agent")
        
        self.__setattr__(name, value)
        
        self._optimizers.append(name)
    
    def add_save_kwargs(self, name: str):
        if not isinstance(name, str):
            raise ValueError("name must be a string")
        
        self.save_kwargs.append(name)

    def save_info(self):
        ret = {
            'episode': self.episode,
            'total_timesteps': self.total_timesteps,
            'scores': self.scores,
            'mean_scores': self.mean_scores,
            'policy': self.policy.state_dict(),
        }

        if len(self._optimizers) > 0:
            ret['optimizers'] = {}
            for name in self._optimizers:
                ret['optimizers'][name] = getattr(self, name).state_dict()

        for name in self.save_kwargs:
            ret[name] = getattr(self, name)

        return ret

    def save(self, dir, *post_names):
        name = self.name

        if not os.path.exists(dir):
            os.makedirs(dir)
        
        for post_name in post_names:
            name += "_" + str(post_name)

        torch.save(self.save_info(), os.path.join(dir, name + ".pth"))
    
    def load(self, dir, *post_names):
        name = self.name
        for post_name in post_names:
            name += "_" + str(post_name)

        checkpoint = torch.load(os.path.join(dir, name + ".pth"), self.device, weights_only=False)

        self.policy.load_state_dict(
            checkpoint['policy']
        )

        self.episode = checkpoint['episode']
        self.total_timesteps = checkpoint['total_timesteps']
        self.scores = checkpoint['scores']
        self.mean_scores = checkpoint['mean_scores']

        for name in self._optimizers:
            getattr(self, name).load_state_dict(checkpoint['optimizers'][name])
        
        for name in self.save_kwargs:
            setattr(self, name, checkpoint[name])


    def reset(self):
        r"""
        This method should call at the start of an episode (after :func:``env.reset``)
        """
        ...
    
    @abstractmethod
    def predict(self, observations: NDArray | dict[str, NDArray], deterministic: bool = True) -> ActType:
        """
        Perform an action based on the given observations.

        Parameters:
            observations (NDArray | dict[str, NDArray]): The input observations which can be either a numpy array or a dictionary
            * ``NDArray`` shape - `[batch, *obs_shape]`
            * ``dict`` shape - `[key: sub_space]` with `sub_space` shape: `[batch, *sub_obs_shape]`
            deterministic (bool, optional): If True, the action is chosen deterministically. Defaults to True.

        Returns:
            ActType: The action to be performed.

        Raises:
            NotImplementedError: This method should be overridden by subclasses.
        """
        raise NotImplementedError

    def train_on_episode(self, deterministic: bool, callbacks: Type[Callbacks] = None):
        raise NotImplementedError

    def fit(self, total_timesteps: int = None, n_games: int = None, deterministic = False, save_best = False, save_every = False, save_dir = "./", progress_bar: Type[tqdm] = tqdm, callbacks: Type[Callbacks] = None):
        if callbacks is None:
            callbacks = Callbacks(self)
        
        if total_timesteps is None and n_games is None:
            raise ValueError("Either total_timesteps or n_games must be provided")

        # Use tqdm for progress bar if provided
        if total_timesteps:
            loop = progress_bar(range(total_timesteps)) if progress_bar else range(total_timesteps)
            used_timesteps = True
        else:
            loop = progress_bar(range(n_games)) if progress_bar else range(n_games)
            used_timesteps = False
        
        callbacks.on_train_begin()

        best_score = float('-inf')

        start_episode = self.episode    
        start_timesteps = self.total_timesteps

        while True:
            score, timesteps = self.train_on_episode(deterministic, callbacks)

            self.scores.append(score)
            avg_score = np.mean(self.scores[-100:])
            self.mean_scores.append(avg_score)

            if save_best:
                if avg_score >= best_score:
                    best_score = avg_score
                    self.save(save_dir, "best")
            
            if save_every:
                if (self.episode+1) % save_every == 0:
                    self.save(save_dir, str(self.episode+1))

            self.total_timesteps += timesteps
            self.episode += 1

            if used_timesteps:
                if progress_bar:
                    loop.update(timesteps)
                    loop.set_postfix({"episode": self.episode, "time_step": self.total_timesteps, "avg_score": avg_score, "score": self.scores[-1]})

                if self.total_timesteps >= total_timesteps + start_timesteps:
                    break
            else:
                if progress_bar:
                    loop.update(1)
                    loop.set_postfix({"episode": self.episode, "time_step": self.total_timesteps, "avg_score": avg_score, "score": self.scores[-1]})

                if self.episode >= n_games + start_episode:
                    break

        callbacks.on_train_end()
    
    def play(self, env: core.EnvWithTransform, stop_if_truncated: bool = True, seed = None) -> float:
        self.eval = True
        import pygame
        
        pygame.init()
        score = 0
        obs = env.reset(seed=seed)[0]
        self.reset()

        done = False
        while not done:
            env.render()
            if isinstance(env.observation_space, gym.spaces.Dict):
                action = self.predict({key: np.expand_dims(obs[key], 0) for key in obs}, True)
            else:
                action = self.predict(np.expand_dims(obs.copy(), 0), True)

            next_obs, reward, terminated, truncated, info = env.step(action[0])

            done = terminated

            if stop_if_truncated:
                done = done or truncated

            obs = next_obs

            score += reward

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

        pygame.quit()

        self.eval = False

        return score
    
    def play_jupyter(self, env: core.EnvWithTransform, stop_if_truncated: bool = True, seed = None, FPS = 30) -> float:
        self.eval = True
        import pygame
        from IPython.display import display
        from PIL import Image

        pygame.init()
        clock = pygame.time.Clock()
        score = 0
        obs = env.reset(seed=seed)[0]
        self.reset()

        done = False
        while not done:
            clock.tick(FPS)
            env.render()
            if isinstance(env.observation_space, gym.spaces.Dict):
                action = self.predict({key: np.expand_dims(obs[key], 0) for key in obs}, True)
            else:
                action = self.predict(np.expand_dims(obs, 0), True)

            next_obs, reward, terminated, truncated, info = env.step(action[0])

            pixel = env.render()

            display(Image.fromarray(pixel), clear=True)

            done = terminated

            if stop_if_truncated:
                done = done or truncated

            obs = next_obs

            score += reward

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True

        pygame.quit()

        return score

class OffPolicyAgent(AgentBase):
    memory: ReplayBuffer
    def __init__(
            self, 
            policy: nn.Module, 
            env: core.EnvWithTransform | gym.Env | str, 
            gamma = 0.99,
            buffer_size = int(1e5),
            batch_size: int = 64,
            update_every: int = 1,
            name = None,
            device = 'auto', 
            seed = None
        ):
        super().__init__(policy, env, name, device, seed)

        self.gamma = gamma

        self.memory = ReplayBuffer(buffer_size = buffer_size, observation_space = self.dummy_env.observation_space, action_space = self.dummy_env.action_space, device = self.device, n_envs = self.n_envs, seed = self.seed)

        self.update_every = update_every

        self.batch_size = batch_size

        self.time_step = 0

    
    @abstractmethod
    def learn(self, observations: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, next_observations: torch.Tensor, terminals: torch.Tensor) -> None:
        """
        Abstract method to be implemented by subclasses for learning from a batch of experiences.

        Args:
            observations (torch.Tensor): The batch of current observations.
            actions (torch.Tensor): The batch of actions taken.
            rewards (torch.Tensor): The batch of rewards received.
            next_observations (torch.Tensor): The batch of next observations resulting from the actions.
            terminals (torch.Tensor): The batch of terminal flags indicating if the next observation is terminal.
        """
        ...
    
    def step(self, observation: NDArray | dict, action: NDArray | dict, reward: NDArray, next_observation: NDArray | dict, terminal: NDArray):
        """
        Perform a single step in the agent's environment.
        This method adds the current experience to the replay buffer, increments the time step,
        and updates the agent by sampling a batch from memory if it's time to update.
        Args:
            observation (NDArray): The current state observation.
            action (NDArray): The action taken by the agent.
            reward (NDArray): The reward received after taking the action.
            next_observation (NDArray): The next state observation after taking the action.
            terminal (NDArray): A flag indicating whether the episode has ended.
        Returns:
            None
        """
        
        # Add the current experience to the replay buffer
        self.memory.add(observation, action, reward, next_observation, terminal)

        # Increment the time step and check if it's time to update
        self.time_step = (self.time_step + 1) % self.update_every

        # If it's time to update, sample a batch from memory and learn from it
        if self.time_step == 0:
            if len(self.memory) >= self.batch_size:
                observations, actions, rewards, next_observations, terminals = self.memory.sample(self.batch_size)
                self.learn(observations, actions, rewards, next_observations, terminals)


    def train_on_episode(self, deterministic: bool, callbacks: Type[Callbacks] = None):
        if callbacks is None:
            callbacks = Callbacks(self)

        done = False
        score = 0
        timesteps = 0
        callbacks.on_episode_begin()
        obs = self.envs.reset()[0]
        self.reset()
        while not done:
            timesteps += 1
            action = self.predict(obs, deterministic)
            next_obs, reward, terminal, truncated, info = self.envs.step(action)

            done = (terminal | truncated).all()
            self.step(obs, action, reward, next_obs, terminal)
            score += reward.mean()
            obs = next_obs

        callbacks.on_episode_end()

        return score, timesteps


class OnPolicyAgent(AgentBase):
    memory: RolloutBuffer

    def __init__(
            self, 
            policy: nn.Module, 
            env: core.EnvWithTransform | gym.Env | str, 
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            buffer_size = int(1e5),
            batch_size: int = 64,
            name = None,
            device = 'auto', 
            seed = None
        ):
        super().__init__(policy, env, name, device, seed)

        self.memory = RolloutBuffer(buffer_size = buffer_size, observation_space = self.dummy_env.observation_space, action_space = self.dummy_env.action_space, device = self.device, n_envs = self.n_envs, gamma = gamma, gae_lambda = gae_lambda, seed = self.seed)

        self.batch_size = batch_size
    
    @abstractmethod
    def predict(self, state: NDArray | dict[str, NDArray], deterministic: bool = True) -> ActType:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, state: NDArray | dict[str, NDArray], deterministic: bool = True) -> tuple[NDArray, NDArray, NDArray]:
        """
        Evaluate the given state and return the action, log probability, and value.

        Parameters:
            state (NDArray | dict[str, NDArray]): The input state which can be either a numpy array or a dictionary.
            deterministic (bool, optional): If True, the action is chosen deterministically. Defaults to True.

        Returns:
            tuple[NDArray, NDArray, NDArray]: A tuple containing:
                - actions (NDArray): The action to be performed.
                - values (NDArray): The estimated value of the state.
                - log_probs (NDArray): The log probability of the action.
        """
        raise NotImplementedError

    @abstractmethod
    def learn(self, memory: RolloutBuffer) -> None:
        raise NotImplementedError
    
    def step(self, observation: NDArray | dict[str, NDArray], action: NDArray | dict[str, NDArray], reward: NDArray, value: NDArray, log_prob: NDArray, terminal: NDArray[np.bool_]):
        self.memory.add(observation, action, reward, value, log_prob, terminal)

    def train_on_episode(self, deterministic: bool, callbacks: Type[Callbacks] = None):
        if callbacks is None:
            callbacks = Callbacks(self)

        done = False
        score = 0
        timesteps = 0
        callbacks.on_episode_begin()

        obs = self.envs.reset()[0]
        self.reset()

        while not done:
            action, value, log_prob = self.evaluate(obs, deterministic)
            next_obs, reward, terminal, truncated, info = self.envs.step(action)

            done = (terminal | truncated).all()

            self.step(obs, action, reward, value, log_prob, terminal)

            score += reward.mean()
            obs = next_obs
            timesteps += 1
        
        self.memory.calc_advantages_and_returns(value, terminal)

        callbacks.on_learn_begin()
        self.learn(self.memory)
        callbacks.on_learn_end()
        self.memory.reset()

        callbacks.on_episode_end()

        return score, timesteps
