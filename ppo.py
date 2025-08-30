import os
import sys
from functools import partial
from typing import Any, Callable, Optional, Union

from dataclasses import dataclass, asdict


import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule

from environment import KRes
from GNNbase import BlockBase, GATBlock, GCNAdjBlock, GCNBlock, ResBn


class RewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        for reward in self.locals.get("rewards", []):
            print(f"Step: {self.num_timesteps} Reward: {reward} ")
        return True


class GraphFeaturesExtractor(BaseFeaturesExtractor):
    """
    GCN-based feature extractor for Stable Baselines3

    This extracts features from graph-structured observations where:
    - obs[:, :, :n_drones] is the adjacency matrix
    - obs[:, :, n_drones:] contains node features
    """

    supported: list[BlockBase] = [GCNBlock, GCNAdjBlock, GATBlock]

    def __init__(
        self,
        observation_space: spaces.Box,
        NNBlock: BlockBase = GCNBlock,
        hidden_dims: list[int] = [64, 64],
        used_res: list[bool] | bool = [True, False],
        norm: Optional[str | Callable] = "layer",
        norm_kwargs: Optional[dict[str, Any]] = None,
        act: Optional[str | Callable] = "relu",
        act_kwargs: Optional[dict[str, Any]] = None,
        gnn_layer_kwargs: dict[str, Any] = None,
        dropout: float = 0.1,
    ):
        # The features_dim is the output dimension of this feature extractor
        super().__init__(observation_space, hidden_dims[-1])

        self.n_drones = observation_space.shape[0]
        self.n_node_features = observation_space.shape[1] - self.n_drones
        self.hidden_dims = hidden_dims

        if isinstance(used_res, bool):
            used_res = [used_res] * len(hidden_dims)

        if len(used_res) < len(hidden_dims):
            used_res.extend([False] * (len(hidden_dims) - len(used_res)))

        # GCN layers
        self.gcn_layers = nn.ModuleList()

        if gnn_layer_kwargs is None:
            gnn_layer_kwargs = {}

        # First layer: node_features -> hidden_dim
        block = NNBlock(
            self.n_node_features,
            self.hidden_dims[0],
            norm=norm,
            norm_kwargs=norm_kwargs,
            act=act,
            act_kwargs=act_kwargs,
            **gnn_layer_kwargs,
        )
        if used_res[0]:
            block = ResBn(block)

        self.gcn_layers.append(block)

        for i in range(1, len(hidden_dims)):
            block = NNBlock(
                self.hidden_dims[i - 1],
                self.hidden_dims[i],
                norm=norm,
                norm_kwargs=norm_kwargs,
                act=act,
                act_kwargs=act_kwargs,
                **gnn_layer_kwargs,
            )
            if used_res[i]:
                block = ResBn(block)
            self.gcn_layers.append(block)

        self.dropout = nn.Dropout(p=dropout) if 0 <= dropout < 1 else nn.Identity()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            observations: [batch_size, n_drones, n_drones + n_node_features]

        Returns:
            features: [batch_size, features_dim]
        """

        # Extract adjacency matrix and node features

        X = observations[:, :, self.n_drones :]
        adj = observations[:, :, : self.n_drones]

        for i, gcn_layer in enumerate(self.gcn_layers):
            X = gcn_layer(X, adj)  # [batch_size, n_drones, hidden_dim]

        X = self.dropout(X)

        return X  # batch_size, n_drones, hidden_dims[-1]


class ValueNet(torch.nn.Module):
    def __init__(self, n_nodes: int, latent_dim: int):
        super(ValueNet, self).__init__()
        self.value_net = torch.nn.Linear(n_nodes * latent_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.value_net(x.view(x.shape[0], -1))


class ActionNet(torch.nn.Module):
    def __init__(self, n_nodes: int, linear_module: torch.nn.Linear):
        super(ActionNet, self).__init__()

        self.action_net = torch.nn.Linear(
            linear_module.in_features, linear_module.out_features // n_nodes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        X = self.action_net(x)
        return X.view(X.shape[0], -1)


class NodeLevelActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        NNBlock: BlockBase,
        hidden_dims: list[int] = [64, 64],
        used_res: list[bool] | bool = True,
        norm: Optional[str | Callable] = "layer",
        norm_kwargs: Optional[dict[str, Any]] = None,
        act: Optional[str | Callable] = "relu",
        act_kwargs: Optional[dict[str, Any]] = None,
        gnn_layer_kwargs: dict[str, Any] = None,
        dropout: float = 0.1,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        activation_fn: type[torch.nn.Module] = torch.nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
    ):
        self.n_nodes = observation_space.shape[0]

        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
            use_sde=use_sde,
            log_std_init=log_std_init,
            full_std=full_std,
            use_expln=use_expln,
            squash_output=squash_output,
            features_extractor_class=GraphFeaturesExtractor,
            features_extractor_kwargs=dict(
                NNBlock=NNBlock,
                used_res=used_res,
                hidden_dims=hidden_dims,
                norm=norm,
                norm_kwargs=norm_kwargs,
                act=act,
                act_kwargs=act_kwargs,
                gnn_layer_kwargs=gnn_layer_kwargs,
                dropout=dropout,
            ),
            share_features_extractor=share_features_extractor,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi,
                latent_sde_dim=latent_dim_pi,
                log_std_init=self.log_std_init,
            )
        elif isinstance(
            self.action_dist,
            (
                CategoricalDistribution,
                MultiCategoricalDistribution,
                BernoulliDistribution,
            ),
        ):
            self.action_net = ActionNet(
                self.n_nodes,
                self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi),
            )
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        self.value_net = ValueNet(self.n_nodes, self.mlp_extractor.latent_dim_vf)

        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
            }
            if not self.share_features_extractor:
                # Note(antonin): this is to keep SB3 results
                # consistent, see GH#1148
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )  # type: ignore[call-arg]

    def forward(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        # Evaluate the values for the given observations
        values = self.value_net(latent_vf)

        distribution = self._get_action_dist_from_latent(latent_pi)

        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))  # type: ignore[misc]
        return actions, values, log_prob

@dataclass
class EnvOptions:
    n_envs: int
    n_drones: int
    k: int
    size: int
    alpha: float = 0.1

@dataclass
class PolicyOptions:
    NNBlock: BlockBase
    hidden_dims: list[int]
    used_res: bool | list[bool]

    norm: str = "layer"
    norm_kwargs: Optional[dict[str, Any]] = None
    act: str = "relu"
    act_kwargs: Optional[dict[str, Any]] = None
    gnn_layer_kwargs: Optional[dict] = None
    dropout: float = 0.5
    net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None

    activation_fn: type[torch.nn.Module] = torch.nn.Tanh
    ortho_init: bool = True
    log_std_init: float = 0.0
    full_std: bool = True
    use_expln: bool = False
    squash_output: bool = False
    share_features_extractor: bool = True
    normalize_images: bool = True
    optimizer_class: type[torch.optim.Optimizer] = torch.optim.Adam
    optimizer_kwargs: Optional[dict[str, Any]] = None

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
    train_options
):
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    def make_env():
        return KRes(n_drones=env_options.n_drones, k=env_options.k, size=env_options.size, alpha=env_options.alpha)
    env = make_vec_env(make_env, n_envs=env_options.n_envs)

    # Create PPO model with custom GCN policy
    model = PPO(
        policy=NodeLevelActorCriticPolicy,
        env=env,
        policy_kwargs=asdict(policy_options),
        **asdict(rl_options),
    )

    print("Node-level GCN policy created successfully for Stable Baselines3!")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Test the policy
    obs = env.reset()
    print(f"Observation shape: {obs.shape}")

    action, _ = model.predict(obs, deterministic=True)
    print(f"Predicted action: {action}")

    # Test a short training run
    print("\nTesting short training run...")
    model.learn(
        **asdict(train_options)
    )
    print("Training completed successfully!")

    return model


def main():
    model = train(
        EnvOptions(
            1,
            5,
            3,
            10,
            0.1
        ),
        PolicyOptions(
            NNBlock=GCNBlock,
            hidden_dims=[64, 64],
            used_res=True,
        ),
        RLOptions(),
        TrainOptions()
    )
    env = KRes(
        n_drones=5,
        k=3,
        size=10,
        return_state="features",
        normalize_features=True,
        render_mode="human",
        render_fps=1,
    )

    obs, _ = env.reset()

    for i in range(10):
        env.render()
        actions, _ = model.predict(obs)
        new_obs, reward, done, terminated, info = env.step(actions)

        if done:
            break

if __name__ == "__main__":
    main()


