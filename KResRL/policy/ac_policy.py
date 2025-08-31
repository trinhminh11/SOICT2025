from functools import partial
from dataclasses import dataclass
from typing import Any, Optional, Union

from gymnasium import spaces
import numpy as np
import torch
from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
)
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

@dataclass(frozen=True)
class PolicyOptions:
    """
    General Policy options for the actor-critic model
    """
    policy_cls: type[ActorCriticPolicy]

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


class NodeLevelFeatureExtractor(BaseFeaturesExtractor):
    """DL Module return [B, N, d]
    B: batch_size
    N: number of nodes
    d: feature dimension

    Args:
        BaseFeaturesExtractor (_type_): _description_
    """
    pass

class NodeLevelActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        activation_fn: type[torch.nn.Module] = torch.nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: type[NodeLevelFeatureExtractor] = NodeLevelFeatureExtractor,
        features_extractor_kwargs: dict[str, Any]=None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
    ):
        self.n_nodes = observation_space.shape[0]

        if not issubclass(features_extractor_class, NodeLevelFeatureExtractor):
            raise ValueError("features_extractor_class must be a subclass of NodeLevelFeatureExtractor")

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
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            share_features_extractor=share_features_extractor,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )
    
    def get_features_extractor_class(self) -> type[torch.nn.Module]:
        raise NotImplementedError("")

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

