# Third-party imports
import torch

import numpy as np
from numpy.typing import NDArray

import gymnasium.spaces as spaces


def get_device(device: torch.device | str = "auto") -> torch.device:
    """
    Retrieve PyTorch device.
    It checks that the requested device is available first.
    For now, it supports only cpu and cuda.
    By default, it tries to use the gpu.

    :param device: One for 'auto', 'cuda', 'cpu'
    :return: Supported Pytorch device
    """
    # Cuda by default
    if device == "auto":
        device = "cuda"
    # Force conversion to th.device
    device = torch.device(device)

    # Cuda not available
    if device.type == torch.device("cuda").type and not torch.cuda.is_available():
        return torch.device("cpu")

    return device


def get_shape(
    space: spaces.Space,
) -> tuple[int, ...] | dict[str, tuple[int, ...]]:
    """
    Get the shape of the observation (useful for the buffers).

    :param space:
    :return:
    """
    if isinstance(space, spaces.Box):
        return space.shape
    elif isinstance(space, spaces.Discrete):
        return ()
    elif isinstance(space, spaces.MultiDiscrete):
        # Number of discrete features
        return (len(space.nvec),)
    elif isinstance(space, spaces.MultiBinary):
        # Number of binary features
        return space.shape
    elif isinstance(space, spaces.Dict):
        return {key: get_shape(subspace) for (key, subspace) in space.spaces.items()}
    else:
        raise NotImplementedError(f"{space} space is not supported")


def get_obs_shape(
    observation_space: spaces.Space,
) -> tuple[int, ...] | dict[str, tuple[int, ...]]:
    """
    Get the shape of the observation (useful for the buffers).

    :param observation_space:
    :return:
    """
    if isinstance(observation_space, spaces.Box):
        return observation_space.shape
    elif isinstance(observation_space, spaces.Discrete):
        # Observation is an int
        return (1,)
    elif isinstance(observation_space, spaces.MultiDiscrete):
        # Number of discrete features
        return (len(observation_space.nvec),)
    elif isinstance(observation_space, spaces.MultiBinary):
        # Number of binary features
        return observation_space.shape
    elif isinstance(observation_space, spaces.Dict):
        return {
            key: get_obs_shape(subspace)
            for (key, subspace) in observation_space.spaces.items()
        }  # type: ignore[misc]

    else:
        raise NotImplementedError(
            f"{observation_space} observation space is not supported"
        )


def get_action_dim(action_space: spaces.Space) -> int:
    """
    Get the dimension of the action space.

    :param action_space:
    :return:
    """
    if isinstance(action_space, spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, spaces.Discrete):
        # Action is an int
        return 1
    elif isinstance(action_space, spaces.MultiDiscrete):
        # Number of discrete actions
        return len(action_space.nvec)
    elif isinstance(action_space, spaces.MultiBinary):
        # Number of binary actions
        assert isinstance(action_space.n, int), (
            f"Multi-dimensional MultiBinary({action_space.n}) action space is not supported. You can flatten it instead."
        )
        return int(action_space.n)
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")


def to_torch(
    x: NDArray | dict[str, NDArray],
    device: str | torch.device,
) -> torch.Tensor | dict[str, torch.Tensor]:
    """
    Convert a tensor or a dictionary of tensors to a torch.Tensor.

    :param x: the input tensor or dictionary of tensors
    :param device: the device to which the tensor(s) will be moved
    :return: the tensor or dictionary of tensors as torch.Tensor(s)
    """
    if isinstance(x, dict):
        return {key: torch.from_numpy(value).to(device) for key, value in x.items()}
    return torch.from_numpy(x).to(device)


def check_for_nested_spaces(space: spaces.Space) -> None:
    """
    Make sure the observation space does not have nested spaces (Dicts/Tuples inside Dicts/Tuples).
    If so, raise an Exception informing that there is no support for this.

    :param space: an observation space
    """
    if isinstance(space, (spaces.Dict, spaces.Tuple)):
        sub_spaces = (
            space.spaces.values() if isinstance(space, spaces.Dict) else space.spaces
        )
        for sub_space in sub_spaces:
            if isinstance(sub_space, (spaces.Dict, spaces.Tuple)):
                raise NotImplementedError(
                    "Nested observation spaces are not supported (Tuple/Dict space inside Tuple/Dict space)."
                )


def to_device(*args, device="cuda"):
    for arg in args:
        arg.to(device)
