import torch
import torch.nn as nn


class BasePolicy(nn.Module):
    def action(self, state: torch.Tensor):
        raise NotImplementedError
    
    def value(self, state: torch.Tensor):
        raise NotImplementedError
    
    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
    