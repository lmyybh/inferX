from abc import ABC, abstractmethod
import torch
from torch import nn
from inferx.runtime.batch import InferenceBatch


class BaseModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, batch: InferenceBatch) -> torch.Tensor: ...

    @abstractmethod
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor: ...
