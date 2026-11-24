import torch
from torch import nn
from inferx.models.base_model import BaseModel
from inferx.models.tiny.config import TinyConfig
from inferx.runtime.batch import InferenceBatch
from inferx.models.registry import register_model


@register_model
class TinyModel(BaseModel):
    def __init__(self, config: TinyConfig):
        super().__init__()

        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

        self.embeddings = nn.Embedding(self.vocab_size, self.hidden_size)
        self.layers = nn.ModuleList(
            [
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.RMSNorm(self.hidden_size),
                nn.Linear(self.hidden_size, self.hidden_size),
            ]
        )
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, batch: InferenceBatch) -> torch.Tensor:
        x = self.embeddings(batch.input_ids)

        for layer in self.layers:
            x = layer(x)

        return x

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)
