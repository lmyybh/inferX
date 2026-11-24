from dataclasses import dataclass
import torch


@dataclass
class InferenceBatch:
    req_ids: list[str]
    batch_size: int
    input_ids: torch.Tensor
    positions: torch.Tensor
    seq_lens: torch.Tensor
