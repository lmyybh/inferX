import torch
from typing import Self
from dataclasses import dataclass
from transformers import PreTrainedConfig, AutoConfig

_STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.float16,
    "float16": torch.float16,
    "float": torch.float32,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


class ModelConfig:
    def __init__(self, model_path: str, dtype: str = "auto"):
        self.model_path = model_path
        self.hf_config = self.load_hf_config(model_path)
        self.dtype = self.get_dtype(self.hf_config, dtype)

    def load_hf_config(self, model_path: str) -> PreTrainedConfig:
        return AutoConfig.from_pretrained(model_path)

    def get_dtype(self, hf_config: PreTrainedConfig, dtype: str) -> torch.dtype:
        if not isinstance(dtype, str):
            raise TypeError(f"dtype must be a string, got {dtype!r}")

        dtype = dtype.lower()
        if dtype == "auto":
            # parse dtype in config.json
            torch_dtype = getattr(hf_config, "dtype", None) or getattr(
                hf_config, "torch_dtype", None
            )
            if isinstance(torch_dtype, str):
                torch_dtype = _STR_DTYPE_TO_TORCH_DTYPE.get(torch_dtype, None)
            if torch_dtype is None:
                torch_dtype = torch.float32
        else:
            if dtype not in _STR_DTYPE_TO_TORCH_DTYPE:
                raise ValueError(f"Unkown dtype: {dtype}")
            torch_dtype = _STR_DTYPE_TO_TORCH_DTYPE[dtype]

        return torch_dtype


@dataclass
class DeviceConfig:
    device: str = "cuda"


@dataclass
class EngineConfig:
    model_config: ModelConfig
    device_config: DeviceConfig
