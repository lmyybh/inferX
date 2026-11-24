from dataclasses import dataclass
from inferx.config import ModelConfig, DeviceConfig, EngineConfig
from transformers import PreTrainedConfig, AutoConfig


@dataclass
class EngineArgs:
    model_path: str
    device: str = "cuda"

    def create_engine_config(self) -> EngineConfig:
        def create_model_config(model_path: str) -> ModelConfig:
            return ModelConfig(model_path=model_path)

        def create_device_config(device: str) -> DeviceConfig:
            return DeviceConfig(device=device)

        return EngineConfig(
            model_config=create_model_config(self.model_path),
            device_config=create_device_config(self.device),
        )
