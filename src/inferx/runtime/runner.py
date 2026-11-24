import torch

from inferx.config import EngineConfig
from inferx.models.registry import get_model_class
from inferx.runtime.batch import InferenceBatch
from inferx.utils import set_default_torch_dtype


class ModelRunner:
    def __init__(self, config: EngineConfig):
        self.config = config

        self.model = self.load_model(config)

    def load_model(self, config: EngineConfig):
        target_device = torch.device(self.config.device_config.device)

        with set_default_torch_dtype(self.config.model_config.dtype):
            with target_device:
                model_cls = get_model_class(config.model_config.hf_config.architectures)
                model = model_cls(config.model_config.hf_config)
        # todo: gc, empty_cache

        return model.eval()

    @torch.inference_mode()
    def forward(self, batch: InferenceBatch):
        hidden_states = self.model(batch)
        logits = self.model.compute_logits(hidden_states)
        return logits
