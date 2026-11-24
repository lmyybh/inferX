from transformers import PreTrainedConfig


class TinyConfig(PreTrainedConfig):
    model_type = "tiny"

    def __init__(self, vocab_size: int = 512, hidden_size: int = 32, **kwargs):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
