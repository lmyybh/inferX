import torch
from pathlib import Path
from transformers import AutoConfig
from inferx.runtime.batch import InferenceBatch
from inferx.config import EngineConfig
from inferx.runtime.runner import ModelRunner
from inferx.engine_args import EngineArgs
from inferx.models.tiny.config import TinyConfig

AutoConfig.register("tiny", TinyConfig)

args = EngineArgs(
    model_path=str(Path(__file__).resolve().parent / "model"), device="cuda"
)
engine_config = args.create_engine_config()

batch = InferenceBatch(
    req_ids=["req01", "req02"],
    batch_size=2,
    input_ids=torch.tensor(
        [2, 3, 4, 11, 12, 13, 14], dtype=torch.int32, device=args.device
    ),
    positions=torch.tensor(
        [0, 1, 2, 0, 1, 2, 3], dtype=torch.int32, device=args.device
    ),
    seq_lens=torch.tensor([3, 4], dtype=torch.int32, device=args.device),
)

runner = ModelRunner(config=engine_config)
logits = runner.forward(batch)

print(logits.shape)
