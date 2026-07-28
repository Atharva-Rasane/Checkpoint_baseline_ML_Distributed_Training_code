from types import SimpleNamespace

import torch
from torch.utils.data import Dataset

from train import RandomTokenDataset, initialize_engine, load_model


def test_loads_model_from_llm_models_folder():
    model = load_model("tiny_gpt", vocab_size=128, seq_len=16)

    assert model.__class__.__name__ == "TinyGPT"


def test_tiny_gpt_forward_returns_loss():
    model = load_model("tiny_gpt", vocab_size=128, seq_len=16)
    input_ids = torch.randint(0, 128, (2, 16), dtype=torch.long)

    output = model(input_ids=input_ids, labels=input_ids)

    assert output["loss"].ndim == 0
    assert output["logits"].shape == (2, 16, 128)


def test_deepspeed_config_is_not_passed_twice():
    class FakeDeepSpeed:
        def initialize(self, **kwargs):
            assert "config" not in kwargs
            assert kwargs["args"].deepspeed_config == "ds_config_zero3.json"
            assert isinstance(kwargs["training_data"], Dataset)
            return "engine"

    model = torch.nn.Linear(2, 2)
    args = SimpleNamespace(deepspeed_config="ds_config_zero3.json")
    dataset = RandomTokenDataset(samples=4, seq_len=2, vocab_size=8)

    result = initialize_engine(FakeDeepSpeed(), args, model, dataset)

    assert result == "engine"
