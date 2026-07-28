import torch

from train import load_model


def test_loads_model_from_llm_models_folder():
    model = load_model("tiny_gpt", vocab_size=128, seq_len=16)

    assert model.__class__.__name__ == "TinyGPT"


def test_tiny_gpt_forward_returns_loss():
    model = load_model("tiny_gpt", vocab_size=128, seq_len=16)
    input_ids = torch.randint(0, 128, (2, 16), dtype=torch.long)

    output = model(input_ids=input_ids, labels=input_ids)

    assert output["loss"].ndim == 0
    assert output["logits"].shape == (2, 16, 128)
