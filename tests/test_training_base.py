import json
from types import SimpleNamespace

import torch
from torch.utils.data import Dataset

from train import RandomTokenDataset, TrainingTrace, initialize_engine, load_model
from visualization.trace_tools import build_dashboard, read_hosts


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


def test_visualization_builds_from_rank_metrics(tmp_path):
    worker_dir = tmp_path / "collected" / "localhost" / "run-1" / "node" / "rank-0"
    worker_dir.mkdir(parents=True)
    events = [
        {"event": "run_start", "model": "tiny_gpt", "rank": 0, "world_size": 1},
        {"event": "step", "step": 1, "loss": 2.5, "duration_ms": 10, "cuda_peak_allocated_bytes": 100},
        {"event": "checkpoint_complete", "step": 1, "tag": "step-1", "duration_ms": 4, "output_dir": "checkpoints"},
    ]
    (worker_dir / "metrics.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )
    (worker_dir / "operator-trace.json").write_text("{}", encoding="utf-8")
    output = tmp_path / "index.html"

    build_dashboard(tmp_path / "collected", output)

    assert "tiny_gpt" in output.read_text(encoding="utf-8")


def test_hostfile_parser_ignores_slots_and_comments(tmp_path):
    hostfile = tmp_path / "hostfile"
    hostfile.write_text("localhost slots=1\n# worker\n10.0.0.2 slots=2\n", encoding="utf-8")

    assert read_hosts(hostfile) == ["localhost", "10.0.0.2"]


def test_training_trace_writes_profiler_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    trace = TrainingTrace(
        enabled=True,
        trace_dir=tmp_path,
        run_id="run-1",
        rank=0,
        world_size=1,
        metadata={"model": "tiny_gpt"},
    )
    with trace.region("test/operator"):
        torch.ones(2) + 1
    trace.log("step", step=1, loss=1.0)
    trace.stop()

    worker_dir = next(tmp_path.glob("run-1/*/rank-0"))
    assert (worker_dir / "operator-trace.json").stat().st_size > 0
    assert (worker_dir / "execution-trace.json").stat().st_size > 0
    assert '"event": "step"' in (worker_dir / "metrics.jsonl").read_text(encoding="utf-8")
