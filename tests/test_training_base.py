import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import Dataset

from train import (
    RandomTokenDataset,
    TrainingTrace,
    directory_stats,
    initialize_engine,
    load_model,
)
from visualization.trace_tools import (
    build_dashboard,
    build_spans,
    cross_node_alignment_figure,
    read_hosts,
    summarize_operator_trace,
    upload_visualization,
    visualization_files,
)


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
    base_ns = 1_785_267_720_000_000_000
    worker_dir = tmp_path / "collected" / "localhost" / "run-1" / "node" / "rank-0"
    worker_dir.mkdir(parents=True)
    events = [
        {
            "event": "run_start",
            "timestamp": "2026-07-28T19:42:00+00:00",
            "model": "tiny_gpt",
            "job": "tiny_gpt",
            "rank": 0,
            "world_size": 1,
        },
        {
            "event": "hardware_profile",
            "gpu_name": "Tesla T4",
            "gpu_total_memory_bytes": 16 * 1024**3,
            "cpu_model": "Test CPU",
            "cpu_count": 4,
            "host_memory_bytes": 32 * 1024**3,
            "network_interface": "ens4",
            "nic_link_speed_mbps": 16000,
            "storage_write_gbps": 1.2,
            "storage_read_gbps": 2.4,
            "host_to_gpu_gbps": 11.0,
            "gpu_to_host_gbps": 10.0,
            "gpu_dram_copy_gbps": 250.0,
            "all_reduce_payload_gbps": 8.0,
            "all_reduce_bus_gbps": 8.0,
            "all_reduce_average_ms": 2.0,
            "model_parameter_bytes": 1024**3,
            "gradient_bytes_estimate": 1024**3,
        },
        {
            "event": "span",
            "host": "node",
            "rank": 0,
            "category": "Training",
            "operation": "Forward",
            "step": 1,
            "start_ns": base_ns,
            "end_ns": base_ns + 5_000_000,
            "duration_ms": 5,
        },
        {"event": "step", "step": 1, "loss": 2.5, "duration_ms": 10, "cuda_peak_allocated_bytes": 100},
        {"event": "checkpoint_complete", "step": 1, "tag": "step-1", "duration_ms": 4, "output_dir": "checkpoints"},
    ]
    (worker_dir / "metrics.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )
    (worker_dir / "operator-trace.json").write_text(
        json.dumps(
            {
                "baseTimeNanoseconds": base_ns,
                "traceEvents": [
                    {"ph": "X", "cat": "cpu_op", "name": "aten::mm", "ts": 100, "dur": 80},
                    {"ph": "X", "cat": "kernel", "name": "gemm", "ts": 200, "dur": 60},
                    {
                        "ph": "X",
                        "cat": "kernel",
                        "name": "ncclAllReduce",
                        "ts": 300,
                        "dur": 50,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    resource_events = [
        {
            "event": "resource_span",
            "host": "node",
            "job": "tiny_gpt",
            "rank": 0,
            "category": "Training",
            "operation": "Forward",
            "resource": resource,
            "start_ns": base_ns,
            "end_ns": base_ns + 5_000_000,
            "duration_ms": duration,
            "measurement": measurement,
        }
        for resource, duration, measurement in (
            ("CPU", 5, "profiler_cpu_total"),
            ("GPU", 4, "profiler_device_total"),
        )
    ]
    (worker_dir / "resource-trace.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in resource_events), encoding="utf-8"
    )
    output = tmp_path / "index.html"

    build_dashboard(tmp_path / "collected", output)

    aggregate = output.read_text(encoding="utf-8")
    node_page = tmp_path / "nodes" / "node.html"
    simulator_profile = json.loads((tmp_path / "simulator-profile.json").read_text())

    assert "Aggregate Trace Explorer" in aggregate
    assert "Cluster activity" in aggregate
    assert "Cross-node start alignment" in aggregate
    assert "CPU, GPU, storage, and network activity" in aggregate
    assert "Node trace: node" in aggregate
    assert "Top CPU operators" in aggregate
    assert '<script src="' not in aggregate
    assert node_page.is_file()
    node_html = node_page.read_text(encoding="utf-8")
    assert "Resource occupancy" in node_html
    assert "CPU, GPU, and gradient synchronization timeline" in node_html
    assert "Simulator hardware profile" in node_html
    assert "16.00 Gbps" in node_html
    assert "Absolute UTC time" in node_html
    assert r"Gradient\u002fNCCL sync" in node_html
    assert r"tiny_gpt \u002f rank 0 \u002f CPU" in node_html
    assert r"tiny_gpt \u002f rank 0 \u002f GPU" in node_html
    assert "CPU/GPU log" in node_html
    assert simulator_profile["trace_start_utc"].startswith("2026-07-28T19:42:00")
    assert simulator_profile["nodes"]["node"]["ranks"][0]["hardware"][
        "nic_link_speed_mbps"
    ] == 16000


def test_hostfile_parser_ignores_slots_and_comments(tmp_path):
    hostfile = tmp_path / "hostfile"
    hostfile.write_text("localhost slots=1\n# worker\n10.0.0.2 slots=2\n", encoding="utf-8")

    assert read_hosts(hostfile) == ["localhost", "10.0.0.2"]


def test_cross_node_timeline_preserves_utc_start_offsets():
    base_ns = 1_785_267_720_000_000_000
    workers = []
    for rank, offset_ms in ((0, 0), (1, 5)):
        start_ns = base_ns + offset_ms * 1_000_000
        workers.append(
            {
                "worker": f"node-{rank}/rank-{rank}",
                "host": f"node-{rank}",
                "started_at": f"2026-07-28T19:42:00.{offset_ms:03d}+00:00",
                "events": [
                    {
                        "event": "span",
                        "rank": rank,
                        "category": "Training",
                        "operation": "Forward",
                        "start_ns": start_ns,
                        "end_ns": start_ns + 2_000_000,
                    }
                ],
            }
        )

    spans = build_spans(workers)
    figure = cross_node_alignment_figure(workers, spans)

    assert spans[1]["start_s"] == 0.005
    assert figure.data[0].customdata[0][1] == 0
    assert round(figure.data[0].customdata[1][1], 3) == 5
    assert figure.layout.xaxis.type == "date"


def test_operator_trace_summary_streams_cpu_gpu_and_communication(tmp_path):
    trace = tmp_path / "operator-trace.json"
    trace.write_text(
        json.dumps(
            {
                "baseTimeNanoseconds": 1_000_000_000,
                "traceEvents": [
                    {"ph": "X", "cat": "cpu_op", "name": "aten::mm", "ts": 10, "dur": 100},
                    {"ph": "X", "cat": "kernel", "name": "gemm", "ts": 20, "dur": 80},
                    {
                        "ph": "X",
                        "cat": "kernel",
                        "name": "ncclAllReduce",
                        "ts": 30,
                        "dur": 20,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    summary = summarize_operator_trace(trace)

    assert summary["counts"] == {"cpu": 1, "gpu": 2, "communication": 1}
    assert summary["cpu"]["aten::mm"] == 100
    assert summary["communication"]["ncclAllReduce"] == 20
    assert summary["timeline_events"][0]["start_ns"] == 1_000_010_000
    assert {event["resource"] for event in summary["timeline_events"]} == {
        "CPU",
        "GPU",
        "Communication",
    }


def test_checkpoint_directory_stats(tmp_path):
    (tmp_path / "rank-0.bin").write_bytes(b"1234")
    (tmp_path / "rank-1.bin").write_bytes(b"56789")

    assert directory_stats(tmp_path) == (2, 9)


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
    with trace.span("Training", "Forward", step=1):
        torch.ones(2) + 1
    trace.log("step", step=1, loss=1.0)
    trace.stop()

    worker_dir = next(tmp_path.glob("run-1/*/rank-0"))
    assert (worker_dir / "operator-trace.json").stat().st_size > 0
    assert (worker_dir / "execution-trace.json").stat().st_size > 0
    resource_events = (worker_dir / "resource-trace.jsonl").read_text(encoding="utf-8")
    assert '"resource": "CPU"' in resource_events
    assert '"operation": "Forward"' in resource_events
    assert '"event": "span"' in (worker_dir / "metrics.jsonl").read_text(encoding="utf-8")
    assert '"event": "step"' in (worker_dir / "metrics.jsonl").read_text(encoding="utf-8")


def test_upload_file_list_contains_dashboard_and_collected_traces(tmp_path):
    (tmp_path / "index.html").write_text("dashboard", encoding="utf-8")
    (tmp_path / "plotly.min.js").write_text("plotly", encoding="utf-8")
    node_page = tmp_path / "nodes" / "node-a.html"
    node_page.parent.mkdir()
    node_page.write_text("node", encoding="utf-8")
    trace = tmp_path / "collected" / "localhost" / "operator-trace.json"
    trace.parent.mkdir(parents=True)
    trace.write_text("{}", encoding="utf-8")

    assert visualization_files(tmp_path) == [
        tmp_path / "index.html",
        tmp_path / "plotly.min.js",
        node_page,
        trace,
    ]


def test_upload_uses_vm_and_timestamp_prefix(tmp_path):
    uploaded = []

    class FakeBlob:
        def __init__(self, name):
            self.name = name
            self.content_disposition = None
            self.cache_control = None

        def upload_from_filename(self, path, content_type=None):
            uploaded.append(
                (
                    self.name,
                    Path(path).name,
                    content_type,
                    self.content_disposition,
                    self.cache_control,
                )
            )

    class FakeBucket:
        def blob(self, name):
            return FakeBlob(name)

    class FakeClient:
        def bucket(self, _):
            return FakeBucket()

    (tmp_path / "index.html").write_text("dashboard", encoding="utf-8")
    trace = tmp_path / "collected" / "node" / "rank-0" / "operator-trace.json"
    trace.parent.mkdir(parents=True)
    trace.write_text("{}", encoding="utf-8")

    upload_visualization(tmp_path, "traces", "atharva-instace_20260728_191530", client=FakeClient())

    assert uploaded[0][0] == "atharva-instace_20260728_191530/index.html"
    assert uploaded[0][2:] == ("text/html", "inline", "no-cache")
    assert uploaded[1][0].endswith("collected/node/rank-0/operator-trace.json")
