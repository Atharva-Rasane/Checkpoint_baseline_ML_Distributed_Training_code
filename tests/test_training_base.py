import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import Dataset

from train import (
    RandomTokenDataset,
    TrainingTrace,
    checkpoint_deepspeed_config,
    directory_stats,
    finish_async_checkpoint,
    initialize_engine,
    load_dataset,
    load_model,
    resume_from_checkpoint,
)
from visualization.trace_tools import (
    build_dashboard,
    build_spans,
    cross_node_alignment_figure,
    read_hosts,
    resource_timeline,
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


def test_checkpoint_entrypoint_prefers_project_train_module(monkeypatch):
    project_root = Path(__file__).resolve().parents[1]
    wrapper_dir = project_root / "checkpointing" / "asynchronous"
    monkeypatch.setattr(sys, "path", [str(wrapper_dir), str(project_root), *sys.path])

    namespace = runpy.run_path(str(wrapper_dir / "train.py"), run_name="entrypoint_test")

    assert sys.path[0] == str(project_root)
    assert namespace["PROJECT_ROOT"] == project_root


def test_synthetic_dataset_is_deterministic_by_sample():
    dataset = RandomTokenDataset(samples=4, seq_len=8, vocab_size=32, seed=7)

    assert torch.equal(dataset[2]["input_ids"], dataset[2]["input_ids"])
    assert not torch.equal(dataset[1]["input_ids"], dataset[2]["input_ids"])


def test_model_module_can_provide_production_dataset(monkeypatch):
    expected = RandomTokenDataset(samples=2, seq_len=4, vocab_size=8)
    module = SimpleNamespace(
        __name__="llm_models.production_model",
        build_dataset=lambda **_kwargs: expected,
    )
    monkeypatch.setattr("train.load_model_module", lambda _name: module)

    dataset = load_dataset(
        "production_model",
        samples=2,
        seq_len=4,
        vocab_size=8,
        seed=1234,
    )

    assert dataset is expected


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


def test_async_checkpoint_config_uses_decoupled_cpu_writer():
    config = checkpoint_deepspeed_config("asynchronous", batch_size=4)

    assert config["zero_optimization"]["stage"] == 3
    assert config["train_micro_batch_size_per_gpu"] == 4
    assert config["gradient_accumulation_steps"] == 1
    assert config["checkpoint"]["writer"] == {
        "type": "PYTHON",
        "decoupled": True,
        "data_parallel": "REPLICA",
        "show_statistics": True,
    }


def test_synchronous_config_has_no_accumulation_or_async_writer():
    config = checkpoint_deepspeed_config("synchronous", batch_size=3)

    assert config["train_micro_batch_size_per_gpu"] == 3
    assert config["gradient_accumulation_steps"] == 1
    assert "checkpoint" not in config


def test_resume_restores_completed_step_from_client_state():
    class FakeEngine:
        def load_checkpoint(self, output_dir, tag=None):
            assert output_dir == "checkpoints"
            assert tag is None
            return "checkpoints/step-20/model.pt", {"completed_steps": 20}

    completed_steps, path = resume_from_checkpoint(
        FakeEngine(),
        "checkpoints",
        "latest",
    )

    assert completed_steps == 20
    assert path == "checkpoints/step-20/model.pt"


def test_async_checkpoint_completion_records_background_cpu_span(tmp_path, monkeypatch):
    class FakeCheckpointEngine:
        def __init__(self):
            self.commit_info = object()

        def get_commit_info(self):
            return self.commit_info

    class FakeEngine:
        def __init__(self):
            self.checkpoint_engine = FakeCheckpointEngine()
            self.commits = 0

        def _commit_decoupled_checkpoint(self):
            self.commits += 1
            self.checkpoint_engine.commit_info = None

    class FakeTrace:
        def __init__(self):
            self.spans = []
            self.events = []

        def record_span(self, category, operation, start_ns, end_ns, **values):
            self.spans.append((category, operation, start_ns, end_ns, values))

        def log(self, event, **values):
            self.events.append((event, values))

    checkpoint_dir = tmp_path / "step-2"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "rank-0.pt").write_bytes(b"checkpoint")
    timestamps = iter((200, 300, 400))
    monkeypatch.setattr("train.time.time_ns", lambda: next(timestamps))
    engine = FakeEngine()
    trace = FakeTrace()
    pending = {
        "step": 2,
        "tag": "step-2",
        "request_ns": 100,
        "staged_ns": 150,
        "staging_duration_ms": 0.00005,
        "worker_pid": 4321,
    }

    result = finish_async_checkpoint(
        engine,
        trace,
        pending,
        tmp_path,
        rank=0,
        force=True,
    )

    assert result is None
    assert engine.commits == 1
    assert [span[1] for span in trace.spans] == [
        "Async checkpoint commit wait",
        "Checkpoint CPU serialization",
        "Checkpoint DRAM to SSD write",
    ]
    serialization, storage = trace.spans[1:]
    assert serialization[2:4] == (150, 400)
    assert serialization[4]["resource_measurement"] == (
        "background_checkpoint_in_flight_wall_clock"
    )
    assert storage[2:4] == (150, 400)
    assert storage[4]["resource_override"] == "Storage"
    assert storage[4]["resource_measurement"] == (
        "checkpoint_storage_in_flight_wall_clock"
    )
    assert trace.events[0][0] == "checkpoint_complete"
    assert trace.events[0][1]["checkpoint_mode"] == "asynchronous"
    assert trace.events[0][1]["checkpoint_worker_pid"] == 4321


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
            "checkpoint_mode": "asynchronous",
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
        {
            "event": "span",
            "host": "node",
            "rank": 0,
            "category": "Training",
            "operation": "Backward",
            "step": 1,
            "start_ns": base_ns + 5_000_000,
            "end_ns": base_ns + 12_000_000,
            "duration_ms": 7,
        },
        {
            "event": "span",
            "host": "node",
            "rank": 0,
            "category": "Optimizer",
            "operation": "Optimizer step",
            "step": 1,
            "start_ns": base_ns + 12_000_000,
            "end_ns": base_ns + 15_000_000,
            "duration_ms": 3,
        },
        {
            "event": "span",
            "host": "node",
            "rank": 0,
            "category": "Checkpoint",
            "operation": "Checkpoint state snapshot to host DRAM",
            "step": 1,
            "tag": "step-1",
            "checkpoint_mode": "asynchronous",
            "start_ns": base_ns + 1_000_000,
            "end_ns": base_ns + 3_000_000,
            "duration_ms": 2,
        },
        {
            "event": "step",
            "step": 1,
            "loss": 2.5,
            "duration_ms": 10,
            "cuda_peak_allocated_bytes": 100,
        },
        {
            "event": "checkpoint_complete",
            "step": 1,
            "tag": "step-1",
            "duration_ms": 4,
            "staging_duration_ms": 1,
            "persistence_duration_ms": 3,
            "checkpoint_mode": "asynchronous",
            "checkpoint_worker_pid": 4321,
            "output_dir": "checkpoints",
        },
    ]
    (worker_dir / "metrics.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )
    (worker_dir / "operator-trace.json").write_text(
        json.dumps(
            {
                "baseTimeNanoseconds": base_ns,
                "traceEvents": [
                    {
                        "ph": "X",
                        "cat": "cpu_op",
                        "name": "aten::mm",
                        "ts": 100,
                        "dur": 80,
                    },
                    {"ph": "X", "cat": "kernel", "name": "gemm", "ts": 200, "dur": 60},
                    {
                        "ph": "X",
                        "cat": "gpu_memcpy",
                        "name": "Memcpy DtoH (Device -> Pinned)",
                        "ts": 1500,
                        "dur": 200,
                    },
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
    phase_resources = (
        ("Forward", "Training", 0, 5),
        ("Backward", "Training", 5, 7),
        ("Optimizer step", "Optimizer", 12, 3),
    )
    resource_events = [
        {
            "schema_version": 2,
            "event": "resource_span",
            "host": "node",
            "job": "tiny_gpt",
            "rank": 0,
            "category": category,
            "operation": operation,
            "resource": resource,
            "start_ns": base_ns + offset_ms * 1_000_000,
            "end_ns": base_ns + (offset_ms + duration_ms) * 1_000_000,
            "resource_start_ns": base_ns + offset_ms * 1_000_000,
            "resource_end_ns": base_ns + (offset_ms + duration_ms) * 1_000_000,
            "duration_ms": duration_ms,
            "measurement": measurement,
            "start_alignment": (
                "observed_wall_clock"
                if resource == "CPU"
                else "enclosing_cpu_phase_start"
            ),
            "start_is_estimated": resource == "GPU",
            "profiler_cpu_total_ms": duration_ms if resource == "CPU" else None,
            "profiler_self_cpu_ms": 0.5 if resource == "CPU" else None,
            "gpu_stream_elapsed_ms": duration_ms if resource == "GPU" else None,
            "profiler_device_total_ms": duration_ms - 0.5 if resource == "GPU" else None,
            "device_kernel_count": 3 if resource == "GPU" else None,
        }
        for operation, category, offset_ms, duration_ms in phase_resources
        for resource, measurement in (
            ("CPU", "wall_clock"),
            ("GPU", "cuda_event_elapsed"),
        )
    ]
    resource_events.extend(
        [
            {
                "schema_version": 2,
                "event": "resource_span",
                "host": "node",
                "job": "tiny_gpt",
                "rank": 0,
                "category": "Checkpoint",
                "operation": "Checkpoint CPU serialization",
                "resource": "CPU",
                "start_ns": base_ns + 2_000_000,
                "end_ns": base_ns + 14_000_000,
                "resource_start_ns": base_ns + 2_000_000,
                "resource_end_ns": base_ns + 14_000_000,
                "duration_ms": 12,
                "measurement": "background_checkpoint_in_flight_wall_clock",
                "start_alignment": "observed_wall_clock",
                "start_is_estimated": False,
                "checkpoint_worker_pid": 4321,
            },
            {
                "schema_version": 2,
                "event": "resource_span",
                "host": "node",
                "job": "tiny_gpt",
                "rank": 0,
                "category": "Checkpoint",
                "operation": "Checkpoint DRAM to SSD write",
                "resource": "Storage",
                "start_ns": base_ns + 2_000_000,
                "end_ns": base_ns + 14_000_000,
                "resource_start_ns": base_ns + 2_000_000,
                "resource_end_ns": base_ns + 14_000_000,
                "duration_ms": 12,
                "measurement": "checkpoint_storage_in_flight_wall_clock",
                "start_alignment": "observed_wall_clock",
                "start_is_estimated": False,
                "checkpoint_worker_pid": 4321,
            },
        ]
    )
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
    assert "Capacity and cross-node alignment" in aggregate
    assert "CPU, GPU, storage, and network activity" in aggregate
    assert "Forward/backward CPU and GPU phases across all ranks" in aggregate
    assert "Forward - CPU" in aggregate
    assert "Forward - GPU" in aggregate
    assert "Backward - CPU" in aggregate
    assert "Backward - GPU" in aggregate
    assert "Optimizer step - CPU" in aggregate
    assert "Optimizer step - GPU" in aggregate
    assert "Checkpoint data path across all ranks" in aggregate
    assert "Checkpoint CPU serialization - CPU" in aggregate
    assert "Checkpoint DRAM to SSD write - Storage" in aggregate
    assert "Observed GPU to host DRAM copy - GPU" in aggregate
    assert "background_checkpoint_in_flight_wall_clock" in aggregate
    assert "checkpoint_storage_in_flight_wall_clock" in aggregate
    assert "kineto_gpu_memcpy" in aggregate
    assert "asynchronous" in aggregate
    assert "4321" in aggregate
    assert "What happened?" in aggregate
    assert "Aggregate metrics" in aggregate
    assert "Collective communication occupancy" in aggregate
    assert 'data-tab-target="node-panel-node"' in aggregate
    assert 'data-plot="aggregate-communication-timeline"' in aggregate
    assert 'data-table-filter="aggregate-slow-span-rows"' in aggregate
    assert "Node trace: node" in aggregate
    assert "Top CPU operators" in aggregate
    assert '<script src="' not in aggregate
    assert node_page.is_file()
    node_html = node_page.read_text(encoding="utf-8")
    assert "Forward/backward CPU and GPU phase timeline" in node_html
    assert "Checkpoint data path" in node_html
    assert "CPU and GPU operation timeline" in node_html
    assert "Collective communication occupancy" in node_html
    assert "Simulator hardware profile" in node_html
    assert "16.00 Gbps" in node_html
    assert "Absolute UTC time" in node_html
    assert r"Gradient\u002fNCCL sync" in node_html
    assert r"tiny_gpt \u002f rank 0 | CPU" in node_html
    assert r"tiny_gpt \u002f rank 0 | GPU" in node_html
    assert "enclosing_cpu_phase_start" in node_html
    assert "kernel total=" in node_html
    assert r"tiny_gpt \u002f rank 0 \u002f collective stream" in node_html
    assert "CPU/GPU log" in node_html
    assert simulator_profile["trace_start_utc"].startswith("2026-07-28T19:42:00")
    assert simulator_profile["checkpoint_mode"] == "asynchronous"
    assert simulator_profile["checkpoints"][0]["staging_duration_ms"] == 1
    assert simulator_profile["checkpoints"][0]["persistence_duration_ms"] == 3
    assert (
        simulator_profile["nodes"]["node"]["ranks"][0]["hardware"][
            "nic_link_speed_mbps"
        ]
        == 16000
    )


def test_hostfile_parser_ignores_slots_and_comments(tmp_path):
    hostfile = tmp_path / "hostfile"
    hostfile.write_text(
        "localhost slots=1\n# worker\n10.0.0.2 slots=2\n", encoding="utf-8"
    )

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


def test_phase_timeline_separates_cpu_and_gpu_for_each_operation():
    spans = []
    for operation, start_s in (("Forward", 0.0), ("Backward", 0.01)):
        for resource_name in ("CPU", "GPU"):
            spans.append(
                {
                    "operation": operation,
                    "resource": resource_name,
                    "category": "Training",
                    "job": "tiny_gpt",
                    "rank": 0,
                    "worker": "node/rank-0",
                    "start_s": start_s,
                    "duration_s": 0.005,
                    "start_utc": f"2026-07-28T19:42:00.{int(start_s * 1000):03d}+00:00",
                    "measurement": (
                        "wall_clock" if resource_name == "CPU" else "cuda_event_elapsed"
                    ),
                }
            )
    spans.append(
        {
            "operation": "Checkpoint CPU serialization",
            "resource": "CPU",
            "category": "Checkpoint",
            "job": "tiny_gpt",
            "rank": 0,
            "worker": "node/rank-0",
            "start_s": 0.002,
            "duration_s": 0.012,
            "start_utc": "2026-07-28T19:42:00.002+00:00",
            "measurement": "background_checkpoint_in_flight_wall_clock",
        }
    )

    figure = resource_timeline(spans, "node")
    traces = {trace.name: trace for trace in figure.data}

    assert set(traces) == {
        "Forward - CPU",
        "Forward - GPU",
        "Backward - CPU",
        "Backward - GPU",
        "Checkpoint CPU serialization - CPU",
    }
    assert traces["Forward - CPU"].marker.pattern.shape == "/"
    assert traces["Forward - GPU"].marker.pattern.shape == ""
    assert traces["Forward - CPU"].marker.opacity < traces["Forward - GPU"].marker.opacity
    assert traces["Forward - CPU"].width[0] == 0.36
    assert traces["Checkpoint CPU serialization - CPU"].width[0] == 0.36
    assert traces["Forward - GPU"].width[0] == 0.68
    assert traces["Forward - CPU"].y[0] != traces["Checkpoint CPU serialization - CPU"].y[0]
    assert list(figure.layout.yaxis.ticktext) == [
        "tiny_gpt / rank 0 | CPU [checkpoint | training]",
        "tiny_gpt / rank 0 | GPU",
    ]
    assert traces["Forward - CPU"].customdata[0][13] == "training half"
    assert traces["Checkpoint CPU serialization - CPU"].customdata[0][13] == (
        "checkpoint half"
    )


def test_operator_trace_summary_streams_cpu_gpu_and_communication(tmp_path):
    trace = tmp_path / "operator-trace.json"
    trace.write_text(
        json.dumps(
            {
                "baseTimeNanoseconds": 1_000_000_000,
                "traceEvents": [
                    {
                        "ph": "X",
                        "cat": "cpu_op",
                        "name": "aten::mm",
                        "ts": 10,
                        "dur": 100,
                    },
                    {"ph": "X", "cat": "kernel", "name": "gemm", "ts": 20, "dur": 80},
                    {
                        "ph": "X",
                        "cat": "kernel",
                        "name": "ncclAllReduce",
                        "ts": 30,
                        "dur": 20,
                    },
                    {
                        "ph": "X",
                        "cat": "gpu_memcpy",
                        "name": "Memcpy DtoH",
                        "ts": 40,
                        "dur": 10,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    summary = summarize_operator_trace(trace)

    assert summary["counts"] == {"cpu": 1, "gpu": 3, "communication": 1}
    assert summary["cpu"]["aten::mm"] == 100
    assert summary["communication"]["ncclAllReduce"] == 20
    assert summary["timeline_events"][0]["start_ns"] == 1_000_010_000
    assert {event["resource"] for event in summary["timeline_events"]} == {
        "CPU",
        "GPU",
        "Communication",
    }
    assert summary["checkpoint_copies"][0]["name"] == "Memcpy DtoH"
    assert summary["checkpoint_copies"][0]["category"] == "gpu_memcpy"


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
    with trace.span("Training", "Warmup", step=1):
        torch.ones(2) + 1
    trace.start_profiler(completed_warmup_steps=3)
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
    assert '"operation": "Warmup"' not in resource_events
    assert '"schema_version": 2' in resource_events
    assert '"start_alignment": "observed_wall_clock"' in resource_events
    assert '"profiler_cpu_total_ms":' in resource_events
    assert '"event": "span"' in (worker_dir / "metrics.jsonl").read_text(
        encoding="utf-8"
    )
    assert '"event": "step"' in (worker_dir / "metrics.jsonl").read_text(
        encoding="utf-8"
    )
    assert '"completed_warmup_steps": 3' in (
        worker_dir / "metrics.jsonl"
    ).read_text(encoding="utf-8")


def test_resource_trace_records_gpu_timing_provenance(tmp_path):
    class FakeCudaEvent:
        def elapsed_time(self, _other):
            return 4.25

    phase_event = SimpleNamespace(
        name="job_span/0",
        cpu_time_total=5100,
        self_cpu_time_total=900,
        device_time_total=3900,
        kernels=["kernel-a", "kernel-b"],
        cpu_children=[],
    )
    trace = TrainingTrace.__new__(TrainingTrace)
    trace.output_dir = tmp_path
    trace.host = "node"
    trace.job = "tiny_gpt"
    trace.rank = 0
    trace.world_size = 1
    trace.profiler = SimpleNamespace(events=lambda: [phase_event])
    trace.span_metadata = {
        0: {
            "category": "Training",
            "operation": "Forward",
            "start_ns": 1_000_000_000,
            "end_ns": 1_006_000_000,
            "status": "ok",
            "step": 1,
        }
    }
    trace.cuda_span_events = {0: (FakeCudaEvent(), FakeCudaEvent())}

    trace._export_resource_trace()

    events = [
        json.loads(line)
        for line in (tmp_path / "resource-trace.jsonl").read_text().splitlines()
    ]
    cpu, gpu = events
    assert cpu["schema_version"] == 2
    assert cpu["phase"] == "Forward"
    assert cpu["profiler_cpu_total_ms"] == 5.1
    assert cpu["start_is_estimated"] is False
    assert gpu["gpu_stream_elapsed_ms"] == 4.25
    assert gpu["profiler_device_total_ms"] == 3.9
    assert gpu["device_kernel_count"] == 2
    assert gpu["resource_end_ns"] == 1_004_250_000
    assert gpu["start_alignment"] == "enclosing_cpu_phase_start"
    assert gpu["start_is_estimated"] is True


def test_resource_trace_can_emit_storage_without_a_synthetic_gpu_span(tmp_path):
    trace = TrainingTrace.__new__(TrainingTrace)
    trace.output_dir = tmp_path
    trace.host = "node"
    trace.job = "tiny_gpt"
    trace.rank = 0
    trace.world_size = 1
    trace.profiler = SimpleNamespace(events=list)
    trace.span_metadata = {
        0: {
            "category": "Checkpoint",
            "operation": "Checkpoint DRAM to SSD write",
            "start_ns": 1_000_000_000,
            "end_ns": 1_010_000_000,
            "status": "ok",
            "resource_override": "Storage",
            "resource_measurement": "checkpoint_storage_in_flight_wall_clock",
            "emit_gpu_resource": False,
        }
    }
    trace.cuda_span_events = {}

    trace._export_resource_trace()

    event = json.loads((tmp_path / "resource-trace.jsonl").read_text())
    assert event["resource"] == "Storage"
    assert event["duration_ms"] == 10
    assert event["cpu_wall_ms"] is None
    assert event["measurement"] == "checkpoint_storage_in_flight_wall_clock"


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

    upload_visualization(
        tmp_path, "traces", "atharva-instace_20260728_191530", client=FakeClient()
    )

    assert uploaded[0][0] == "atharva-instace_20260728_191530/index.html"
    assert uploaded[0][2:] == ("text/html", "inline", "no-cache")
    assert uploaded[1][0].endswith("collected/node/rank-0/operator-trace.json")
