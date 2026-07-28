import argparse
import importlib
import json
import os
import platform
import random
import resource
import socket
import time
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import Dataset


class TrainingTrace:
    def __init__(self, enabled, trace_dir, run_id, rank, world_size, metadata):
        self.enabled = enabled
        self.profiler = None
        self.events_file = None
        self.host = socket.gethostname().split(".")[0]
        self.rank = rank
        self.world_size = world_size
        self.job = metadata.get("job", metadata.get("model", "training"))
        self.span_metadata = {}
        self.cuda_span_events = {}
        self.next_span_id = 0
        self.cuda_timing = enabled and torch.cuda.is_available()
        if not enabled:
            return

        self.output_dir = Path(trace_dir) / run_id / self.host / f"rank-{rank}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.events_file = (self.output_dir / "metrics.jsonl").open("a", encoding="utf-8", buffering=1)

        observer = torch.profiler.ExecutionTraceObserver().register_callback(
            str(self.output_dir / "execution-trace.json")
        )
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        self.profiler = torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
            execution_trace_observer=observer,
        )
        self.profiler.start()
        self.profiler.add_metadata_json("run", json.dumps(metadata))
        self.log("run_start", rank=rank, world_size=world_size, **metadata)

    def region(self, name):
        if not self.enabled:
            return nullcontext()
        return torch.profiler.record_function(name)

    @contextmanager
    def span(self, category, operation, **values):
        if not self.enabled:
            yield
            return

        start_ns = time.time_ns()
        span_id = self.next_span_id
        self.next_span_id += 1
        cuda_start = None
        cuda_end = None
        if self.cuda_timing:
            cuda_start = torch.cuda.Event(enable_timing=True)
            cuda_end = torch.cuda.Event(enable_timing=True)
            cuda_start.record()
        error = None
        try:
            with self.region(f"job_span/{span_id}"):
                yield
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if cuda_end is not None:
                cuda_end.record()
                self.cuda_span_events[span_id] = (cuda_start, cuda_end)
            end_ns = time.time_ns()
            self.record_span(
                category,
                operation,
                start_ns,
                end_ns,
                span_id=span_id,
                status="error" if error else "ok",
                error=error,
                **values,
            )

    def record_span(self, category, operation, start_ns, end_ns, span_id=None, **values):
        if not self.enabled:
            return
        if span_id is None:
            span_id = self.next_span_id
            self.next_span_id += 1
        metadata = {
            "category": category,
            "operation": operation,
            "start_ns": start_ns,
            "end_ns": end_ns,
            "status": values.pop("status", "ok"),
            **values,
        }
        self.span_metadata[span_id] = metadata
        self.log(
            "span",
            span_id=span_id,
            duration_ms=(end_ns - start_ns) / 1_000_000,
            **metadata,
        )

    def log(self, event, **values):
        if not self.enabled:
            return
        payload = {
            "event": event,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "host": self.host,
            "job": self.job,
            "rank": self.rank,
            "world_size": self.world_size,
            **values,
        }
        self.events_file.write(json.dumps(payload, sort_keys=True) + "\n")

    def stop(self, error=None):
        if not self.enabled:
            return
        self.log("run_end" if error is None else "run_error", error=error)
        try:
            self.profiler.stop()
            self.profiler.export_chrome_trace(str(self.output_dir / "operator-trace.json"))
            if self.cuda_timing:
                torch.cuda.synchronize()
            self._export_resource_trace()
        finally:
            self.events_file.close()

    def _export_resource_trace(self):
        def kernel_count(event):
            if event is None:
                return None
            count = 0
            pending = [event]
            while pending:
                current = pending.pop()
                count += len(current.kernels)
                pending.extend(current.cpu_children)
            return count

        profiler_spans = {}
        try:
            for event in self.profiler.events():
                if not event.name.startswith("job_span/"):
                    continue
                span_id = int(event.name.removeprefix("job_span/"))
                profiler_spans[span_id] = event
        except (AttributeError, RuntimeError, ValueError):
            pass

        output = self.output_dir / "resource-trace.jsonl"
        with output.open("w", encoding="utf-8") as resource_file:
            for span_id, metadata in self.span_metadata.items():
                cuda_events = self.cuda_span_events.get(span_id)
                profiler_span = profiler_spans.get(span_id)
                cpu_wall_ms = (metadata["end_ns"] - metadata["start_ns"]) / 1_000_000
                common = {
                    "schema_version": 2,
                    "event": "resource_span",
                    "span_id": span_id,
                    "phase": metadata["operation"],
                    "host": self.host,
                    "job": self.job,
                    "rank": self.rank,
                    "world_size": self.world_size,
                    **metadata,
                }
                primary_resource = metadata.get("resource_override", "CPU")
                primary = {
                    **common,
                    "resource": primary_resource,
                    "resource_start_ns": metadata["start_ns"],
                    "resource_end_ns": metadata["end_ns"],
                    "duration_ms": cpu_wall_ms,
                    "cpu_wall_ms": (
                        cpu_wall_ms if primary_resource == "CPU" else None
                    ),
                    "profiler_cpu_total_ms": (
                        float(profiler_span.cpu_time_total) / 1000
                        if profiler_span is not None and primary_resource == "CPU"
                        else None
                    ),
                    "profiler_self_cpu_ms": (
                        float(profiler_span.self_cpu_time_total) / 1000
                        if profiler_span is not None and primary_resource == "CPU"
                        else None
                    ),
                    "measurement": metadata.get(
                        "resource_measurement", "phase_wall_clock"
                    ),
                    "start_alignment": "observed_wall_clock",
                    "start_is_estimated": False,
                }
                resource_file.write(json.dumps(primary, sort_keys=True) + "\n")
                if cuda_events is not None and metadata.get(
                    "emit_gpu_resource", True
                ):
                    gpu_duration_ms = float(cuda_events[0].elapsed_time(cuda_events[1]))
                    gpu_start_ns = metadata["start_ns"]
                    gpu = {
                        **common,
                        "resource": "GPU",
                        "resource_start_ns": gpu_start_ns,
                        "resource_end_ns": gpu_start_ns + int(gpu_duration_ms * 1_000_000),
                        "duration_ms": gpu_duration_ms,
                        "gpu_stream_elapsed_ms": gpu_duration_ms,
                        "profiler_device_total_ms": (
                            float(profiler_span.device_time_total) / 1000
                            if profiler_span is not None
                            else None
                        ),
                        "device_kernel_count": kernel_count(profiler_span),
                        "measurement": "cuda_stream_elapsed",
                        "start_alignment": "enclosing_cpu_phase_start",
                        "start_is_estimated": True,
                    }
                    resource_file.write(json.dumps(gpu, sort_keys=True) + "\n")


class RandomTokenDataset(Dataset):
    def __init__(self, samples, seq_len, vocab_size, seed=1234):
        self.samples = samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.seed = seed

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(self.seed + int(index))
        tokens = torch.randint(
            0,
            self.vocab_size,
            (self.seq_len,),
            dtype=torch.long,
            generator=generator,
        )
        return {"input_ids": tokens, "labels": tokens.clone()}


def system_metrics(device):
    usage = resource.getrusage(resource.RUSAGE_SELF)
    values = {
        "host_load_1m": os.getloadavg()[0],
        "process_rss_bytes": int(usage.ru_maxrss) * 1024,
    }
    cuda_metrics = {
        "utilization": "cuda_utilization_percent",
        "memory_usage": "cuda_memory_usage_percent",
        "temperature": "cuda_temperature_c",
        "power_draw": "cuda_power_draw_mw",
        "clock_rate": "cuda_clock_mhz",
    }
    for function_name, output_name in cuda_metrics.items():
        function = getattr(torch.cuda, function_name, None)
        if function is None:
            continue
        try:
            values[output_name] = function(device)
        except (ImportError, ModuleNotFoundError, RuntimeError):
            continue
    return values


def directory_stats(path):
    try:
        file_count = 0
        total_bytes = 0
        for file in Path(path).rglob("*"):
            if file.is_file():
                file_count += 1
                total_bytes += file.stat().st_size
        return file_count, total_bytes
    except OSError:
        return 0, 0


def process_io_counters(pid):
    if not pid:
        return {}
    try:
        return {
            key.rstrip(":"): int(value)
            for key, value in (
                line.split(maxsplit=1)
                for line in Path(f"/proc/{pid}/io")
                .read_text(encoding="utf-8")
                .splitlines()
            )
        }
    except (OSError, ValueError):
        return {}


def _default_network_interface():
    try:
        for line in Path("/proc/net/route").read_text(encoding="utf-8").splitlines()[1:]:
            fields = line.split()
            if len(fields) > 1 and fields[1] == "00000000":
                return fields[0]
    except OSError:
        return None
    return None


def _cpu_model():
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _storage_benchmark(directory, size_mb, host):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f".trace-storage-benchmark-{host}"
    total_bytes = max(1, size_mb) * 1024**2
    block = bytes(min(4 * 1024**2, total_bytes))
    try:
        started = time.perf_counter()
        with path.open("wb", buffering=0) as output:
            remaining = total_bytes
            while remaining:
                chunk = block[: min(len(block), remaining)]
                output.write(chunk)
                remaining -= len(chunk)
            os.fsync(output.fileno())
        write_seconds = time.perf_counter() - started

        started = time.perf_counter()
        with path.open("rb", buffering=0) as source:
            while source.read(len(block)):
                pass
        read_seconds = time.perf_counter() - started
        return {
            "storage_benchmark_bytes": total_bytes,
            "storage_write_gbps": total_bytes / max(write_seconds, 1e-9) / 1e9,
            "storage_read_gbps": total_bytes / max(read_seconds, 1e-9) / 1e9,
        }
    except OSError as exc:
        return {"storage_benchmark_error": f"{type(exc).__name__}: {exc}"}
    finally:
        path.unlink(missing_ok=True)


def _gpu_bandwidth_benchmark(device, size_mb):
    total_bytes = max(1, size_mb) * 1024**2
    elements = total_bytes // 4
    iterations = 3
    host_source = torch.empty(elements, dtype=torch.float32, pin_memory=True)
    host_destination = torch.empty_like(host_source, pin_memory=True)
    device_source = torch.empty(elements, dtype=torch.float32, device=device)
    device_destination = torch.empty_like(device_source)

    def measure(operation):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        for _ in range(iterations):
            operation()
        torch.cuda.synchronize(device)
        return total_bytes * iterations / max(time.perf_counter() - started, 1e-9) / 1e9

    values = {
        "host_to_gpu_gbps": measure(
            lambda: device_source.copy_(host_source, non_blocking=True)
        ),
        "gpu_to_host_gbps": measure(
            lambda: host_destination.copy_(device_source, non_blocking=True)
        ),
        "gpu_dram_copy_gbps": measure(lambda: device_destination.copy_(device_source)),
        "gpu_bandwidth_benchmark_bytes": total_bytes,
    }
    return values


def distributed_bandwidth_benchmark(device, size_mb):
    world_size = torch.distributed.get_world_size()
    if world_size <= 1:
        return {}
    total_bytes = max(1, size_mb) * 1024**2
    tensor = torch.ones(total_bytes // 4, dtype=torch.float32, device=device)
    iterations = 3
    torch.distributed.all_reduce(tensor)
    torch.cuda.synchronize(device)
    torch.distributed.barrier()
    started = time.perf_counter()
    for _ in range(iterations):
        torch.distributed.all_reduce(tensor)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    average_seconds = elapsed / iterations
    payload_gbps = total_bytes / max(average_seconds, 1e-9) / 1e9
    return {
        "all_reduce_benchmark_bytes": total_bytes,
        "all_reduce_average_ms": average_seconds * 1000,
        "all_reduce_payload_gbps": payload_gbps,
        "all_reduce_bus_gbps": payload_gbps * 2 * (world_size - 1) / world_size,
    }


def hardware_profile(device, storage_dir, benchmark_mb, run_benchmarks):
    host = socket.gethostname().split(".")[0]
    interface = _default_network_interface()
    Path(storage_dir).mkdir(parents=True, exist_ok=True)
    storage = os.statvfs(storage_dir)
    memory_kib = 0
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                memory_kib = int(line.split()[1])
                break
    except OSError:
        pass

    properties = torch.cuda.get_device_properties(device)
    values = {
        "cpu_count": os.cpu_count(),
        "cpu_model": _cpu_model(),
        "host_memory_bytes": memory_kib * 1024,
        "network_interface": interface,
        "storage_path": str(Path(storage_dir).resolve()),
        "storage_capacity_bytes": storage.f_blocks * storage.f_frsize,
        "storage_free_bytes": storage.f_bavail * storage.f_frsize,
        "gpu_name": properties.name,
        "gpu_total_memory_bytes": properties.total_memory,
        "gpu_compute_capability": f"{properties.major}.{properties.minor}",
        "gpu_multiprocessor_count": properties.multi_processor_count,
    }
    if interface:
        try:
            link_speed = int(
                Path(f"/sys/class/net/{interface}/speed").read_text(encoding="utf-8").strip()
            )
            if link_speed > 0:
                values["nic_link_speed_mbps"] = link_speed
        except (OSError, ValueError):
            pass

    memory_clock_khz = getattr(properties, "memory_clock_rate", None)
    memory_bus_bits = getattr(properties, "memory_bus_width", None)
    if memory_clock_khz and memory_bus_bits:
        values["gpu_dram_theoretical_gbps"] = (
            memory_clock_khz * 1000 * (memory_bus_bits / 8) * 2 / 1e9
        )
    if run_benchmarks:
        values.update(_storage_benchmark(Path(storage_dir), benchmark_mb, host))
        values.update(_gpu_bandwidth_benchmark(device, benchmark_mb))
    return values


def load_model_module(model_name):
    module_path = f"llm_models.{model_name}"
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        if exc.name == module_path:
            available = sorted(p.stem for p in Path("llm_models").glob("*.py") if p.name != "__init__.py")
            raise SystemExit(f"Unknown model '{model_name}'. Available models: {available}") from exc
        raise

    return module


def load_model(model_name, **kwargs):
    module = load_model_module(model_name)
    module_path = module.__name__
    if not hasattr(module, "build_model"):
        raise SystemExit(f"{module_path} must define build_model(**kwargs)")
    return module.build_model(**kwargs)


def load_dataset(model_name, **kwargs):
    module = load_model_module(model_name)
    if hasattr(module, "build_dataset"):
        dataset = module.build_dataset(**kwargs)
    else:
        dataset = RandomTokenDataset(**kwargs)
    if not isinstance(dataset, Dataset):
        raise TypeError(
            f"{module.__name__}.build_dataset(**kwargs) must return a torch Dataset"
        )
    return dataset


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    import deepspeed

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="tiny_gpt", help="Loads llm_models/<model_name>.py")
    parser.add_argument("--job_name", help="Trace label for this distributed training job")
    parser.add_argument("--output_dir", default="checkpoints")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--resume",
        help="Resume from 'latest' or a checkpoint tag such as step-100",
    )
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--vocab_size", type=int, default=50304)
    parser.add_argument("--dataset_samples", type=int, default=10000)
    parser.add_argument("--hardware_benchmark_mb", type=int, default=64)
    parser.add_argument("--trace", action="store_true", help="Collect full per-rank CPU/CUDA traces")
    parser.add_argument("--trace_dir", default="visualization/traces")
    parser.add_argument("--run_id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", "-1")))
    parser = deepspeed.add_config_arguments(parser)
    parser.set_defaults(deepspeed_config="ds_config_zero3.json")
    return parser.parse_args()


def checkpoint_deepspeed_config(checkpoint_mode, batch_size=None):
    if checkpoint_mode not in {"synchronous", "asynchronous"}:
        raise ValueError(f"Unsupported checkpoint mode: {checkpoint_mode}")
    config_path = Path(__file__).with_name("ds_config_zero3.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if batch_size is not None:
        config["train_micro_batch_size_per_gpu"] = batch_size
    config["gradient_accumulation_steps"] = 1
    if checkpoint_mode == "synchronous":
        return config
    config["checkpoint"] = {
        "tag_validation": "WARN",
        "checkpoint_serialization": True,
        "writer": {
            "type": "PYTHON",
            "decoupled": True,
            "data_parallel": "REPLICA",
            "show_statistics": True,
        },
    }
    return config


def resume_from_checkpoint(engine, output_dir, resume):
    requested_tag = None if resume == "latest" else resume
    load_path, client_state = engine.load_checkpoint(output_dir, tag=requested_tag)
    if load_path is None:
        raise FileNotFoundError(
            f"Unable to load checkpoint {resume!r} from {output_dir}"
        )
    client_state = client_state or {}
    completed_steps = client_state.get("completed_steps")
    if completed_steps is None:
        candidates = [resume, *Path(load_path).parts]
        step_tag = next(
            (
                value
                for value in reversed(candidates)
                if isinstance(value, str) and value.startswith("step-")
            ),
            None,
        )
        if step_tag is not None and step_tag.removeprefix("step-").isdigit():
            completed_steps = int(step_tag.removeprefix("step-"))
    if completed_steps is None:
        raise ValueError(
            "Checkpoint has no completed_steps client state or step-N tag"
        )
    completed_steps = int(completed_steps)
    if completed_steps < 0:
        raise ValueError("Checkpoint completed_steps cannot be negative")
    return completed_steps, load_path


def initialize_engine(deepspeed, args, model, dataset, config=None):
    initialize_args = {
        "args": args,
        "model": model,
        "model_parameters": model.parameters(),
        "training_data": dataset,
    }
    if config is not None:
        initialize_args["config"] = config
    return deepspeed.initialize(**initialize_args)


def finish_async_checkpoint(engine, trace, pending, output_dir, rank, force=False):
    if pending is None:
        return None
    commit_info = engine.checkpoint_engine.get_commit_info()
    if commit_info is not None and not force:
        return pending
    if commit_info is not None:
        wait_start_ns = time.time_ns()
        engine._commit_decoupled_checkpoint()
        wait_end_ns = time.time_ns()
        trace.record_span(
            "Checkpoint",
            "Async checkpoint commit wait",
            wait_start_ns,
            wait_end_ns,
            step=pending["step"],
            tag=pending["tag"],
            checkpoint_mode="asynchronous",
            checkpoint_worker_pid=pending["worker_pid"],
            resource_measurement="checkpoint_commit_wait_wall_clock",
        )

    completed_ns = time.time_ns()
    persistence_start_ns = pending["staged_ns"]
    total_duration_ms = (completed_ns - pending["request_ns"]) / 1_000_000
    persistence_duration_ms = (completed_ns - persistence_start_ns) / 1_000_000
    checkpoint_files, checkpoint_bytes = (
        directory_stats(Path(output_dir) / pending["tag"]) if rank == 0 else (0, 0)
    )
    completed_io = process_io_counters(pending["worker_pid"])
    storage_write_bytes = None
    initial_io = pending.get("worker_io", {})
    if "write_bytes" in initial_io and "write_bytes" in completed_io:
        storage_write_bytes = max(
            0,
            completed_io["write_bytes"] - initial_io["write_bytes"],
        )
    checkpoint_throughput_mib_s = (
        checkpoint_bytes / 1024**2 / (persistence_duration_ms / 1000)
        if persistence_duration_ms > 0
        else 0
    )
    background_values = {
        "step": pending["step"],
        "tag": pending["tag"],
        "checkpoint_mode": "asynchronous",
        "checkpoint_worker_pid": pending["worker_pid"],
        "checkpoint_size_bytes": checkpoint_bytes,
        "checkpoint_storage_write_bytes": storage_write_bytes,
        "checkpoint_throughput_mib_s": checkpoint_throughput_mib_s,
    }
    trace.record_span(
        "Checkpoint",
        "Checkpoint CPU serialization",
        persistence_start_ns,
        completed_ns,
        resource_measurement="background_checkpoint_in_flight_wall_clock",
        **background_values,
    )
    trace.record_span(
        "Checkpoint",
        "Checkpoint DRAM to SSD write",
        persistence_start_ns,
        completed_ns,
        resource_override="Storage",
        resource_measurement="checkpoint_storage_in_flight_wall_clock",
        **background_values,
    )
    trace.log(
        "checkpoint_complete",
        step=pending["step"],
        tag=pending["tag"],
        output_dir=output_dir,
        checkpoint_mode="asynchronous",
        checkpoint_worker_pid=pending["worker_pid"],
        duration_ms=total_duration_ms,
        staging_duration_ms=pending["staging_duration_ms"],
        persistence_duration_ms=persistence_duration_ms,
        checkpoint_file_count=checkpoint_files,
        checkpoint_size_bytes=checkpoint_bytes,
        checkpoint_storage_write_bytes=storage_write_bytes,
        checkpoint_throughput_mib_s=checkpoint_throughput_mib_s,
    )
    return None


def start_async_checkpoint(engine, trace, step, output_dir):
    tag = f"step-{step}"
    request_ns = time.time_ns()
    trace.log(
        "checkpoint_start",
        step=step,
        tag=tag,
        output_dir=output_dir,
        checkpoint_mode="asynchronous",
    )
    with trace.span(
        "Checkpoint",
        "Checkpoint state snapshot to host DRAM",
        step=step,
        tag=tag,
        checkpoint_mode="asynchronous",
        resource_measurement="checkpoint_state_snapshot_wall_clock",
        emit_gpu_resource=False,
    ):
        engine.save_checkpoint(
            output_dir,
            tag=tag,
            client_state={"completed_steps": step},
        )
    staged_ns = time.time_ns()
    process = getattr(engine.checkpoint_engine, "ckpt_process", None)
    worker_pid = getattr(process, "pid", None)
    staging_duration_ms = (staged_ns - request_ns) / 1_000_000
    trace.log(
        "checkpoint_queued",
        step=step,
        tag=tag,
        output_dir=output_dir,
        checkpoint_mode="asynchronous",
        checkpoint_worker_pid=worker_pid,
        staging_duration_ms=staging_duration_ms,
    )
    return {
        "step": step,
        "tag": tag,
        "request_ns": request_ns,
        "staged_ns": staged_ns,
        "staging_duration_ms": staging_duration_ms,
        "worker_pid": worker_pid,
        "worker_io": process_io_counters(worker_pid),
    }


def main(checkpoint_mode="synchronous"):
    import deepspeed

    args = parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.save_every < 1:
        raise ValueError("steps, batch_size, and save_every must all be positive")
    deepspeed_config = checkpoint_deepspeed_config(
        checkpoint_mode,
        batch_size=args.batch_size,
    )
    args.deepspeed_config = None
    deepspeed.init_distributed()

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(args.local_rank)
    seed_everything(args.seed)
    global_batch_size = args.batch_size * world_size

    trace = TrainingTrace(
        args.trace,
        args.trace_dir,
        args.run_id,
        rank,
        world_size,
        {
            "job": args.job_name or args.model_name,
            "model": args.model_name,
            "steps": args.steps,
            "batch_size_per_gpu": args.batch_size,
            "global_batch_size": global_batch_size,
            "gradient_accumulation_steps": 1,
            "seed": args.seed,
            "resume": args.resume,
            "save_every": args.save_every,
            "checkpoint_mode": checkpoint_mode,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
    )

    error = None
    start_step = 1
    resume_path = None
    try:
        with trace.span("Setup", "Model and DeepSpeed initialization"):
            model = load_model(args.model_name, vocab_size=args.vocab_size, seq_len=args.seq_len)
            parameter_count = sum(parameter.numel() for parameter in model.parameters())
            trainable_parameter_count = sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            )
            parameter_bytes = sum(
                parameter.numel() * parameter.element_size() for parameter in model.parameters()
            )
            dataset = load_dataset(
                args.model_name,
                samples=args.dataset_samples,
                seq_len=args.seq_len,
                vocab_size=args.vocab_size,
                seed=args.seed,
            )
            engine, _, _, _ = initialize_engine(
                deepspeed,
                args,
                model,
                dataset,
                config=deepspeed_config,
            )
            if checkpoint_mode == "asynchronous" and not engine.checkpoint_engine.is_decoupled():
                raise RuntimeError("DeepSpeed did not initialize its decoupled checkpoint engine")
            seed_everything(args.seed + rank)

            if args.resume:
                with trace.span(
                    "Setup",
                    "Load checkpoint",
                    requested_checkpoint=args.resume,
                ):
                    completed_steps, resume_path = resume_from_checkpoint(
                        engine,
                        args.output_dir,
                        args.resume,
                    )
                start_step = completed_steps + 1
                trace.log(
                    "checkpoint_loaded",
                    requested_checkpoint=args.resume,
                    checkpoint_path=resume_path,
                    completed_steps=completed_steps,
                    next_step=start_step,
                )

            if args.trace:
                torch.distributed.barrier()
                profile = hardware_profile(
                    engine.device,
                    args.output_dir,
                    args.hardware_benchmark_mb,
                    run_benchmarks=args.local_rank == 0,
                )
                profile.update(
                    distributed_bandwidth_benchmark(
                        engine.device,
                        args.hardware_benchmark_mb,
                    )
                )
                trace.log(
                    "hardware_profile",
                    local_rank=args.local_rank,
                    model_parameter_count=parameter_count,
                    model_trainable_parameter_count=trainable_parameter_count,
                    model_parameter_bytes=parameter_bytes,
                    gradient_bytes_estimate=parameter_bytes,
                    sequence_length=args.seq_len,
                    vocabulary_size=args.vocab_size,
                    dataset_samples=args.dataset_samples,
                    dataset_type=type(dataset).__name__,
                    synthetic_dataset=isinstance(dataset, RandomTokenDataset),
                    batch_size_per_gpu=args.batch_size,
                    global_batch_size=global_batch_size,
                    gradient_accumulation_steps=1,
                    **profile,
                )
                torch.distributed.barrier()

        engine.train()
        data_iter = iter(engine.training_dataloader)
        pending_checkpoint = None

        if rank == 0:
            print(
                f"training model={args.model_name} steps={start_step}-{args.steps} "
                f"batch_per_gpu={args.batch_size} global_batch={global_batch_size} "
                "gradient_accumulation=1"
            )

        for step in range(start_step, args.steps + 1):
            if args.trace:
                torch.cuda.synchronize(engine.device)
            step_started = time.perf_counter()
            step_started_ns = time.time_ns()

            with trace.span("Input", "Data loading", step=step):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(engine.training_dataloader)
                    batch = next(data_iter)
                batch = {key: value.to(engine.device) for key, value in batch.items()}

            with trace.span("Training", "Forward", step=step):
                loss = engine(**batch)["loss"]
                if not bool(torch.isfinite(loss.detach()).all()):
                    raise FloatingPointError(f"Non-finite loss at step {step}")
            with trace.span("Training", "Backward", step=step):
                engine.backward(loss)
            with trace.span("Optimizer", "Optimizer step", step=step):
                engine.step()
            if checkpoint_mode == "asynchronous":
                pending_checkpoint = finish_async_checkpoint(
                    engine,
                    trace,
                    pending_checkpoint,
                    args.output_dir,
                    rank,
                )

            if args.trace:
                torch.cuda.synchronize(engine.device)
                trace.log(
                    "step",
                    step=step,
                    loss=loss.item(),
                    learning_rate=engine.get_lr()[0],
                    start_ns=step_started_ns,
                    end_ns=time.time_ns(),
                    duration_ms=(time.perf_counter() - step_started) * 1000,
                    batch_size_per_gpu=args.batch_size,
                    global_batch_size=global_batch_size,
                    optimizer_update=True,
                    cuda_allocated_bytes=torch.cuda.memory_allocated(engine.device),
                    cuda_reserved_bytes=torch.cuda.memory_reserved(engine.device),
                    cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(engine.device),
                    **system_metrics(engine.device),
                )

            if rank == 0 and step % 10 == 0:
                print(f"step={step} loss={loss.item():.4f}")

            if step % args.save_every == 0:
                if checkpoint_mode == "asynchronous":
                    pending_checkpoint = finish_async_checkpoint(
                        engine,
                        trace,
                        pending_checkpoint,
                        args.output_dir,
                        rank,
                        force=True,
                    )
                    pending_checkpoint = start_async_checkpoint(
                        engine,
                        trace,
                        step,
                        args.output_dir,
                    )
                else:
                    checkpoint_started = time.perf_counter()
                    checkpoint_started_ns = time.time_ns()
                    tag = f"step-{step}"
                    trace.log(
                        "checkpoint_start",
                        step=step,
                        tag=tag,
                        output_dir=args.output_dir,
                        checkpoint_mode="synchronous",
                    )
                    with trace.span(
                        "Checkpoint",
                        "Synchronous checkpoint save pipeline",
                        step=step,
                        tag=tag,
                        checkpoint_mode="synchronous",
                        resource_measurement="synchronous_checkpoint_wall_clock",
                        emit_gpu_resource=False,
                    ):
                        engine.save_checkpoint(
                            args.output_dir,
                            tag=tag,
                            client_state={"completed_steps": step},
                        )
                    checkpoint_completed_ns = time.time_ns()
                    checkpoint_duration_ms = (
                        time.perf_counter() - checkpoint_started
                    ) * 1000
                    checkpoint_files, checkpoint_bytes = (
                        directory_stats(Path(args.output_dir) / tag)
                        if rank == 0
                        else (0, 0)
                    )
                    trace.record_span(
                        "Checkpoint",
                        "Checkpoint DRAM to SSD write",
                        checkpoint_started_ns,
                        checkpoint_completed_ns,
                        step=step,
                        tag=tag,
                        checkpoint_mode="synchronous",
                        checkpoint_size_bytes=checkpoint_bytes,
                        resource_override="Storage",
                        resource_measurement=(
                            "synchronous_checkpoint_pipeline_wall_clock"
                        ),
                    )
                    trace.log(
                        "checkpoint_complete",
                        step=step,
                        tag=tag,
                        output_dir=args.output_dir,
                        checkpoint_mode="synchronous",
                        duration_ms=checkpoint_duration_ms,
                        checkpoint_file_count=checkpoint_files,
                        checkpoint_size_bytes=checkpoint_bytes,
                        checkpoint_throughput_mib_s=(
                            checkpoint_bytes / 1024**2 / (checkpoint_duration_ms / 1000)
                            if checkpoint_duration_ms > 0
                            else 0
                        ),
                    )

        if checkpoint_mode == "asynchronous":
            finish_async_checkpoint(
                engine,
                trace,
                pending_checkpoint,
                args.output_dir,
                rank,
                force=True,
            )
            engine.checkpoint_engine.cleanup()

        if rank == 0:
            print("done")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            trace.stop(error)
        finally:
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
