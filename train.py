import argparse
import importlib
import json
import os
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
            self.span_metadata[span_id] = {
                "category": category,
                "operation": operation,
                "start_ns": start_ns,
                "end_ns": end_ns,
                "status": "error" if error else "ok",
                **values,
            }
            self.log(
                "span",
                span_id=span_id,
                category=category,
                operation=operation,
                start_ns=start_ns,
                end_ns=end_ns,
                duration_ms=(end_ns - start_ns) / 1_000_000,
                status="error" if error else "ok",
                error=error,
                **values,
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
        output = self.output_dir / "resource-trace.jsonl"
        with output.open("w", encoding="utf-8") as resource_file:
            for span_id, metadata in self.span_metadata.items():
                cuda_events = self.cuda_span_events.get(span_id)
                common = {
                    "event": "resource_span",
                    "span_id": span_id,
                    "host": self.host,
                    "job": self.job,
                    "rank": self.rank,
                    "world_size": self.world_size,
                    **metadata,
                }
                cpu = {
                    **common,
                    "resource": "CPU",
                    "duration_ms": (metadata["end_ns"] - metadata["start_ns"])
                    / 1_000_000,
                    "measurement": "wall_clock",
                }
                resource_file.write(json.dumps(cpu, sort_keys=True) + "\n")
                if cuda_events is not None:
                    gpu_duration_ms = float(cuda_events[0].elapsed_time(cuda_events[1]))
                    gpu = {
                        **common,
                        "resource": "GPU",
                        "duration_ms": gpu_duration_ms,
                        "measurement": "cuda_event_elapsed",
                    }
                    resource_file.write(json.dumps(gpu, sort_keys=True) + "\n")


class RandomTokenDataset(Dataset):
    def __init__(self, samples, seq_len, vocab_size):
        self.samples = samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self):
        return self.samples

    def __getitem__(self, _):
        tokens = torch.randint(0, self.vocab_size, (self.seq_len,), dtype=torch.long)
        return {"input_ids": tokens, "labels": tokens.clone()}


def load_model(model_name, **kwargs):
    module_path = f"llm_models.{model_name}"
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        if exc.name == module_path:
            available = sorted(p.stem for p in Path("llm_models").glob("*.py") if p.name != "__init__.py")
            raise SystemExit(f"Unknown model '{model_name}'. Available models: {available}") from exc
        raise

    if not hasattr(module, "build_model"):
        raise SystemExit(f"{module_path} must define build_model(**kwargs)")
    return module.build_model(**kwargs)


def parse_args():
    import deepspeed

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="tiny_gpt", help="Loads llm_models/<model_name>.py")
    parser.add_argument("--job_name", help="Trace label for this distributed training job")
    parser.add_argument("--output_dir", default="checkpoints")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--vocab_size", type=int, default=50304)
    parser.add_argument("--dataset_samples", type=int, default=10000)
    parser.add_argument("--trace", action="store_true", help="Collect full per-rank CPU/CUDA traces")
    parser.add_argument("--trace_dir", default="visualization/traces")
    parser.add_argument("--run_id", default=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", "-1")))
    parser = deepspeed.add_config_arguments(parser)
    parser.set_defaults(deepspeed_config="ds_config_zero3.json")
    return parser.parse_args()


def initialize_engine(deepspeed, args, model, dataset):
    return deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        training_data=dataset,
    )


def main():
    import deepspeed

    args = parse_args()
    deepspeed.init_distributed()

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(args.local_rank)

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
            "save_every": args.save_every,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
    )

    error = None
    try:
        with trace.span("Setup", "Model and DeepSpeed initialization"):
            model = load_model(args.model_name, vocab_size=args.vocab_size, seq_len=args.seq_len)
            dataset = RandomTokenDataset(args.dataset_samples, args.seq_len, args.vocab_size)
            engine, _, _, _ = initialize_engine(deepspeed, args, model, dataset)

        engine.train()
        data_iter = iter(engine.training_dataloader)

        for step in range(1, args.steps + 1):
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
            with trace.span("Training", "Backward", step=step):
                engine.backward(loss)
            with trace.span("Optimizer", "Optimizer step", step=step):
                engine.step()

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
                    cuda_allocated_bytes=torch.cuda.memory_allocated(engine.device),
                    cuda_reserved_bytes=torch.cuda.memory_reserved(engine.device),
                    cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(engine.device),
                )

            if rank == 0 and step % 10 == 0:
                print(f"step={step} loss={loss.item():.4f}")

            if step % args.save_every == 0:
                checkpoint_started = time.perf_counter()
                tag = f"step-{step}"
                trace.log("checkpoint_start", step=step, tag=tag, output_dir=args.output_dir)
                with trace.span("Checkpoint", "Save checkpoint", step=step, tag=tag):
                    engine.save_checkpoint(args.output_dir, tag=tag)
                trace.log(
                    "checkpoint_complete",
                    step=step,
                    tag=tag,
                    output_dir=args.output_dir,
                    duration_ms=(time.perf_counter() - checkpoint_started) * 1000,
                )

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
