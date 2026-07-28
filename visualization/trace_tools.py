import argparse
import heapq
import html
import json
import math
import mimetypes
import os
import re
import shutil
import socket
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote

import ijson
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

CATEGORY_COLORS = {
    "Setup": "#64748b",
    "Input": "#d6a21d",
    "Training": "#2563eb",
    "Optimizer": "#0f8a72",
    "Checkpoint": "#e36a2e",
    "Failure": "#dc2626",
    "Other": "#7c3aed",
}
OPERATION_COLORS = {
    "Model and DeepSpeed initialization": "#64748b",
    "Data loading": "#d6a21d",
    "Forward": "#0072b2",
    "Backward": "#d55e00",
    "Optimizer step": "#009e73",
    "Save checkpoint": "#cc79a7",
    "Async checkpoint staging": "#b45309",
    "Async checkpoint persistence": "#7c3aed",
    "Async checkpoint commit wait": "#dc2626",
    "Full step": "#2563eb",
}
RESOURCE_PATTERNS = {
    "CPU": "/",
    "GPU": "",
    "Storage": ".",
    "Network": "|",
}
RESOURCE_OPACITY = {
    "CPU": 0.72,
    "GPU": 1.0,
    "Storage": 0.88,
    "Network": 0.88,
}
PLOT_CONFIG = {
    "responsive": True,
    "displaylogo": False,
    "scrollZoom": True,
}
COMMUNICATION_TERMS = (
    "nccl",
    "all_reduce",
    "allreduce",
    "all_gather",
    "allgather",
    "reduce_scatter",
    "broadcast",
    "c10d",
    "send",
    "recv",
)
RESOURCE_COLORS = {
    "CPU": "#2563eb",
    "GPU": "#0f8a72",
    "Storage": "#e36a2e",
    "Network": "#7c3aed",
}
COMMUNICATION_COLORS = {
    "All-reduce": "#7c3aed",
    "All-gather": "#2563eb",
    "Reduce-scatter": "#0f8a72",
    "Broadcast": "#d6a21d",
    "Send/receive": "#e36a2e",
    "Other collective": "#64748b",
}


def read_hosts(hostfile):
    hosts = []
    for raw_line in Path(hostfile).read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if line:
            hosts.append(line.split()[0])
    return list(dict.fromkeys(hosts))


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value)


def is_local_host(host):
    return host in {"localhost", "127.0.0.1", socket.gethostname(), socket.getfqdn()}


def collect_host(host, source, output, remote_dir):
    destination = output / safe_name(host)
    destination.mkdir(parents=True, exist_ok=True)
    if is_local_host(host):
        if not source.exists():
            raise FileNotFoundError(f"No local traces found at {source}")
        shutil.copytree(source, destination, dirs_exist_ok=True)
        return

    remote_source = (
        f"{host}:{remote_dir.rstrip('/')}/{source.as_posix().lstrip('./')}/."
    )
    subprocess.run(["scp", "-r", remote_source, str(destination)], check=True)


def collect(hostfile, source, output, remote_dir):
    hosts = read_hosts(hostfile)
    if not hosts:
        raise ValueError(f"No hosts found in {hostfile}")
    output.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=len(hosts)) as pool:
        futures = [
            pool.submit(collect_host, host, source, output, remote_dir)
            for host in hosts
        ]
        for future in futures:
            future.result()
    print(f"Collected traces from {len(hosts)} host(s) into {output}")


def _read_jsonl(path):
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _timestamp_ns(value):
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    parsed = parsed.astimezone(timezone.utc)
    return int(parsed.timestamp()) * 1_000_000_000 + parsed.microsecond * 1000


def _iso_from_ns(value):
    return datetime.fromtimestamp(int(value) / 1_000_000_000, timezone.utc).isoformat(
        timespec="microseconds"
    )


def summarize_operator_trace(path):
    summary = {
        "cpu": Counter(),
        "gpu": Counter(),
        "communication": Counter(),
        "counts": Counter(),
        "memory_peak_bytes": 0,
        "memory_events": 0,
        "timeline_events": [],
    }
    try:
        base_time_ns = None
        with path.open("rb") as trace_file:
            for prefix, event_type, value in ijson.parse(trace_file):
                if prefix == "baseTimeNanoseconds":
                    base_time_ns = int(value)
                if prefix == "traceEvents" and event_type == "start_array":
                    break

        timeline_heaps = {"CPU": [], "GPU": [], "Communication": []}
        sequence = 0
        with path.open("rb") as trace_file:
            for event in ijson.items(trace_file, "traceEvents.item"):
                name = str(event.get("name") or "unknown")
                category = str(event.get("cat") or "").lower()
                duration_us = float(event.get("dur") or 0)
                lower_name = name.lower()
                is_cpu = category == "cpu_op"
                is_gpu = category in {"kernel", "gpu_memcpy", "gpu_memset"}
                is_communication = category in {
                    "cpu_op",
                    "kernel",
                    "user_annotation",
                    "cuda_runtime",
                } and any(term in lower_name for term in COMMUNICATION_TERMS)
                if is_cpu:
                    summary["cpu"][name] += duration_us
                    summary["counts"]["cpu"] += 1
                if is_gpu:
                    summary["gpu"][name] += duration_us
                    summary["counts"]["gpu"] += 1
                if is_communication:
                    summary["communication"][name] += duration_us
                    summary["counts"]["communication"] += 1
                if event.get("ph") == "X" and (is_cpu or is_gpu or is_communication):
                    resource_name = (
                        "Communication"
                        if is_communication
                        else "GPU"
                        if is_gpu
                        else "CPU"
                    )
                    trace_timestamp_us = float(event.get("ts") or 0)
                    timeline_event = {
                        "name": name,
                        "resource": resource_name,
                        "duration_us": duration_us,
                        "start_ns": (
                            base_time_ns + int(trace_timestamp_us * 1000)
                            if base_time_ns is not None
                            else None
                        ),
                        "trace_timestamp_us": trace_timestamp_us,
                    }
                    item = (duration_us, sequence, timeline_event)
                    timeline_heap = timeline_heaps[resource_name]
                    limit = 5000 if resource_name == "Communication" else 3000
                    if len(timeline_heap) < limit:
                        heapq.heappush(timeline_heap, item)
                    elif duration_us > timeline_heap[0][0]:
                        heapq.heapreplace(timeline_heap, item)
                    sequence += 1
                if name == "[memory]":
                    args = event.get("args") or {}
                    if int(args.get("Device Type") or 0) == 1:
                        summary["memory_peak_bytes"] = max(
                            summary["memory_peak_bytes"],
                            int(args.get("Total Allocated") or 0),
                        )
                        summary["memory_events"] += 1
        summary["timeline_events"] = [
            item[2]
            for item in sorted(
                [
                    item
                    for timeline_heap in timeline_heaps.values()
                    for item in timeline_heap
                ],
                key=lambda item: (
                    item[2]["start_ns"] or item[2]["trace_timestamp_us"],
                    item[1],
                ),
            )
        ]
    except (OSError, ijson.JSONError, ValueError):
        summary["parse_error"] = True
    return summary


def load_workers(data_dir):
    workers = []
    for metrics_file in sorted(data_dir.rglob("metrics.jsonl")):
        events = _read_jsonl(metrics_file)
        start = next(
            (event for event in events if event.get("event") == "run_start"), {}
        )
        worker_dir = metrics_file.parent
        host = str(start.get("host") or worker_dir.parent.name)
        run_id = worker_dir.parent.parent.name
        operator_trace = worker_dir / "operator-trace.json"
        workers.append(
            {
                "worker": f"{host}/rank-{start.get('rank', worker_dir.name.removeprefix('rank-'))}",
                "host": host,
                "model": start.get("model", "unknown"),
                "job": start.get("job", start.get("model", "training")),
                "checkpoint_mode": start.get("checkpoint_mode", "synchronous"),
                "rank": int(start.get("rank", worker_dir.name.removeprefix("rank-"))),
                "world_size": start.get("world_size", "?"),
                "run_id": run_id,
                "started_at": start.get("timestamp", ""),
                "events": events,
                "steps": [event for event in events if event.get("event") == "step"],
                "checkpoints": [
                    event
                    for event in events
                    if event.get("event") == "checkpoint_complete"
                ],
                "hardware": next(
                    (
                        event
                        for event in events
                        if event.get("event") == "hardware_profile"
                    ),
                    {},
                ),
                "operator_trace": operator_trace if operator_trace.exists() else None,
                "operator_summary": summarize_operator_trace(operator_trace),
                "execution_trace": (
                    worker_dir / "execution-trace.json"
                    if (worker_dir / "execution-trace.json").exists()
                    else None
                ),
                "resource_trace": (
                    worker_dir / "resource-trace.jsonl"
                    if (worker_dir / "resource-trace.jsonl").exists()
                    else None
                ),
            }
        )
    return workers


def _latest_run(workers):
    by_run = defaultdict(list)
    for worker in workers:
        by_run[worker["run_id"]].append(worker)
    if not by_run:
        return "", []
    run_id = max(
        by_run,
        key=lambda value: (
            max((worker["started_at"] for worker in by_run[value]), default=""),
            value,
        ),
    )
    return run_id, by_run[run_id]


def _resource(category):
    if category in {"Training", "Optimizer"}:
        return "GPU"
    if category == "Checkpoint":
        return "Storage"
    return "CPU"


def build_spans(workers):
    raw_spans = []
    for worker in workers:
        worker_start_ns = _timestamp_ns(worker["started_at"])
        structured = [
            event for event in worker["events"] if event.get("event") == "span"
        ]
        if structured:
            for event in structured:
                if event.get("start_ns") is None or event.get("end_ns") is None:
                    continue
                raw_spans.append(
                    {**event, "worker": worker["worker"], "host": worker["host"]}
                )
            continue

        cursor_ms = 0.0
        for event in worker["events"]:
            if event.get("event") == "step":
                duration = float(event.get("duration_ms") or 0.0)
                start_ns = event.get("start_ns") or (
                    worker_start_ns + int(cursor_ms * 1_000_000)
                    if worker_start_ns
                    else None
                )
                end_ns = event.get("end_ns") or (
                    int(start_ns) + int(duration * 1_000_000) if start_ns else None
                )
                raw_spans.append(
                    {
                        **event,
                        "event": "span",
                        "category": "Training",
                        "operation": "Full step",
                        "start_ns": start_ns,
                        "end_ns": end_ns,
                        "synthetic_start_ms": cursor_ms,
                        "duration_ms": duration,
                        "worker": worker["worker"],
                        "host": worker["host"],
                    }
                )
                cursor_ms += duration
            elif event.get("event") == "checkpoint_complete":
                duration = float(event.get("duration_ms") or 0.0)
                start_ns = (
                    worker_start_ns + int(cursor_ms * 1_000_000)
                    if worker_start_ns
                    else None
                )
                raw_spans.append(
                    {
                        **event,
                        "event": "span",
                        "category": "Checkpoint",
                        "operation": "Save checkpoint",
                        "start_ns": start_ns,
                        "end_ns": (
                            int(start_ns) + int(duration * 1_000_000)
                            if start_ns
                            else None
                        ),
                        "synthetic_start_ms": cursor_ms,
                        "duration_ms": duration,
                        "worker": worker["worker"],
                        "host": worker["host"],
                    }
                )
                cursor_ms += duration

    timestamps = [
        int(span["start_ns"]) for span in raw_spans if span.get("start_ns") is not None
    ]
    base_ns = min(timestamps, default=0)
    spans = []
    for span in raw_spans:
        duration_ms = float(span.get("duration_ms") or 0.0)
        if span.get("start_ns") is not None:
            start_ns = int(span["start_ns"])
            start_s = (start_ns - base_ns) / 1_000_000_000
            if span.get("end_ns") is not None:
                duration_ms = (int(span["end_ns"]) - start_ns) / 1_000_000
        else:
            start_ns = base_ns + int(
                float(span.get("synthetic_start_ms") or 0.0) * 1_000_000
            )
            start_s = float(span.get("synthetic_start_ms") or 0.0) / 1000
        end_ns = start_ns + int(max(0.0, duration_ms) * 1_000_000)
        category = str(span.get("category") or "Other")
        spans.append(
            {
                **span,
                "category": category,
                "operation": str(span.get("operation") or "Unknown"),
                "resource": _resource(category),
                "start_s": start_s,
                "duration_s": max(0.0, duration_ms / 1000),
                "end_s": start_s + max(0.0, duration_ms / 1000),
                "start_epoch_s": start_ns / 1_000_000_000,
                "end_epoch_s": end_ns / 1_000_000_000,
                "start_utc": _iso_from_ns(start_ns),
                "end_utc": _iso_from_ns(end_ns),
            }
        )
    return spans


def build_resource_spans(workers, logical_spans):
    timestamps = [
        int(span["start_ns"])
        for span in logical_spans
        if span.get("start_ns") is not None
    ]
    base_ns = min(timestamps, default=0)
    resource_spans = []
    for worker in workers:
        worker_resource_count = 0
        path = worker["resource_trace"]
        if path is not None:
            for event in _read_jsonl(path):
                if event.get("event") != "resource_span":
                    continue
                event_start_ns = event.get("resource_start_ns", event.get("start_ns"))
                start_s = (
                    (int(event_start_ns) - base_ns) / 1_000_000_000
                    if event_start_ns is not None
                    else 0.0
                )
                duration_s = max(0.0, float(event.get("duration_ms") or 0) / 1000)
                start_ns = int(event_start_ns or base_ns)
                event_end_ns = event.get("resource_end_ns")
                end_ns = (
                    int(event_end_ns)
                    if event_end_ns is not None
                    else start_ns + int(duration_s * 1_000_000_000)
                )
                resource_spans.append(
                    {
                        **event,
                        "worker": worker["worker"],
                        "host": worker["host"],
                        "job": worker["job"],
                        "start_s": start_s,
                        "duration_s": duration_s,
                        "end_s": start_s + duration_s,
                        "start_epoch_s": start_ns / 1_000_000_000,
                        "end_epoch_s": end_ns / 1_000_000_000,
                        "start_utc": _iso_from_ns(start_ns),
                        "end_utc": _iso_from_ns(end_ns),
                    }
                )
                worker_resource_count += 1
        if worker_resource_count == 0:
            resource_spans.extend(
                {
                    **span,
                    "job": span.get("job", worker["job"]),
                }
                for span in logical_spans
                if span["worker"] == worker["worker"]
            )
    return resource_spans


def _figure_html(figure, div_id):
    return pio.to_html(
        figure,
        include_plotlyjs=False,
        full_html=False,
        config=PLOT_CONFIG,
        div_id=div_id,
    )


def _layout(title, height=430):
    return {
        "title": {"text": title, "x": 0.01, "xanchor": "left"},
        "height": height,
        "paper_bgcolor": "#ffffff",
        "plot_bgcolor": "#f7f9fb",
        "font": {"family": "Inter, Segoe UI, sans-serif", "color": "#24313a"},
        "margin": {"l": 70, "r": 35, "t": 70, "b": 60},
        "hoverlabel": {"font": {"family": "Consolas, monospace"}},
        "legend": {"orientation": "h", "y": 1.12, "x": 0},
    }


def _resource_measurement_summary(row):
    if row.get("resource") == "CPU":
        details = [f"wall={row['duration_s'] * 1000:.3f}ms"]
        if row.get("profiler_cpu_total_ms") is not None:
            details.append(f"profiler total={float(row['profiler_cpu_total_ms']):.3f}ms")
        if row.get("profiler_self_cpu_ms") is not None:
            details.append(f"self CPU={float(row['profiler_self_cpu_ms']):.3f}ms")
        return ", ".join(details)
    if row.get("resource") == "GPU":
        details = [f"stream elapsed={row['duration_s'] * 1000:.3f}ms"]
        if row.get("profiler_device_total_ms") is not None:
            details.append(
                f"kernel total={float(row['profiler_device_total_ms']):.3f}ms"
            )
        if row.get("device_kernel_count") is not None:
            details.append(f"direct kernels={int(row['device_kernel_count'])}")
        return ", ".join(details)
    return f"elapsed={row['duration_s'] * 1000:.3f}ms"


def resource_timeline(spans, host):
    figure = go.Figure()

    def lane_name(span):
        return f"{span['job']} / rank {span['rank']} | {span['resource']}"

    lanes = sorted(
        {lane_name(span) for span in spans},
        key=lambda lane: (
            int(re.search(r"rank (\d+)", lane).group(1)),
            {"CPU": 0, "GPU": 1, "Storage": 2, "Network": 3}.get(
                lane.rsplit("|", 1)[-1].strip(), 4
            ),
            lane,
        ),
    )
    lane_positions = {lane: index for index, lane in enumerate(lanes)}
    split_cpu_lanes = {
        lane_name(span)
        for span in spans
        if span["resource"] == "CPU" and span["category"] == "Checkpoint"
    }

    def lane_geometry(row):
        lane = lane_name(row)
        center = lane_positions[lane]
        if lane not in split_cpu_lanes:
            return center, 0.68, "full resource lane"
        if row["category"] == "Checkpoint":
            return center - 0.2, 0.36, "checkpoint half"
        return center + 0.2, 0.36, "training half"

    grouped = defaultdict(list)
    for span in spans:
        grouped[(span["operation"], span["resource"])].append(span)
    for (operation, resource_name), rows in sorted(
        grouped.items(),
        key=lambda item: (
            min(row["start_s"] for row in item[1]),
            item[0][0],
            item[0][1],
        ),
    ):
        color = OPERATION_COLORS.get(
            operation, CATEGORY_COLORS.get(rows[0]["category"], "#64748b")
        )
        figure.add_trace(
            go.Bar(
                name=f"{operation} - {resource_name}",
                legendgroup=operation,
                orientation="h",
                y=[lane_geometry(row)[0] for row in rows],
                width=[lane_geometry(row)[1] for row in rows],
                x=[row["duration_s"] * 1000 for row in rows],
                base=[row["start_utc"] for row in rows],
                marker={
                    "color": color,
                    "opacity": RESOURCE_OPACITY.get(resource_name, 1.0),
                    "line": {"color": "#ffffff", "width": 0.9},
                    "pattern": {
                        "shape": RESOURCE_PATTERNS.get(resource_name, ""),
                        "solidity": 0.25,
                    },
                },
                customdata=[
                    [
                        operation,
                        row["resource"],
                        row["category"],
                        row.get("step", "-"),
                        row["duration_s"] * 1000,
                        row.get("status", "ok"),
                        row.get("measurement", "wall_clock"),
                        row["start_utc"],
                        row["worker"],
                        row.get("start_alignment", "legacy_phase_start"),
                        "yes" if row.get("start_is_estimated") else "no",
                        _resource_measurement_summary(row),
                        row.get("checkpoint_worker_pid", "-"),
                        lane_geometry(row)[2],
                    ]
                    for row in rows
                ],
                hovertemplate=(
                    "phase=%{customdata[0]}<br>resource=%{customdata[1]}"
                    "<br>category=%{customdata[2]}<br>worker=%{customdata[8]}"
                    "<br>step=%{customdata[3]}<br>start UTC=%{customdata[7]}"
                    "<br>duration=%{customdata[4]:.3f}ms"
                    "<br>status=%{customdata[5]}"
                    "<br>measurement=%{customdata[6]}"
                    "<br>start alignment=%{customdata[9]}"
                    "<br>estimated start=%{customdata[10]}"
                    "<br>timing detail=%{customdata[11]}"
                    "<br>checkpoint process PID=%{customdata[12]}"
                    "<br>CPU lane strip=%{customdata[13]}<extra></extra>"
                ),
            )
        )
    layout = _layout(
        f"{host} forward/backward CPU and GPU phase timeline",
        max(500, 48 * len(lanes) + 210),
    )
    layout.update(
        {
            "barmode": "overlay",
            "bargap": 0.18,
            "xaxis": {
                "title": "Absolute UTC time",
                "type": "date",
                "tickformat": "%H:%M:%S.%L<br>%Y-%m-%d",
                "rangeslider": {"visible": True, "thickness": 0.07},
            },
            "yaxis": {
                "title": "Job / rank | resource",
                "tickmode": "array",
                "tickvals": list(range(len(lanes))),
                "ticktext": [
                    f"{lane} [checkpoint | training]"
                    if lane in split_cpu_lanes
                    else lane
                    for lane in lanes
                ],
                "autorange": "reversed",
                "automargin": True,
            },
            "shapes": [
                {
                    "type": "line",
                    "xref": "paper",
                    "x0": 0,
                    "x1": 1,
                    "yref": "y",
                    "y0": lane_positions[lane],
                    "y1": lane_positions[lane],
                    "line": {"color": "#cbd5e1", "width": 1},
                    "layer": "below",
                }
                for lane in split_cpu_lanes
            ],
        }
    )
    figure.update_layout(**layout)
    return figure


def cross_node_alignment_figure(workers, spans):
    rows = []
    starts = []
    for worker in workers:
        worker_spans = [span for span in spans if span["worker"] == worker["worker"]]
        fallback_start = min(
            (span["start_epoch_s"] for span in worker_spans), default=0
        )
        start_ns = _timestamp_ns(worker["started_at"])
        start_epoch = start_ns / 1_000_000_000 if start_ns else fallback_start
        end_epoch = max(
            (span["end_epoch_s"] for span in worker_spans), default=start_epoch
        )
        starts.append(start_epoch)
        rows.append((worker, start_epoch, end_epoch))
    earliest = min(starts, default=0)
    figure = go.Figure(
        go.Bar(
            name="Process lifetime",
            orientation="h",
            y=[row[0]["worker"] for row in rows],
            x=[(row[2] - row[1]) * 1000 for row in rows],
            base=[
                datetime.fromtimestamp(row[1], timezone.utc).isoformat(
                    timespec="microseconds"
                )
                for row in rows
            ],
            marker={"color": "#2563eb"},
            customdata=[
                [
                    datetime.fromtimestamp(row[1], timezone.utc).isoformat(
                        timespec="microseconds"
                    ),
                    (row[1] - earliest) * 1000,
                    (row[2] - row[1]) * 1000,
                    row[0]["host"],
                ]
                for row in rows
            ],
            hovertemplate=(
                "worker=%{y}<br>host=%{customdata[3]}<br>start UTC=%{customdata[0]}"
                "<br>start offset=%{customdata[1]:.3f}ms"
                "<br>recorded lifetime=%{customdata[2]:.3f}ms<extra></extra>"
            ),
        )
    )
    layout = _layout("Cross-node trace start alignment", max(400, 34 * len(rows) + 220))
    layout.update(
        {
            "xaxis": {
                "title": "Absolute UTC time",
                "type": "date",
                "tickformat": "%H:%M:%S.%L<br>%Y-%m-%d",
                "rangeslider": {"visible": True, "thickness": 0.08},
            },
            "yaxis": {"title": "Worker", "automargin": True, "autorange": "reversed"},
        }
    )
    figure.update_layout(**layout)
    return figure


def resource_activity_figure(spans, buckets=200):
    figure = go.Figure()
    if not spans:
        figure.update_layout(**_layout("CPU, GPU, storage, and network activity"))
        return figure
    start_epoch = min(span["start_epoch_s"] for span in spans)
    end_epoch = max(span["end_epoch_s"] for span in spans)
    width = max((end_epoch - start_epoch) / buckets, 1e-6)
    centers = [
        datetime.fromtimestamp(
            start_epoch + (index + 0.5) * width, timezone.utc
        ).isoformat(timespec="microseconds")
        for index in range(buckets)
    ]
    values = {resource_name: [0.0] * buckets for resource_name in RESOURCE_COLORS}
    for span in spans:
        first = max(
            0, min(buckets - 1, int((span["start_epoch_s"] - start_epoch) / width))
        )
        last = max(
            0,
            min(
                buckets - 1,
                int(
                    (
                        max(span["start_epoch_s"], span["end_epoch_s"] - 1e-12)
                        - start_epoch
                    )
                    / width
                ),
            ),
        )
        resource_name = span.get("resource", "CPU")
        values.setdefault(resource_name, [0.0] * buckets)
        for index in range(first, last + 1):
            left = start_epoch + index * width
            right = left + width
            overlap = max(
                0.0,
                min(span["end_epoch_s"], right) - max(span["start_epoch_s"], left),
            )
            values[resource_name][index] += overlap / width
    for resource_name, color in RESOURCE_COLORS.items():
        if not any(values.get(resource_name, [])):
            continue
        figure.add_trace(
            go.Scatter(
                x=centers,
                y=values[resource_name],
                name=resource_name,
                mode="lines",
                stackgroup="resources",
                line={"color": color, "width": 1},
                hovertemplate=(
                    "UTC=%{x}<br>active resource-equivalents=%{y:.2f}"
                    "<extra>%{fullData.name}</extra>"
                ),
            )
        )
    figure.update_layout(**_layout("CPU, GPU, storage, and network activity", 470))
    figure.update_xaxes(
        title="Absolute UTC time",
        type="date",
        tickformat="%H:%M:%S.%L<br>%Y-%m-%d",
    )
    figure.update_yaxes(title="Active resource-equivalents")
    return figure


def operator_timeline_figure(workers, title):
    rows = []
    for worker in workers:
        events = worker["operator_summary"].get("timeline_events", [])
        fallback_ns = _timestamp_ns(worker["started_at"])
        trace_origin = min((event["trace_timestamp_us"] for event in events), default=0)
        for event in events:
            start_ns = event.get("start_ns")
            if start_ns is None and fallback_ns is not None:
                start_ns = fallback_ns + int(
                    (event["trace_timestamp_us"] - trace_origin) * 1000
                )
            if start_ns is None:
                continue
            rows.append(
                {
                    **event,
                    "worker": worker["worker"],
                    "rank": worker["rank"],
                    "job": worker["job"],
                    "start_ns": start_ns,
                    "start_utc": _iso_from_ns(start_ns),
                }
            )
    rows = [
        row
        for resource_name in ("CPU", "GPU", "Communication")
        for row in heapq.nlargest(
            5000 if resource_name == "Communication" else 3000,
            (item for item in rows if item["resource"] == resource_name),
            key=lambda item: item["duration_us"],
        )
    ]
    rows.sort(key=lambda row: row["start_ns"])
    figure = go.Figure()
    for resource_name in ("CPU", "GPU", "Communication"):
        resource_rows = [row for row in rows if row["resource"] == resource_name]
        if not resource_rows:
            continue
        lane_name = (
            "Gradient/NCCL sync" if resource_name == "Communication" else resource_name
        )
        figure.add_trace(
            go.Bar(
                name=lane_name,
                orientation="h",
                y=[
                    f"{row['job']} / rank {row['rank']} / {lane_name}"
                    for row in resource_rows
                ],
                x=[row["duration_us"] / 1000 for row in resource_rows],
                base=[row["start_utc"] for row in resource_rows],
                marker={"color": RESOURCE_COLORS.get(resource_name, "#7c3aed")},
                customdata=[
                    [row["name"], row["start_utc"], row["duration_us"], row["worker"]]
                    for row in resource_rows
                ],
                hovertemplate=(
                    "operation=%{customdata[0]}<br>worker=%{customdata[3]}"
                    "<br>start UTC=%{customdata[1]}<br>duration=%{customdata[2]:.3f}us"
                    "<extra>%{fullData.name}</extra>"
                ),
            )
        )
    lanes = sorted(
        {trace.y[index] for trace in figure.data for index in range(len(trace.y))},
        key=lambda lane: (int(re.search(r"rank (\d+)", lane).group(1)), lane),
    )
    layout = _layout(title, max(500, 42 * len(lanes) + 220))
    layout.update(
        {
            "barmode": "overlay",
            "bargap": 0.14,
            "xaxis": {
                "title": "Absolute UTC time",
                "type": "date",
                "tickformat": "%H:%M:%S.%L<br>%Y-%m-%d",
                "rangeslider": {"visible": True, "thickness": 0.07},
            },
            "yaxis": {
                "title": "Job / rank / resource",
                "categoryorder": "array",
                "categoryarray": lanes,
                "autorange": "reversed",
                "automargin": True,
            },
        }
    )
    figure.update_layout(**layout)
    return figure


def _communication_kind(name):
    normalized = name.lower().replace("_", "")
    if "allreduce" in normalized:
        return "All-reduce"
    if "allgather" in normalized:
        return "All-gather"
    if "reducescatter" in normalized:
        return "Reduce-scatter"
    if "broadcast" in normalized:
        return "Broadcast"
    if "send" in normalized or "recv" in normalized:
        return "Send/receive"
    return "Other collective"


def communication_timeline_figure(workers, title):
    rows = []
    for worker in workers:
        events = worker["operator_summary"].get("timeline_events", [])
        fallback_ns = _timestamp_ns(worker["started_at"])
        trace_origin = min((event["trace_timestamp_us"] for event in events), default=0)
        estimated_bytes = worker.get("hardware", {}).get("gradient_bytes_estimate")
        for event in events:
            if event.get("resource") != "Communication":
                continue
            start_ns = event.get("start_ns")
            if start_ns is None and fallback_ns is not None:
                start_ns = fallback_ns + int(
                    (event["trace_timestamp_us"] - trace_origin) * 1000
                )
            if start_ns is None:
                continue
            rows.append(
                {
                    **event,
                    "kind": _communication_kind(event["name"]),
                    "worker": worker["worker"],
                    "host": worker["host"],
                    "rank": worker["rank"],
                    "job": worker["job"],
                    "start_ns": start_ns,
                    "start_utc": _iso_from_ns(start_ns),
                    "estimated_bytes": estimated_bytes,
                }
            )

    figure = go.Figure()
    for kind, color in COMMUNICATION_COLORS.items():
        kind_rows = [row for row in rows if row["kind"] == kind]
        if not kind_rows:
            continue
        figure.add_trace(
            go.Bar(
                name=kind,
                orientation="h",
                y=[
                    f"{row['job']} / rank {row['rank']} / collective stream"
                    for row in kind_rows
                ],
                x=[row["duration_us"] / 1000 for row in kind_rows],
                base=[row["start_utc"] for row in kind_rows],
                marker={"color": color, "line": {"color": "white", "width": 0.7}},
                customdata=[
                    [
                        row["name"],
                        row["worker"],
                        row["host"],
                        row["rank"],
                        row["start_utc"],
                        row["duration_us"],
                        (
                            float(row["estimated_bytes"]) / 1024**3
                            if row["estimated_bytes"] is not None
                            else None
                        ),
                    ]
                    for row in kind_rows
                ],
                hovertemplate=(
                    "collective=%{fullData.name}<br>operation=%{customdata[0]}"
                    "<br>worker=%{customdata[1]}<br>node=%{customdata[2]}"
                    "<br>rank=%{customdata[3]}<br>start UTC=%{customdata[4]}"
                    "<br>duration=%{customdata[5]:.3f}us"
                    "<br>estimated gradient payload=%{customdata[6]:.3f}GiB<extra></extra>"
                ),
            )
        )
    lanes = sorted(
        {str(value) for trace in figure.data for value in trace.y},
        key=lambda lane: (int(re.search(r"rank (\d+)", lane).group(1)), lane),
    )
    layout = _layout(title, max(430, 44 * len(lanes) + 220))
    layout.update(
        {
            "barmode": "overlay",
            "bargap": 0.16,
            "xaxis": {
                "title": "Absolute UTC time",
                "type": "date",
                "tickformat": "%H:%M:%S.%L<br>%Y-%m-%d",
                "rangeslider": {"visible": True, "thickness": 0.08},
            },
            "yaxis": {
                "title": "Job / rank / observed collective stream",
                "categoryorder": "array",
                "categoryarray": lanes,
                "autorange": "reversed",
                "automargin": True,
            },
        }
    )
    figure.update_layout(**layout)
    return figure


def step_performance_figure(workers, title):
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    for worker in workers:
        steps = worker["steps"]
        if not steps:
            continue
        figure.add_trace(
            go.Scatter(
                x=[row.get("step") for row in steps],
                y=[row.get("loss") for row in steps],
                name=f"{worker['worker']} loss",
                mode="lines+markers",
                hovertemplate="step=%{x}<br>loss=%{y:.5f}<extra>%{fullData.name}</extra>",
            ),
            secondary_y=False,
        )
        figure.add_trace(
            go.Scatter(
                x=[row.get("step") for row in steps],
                y=[row.get("duration_ms") for row in steps],
                name=f"{worker['worker']} latency",
                mode="lines",
                line={"dash": "dot"},
                hovertemplate="step=%{x}<br>latency=%{y:.3f}ms<extra>%{fullData.name}</extra>",
            ),
            secondary_y=True,
        )
    figure.update_layout(**_layout(title, 470))
    figure.update_xaxes(title_text="Training step")
    figure.update_yaxes(title_text="Loss", secondary_y=False)
    figure.update_yaxes(title_text="Step latency (ms)", secondary_y=True)
    return figure


def memory_figure(workers):
    figure = go.Figure()
    fields = [
        ("cuda_allocated_bytes", "allocated", "#2563eb"),
        ("cuda_reserved_bytes", "reserved", "#7c3aed"),
        ("cuda_peak_allocated_bytes", "peak allocated", "#e36a2e"),
    ]
    for worker in workers:
        for field, label, color in fields:
            figure.add_trace(
                go.Scatter(
                    x=[row.get("step") for row in worker["steps"]],
                    y=[float(row.get(field) or 0) / 1024**3 for row in worker["steps"]],
                    name=f"rank {worker['rank']} {label}",
                    mode="lines",
                    line={
                        "color": color,
                        "dash": "dot" if label == "reserved" else "solid",
                    },
                    hovertemplate="step=%{x}<br>%{y:.3f} GiB<extra>%{fullData.name}</extra>",
                )
            )
    figure.update_layout(**_layout("CUDA memory by rank", 420))
    figure.update_xaxes(title="Training step")
    figure.update_yaxes(title="GiB")
    return figure


def system_metrics_figure(workers):
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.14,
        specs=[[{"secondary_y": True}], [{"secondary_y": True}]],
        subplot_titles=("GPU health", "Host process health"),
    )
    gpu_fields = [
        ("cuda_utilization_percent", "GPU utilization", 1, False),
        ("cuda_memory_usage_percent", "GPU memory utilization", 1, False),
        ("cuda_temperature_c", "GPU temperature", 1, False),
        ("cuda_power_draw_mw", "GPU power", 1, True),
    ]
    host_fields = [
        ("process_rss_bytes", "Process RSS", 2, False),
        ("host_load_1m", "Host load 1m", 2, True),
    ]
    for worker in workers:
        for field, label, row, secondary in [*gpu_fields, *host_fields]:
            values = []
            steps = []
            for step in worker["steps"]:
                if step.get(field) is None:
                    continue
                value = float(step[field])
                if field == "cuda_power_draw_mw":
                    value /= 1000
                elif field == "process_rss_bytes":
                    value /= 1024**3
                steps.append(step.get("step"))
                values.append(value)
            if not values:
                continue
            figure.add_trace(
                go.Scatter(
                    x=steps,
                    y=values,
                    name=f"{worker['worker']} {label}",
                    mode="lines+markers",
                    hovertemplate="step=%{x}<br>value=%{y:.3f}<extra>%{fullData.name}</extra>",
                ),
                row=row,
                col=1,
                secondary_y=secondary,
            )
    figure.update_layout(**_layout("GPU and host telemetry", 680))
    figure.update_xaxes(title="Training step", row=2, col=1)
    figure.update_yaxes(title="Percent / C", row=1, col=1, secondary_y=False)
    figure.update_yaxes(title="Power (W)", row=1, col=1, secondary_y=True)
    figure.update_yaxes(title="Process RSS (GiB)", row=2, col=1, secondary_y=False)
    figure.update_yaxes(title="Host load", row=2, col=1, secondary_y=True)
    return figure


def operator_figure(workers, key, title):
    totals = Counter()
    counts = Counter()
    for worker in workers:
        totals.update(worker["operator_summary"].get(key, {}))
        if key == "cpu":
            counts["events"] += worker["operator_summary"]["counts"]["cpu"]
        elif key == "gpu":
            counts["events"] += worker["operator_summary"]["counts"]["gpu"]
        else:
            counts["events"] += worker["operator_summary"]["counts"]["communication"]
    top = totals.most_common(25)
    figure = go.Figure(
        go.Bar(
            x=[duration / 1000 for _, duration in reversed(top)],
            y=[name for name, _ in reversed(top)],
            orientation="h",
            marker={
                "color": CATEGORY_COLORS["Training"] if key == "cpu" else "#7c3aed"
            },
            hovertemplate="%{y}<br>total=%{x:.3f}ms<extra></extra>",
        )
    )
    figure.update_layout(**_layout(f"{title} ({counts['events']:,} events)", 620))
    figure.update_xaxes(title="Cumulative duration (ms)")
    figure.update_yaxes(automargin=True)
    return figure


def aggregate_activity_figure(spans, buckets=160):
    figure = go.Figure()
    if not spans:
        figure.update_layout(**_layout("Cluster activity"))
        return figure
    start = min(span["start_s"] for span in spans)
    end = max(span["end_s"] for span in spans)
    start_epoch = min(span["start_epoch_s"] for span in spans)
    width = max((end - start) / buckets, 1e-6)
    centers = [
        datetime.fromtimestamp(
            start_epoch + (index + 0.5) * width, timezone.utc
        ).isoformat(timespec="microseconds")
        for index in range(buckets)
    ]
    values = {category: [0.0] * buckets for category in CATEGORY_COLORS}
    for span in spans:
        first = max(0, min(buckets - 1, int((span["start_s"] - start) / width)))
        last = max(
            0,
            min(
                buckets - 1,
                int((max(span["start_s"], span["end_s"] - 1e-12) - start) / width),
            ),
        )
        for index in range(first, last + 1):
            left = start + index * width
            right = left + width
            overlap = max(0.0, min(span["end_s"], right) - max(span["start_s"], left))
            values.setdefault(span["category"], [0.0] * buckets)[index] += (
                overlap / width
            )
    for category, color in CATEGORY_COLORS.items():
        if not any(values[category]):
            continue
        figure.add_trace(
            go.Scatter(
                x=centers,
                y=values[category],
                name=category,
                mode="lines",
                stackgroup="activity",
                line={"color": color, "width": 1},
                hovertemplate=(
                    "UTC=%{x}<br>active rank-equivalents=%{y:.2f}"
                    "<extra>%{fullData.name}</extra>"
                ),
            )
        )
    figure.update_layout(**_layout("Cluster activity by category", 460))
    figure.update_xaxes(
        title="Absolute UTC time",
        type="date",
        tickformat="%H:%M:%S.%L<br>%Y-%m-%d",
    )
    figure.update_yaxes(title="Active rank-equivalents")
    return figure


def latency_heatmap(workers):
    rows = []
    labels = []
    max_step = max(
        (int(step.get("step") or 0) for worker in workers for step in worker["steps"]),
        default=0,
    )
    for worker in workers:
        by_step = {int(step.get("step") or 0): step for step in worker["steps"]}
        rows.append(
            [
                float(by_step[step].get("duration_ms") or 0)
                if step in by_step
                else None
                for step in range(1, max_step + 1)
            ]
        )
        labels.append(worker["worker"])
    figure = go.Figure(
        go.Heatmap(
            z=rows,
            x=list(range(1, max_step + 1)),
            y=labels,
            colorscale="Viridis",
            colorbar={"title": "ms"},
            hovertemplate="worker=%{y}<br>step=%{x}<br>latency=%{z:.3f}ms<extra></extra>",
        )
    )
    figure.update_layout(
        **_layout("Step latency heatmap", max(380, 34 * len(labels) + 210))
    )
    figure.update_xaxes(title="Training step")
    figure.update_yaxes(title="Worker", automargin=True)
    return figure


def checkpoint_figure(workers):
    figure = go.Figure()
    for worker in workers:
        rows = worker["checkpoints"]
        if rows:
            figure.add_trace(
                go.Bar(
                    name=worker["worker"],
                    x=[row.get("step") for row in rows],
                    y=[row.get("duration_ms") for row in rows],
                    customdata=[
                        [
                            row.get("tag", ""),
                            row.get("output_dir", ""),
                            float(row.get("checkpoint_size_bytes") or 0) / 1024**3,
                            float(row.get("checkpoint_throughput_mib_s") or 0),
                            row.get("checkpoint_mode", "synchronous"),
                            row.get("staging_duration_ms"),
                            row.get("persistence_duration_ms"),
                        ]
                        for row in rows
                    ],
                    hovertemplate=(
                        "step=%{x}<br>duration=%{y:.3f}ms<br>tag=%{customdata[0]}"
                        "<br>directory=%{customdata[1]}<br>size=%{customdata[2]:.3f}GiB"
                        "<br>throughput=%{customdata[3]:.2f}MiB/s"
                        "<br>mode=%{customdata[4]}<br>staging=%{customdata[5]}ms"
                        "<br>background persistence=%{customdata[6]}ms"
                        "<extra>%{fullData.name}</extra>"
                    ),
                )
            )
    figure.update_layout(**_layout("Checkpoint duration", 400))
    figure.update_xaxes(title="Training step")
    figure.update_yaxes(title="Duration (ms)")
    return figure


def _percentile(values, percentile):
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def write_simulator_profile(path, run_id, workers, spans):
    phase_durations = defaultdict(list)
    for span in spans:
        phase_durations[span["operation"]].append(span["duration_s"] * 1000)
    nodes = {}
    for host in sorted({worker["host"] for worker in workers}):
        host_workers = [worker for worker in workers if worker["host"] == host]
        step_durations = [
            float(step.get("duration_ms") or 0)
            for worker in host_workers
            for step in worker["steps"]
        ]
        nodes[host] = {
            "step_latency_ms": {
                "count": len(step_durations),
                "mean": sum(step_durations) / len(step_durations)
                if step_durations
                else 0,
                "p50": _percentile(step_durations, 0.5),
                "p95": _percentile(step_durations, 0.95),
            },
            "ranks": [
                {
                    "rank": worker["rank"],
                    "job": worker["job"],
                    "checkpoint_mode": worker["checkpoint_mode"],
                    "hardware": worker["hardware"],
                }
                for worker in sorted(host_workers, key=lambda item: item["rank"])
            ],
        }
    checkpoint_events = [
        {
            "worker": worker["worker"],
            "step": checkpoint.get("step"),
            "duration_ms": checkpoint.get("duration_ms"),
            "size_bytes": checkpoint.get("checkpoint_size_bytes", 0),
            "file_count": checkpoint.get("checkpoint_file_count", 0),
            "throughput_mib_s": checkpoint.get("checkpoint_throughput_mib_s", 0),
            "checkpoint_mode": checkpoint.get(
                "checkpoint_mode", worker["checkpoint_mode"]
            ),
            "staging_duration_ms": checkpoint.get("staging_duration_ms"),
            "persistence_duration_ms": checkpoint.get("persistence_duration_ms"),
            "checkpoint_worker_pid": checkpoint.get("checkpoint_worker_pid"),
        }
        for worker in workers
        for checkpoint in worker["checkpoints"]
    ]
    profile = {
        "schema_version": 1,
        "run_id": run_id,
        "model": workers[0]["model"],
        "checkpoint_mode": workers[0]["checkpoint_mode"],
        "world_size": len(workers),
        "trace_start_utc": min((span["start_utc"] for span in spans), default=None),
        "trace_duration_s": max((span["end_s"] for span in spans), default=0),
        "phase_duration_ms": {
            operation: {
                "count": len(durations),
                "mean": sum(durations) / len(durations),
                "p50": _percentile(durations, 0.5),
                "p95": _percentile(durations, 0.95),
            }
            for operation, durations in sorted(phase_durations.items())
        },
        "checkpoints": checkpoint_events,
        "nodes": nodes,
    }
    path.write_text(
        json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _cards(items):
    return "".join(
        f'<div class="metric"><span>{html.escape(str(label))}</span>'
        f"<strong>{html.escape(str(value))}</strong></div>"
        for label, value in items
    )


def _finding_cards(workers, spans, resource_spans, start_skew_ms):
    operation_time = Counter()
    resource_time = Counter()
    for span in spans:
        operation_time[span["operation"]] += span["duration_s"]
    for span in resource_spans:
        resource_time[span.get("resource", "CPU")] += span["duration_s"]

    worker_latencies = []
    for worker in workers:
        durations = [float(step.get("duration_ms") or 0) for step in worker["steps"]]
        if durations:
            worker_latencies.append((sum(durations) / len(durations), worker["worker"]))
    slowest_latency, slowest_worker = max(
        worker_latencies, default=(0.0, "No step data")
    )
    dominant_operation, dominant_seconds = (
        operation_time.most_common(1)[0] if operation_time else ("No spans", 0)
    )
    busiest_resource, resource_seconds = (
        resource_time.most_common(1)[0] if resource_time else ("No resource spans", 0)
    )
    checkpoints = [
        checkpoint for worker in workers for checkpoint in worker["checkpoints"]
    ]
    checkpoint_bytes = sum(
        float(row.get("checkpoint_size_bytes") or 0) for row in checkpoints
    )
    checkpoint_ms = sum(float(row.get("duration_ms") or 0) for row in checkpoints)
    return "".join(
        [
            (
                '<article class="finding"><span>Slowest observed rank</span>'
                f"<strong>{html.escape(slowest_worker)}</strong>"
                f"<p>{slowest_latency:.3f} ms mean recorded step latency.</p></article>"
            ),
            (
                '<article class="finding"><span>Largest logical phase</span>'
                f"<strong>{html.escape(dominant_operation)}</strong>"
                f"<p>{dominant_seconds:.3f} rank-seconds across recorded spans.</p></article>"
            ),
            (
                '<article class="finding"><span>Largest resource occupancy</span>'
                f"<strong>{html.escape(busiest_resource)}</strong>"
                f"<p>{resource_seconds:.3f} resource-seconds; overlapping work is retained.</p></article>"
            ),
            (
                '<article class="finding"><span>Checkpoint behavior</span>'
                f"<strong>{len(checkpoints)} completed</strong>"
                f"<p>{checkpoint_bytes / 1024**3:.3f} GiB in {checkpoint_ms:.3f} ms total.</p></article>"
            ),
            (
                '<article class="finding"><span>Cross-node alignment</span>'
                f"<strong>{start_skew_ms:.3f} ms skew</strong>"
                "<p>Difference between earliest and latest recorded worker start.</p></article>"
            ),
        ]
    )


def _metric_definitions(workers, spans, resource_spans):
    trace_duration = max((span["end_s"] for span in spans), default=0.0)
    recorded_work = sum(span["duration_s"] for span in spans)
    allocation = trace_duration * len(workers)
    training_work = sum(
        span["duration_s"]
        for span in spans
        if span["category"] in {"Training", "Optimizer"}
    )
    resource_work = sum(span["duration_s"] for span in resource_spans)
    checkpoints = [
        checkpoint for worker in workers for checkpoint in worker["checkpoints"]
    ]
    checkpoint_bytes = sum(
        float(row.get("checkpoint_size_bytes") or 0) for row in checkpoints
    )
    checkpoint_seconds = (
        sum(float(row.get("duration_ms") or 0) for row in checkpoints) / 1000
    )
    rows = [
        (
            "Recorded wall time",
            f"{trace_duration:.3f} s",
            "Earliest observed span start to latest observed span end.",
        ),
        (
            "Rank allocation",
            f"{allocation:.3f} rank-s",
            "Recorded wall time multiplied by participating ranks.",
        ),
        (
            "Logical work",
            f"{recorded_work:.3f} rank-s",
            "Sum of logical span durations; nested spans may overlap.",
        ),
        (
            "Training goodput",
            f"{100 * training_work / allocation:.2f}%" if allocation else "-",
            "Training and optimizer span time divided by rank allocation.",
        ),
        (
            "Resource occupancy",
            f"{resource_work:.3f} resource-s",
            "Sum of measured CPU, GPU, network, and storage spans; overlap is retained.",
        ),
        (
            "Completed checkpoints",
            str(len(checkpoints)),
            "Checkpoint completion records emitted by DeepSpeed.",
        ),
        (
            "Checkpoint traffic",
            f"{checkpoint_bytes / 1024**3:.3f} GiB",
            "Total on-disk size reported after completed checkpoints.",
        ),
        (
            "Checkpoint throughput",
            f"{checkpoint_bytes / 1024**2 / checkpoint_seconds:.2f} MiB/s"
            if checkpoint_seconds
            else "-",
            "Total checkpoint bytes divided by observed checkpoint duration.",
        ),
    ]
    return "".join(
        "<tr>"
        f"<td>{html.escape(metric)}</td><td>{html.escape(value)}</td>"
        f"<td>{html.escape(definition)}</td></tr>"
        for metric, value, definition in rows
    )


def _timeline_panel(title, subtitle, figure, div_id):
    escaped_id = html.escape(div_id, quote=True)
    controls = "".join(
        f'<button type="button" data-plot="{escaped_id}" data-axis="{axis}" '
        f'data-action="{action}" title="{html.escape(label)}">{html.escape(text)}</button>'
        for axis, action, label, text in (
            ("x", "in", "Zoom in on the time axis", "X +"),
            ("x", "out", "Zoom out on the time axis", "X -"),
            ("x", "reset", "Reset the time axis", "Reset X"),
            ("y", "in", "Zoom in on the lane axis", "Y +"),
            ("y", "out", "Zoom out on the lane axis", "Y -"),
            ("y", "reset", "Reset the lane axis", "Reset Y"),
        )
    )
    return (
        '<div class="section-heading"><div>'
        f"<h2>{html.escape(title)}</h2><p>{html.escape(subtitle)}</p></div>"
        f'<div class="axis-controls" aria-label="Chart zoom controls">{controls}</div></div>'
        + _figure_html(figure, div_id)
    )


def _artifact_link(path, page, label, perfetto=False):
    if path is None:
        return "-"
    relative = Path(os.path.relpath(path.resolve(), page.parent.resolve())).as_posix()
    escaped = html.escape(relative, quote=True)
    if perfetto:
        return f'<a class="perfetto" data-trace="{escaped}" target="_blank">{label}</a>'
    return f'<a href="{escaped}" download>{label}</a>'


def _worker_table(workers, page):
    rows = []
    for worker in sorted(workers, key=lambda item: item["rank"]):
        durations = [float(step.get("duration_ms") or 0) for step in worker["steps"]]
        peak = max(
            (
                float(step.get("cuda_peak_allocated_bytes") or 0)
                for step in worker["steps"]
            ),
            default=0,
        )
        traces = " / ".join(
            [
                _artifact_link(
                    worker["operator_trace"], page, "Perfetto", perfetto=True
                ),
                _artifact_link(worker["execution_trace"], page, "Execution graph"),
                _artifact_link(worker["resource_trace"], page, "CPU/GPU log"),
            ]
        )
        mean_duration = (
            f"{sum(durations) / len(durations):.3f} ms" if durations else "-"
        )
        rows.append(
            "<tr>"
            f"<td>{worker['rank']}</td><td>{len(worker['steps'])}</td>"
            f"<td>{mean_duration}</td>"
            f"<td>{_percentile(durations, 0.95):.3f} ms</td>"
            f"<td>{peak / 1024**3:.3f} GiB</td><td>{len(worker['checkpoints'])}</td>"
            f"<td>{traces}</td></tr>"
        )
    return "".join(rows)


def _checkpoint_table(workers):
    rows = []
    for worker in workers:
        for checkpoint in worker["checkpoints"]:
            staging_ms = checkpoint.get("staging_duration_ms")
            persistence_ms = checkpoint.get("persistence_duration_ms")
            rows.append(
                "<tr>"
                f"<td>{html.escape(worker['worker'])}</td>"
                f"<td>{checkpoint.get('step', '-')}</td>"
                f"<td>{html.escape(str(checkpoint.get('tag', '-')))}</td>"
                f"<td>{html.escape(str(checkpoint.get('checkpoint_mode', 'synchronous')))}</td>"
                f"<td>{float(checkpoint.get('duration_ms') or 0):.3f} ms</td>"
                f"<td>{f'{float(staging_ms):.3f} ms' if staging_ms is not None else '-'}</td>"
                f"<td>{f'{float(persistence_ms):.3f} ms' if persistence_ms is not None else '-'}</td>"
                f"<td>{float(checkpoint.get('checkpoint_size_bytes') or 0) / 1024**3:.3f} GiB</td>"
                f"<td>{int(checkpoint.get('checkpoint_file_count') or 0)}</td>"
                f"<td>{float(checkpoint.get('checkpoint_throughput_mib_s') or 0):.2f} MiB/s</td>"
                f"<td><code>{html.escape(str(checkpoint.get('output_dir', '-')))}</code></td>"
                "</tr>"
            )
    return (
        "".join(rows)
        or '<tr><td colspan="11" class="empty">No checkpoints recorded.</td></tr>'
    )


def _profile_number(profile, name, scale=1.0, suffix=""):
    value = profile.get(name)
    return f"{float(value) / scale:,.2f}{suffix}" if value is not None else "-"


def _hardware_table(workers):
    rows = []
    for worker in sorted(workers, key=lambda item: (item["host"], item["rank"])):
        profile = worker["hardware"]
        if not profile:
            continue

        rows.append(
            "<tr>"
            f"<td>{html.escape(worker['host'])}</td><td>{worker['rank']}</td>"
            f"<td>{html.escape(str(profile.get('gpu_name', '-')))}</td>"
            f"<td>{_profile_number(profile, 'gpu_total_memory_bytes', 1024**3, ' GiB')}</td>"
            f"<td>{html.escape(str(profile.get('cpu_model', '-')))}</td>"
            f"<td>{profile.get('cpu_count', '-')}</td>"
            f"<td>{_profile_number(profile, 'host_memory_bytes', 1024**3, ' GiB')}</td>"
            f"<td>{html.escape(str(profile.get('network_interface', '-')))}</td>"
            f"<td>{_profile_number(profile, 'nic_link_speed_mbps', 1000, ' Gbps')}</td>"
            f"<td>{_profile_number(profile, 'storage_write_gbps', suffix=' GB/s')}</td>"
            f"<td>{_profile_number(profile, 'storage_read_gbps', suffix=' GB/s')}</td>"
            f"<td>{_profile_number(profile, 'host_to_gpu_gbps', suffix=' GB/s')}</td>"
            f"<td>{_profile_number(profile, 'gpu_to_host_gbps', suffix=' GB/s')}</td>"
            f"<td>{_profile_number(profile, 'gpu_dram_copy_gbps', suffix=' GB/s')}</td>"
            f"<td>{_profile_number(profile, 'gpu_dram_theoretical_gbps', suffix=' GB/s')}</td>"
            f"<td>{_profile_number(profile, 'all_reduce_payload_gbps', suffix=' GB/s')}</td>"
            f"<td>{_profile_number(profile, 'all_reduce_bus_gbps', suffix=' GB/s')}</td>"
            f"<td>{_profile_number(profile, 'all_reduce_average_ms', suffix=' ms')}</td>"
            f"<td>{_profile_number(profile, 'model_parameter_bytes', 1024**3, ' GiB')}</td>"
            f"<td>{_profile_number(profile, 'gradient_bytes_estimate', 1024**3, ' GiB')}</td>"
            "</tr>"
        )
    return "".join(rows) or (
        '<tr><td colspan="20" class="empty">No hardware profile in this trace.</td></tr>'
    )


HARDWARE_TABLE_HEADER = (
    "<th>Node</th><th>Rank</th><th>GPU</th><th>GPU memory</th><th>CPU</th>"
    "<th>CPU cores</th><th>Host RAM</th><th>NIC</th><th>NIC link</th>"
    "<th>SSD write</th><th>SSD read</th><th>Host to GPU</th><th>GPU to host</th>"
    "<th>GPU DRAM measured</th><th>GPU DRAM theoretical</th>"
    "<th>All-reduce payload</th><th>All-reduce bus</th><th>All-reduce latency</th>"
    "<th>Parameters</th><th>Gradients estimate</th>"
)


def _slow_span_table(spans, limit=100):
    rows = []
    for span in sorted(spans, key=lambda item: item["duration_s"], reverse=True)[
        :limit
    ]:
        rows.append(
            "<tr>"
            f"<td>{html.escape(span['start_utc'])}</td><td>{span['start_s']:.6f} s</td>"
            f"<td>{span['duration_s'] * 1000:.3f} ms</td>"
            f"<td>{html.escape(span['worker'])}</td><td>{html.escape(span['category'])}</td>"
            f"<td>{html.escape(span['operation'])}</td><td>{span.get('step', '-')}</td>"
            f"<td>{html.escape(str(span.get('status', 'ok')))}</td></tr>"
        )
    return "".join(rows)


def _slow_span_explorer(spans, prefix):
    escaped_prefix = html.escape(prefix, quote=True)
    return (
        '<div class="table-tools">'
        f'<input type="search" data-table-filter="{escaped_prefix}" '
        'placeholder="Filter worker, category, operation, step, or status" '
        'aria-label="Filter slow spans"></div>'
        '<div class="table-wrap"><table>'
        "<thead><tr><th>Start UTC</th><th>Offset</th><th>Duration</th><th>Worker</th>"
        "<th>Category</th><th>Operation</th><th>Step</th><th>Status</th></tr></thead>"
        f'<tbody id="{escaped_prefix}">{_slow_span_table(spans)}</tbody>'
        "</table></div>"
    )


BASE_CSS = """
:root{font-family:Inter,Segoe UI,sans-serif;color:#1f2937;background:#f4f7f8}
*{box-sizing:border-box}body{margin:0;background:#f4f7f8}header{position:sticky;top:0;z-index:50;background:#073b4c;color:#fff;border-bottom:1px solid #052f3d;padding:18px max(24px,4vw)}
.header-inner,main{width:min(1560px,100%);margin:0 auto}header h1{font-size:26px;margin:0 0 4px;letter-spacing:0}header p{color:#cde0e5;margin:0;font-size:13px}
nav{display:flex;gap:6px 18px;flex-wrap:wrap;margin-top:13px}nav a{color:#eaf6f8;font-size:12px;font-weight:700;text-transform:uppercase}nav a:hover{color:#fff}
a{color:#0969a8;text-decoration:none}a:hover{text-decoration:underline}main{padding:24px max(24px,4vw) 56px}
.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:1px;background:#cbd5dc;border:1px solid #cbd5dc}
.metric{background:#fff;padding:15px;min-height:82px}.metric span,.finding span{display:block;color:#60717d;font-size:11px;font-weight:700;text-transform:uppercase}.metric strong{display:block;font-size:21px;margin-top:6px;overflow-wrap:anywhere}
.findings{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px}.finding{background:#fff;border:1px solid #d7e0e5;border-radius:6px;padding:15px}.finding strong{display:block;font-size:17px;margin-top:7px;overflow-wrap:anywhere}.finding p{color:#60717d;font-size:12px;line-height:1.45;margin:7px 0 0}
section{background:#fff;border-top:1px solid #d9e0e5;margin-top:26px;padding:19px 0 4px;scroll-margin-top:125px}section>h2,section>.section-heading{margin:0 18px 12px}section>h2{font-size:18px}.section-heading{display:flex;align-items:flex-start;justify-content:space-between;gap:18px}.section-heading h2{font-size:18px;margin:0}.section-heading p,.section-note{color:#60717d;font-size:12px;margin:5px 0 0;line-height:1.45}
.axis-controls,.tab-list{display:flex;gap:5px;flex-wrap:wrap}.axis-controls button,.tab-button{border:1px solid #aab8c1;background:#fff;color:#334155;border-radius:5px;padding:7px 9px;font:600 12px Inter,Segoe UI,sans-serif;cursor:pointer}.axis-controls button:hover,.tab-button:hover{background:#eef4f6}.tab-button[aria-selected="true"]{background:#073b4c;color:#fff;border-color:#073b4c}
.table-wrap{overflow:auto;padding:0 18px 16px}table{width:100%;border-collapse:collapse;font-size:13px;white-space:nowrap}th,td{text-align:left;padding:9px;border-bottom:1px solid #e3e8ec}th{color:#52636f;font-weight:700;background:#f7f9fb;position:sticky;top:0}.definition-table td:last-child{white-space:normal;min-width:360px;color:#52636f}
code{font-family:Consolas,monospace}.empty{color:#60717d}.node-links{display:flex;gap:12px;flex-wrap:wrap;padding:0 18px 16px}.node-tabs{margin-top:26px}.node-tabs .tab-list{padding:0 0 12px}.node-tab-panel[hidden]{display:none}.node-report{border-top:3px solid #073b4c;padding-top:20px}.node-report>h2{font-size:22px;margin:0 0 8px}.node-report>.subtitle{color:#60717d;margin:0 0 18px}
.table-tools{padding:0 18px 10px}.table-tools input{width:min(560px,100%);border:1px solid #aab8c1;border-radius:5px;padding:9px 11px;font:13px Inter,Segoe UI,sans-serif}.raw-note{padding:0 18px 14px;color:#60717d;font-size:12px}.raw-note code{color:#334155}
@media(max-width:760px){header{position:static}.section-heading{display:block}.axis-controls{margin-top:10px}.definition-table td:last-child{min-width:240px}main{padding-left:12px;padding-right:12px}}
"""


INTERACTION_SCRIPT = """
<script>
for(const link of document.querySelectorAll('.perfetto')){
  link.href='https://ui.perfetto.dev/#!/?url='+encodeURIComponent(new URL(link.dataset.trace,location.href).href);
}
for(const button of document.querySelectorAll('[data-tab-target]')){
  button.addEventListener('click',()=>{
    const group=button.closest('.node-tabs');
    for(const peer of group.querySelectorAll('[data-tab-target]')) peer.setAttribute('aria-selected','false');
    for(const panel of group.querySelectorAll('.node-tab-panel')) panel.hidden=true;
    button.setAttribute('aria-selected','true');
    const panel=document.getElementById(button.dataset.tabTarget);
    panel.hidden=false;
    for(const plot of panel.querySelectorAll('.plotly-graph-div')) Plotly.Plots.resize(plot);
  });
}
function zoomAxis(plot,axis,action){
  const key=axis+'axis';
  if(action==='reset'){
    Plotly.relayout(plot,{[key+'.autorange']:true});
    return;
  }
  const full=plot._fullLayout && plot._fullLayout[key];
  if(!full || !full.range) return;
  let start=full.range[0],end=full.range[1],isDate=full.type==='date';
  if(isDate){start=new Date(start).getTime();end=new Date(end).getTime();}
  const center=(start+end)/2,factor=action==='in'?0.7:1.4,half=(end-start)*factor/2;
  const range=isDate?[new Date(center-half).toISOString(),new Date(center+half).toISOString()]:[center-half,center+half];
  Plotly.relayout(plot,{[key+'.range']:range});
}
for(const button of document.querySelectorAll('[data-plot]')){
  button.addEventListener('click',()=>{
    const plot=document.getElementById(button.dataset.plot);
    if(plot) zoomAxis(plot,button.dataset.axis,button.dataset.action);
  });
}
for(const input of document.querySelectorAll('[data-table-filter]')){
  input.addEventListener('input',()=>{
    const needle=input.value.trim().toLowerCase();
    const body=document.getElementById(input.dataset.tableFilter);
    for(const row of body.querySelectorAll('tr')) row.hidden=Boolean(needle)&&!row.textContent.toLowerCase().includes(needle);
  });
}
</script>
"""


def _document(title, subtitle, body, script_path=None, nav="", inline_plotly=False):
    plotly_script = (
        f"<script>{get_plotlyjs()}</script>"
        if inline_plotly
        else f'<script src="{script_path}"></script>'
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>{BASE_CSS}</style>{plotly_script}</head>
<body><header><div class="header-inner"><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p>{nav}</div></header>
<main>{body}</main>{INTERACTION_SCRIPT}</body></html>"""


def node_body(
    host, workers, spans, resource_spans, page, run_id, prefix, heading=False
):
    host_spans = [span for span in spans if span["host"] == host]
    host_resource_spans = [span for span in resource_spans if span["host"] == host]
    all_steps = [step for worker in workers for step in worker["steps"]]
    durations = [float(step.get("duration_ms") or 0) for step in all_steps]
    peak = max(
        (float(step.get("cuda_peak_allocated_bytes") or 0) for step in all_steps),
        default=0,
    )
    global_start = min((span["start_epoch_s"] for span in spans), default=0)
    host_start = min(
        (span["start_epoch_s"] for span in host_spans), default=global_start
    )
    host_start_utc = datetime.fromtimestamp(host_start, timezone.utc).isoformat(
        timespec="milliseconds"
    )
    heading_html = (
        f'<div id="node-{safe_name(host)}" class="node-report"><h2>Node trace: '
        f'{html.escape(host)}</h2><p class="subtitle">{html.escape(workers[0]["model"])} | '
        f"{html.escape(run_id)}</p>"
        if heading
        else ""
    )
    body = (
        heading_html
        + '<div class="metrics">'
        + _cards(
            [
                ("Run", run_id),
                ("Node start UTC", host_start_utc),
                ("Start offset", f"{(host_start - global_start) * 1000:.3f} ms"),
                ("Ranks", len(workers)),
                ("Recorded steps", len(all_steps)),
                ("P95 step", f"{_percentile(durations, 0.95):.3f} ms"),
                ("Peak CUDA", f"{peak / 1024**3:.3f} GiB"),
                ("Checkpoints", sum(len(worker["checkpoints"]) for worker in workers)),
            ]
        )
        + "</div>"
        + f'<section id="{prefix}-resource-occupancy">'
        + _timeline_panel(
            "Forward/backward CPU and GPU phase timeline",
            "Each rank has separate CPU and GPU lanes; phase color identifies data loading, forward, backward, optimizer, synchronization, or checkpoint work.",
            resource_timeline(host_resource_spans, host),
            f"{prefix}-resource-timeline",
        )
        + "</section>"
        + f'<section id="{prefix}-operations">'
        + _timeline_panel(
            "CPU and GPU operation timeline",
            "Kineto CPU operations and GPU kernels retain their absolute UTC timestamps.",
            operator_timeline_figure(
                workers,
                f"{host} CPU operations, GPU kernels, and gradient/NCCL synchronization",
            ),
            f"{prefix}-operator-timeline",
        )
        + "</section>"
        + f'<section id="{prefix}-communication">'
        + _timeline_panel(
            "Collective communication occupancy",
            "Observed NCCL and distributed operations by rank. Collective routes are not inferred when peer endpoints are absent.",
            communication_timeline_figure(
                workers, f"{host} collective occupancy timeline"
            ),
            f"{prefix}-communication-timeline",
        )
        + "</section>"
        + f'<section id="{prefix}-performance"><h2>Step performance</h2>'
        + _figure_html(
            step_performance_figure(workers, "Loss and latency by rank"),
            f"{prefix}-step-performance",
        )
        + "</section>"
        + f'<section id="{prefix}-hardware-profile"><h2>Simulator hardware profile</h2><div class="table-wrap"><table>'
        + "<thead><tr>"
        + HARDWARE_TABLE_HEADER
        + "</tr></thead><tbody>"
        + _hardware_table(workers)
        + "</tbody></table></div></section>"
        + "<section><h2>CUDA memory</h2>"
        + _figure_html(memory_figure(workers), f"{prefix}-cuda-memory")
        + "</section>"
        + "<section><h2>GPU and host telemetry</h2>"
        + _figure_html(system_metrics_figure(workers), f"{prefix}-system-metrics")
        + "</section>"
        + "<section><h2>Top CPU operators</h2>"
        + _figure_html(
            operator_figure(workers, "cpu", "CPU operators"), f"{prefix}-cpu-ops"
        )
        + "</section>"
        + "<section><h2>Top GPU kernels</h2>"
        + _figure_html(
            operator_figure(workers, "gpu", "GPU kernels"), f"{prefix}-gpu-kernels"
        )
        + "</section>"
        + "<section><h2>Distributed communication</h2>"
        + _figure_html(
            operator_figure(
                workers, "communication", "NCCL and distributed operations"
            ),
            f"{prefix}-communication-summary",
        )
        + "</section>"
        + f'<section id="{prefix}-raw-traces"><h2>Ranks and raw traces</h2>'
        + '<p class="raw-note">Perfetto opens the complete PyTorch operator trace. The execution graph and CPU/GPU resource log remain downloadable simulator inputs.</p>'
        + '<div class="table-wrap"><table><thead><tr>'
        + "<th>Rank</th><th>Steps</th><th>Mean step</th><th>P95 step</th><th>Peak CUDA</th>"
        + "<th>Checkpoints</th><th>Artifacts</th></tr></thead><tbody>"
        + _worker_table(workers, page)
        + "</tbody></table></div></section>"
        + f'<section id="{prefix}-checkpoint-events"><h2>Checkpoint events</h2><div class="table-wrap"><table><thead><tr>'
        + "<th>Worker</th><th>Step</th><th>Tag</th><th>Mode</th><th>Total</th>"
        + "<th>Staging</th><th>Background</th><th>Size</th>"
        + "<th>Files</th><th>Throughput</th><th>Directory</th>"
        + "</tr></thead><tbody>"
        + _checkpoint_table(workers)
        + "</tbody></table></div></section>"
        + f'<section id="{prefix}-slow-spans"><h2>Slow-span explorer</h2>'
        + _slow_span_explorer(host_spans, f"{prefix}-slow-span-rows")
        + "</section>"
        + ("</div>" if heading else "")
    )
    return body


def write_node_page(host, workers, spans, resource_spans, output_file, run_id):
    prefix = safe_name(host)
    body = node_body(
        host,
        workers,
        spans,
        resource_spans,
        output_file,
        run_id,
        prefix,
    )
    nav = (
        '<nav><a href="../index.html">Aggregate</a>'
        f'<a href="#{prefix}-resource-occupancy">Resources</a>'
        f'<a href="#{prefix}-operations">Operations</a>'
        f'<a href="#{prefix}-communication">Communication</a>'
        f'<a href="#{prefix}-checkpoint-events">Checkpointing</a>'
        f'<a href="#{prefix}-hardware-profile">Hardware</a>'
        f'<a href="#{prefix}-slow-spans">Slow spans</a></nav>'
    )
    output_file.write_text(
        _document(
            f"Node trace: {host}",
            f"{workers[0]['model']} | {run_id}",
            body,
            "../plotly.min.js",
            nav,
        ),
        encoding="utf-8",
    )


def build_dashboard(data_dir, output_file):
    all_workers = load_workers(data_dir)
    if not all_workers:
        raise FileNotFoundError(
            f"No metrics.jsonl files found under {data_dir}; run make collect first"
        )
    run_id, workers = _latest_run(all_workers)
    spans = build_spans(workers)
    resource_spans = build_resource_spans(workers, spans)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    write_simulator_profile(
        output_file.parent / "simulator-profile.json",
        run_id,
        workers,
        spans,
    )
    nodes_dir = output_file.parent / "nodes"
    if nodes_dir.exists():
        shutil.rmtree(nodes_dir)
    nodes_dir.mkdir(parents=True)
    (output_file.parent / "plotly.min.js").write_text(get_plotlyjs(), encoding="utf-8")

    by_host = defaultdict(list)
    for worker in workers:
        by_host[worker["host"]].append(worker)
    node_buttons = []
    node_panels = []
    for index, (host, host_workers) in enumerate(sorted(by_host.items())):
        host_id = safe_name(host)
        node_page = nodes_dir / f"{safe_name(host)}.html"
        write_node_page(host, host_workers, spans, resource_spans, node_page, run_id)
        node_buttons.append(
            f'<button type="button" class="tab-button" data-tab-target="node-panel-{host_id}" '
            f'aria-selected="{str(index == 0).lower()}">{html.escape(host)}</button>'
        )
        panel_body = node_body(
            host,
            host_workers,
            spans,
            resource_spans,
            output_file,
            run_id,
            f"standalone-{host_id}",
            heading=True,
        )
        node_panels.append(
            f'<div id="node-panel-{host_id}" class="node-tab-panel"'
            + ("" if index == 0 else " hidden")
            + f'><p class="raw-note"><a href="nodes/{host_id}.html">Open {html.escape(host)} as a standalone report</a></p>'
            + panel_body
            + "</div>"
        )

    all_steps = [step for worker in workers for step in worker["steps"]]
    durations = [float(step.get("duration_ms") or 0) for step in all_steps]
    peak = max(
        (float(step.get("cuda_peak_allocated_bytes") or 0) for step in all_steps),
        default=0,
    )
    trace_duration = max((span["end_s"] for span in spans), default=0.0)
    cpu_events = sum(worker["operator_summary"]["counts"]["cpu"] for worker in workers)
    gpu_events = sum(worker["operator_summary"]["counts"]["gpu"] for worker in workers)
    communication_events = sum(
        worker["operator_summary"]["counts"]["communication"] for worker in workers
    )
    checkpoint_bytes = sum(
        float(checkpoint.get("checkpoint_size_bytes") or 0)
        for worker in workers
        for checkpoint in worker["checkpoints"]
    )
    trace_start_epoch = min((span["start_epoch_s"] for span in spans), default=0)
    trace_start_utc = datetime.fromtimestamp(trace_start_epoch, timezone.utc).isoformat(
        timespec="milliseconds"
    )
    worker_start_epochs = []
    for worker in workers:
        start_ns = _timestamp_ns(worker["started_at"])
        if start_ns is not None:
            worker_start_epochs.append(start_ns / 1_000_000_000)
    start_skew_ms = (
        (max(worker_start_epochs) - min(worker_start_epochs)) * 1000
        if worker_start_epochs
        else 0
    )
    body = (
        '<div class="metrics">'
        + _cards(
            [
                ("Run", run_id),
                ("Trace start UTC", trace_start_utc),
                ("Node start skew", f"{start_skew_ms:.3f} ms"),
                ("Nodes", len(by_host)),
                ("Ranks", len(workers)),
                ("Trace duration", f"{trace_duration:.3f} s"),
                ("P95 step", f"{_percentile(durations, 0.95):.3f} ms"),
                ("Peak CUDA", f"{peak / 1024**3:.3f} GiB"),
                ("Checkpoints", sum(len(worker["checkpoints"]) for worker in workers)),
                ("Checkpoint data", f"{checkpoint_bytes / 1024**3:.3f} GiB"),
                ("CPU operators", f"{cpu_events:,}"),
                ("GPU kernels", f"{gpu_events:,}"),
                ("Communication ops", f"{communication_events:,}"),
            ]
        )
        + "</div>"
        + '<section id="summary"><h2>What happened?</h2><div class="findings">'
        + _finding_cards(workers, spans, resource_spans, start_skew_ms)
        + "</div></section>"
        + '<section id="metrics"><h2>Aggregate metrics</h2>'
        + '<div class="table-wrap"><table class="definition-table"><thead><tr>'
        + "<th>Metric</th><th>Aggregate value</th><th>Definition</th>"
        + "</tr></thead><tbody>"
        + _metric_definitions(workers, spans, resource_spans)
        + "</tbody></table></div></section>"
        + '<section id="capacity">'
        + _timeline_panel(
            "Capacity and cross-node alignment",
            "Worker lifetimes share one absolute UTC axis so start skew is directly comparable.",
            cross_node_alignment_figure(workers, spans),
            "cross-node-alignment",
        )
        + "</section>"
        + '<section id="phase-occupancy">'
        + _timeline_panel(
            "Forward/backward CPU and GPU phases across all ranks",
            "Paired CPU and GPU lanes expose phase overlap and imbalance on one absolute UTC axis.",
            resource_timeline(resource_spans, "All nodes"),
            "aggregate-phase-resource-timeline",
        )
        + "</section>"
        + '<section id="resource-activity"><h2>CPU, GPU, storage, and network activity</h2>'
        + _figure_html(
            resource_activity_figure(resource_spans), "aggregate-resource-activity"
        )
        + "</section>"
        + '<section id="operations">'
        + _timeline_panel(
            "Aggregate CPU and GPU operation timeline",
            "The longest recorded Kineto events from every rank, aligned in UTC.",
            operator_timeline_figure(
                workers, "Aggregate CPU, GPU, and synchronization timeline"
            ),
            "aggregate-operator-timeline",
        )
        + "</section>"
        + "<section><h2>Cluster activity</h2>"
        + _figure_html(aggregate_activity_figure(spans), "cluster-activity")
        + "</section>"
        + "<section><h2>Aggregate training performance</h2>"
        + _figure_html(
            step_performance_figure(workers, "Loss and latency across all ranks"),
            "aggregate-performance",
        )
        + "</section>"
        + "<section><h2>Rank latency comparison</h2>"
        + _figure_html(latency_heatmap(workers), "latency-heatmap")
        + "</section>"
        + "<section><h2>Top CPU operators</h2>"
        + _figure_html(
            operator_figure(workers, "cpu", "CPU operators"), "aggregate-cpu-ops"
        )
        + "</section>"
        + "<section><h2>Top GPU kernels</h2>"
        + _figure_html(
            operator_figure(workers, "gpu", "GPU kernels"), "aggregate-gpu-kernels"
        )
        + "</section>"
        + '<section id="communication">'
        + _timeline_panel(
            "Collective communication occupancy",
            "Observed NCCL and distributed operations use rank-scoped collective lanes.",
            communication_timeline_figure(
                workers, "Aggregate collective occupancy timeline"
            ),
            "aggregate-communication-timeline",
        )
        + "</section>"
        + "<section><h2>Communication work by operation</h2>"
        + _figure_html(
            operator_figure(
                workers, "communication", "NCCL and distributed operations"
            ),
            "aggregate-communication",
        )
        + "</section>"
        + '<section id="checkpointing"><h2>Checkpoint performance</h2>'
        + _figure_html(checkpoint_figure(workers), "checkpoint-performance")
        + '<div class="table-wrap"><table><thead><tr>'
        + "<th>Worker</th><th>Step</th><th>Tag</th><th>Mode</th><th>Total</th>"
        + "<th>Staging</th><th>Background</th><th>Size</th>"
        + "<th>Files</th><th>Throughput</th><th>Directory</th>"
        + "</tr></thead><tbody>"
        + _checkpoint_table(workers)
        + "</tbody></table></div></section>"
        + '<section id="hardware"><h2>Simulator hardware profiles</h2>'
        + '<div class="table-wrap"><table><thead><tr>'
        + HARDWARE_TABLE_HEADER
        + "</tr></thead><tbody>"
        + _hardware_table(workers)
        + "</tbody></table></div></section>"
        + "<section><h2>GPU and host telemetry</h2>"
        + _figure_html(system_metrics_figure(workers), "aggregate-system-metrics")
        + "</section>"
        + '<section id="slow-spans"><h2>Slow-span explorer</h2>'
        + _slow_span_explorer(spans, "aggregate-slow-span-rows")
        + "</section>"
        + '<div id="nodes" class="node-tabs"><div class="tab-list" role="tablist">'
        + "".join(node_buttons)
        + "</div>"
        + "".join(node_panels)
        + "</div>"
    )
    nav = (
        '<nav><a href="#summary">Summary</a><a href="#metrics">Metrics</a>'
        '<a href="#capacity">Capacity</a><a href="#phase-occupancy">CPU/GPU phases</a>'
        '<a href="#operations">Operations</a>'
        '<a href="#communication">Communication</a>'
        '<a href="#checkpointing">Checkpointing</a><a href="#hardware">Hardware</a>'
        '<a href="#slow-spans">Slow spans</a><a href="#nodes">Per node</a></nav>'
    )
    output_file.write_text(
        _document(
            "Aggregate Trace Explorer",
            f"{workers[0]['model']} | {run_id} | wall-clock aligned across nodes",
            body,
            nav=nav,
            inline_plotly=True,
        ),
        encoding="utf-8",
    )
    print(
        f"Built {output_file} and {len(by_host)} node dashboard(s) "
        f"for {len(workers)} worker trace(s) from run {run_id}"
    )


def visualization_files(source):
    index = source / "index.html"
    collected = source / "collected"
    if not index.is_file() or not collected.is_dir():
        raise FileNotFoundError("Visualization is incomplete; run make visualize first")
    generated = [
        source / "simulator-profile.json",
        source / "plotly.min.js",
        *sorted((source / "nodes").glob("*.html")),
    ]
    traces = [path for path in sorted(collected.rglob("*")) if path.is_file()]
    return [index, *(path for path in generated if path.is_file()), *traces]


def upload_visualization(source, bucket_name, prefix, client=None):
    if client is None:
        try:
            from google.cloud import storage
        except ModuleNotFoundError as exc:
            raise SystemExit("google-cloud-storage is missing; run make setup") from exc
        client = storage.Client()

    bucket = client.bucket(bucket_name)
    files = visualization_files(source)
    report_object = f"{prefix.strip('/')}/index.html"
    for path in files:
        object_name = f"{prefix.strip('/')}/{path.relative_to(source).as_posix()}"
        content_type, _ = mimetypes.guess_type(path.name)
        blob = bucket.blob(object_name)
        if path.suffix == ".html":
            blob.content_disposition = "inline"
            blob.cache_control = "no-cache"
        try:
            blob.upload_from_filename(path, content_type=content_type)
        except Exception as exc:
            if getattr(exc, "code", None) == 403:
                raise SystemExit(
                    "GCS upload denied (403). Verify bucket IAM and set the GCE VM OAuth "
                    "scope to cloud-platform."
                ) from exc
            raise
        print(f"Uploaded gs://{bucket_name}/{object_name}")
    print(f"Uploaded {len(files)} file(s) to gs://{bucket_name}/{prefix.strip('/')}/")
    authenticated_url = f"https://storage.cloud.google.com/{bucket_name}/{quote(report_object, safe='/')}"
    print(f"Standalone report: {authenticated_url}")


class TraceHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def serve(directory, host, port):
    handler = lambda *args, **kwargs: TraceHandler(
        *args, directory=str(directory), **kwargs
    )
    server = ThreadingHTTPServer((host, port), handler)
    print(f"Serving trace dashboard at http://{host}:{port}")
    server.serve_forever()


def parse_args():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    collect_parser = commands.add_parser("collect")
    collect_parser.add_argument("--hostfile", default="hostfile")
    collect_parser.add_argument(
        "--source", type=Path, default=Path("visualization/traces")
    )
    collect_parser.add_argument(
        "--output", type=Path, default=Path("visualization/collected")
    )
    collect_parser.add_argument("--remote-dir", required=True)
    build_parser = commands.add_parser("build")
    build_parser.add_argument(
        "--data", type=Path, default=Path("visualization/collected")
    )
    build_parser.add_argument(
        "--output", type=Path, default=Path("visualization/index.html")
    )
    serve_parser = commands.add_parser("serve")
    serve_parser.add_argument("--directory", type=Path, default=Path("visualization"))
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    upload_parser = commands.add_parser("upload")
    upload_parser.add_argument("--source", type=Path, default=Path("visualization"))
    upload_parser.add_argument("--bucket", required=True)
    upload_parser.add_argument("--prefix", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "collect":
        collect(args.hostfile, args.source, args.output, args.remote_dir)
    elif args.command == "build":
        build_dashboard(args.data, args.output)
    elif args.command == "serve":
        serve(args.directory, args.host, args.port)
    else:
        upload_visualization(args.source, args.bucket, args.prefix)


if __name__ == "__main__":
    main()
