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
    "Forward": "#3b82f6",
    "Backward": "#1d4ed8",
    "Optimizer step": "#0f8a72",
    "Save checkpoint": "#e36a2e",
    "Full step": "#2563eb",
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

    remote_source = f"{host}:{remote_dir.rstrip('/')}/{source.as_posix().lstrip('./')}/."
    subprocess.run(["scp", "-r", remote_source, str(destination)], check=True)


def collect(hostfile, source, output, remote_dir):
    hosts = read_hosts(hostfile)
    if not hosts:
        raise ValueError(f"No hosts found in {hostfile}")
    output.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=len(hosts)) as pool:
        futures = [pool.submit(collect_host, host, source, output, remote_dir) for host in hosts]
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
                        "Communication" if is_communication else "GPU" if is_gpu else "CPU"
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
                [item for timeline_heap in timeline_heaps.values() for item in timeline_heap],
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
        start = next((event for event in events if event.get("event") == "run_start"), {})
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
                "rank": int(start.get("rank", worker_dir.name.removeprefix("rank-"))),
                "world_size": start.get("world_size", "?"),
                "run_id": run_id,
                "started_at": start.get("timestamp", ""),
                "events": events,
                "steps": [event for event in events if event.get("event") == "step"],
                "checkpoints": [
                    event for event in events if event.get("event") == "checkpoint_complete"
                ],
                "hardware": next(
                    (event for event in events if event.get("event") == "hardware_profile"),
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
        structured = [event for event in worker["events"] if event.get("event") == "span"]
        if structured:
            for event in structured:
                if event.get("start_ns") is None or event.get("end_ns") is None:
                    continue
                raw_spans.append({**event, "worker": worker["worker"], "host": worker["host"]})
            continue

        cursor_ms = 0.0
        for event in worker["events"]:
            if event.get("event") == "step":
                duration = float(event.get("duration_ms") or 0.0)
                start_ns = event.get("start_ns") or (
                    worker_start_ns + int(cursor_ms * 1_000_000) if worker_start_ns else None
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
                    worker_start_ns + int(cursor_ms * 1_000_000) if worker_start_ns else None
                )
                raw_spans.append(
                    {
                        **event,
                        "event": "span",
                        "category": "Checkpoint",
                        "operation": "Save checkpoint",
                        "start_ns": start_ns,
                        "end_ns": (
                            int(start_ns) + int(duration * 1_000_000) if start_ns else None
                        ),
                        "synthetic_start_ms": cursor_ms,
                        "duration_ms": duration,
                        "worker": worker["worker"],
                        "host": worker["host"],
                    }
                )
                cursor_ms += duration

    timestamps = [int(span["start_ns"]) for span in raw_spans if span.get("start_ns") is not None]
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
            start_ns = base_ns + int(float(span.get("synthetic_start_ms") or 0.0) * 1_000_000)
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
                start_s = (
                    (int(event["start_ns"]) - base_ns) / 1_000_000_000
                    if event.get("start_ns") is not None
                    else 0.0
                )
                duration_s = max(0.0, float(event.get("duration_ms") or 0) / 1000)
                start_ns = int(event.get("start_ns") or base_ns)
                end_ns = start_ns + int(duration_s * 1_000_000_000)
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


def resource_timeline(spans, host):
    figure = go.Figure()
    grouped = defaultdict(list)
    for span in spans:
        grouped[span["operation"]].append(span)
    for operation, rows in sorted(
        grouped.items(), key=lambda item: min(row["start_s"] for row in item[1])
    ):
        figure.add_trace(
            go.Bar(
                name=operation,
                orientation="h",
                y=[f"{row['job']} / rank {row['rank']} / {row['resource']}" for row in rows],
                x=[row["duration_s"] * 1000 for row in rows],
                base=[row["start_utc"] for row in rows],
                marker={
                    "color": OPERATION_COLORS.get(
                        operation, CATEGORY_COLORS.get(rows[0]["category"], "#64748b")
                    ),
                    "line": {"color": "white", "width": 0.7},
                },
                customdata=[
                    [
                        row["category"],
                        row.get("step", "-"),
                        row["duration_s"] * 1000,
                        row.get("status", "ok"),
                        row.get("measurement", "wall_clock"),
                        row["start_utc"],
                    ]
                    for row in rows
                ],
                hovertemplate=(
                    "operation=%{fullData.name}<br>category=%{customdata[0]}"
                    "<br>step=%{customdata[1]}<br>start UTC=%{customdata[5]}"
                    "<br>duration=%{customdata[2]:.3f}ms"
                    "<br>status=%{customdata[3]}<br>measurement=%{customdata[4]}<extra></extra>"
                ),
            )
        )
    lanes = sorted(
        {f"{span['job']} / rank {span['rank']} / {span['resource']}" for span in spans},
        key=lambda lane: (int(re.search(r"rank (\d+)", lane).group(1)), lane),
    )
    layout = _layout(f"{host} resource occupancy", max(480, 46 * len(lanes) + 180))
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
                "title": "Rank resource",
                "categoryorder": "array",
                "categoryarray": lanes,
                "autorange": "reversed",
                "automargin": True,
            },
        }
    )
    figure.update_layout(**layout)
    return figure


def cross_node_alignment_figure(workers, spans):
    rows = []
    starts = []
    for worker in workers:
        worker_spans = [span for span in spans if span["worker"] == worker["worker"]]
        fallback_start = min((span["start_epoch_s"] for span in worker_spans), default=0)
        start_ns = _timestamp_ns(worker["started_at"])
        start_epoch = start_ns / 1_000_000_000 if start_ns else fallback_start
        end_epoch = max((span["end_epoch_s"] for span in worker_spans), default=start_epoch)
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
                datetime.fromtimestamp(row[1], timezone.utc).isoformat(timespec="microseconds")
                for row in rows
            ],
            marker={"color": "#2563eb"},
            customdata=[
                [
                    datetime.fromtimestamp(row[1], timezone.utc).isoformat(timespec="microseconds"),
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
        datetime.fromtimestamp(start_epoch + (index + 0.5) * width, timezone.utc).isoformat(
            timespec="microseconds"
        )
        for index in range(buckets)
    ]
    values = {resource_name: [0.0] * buckets for resource_name in RESOURCE_COLORS}
    for span in spans:
        first = max(0, min(buckets - 1, int((span["start_epoch_s"] - start_epoch) / width)))
        last = max(
            0,
            min(
                buckets - 1,
                int((max(span["start_epoch_s"], span["end_epoch_s"] - 1e-12) - start_epoch) / width),
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
        lane_name = "Gradient/NCCL sync" if resource_name == "Communication" else resource_name
        figure.add_trace(
            go.Bar(
                name=lane_name,
                orientation="h",
                y=[f"{row['job']} / rank {row['rank']} / {lane_name}" for row in resource_rows],
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
                    line={"color": color, "dash": "dot" if label == "reserved" else "solid"},
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
            marker={"color": CATEGORY_COLORS["Training"] if key == "cpu" else "#7c3aed"},
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
        datetime.fromtimestamp(start_epoch + (index + 0.5) * width, timezone.utc).isoformat(
            timespec="microseconds"
        )
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
            values.setdefault(span["category"], [0.0] * buckets)[index] += overlap / width
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
            [float(by_step[step].get("duration_ms") or 0) if step in by_step else None for step in range(1, max_step + 1)]
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
    figure.update_layout(**_layout("Step latency heatmap", max(380, 34 * len(labels) + 210)))
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
                        ]
                        for row in rows
                    ],
                    hovertemplate=(
                        "step=%{x}<br>duration=%{y:.3f}ms<br>tag=%{customdata[0]}"
                        "<br>directory=%{customdata[1]}<br>size=%{customdata[2]:.3f}GiB"
                        "<br>throughput=%{customdata[3]:.2f}MiB/s<extra>%{fullData.name}</extra>"
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
                "mean": sum(step_durations) / len(step_durations) if step_durations else 0,
                "p50": _percentile(step_durations, 0.5),
                "p95": _percentile(step_durations, 0.95),
            },
            "ranks": [
                {
                    "rank": worker["rank"],
                    "job": worker["job"],
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
        }
        for worker in workers
        for checkpoint in worker["checkpoints"]
    ]
    profile = {
        "schema_version": 1,
        "run_id": run_id,
        "model": workers[0]["model"],
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
    path.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cards(items):
    return "".join(
        f'<div class="metric"><span>{html.escape(str(label))}</span>'
        f'<strong>{html.escape(str(value))}</strong></div>'
        for label, value in items
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
            (float(step.get("cuda_peak_allocated_bytes") or 0) for step in worker["steps"]),
            default=0,
        )
        traces = " / ".join(
            [
                _artifact_link(worker["operator_trace"], page, "Perfetto", perfetto=True),
                _artifact_link(worker["execution_trace"], page, "Execution graph"),
                _artifact_link(worker["resource_trace"], page, "CPU/GPU log"),
            ]
        )
        mean_duration = f"{sum(durations) / len(durations):.3f} ms" if durations else "-"
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
            rows.append(
                "<tr>"
                f"<td>{html.escape(worker['worker'])}</td>"
                f"<td>{checkpoint.get('step', '-')}</td>"
                f"<td>{html.escape(str(checkpoint.get('tag', '-')))}</td>"
                f"<td>{float(checkpoint.get('duration_ms') or 0):.3f} ms</td>"
                f"<td>{float(checkpoint.get('checkpoint_size_bytes') or 0) / 1024**3:.3f} GiB</td>"
                f"<td>{int(checkpoint.get('checkpoint_file_count') or 0)}</td>"
                f"<td>{float(checkpoint.get('checkpoint_throughput_mib_s') or 0):.2f} MiB/s</td>"
                f"<td><code>{html.escape(str(checkpoint.get('output_dir', '-')))}</code></td>"
                "</tr>"
            )
    return "".join(rows) or '<tr><td colspan="8" class="empty">No checkpoints recorded.</td></tr>'


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
    for span in sorted(spans, key=lambda item: item["duration_s"], reverse=True)[:limit]:
        rows.append(
            "<tr>"
            f"<td>{html.escape(span['start_utc'])}</td><td>{span['start_s']:.6f} s</td>"
            f"<td>{span['duration_s'] * 1000:.3f} ms</td>"
            f"<td>{html.escape(span['worker'])}</td><td>{html.escape(span['category'])}</td>"
            f"<td>{html.escape(span['operation'])}</td><td>{span.get('step', '-')}</td>"
            f"<td>{html.escape(str(span.get('status', 'ok')))}</td></tr>"
        )
    return "".join(rows)


BASE_CSS = """
:root{font-family:Inter,Segoe UI,sans-serif;color:#24313a;background:#f7f9fb}
*{box-sizing:border-box}body{margin:0}header{background:#fff;border-bottom:1px solid #d9e0e5;padding:24px max(24px,4vw)}
header h1{font-size:27px;margin:0 0 5px;letter-spacing:0}header p{color:#60717d;margin:0}nav{margin-top:14px}
a{color:#0969a8;text-decoration:none}a:hover{text-decoration:underline}main{padding:22px max(24px,4vw) 50px}
.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:1px;background:#d9e0e5;border:1px solid #d9e0e5}
.metric{background:#fff;padding:15px}.metric span{display:block;color:#60717d;font-size:12px}.metric strong{display:block;font-size:22px;margin-top:5px}
section{background:#fff;border-top:1px solid #d9e0e5;margin-top:24px;padding:18px 0 4px}section h2{font-size:17px;margin:0 18px 12px}
.table-wrap{overflow:auto;padding:0 18px 16px}table{width:100%;border-collapse:collapse;font-size:13px;white-space:nowrap}
th,td{text-align:left;padding:9px;border-bottom:1px solid #e3e8ec}th{color:#60717d;font-weight:600;background:#f7f9fb;position:sticky;top:0}
code{font-family:Consolas,monospace}.empty{color:#60717d}.node-links{display:flex;gap:12px;flex-wrap:wrap;padding:0 18px 16px}
.node-report{border-top:3px solid #24313a;margin-top:42px;padding-top:20px}.node-report>h2{font-size:22px;margin:0 0 8px}.node-report>.subtitle{color:#60717d;margin:0 0 18px}
"""


PERFETTO_SCRIPT = """
<script>
for(const link of document.querySelectorAll('.perfetto')){
  link.href='https://ui.perfetto.dev/#!/?url='+encodeURIComponent(new URL(link.dataset.trace,location.href).href);
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
<body><header><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p>{nav}</header>
<main>{body}</main>{PERFETTO_SCRIPT}</body></html>"""


def node_body(host, workers, spans, resource_spans, page, run_id, prefix, heading=False):
    host_spans = [span for span in spans if span["host"] == host]
    host_resource_spans = [span for span in resource_spans if span["host"] == host]
    all_steps = [step for worker in workers for step in worker["steps"]]
    durations = [float(step.get("duration_ms") or 0) for step in all_steps]
    peak = max(
        (float(step.get("cuda_peak_allocated_bytes") or 0) for step in all_steps),
        default=0,
    )
    global_start = min((span["start_epoch_s"] for span in spans), default=0)
    host_start = min((span["start_epoch_s"] for span in host_spans), default=global_start)
    host_start_utc = datetime.fromtimestamp(host_start, timezone.utc).isoformat(
        timespec="milliseconds"
    )
    heading_html = (
        f'<div id="node-{safe_name(host)}" class="node-report"><h2>Node trace: '
        f'{html.escape(host)}</h2><p class="subtitle">{html.escape(workers[0]["model"])} | '
        f'{html.escape(run_id)}</p>'
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
        + '<section><h2>Resource occupancy</h2>'
        + _figure_html(resource_timeline(host_resource_spans, host), f"{prefix}-resource-timeline")
        + "</section>"
        + '<section><h2>CPU, GPU, and gradient synchronization timeline</h2>'
        + _figure_html(
            operator_timeline_figure(
                workers,
                f"{host} CPU operations, GPU kernels, and gradient/NCCL synchronization",
            ),
            f"{prefix}-operator-timeline",
        )
        + "</section>"
        + '<section><h2>Step performance</h2>'
        + _figure_html(
            step_performance_figure(workers, "Loss and latency by rank"),
            f"{prefix}-step-performance",
        )
        + "</section>"
        + '<section><h2>Simulator hardware profile</h2><div class="table-wrap"><table>'
        + "<thead><tr>"
        + HARDWARE_TABLE_HEADER
        + "</tr></thead><tbody>"
        + _hardware_table(workers)
        + "</tbody></table></div></section>"
        + '<section><h2>CUDA memory</h2>'
        + _figure_html(memory_figure(workers), f"{prefix}-cuda-memory")
        + "</section>"
        + '<section><h2>GPU and host telemetry</h2>'
        + _figure_html(system_metrics_figure(workers), f"{prefix}-system-metrics")
        + "</section>"
        + '<section><h2>Top CPU operators</h2>'
        + _figure_html(operator_figure(workers, "cpu", "CPU operators"), f"{prefix}-cpu-ops")
        + "</section>"
        + '<section><h2>Top GPU kernels</h2>'
        + _figure_html(operator_figure(workers, "gpu", "GPU kernels"), f"{prefix}-gpu-kernels")
        + "</section>"
        + '<section><h2>Distributed communication</h2>'
        + _figure_html(
            operator_figure(workers, "communication", "NCCL and distributed operations"),
            f"{prefix}-communication",
        )
        + "</section>"
        + '<section><h2>Ranks and raw traces</h2><div class="table-wrap"><table><thead><tr>'
        + "<th>Rank</th><th>Steps</th><th>Mean step</th><th>P95 step</th><th>Peak CUDA</th>"
        + "<th>Checkpoints</th><th>Artifacts</th></tr></thead><tbody>"
        + _worker_table(workers, page)
        + "</tbody></table></div></section>"
        + '<section><h2>Checkpoint events</h2><div class="table-wrap"><table><thead><tr>'
        + "<th>Worker</th><th>Step</th><th>Tag</th><th>Duration</th><th>Size</th>"
        + "<th>Files</th><th>Throughput</th><th>Directory</th>"
        + "</tr></thead><tbody>"
        + _checkpoint_table(workers)
        + "</tbody></table></div></section>"
        + '<section><h2>Slow-span explorer</h2><div class="table-wrap"><table><thead><tr>'
        + "<th>Start UTC</th><th>Offset</th><th>Duration</th><th>Worker</th>"
        + "<th>Category</th><th>Operation</th>"
        + "<th>Step</th><th>Status</th></tr></thead><tbody>"
        + _slow_span_table(host_spans)
        + "</tbody></table></div></section>"
        + ("</div>" if heading else "")
    )
    return body


def write_node_page(host, workers, spans, resource_spans, output_file, run_id):
    body = node_body(
        host,
        workers,
        spans,
        resource_spans,
        output_file,
        run_id,
        safe_name(host),
    )
    nav = '<nav><a href="../index.html">Aggregate trace</a></nav>'
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
        raise FileNotFoundError(f"No metrics.jsonl files found under {data_dir}; run make collect first")
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
    node_links = []
    node_sections = []
    for host, host_workers in sorted(by_host.items()):
        node_page = nodes_dir / f"{safe_name(host)}.html"
        write_node_page(host, host_workers, spans, resource_spans, node_page, run_id)
        node_links.append(f'<a href="#node-{safe_name(host)}">{html.escape(host)} detail</a>')
        node_sections.append(
            node_body(
                host,
                host_workers,
                spans,
                resource_spans,
                output_file,
                run_id,
                f"standalone-{safe_name(host)}",
                heading=True,
            )
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
        + '<section><h2>Node traces</h2><div class="node-links">'
        + "".join(node_links)
        + "</div></section>"
        + '<section><h2>Cross-node start alignment</h2>'
        + _figure_html(cross_node_alignment_figure(workers, spans), "cross-node-alignment")
        + "</section>"
        + '<section><h2>CPU, GPU, storage, and network activity</h2>'
        + _figure_html(
            resource_activity_figure(resource_spans),
            "aggregate-resource-activity",
        )
        + "</section>"
        + '<section><h2>Cluster activity</h2>'
        + _figure_html(aggregate_activity_figure(spans), "cluster-activity")
        + "</section>"
        + '<section><h2>Aggregate training performance</h2>'
        + _figure_html(step_performance_figure(workers, "Loss and latency across all ranks"), "aggregate-performance")
        + "</section>"
        + '<section><h2>Rank latency comparison</h2>'
        + _figure_html(latency_heatmap(workers), "latency-heatmap")
        + "</section>"
        + '<section><h2>Checkpoint performance</h2>'
        + _figure_html(checkpoint_figure(workers), "checkpoint-performance")
        + "</section>"
        + '<section><h2>Simulator hardware profiles</h2><div class="table-wrap"><table>'
        + "<thead><tr>"
        + HARDWARE_TABLE_HEADER
        + "</tr></thead><tbody>"
        + _hardware_table(workers)
        + "</tbody></table></div></section>"
        + '<section><h2>GPU and host telemetry</h2>'
        + _figure_html(system_metrics_figure(workers), "aggregate-system-metrics")
        + "</section>"
        + '<section><h2>Top CPU operators</h2>'
        + _figure_html(operator_figure(workers, "cpu", "CPU operators"), "aggregate-cpu-ops")
        + "</section>"
        + '<section><h2>Top GPU kernels</h2>'
        + _figure_html(operator_figure(workers, "gpu", "GPU kernels"), "aggregate-gpu-kernels")
        + "</section>"
        + '<section><h2>Distributed communication</h2>'
        + _figure_html(
            operator_figure(workers, "communication", "NCCL and distributed operations"),
            "aggregate-communication",
        )
        + "</section>"
        + '<section><h2>Slow-span explorer</h2><div class="table-wrap"><table><thead><tr>'
        + "<th>Start UTC</th><th>Offset</th><th>Duration</th><th>Worker</th>"
        + "<th>Category</th><th>Operation</th>"
        + "<th>Step</th><th>Status</th></tr></thead><tbody>"
        + _slow_span_table(spans)
        + "</tbody></table></div></section>"
        + "".join(node_sections)
    )
    output_file.write_text(
        _document(
            "Aggregate Trace Explorer",
            f"{workers[0]['model']} | {run_id} | wall-clock aligned across nodes",
            body,
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
    authenticated_url = (
        f"https://storage.cloud.google.com/{bucket_name}/{quote(report_object, safe='/')}"
    )
    print(f"Standalone report: {authenticated_url}")


class TraceHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def serve(directory, host, port):
    handler = lambda *args, **kwargs: TraceHandler(*args, directory=str(directory), **kwargs)
    server = ThreadingHTTPServer((host, port), handler)
    print(f"Serving trace dashboard at http://{host}:{port}")
    server.serve_forever()


def parse_args():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    collect_parser = commands.add_parser("collect")
    collect_parser.add_argument("--hostfile", default="hostfile")
    collect_parser.add_argument("--source", type=Path, default=Path("visualization/traces"))
    collect_parser.add_argument("--output", type=Path, default=Path("visualization/collected"))
    collect_parser.add_argument("--remote-dir", required=True)
    build_parser = commands.add_parser("build")
    build_parser.add_argument("--data", type=Path, default=Path("visualization/collected"))
    build_parser.add_argument("--output", type=Path, default=Path("visualization/index.html"))
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
