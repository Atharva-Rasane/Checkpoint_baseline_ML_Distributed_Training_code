import argparse
import json
import mimetypes
import re
import shutil
import socket
import subprocess
from concurrent.futures import ThreadPoolExecutor
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


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


def load_workers(data_dir):
    workers = []
    for metrics_file in sorted(data_dir.rglob("metrics.jsonl")):
        events = []
        for line in metrics_file.read_text(encoding="utf-8").splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        steps = [event for event in events if event.get("event") == "step"]
        checkpoints = [event for event in events if event.get("event") == "checkpoint_complete"]
        start = next((event for event in events if event.get("event") == "run_start"), {})
        worker_dir = metrics_file.parent
        operator_trace = worker_dir / "operator-trace.json"
        execution_trace = worker_dir / "execution-trace.json"
        workers.append(
            {
                "worker": f"{worker_dir.parent.name}/{worker_dir.name}",
                "model": start.get("model", "unknown"),
                "rank": start.get("rank", "?"),
                "world_size": start.get("world_size", "?"),
                "steps": steps,
                "checkpoints": checkpoints,
                "operator_trace": operator_trace.as_posix() if operator_trace.exists() else None,
                "execution_trace": execution_trace.as_posix() if execution_trace.exists() else None,
            }
        )
    return workers


def build_dashboard(data_dir, output_file):
    workers = load_workers(data_dir)
    if not workers:
        raise FileNotFoundError(f"No metrics.jsonl files found under {data_dir}; run make collect first")

    root = output_file.parent.resolve()
    for worker in workers:
        for key in ("operator_trace", "execution_trace"):
            if worker[key]:
                worker[key] = Path(worker[key]).resolve().relative_to(root).as_posix()

    data = json.dumps(workers).replace("</", "<\\/")
    html = DASHBOARD_HTML.replace("__TRACE_DATA__", data)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(html, encoding="utf-8")
    print(f"Built {output_file} for {len(workers)} worker trace(s)")


def visualization_files(source):
    index = source / "index.html"
    collected = source / "collected"
    if not index.is_file() or not collected.is_dir():
        raise FileNotFoundError("Visualization is incomplete; run make visualize first")
    return [index, *(path for path in sorted(collected.rglob("*")) if path.is_file())]


def upload_visualization(source, bucket_name, prefix, client=None):
    if client is None:
        try:
            from google.cloud import storage
        except ModuleNotFoundError as exc:
            raise SystemExit("google-cloud-storage is missing; run make setup") from exc
        client = storage.Client()

    bucket = client.bucket(bucket_name)
    files = visualization_files(source)
    for path in files:
        object_name = f"{prefix.strip('/')}/{path.relative_to(source).as_posix()}"
        content_type, _ = mimetypes.guess_type(path.name)
        bucket.blob(object_name).upload_from_filename(path, content_type=content_type)
        print(f"Uploaded gs://{bucket_name}/{object_name}")
    print(f"Uploaded {len(files)} file(s) to gs://{bucket_name}/{prefix.strip('/')}/")


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


DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Distributed Training Traces</title>
<style>
:root{color-scheme:dark;font-family:Inter,system-ui,sans-serif;background:#101214;color:#edf1f3}
body{margin:0}header{padding:28px max(24px,5vw);border-bottom:1px solid #30363b;background:#171a1d}
h1{font-size:28px;margin:0 0 6px;letter-spacing:0}p{margin:0;color:#aab4ba}
main{padding:24px max(24px,5vw);display:grid;gap:24px}.summary{display:flex;gap:28px;flex-wrap:wrap}
.metric strong{font-size:24px;display:block;color:#80d6b2}.metric span{font-size:12px;color:#aab4ba}
section{border-top:1px solid #30363b;padding-top:20px}h2{font-size:17px;margin:0 0 14px}
canvas{width:100%;height:260px;background:#171a1d;border:1px solid #30363b;border-radius:6px}
table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;padding:10px;border-bottom:1px solid #30363b}
th{color:#8fa0aa;font-weight:600}a{color:#65bff3;text-decoration:none}a:hover{text-decoration:underline}
.empty{color:#8fa0aa}code{color:#f0c674}
</style>
</head>
<body><header><h1>Distributed Training Traces</h1><p>Per-rank performance, memory, checkpoints, and operator timelines</p></header>
<main><div class="summary" id="summary"></div>
<section><h2>Loss by step</h2><canvas id="loss" width="1200" height="260"></canvas></section>
<section><h2>Workers</h2><table><thead><tr><th>Worker</th><th>Model</th><th>Steps</th><th>Last loss</th><th>Avg step</th><th>Peak CUDA</th><th>Checkpoints</th><th>Traces</th></tr></thead><tbody id="workers"></tbody></table></section>
<section><h2>Checkpoint events</h2><table><thead><tr><th>Worker</th><th>Step</th><th>Tag</th><th>Duration</th><th>Directory</th></tr></thead><tbody id="checkpoints"></tbody></table></section>
</main><script>
const workers=__TRACE_DATA__;
const allSteps=workers.flatMap(w=>w.steps), checkpoints=workers.flatMap(w=>w.checkpoints);
const peak=Math.max(0,...allSteps.map(s=>s.cuda_peak_allocated_bytes||0));
const metrics=[['Workers',workers.length],['Recorded steps',allSteps.length],['Checkpoints',checkpoints.length],['Peak CUDA',(peak/1073741824).toFixed(2)+' GiB']];
document.querySelector('#summary').innerHTML=metrics.map(([k,v])=>`<div class="metric"><strong>${v}</strong><span>${k}</span></div>`).join('');
const perfetto=path=>`https://ui.perfetto.dev/#!/?url=${encodeURIComponent(new URL(path,location.href).href)}`;
const rows=workers.map(w=>{const last=w.steps.at(-1)||{},avg=w.steps.length?w.steps.reduce((n,s)=>n+(s.duration_ms||0),0)/w.steps.length:0;const links=[w.operator_trace?`<a target="_blank" href="${perfetto(w.operator_trace)}">Perfetto</a>`:'',w.execution_trace?`<a download href="${w.execution_trace}">Graph</a>`:''].filter(Boolean).join(' / ');return `<tr><td>${w.worker}</td><td>${w.model}</td><td>${w.steps.length}</td><td>${last.loss?.toFixed(4)??'-'}</td><td>${avg.toFixed(1)} ms</td><td>${((last.cuda_peak_allocated_bytes||0)/1073741824).toFixed(2)} GiB</td><td>${w.checkpoints.length}</td><td>${links||'-'}</td></tr>`}).join('');
document.querySelector('#workers').innerHTML=rows;
document.querySelector('#checkpoints').innerHTML=workers.flatMap(w=>w.checkpoints.map(c=>`<tr><td>${w.worker}</td><td>${c.step}</td><td>${c.tag}</td><td>${c.duration_ms.toFixed(1)} ms</td><td><code>${c.output_dir}</code></td></tr>`)).join('')||'<tr><td colspan="5" class="empty">No checkpoint captured in this run.</td></tr>';
const canvas=document.querySelector('#loss'),ctx=canvas.getContext('2d'),pad=36,W=canvas.width-pad*2,H=canvas.height-pad*2;
ctx.strokeStyle='#3b444a';ctx.strokeRect(pad,pad,W,H);const losses=allSteps.map(s=>s.loss);const maxStep=Math.max(1,...allSteps.map(s=>s.step)),minLoss=Math.min(...losses),maxLoss=Math.max(...losses, minLoss+1e-6);const colors=['#65bff3','#80d6b2','#f0c674','#e58b8b'];
workers.forEach((w,i)=>{ctx.beginPath();ctx.strokeStyle=colors[i%colors.length];ctx.lineWidth=2;w.steps.forEach((s,j)=>{const x=pad+(s.step/maxStep)*W,y=pad+(1-(s.loss-minLoss)/(maxLoss-minLoss))*H;j?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke()});
</script></body></html>"""


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
