# Minimal PyTorch + DeepSpeed ZeRO-3 Training Base

Put model files in `llm_models/`. A model named `my_model` should live at:

```text
llm_models/my_model.py
```

Each model file must define:

```python
def build_model(**kwargs):
    return model
```

For real training, the same model file should also define a dataset factory:

```python
def build_dataset(samples, seq_len, vocab_size, seed, **kwargs):
    return training_dataset
```

It must return a `torch.utils.data.Dataset`. `tiny_gpt` intentionally falls back to deterministic random tokens and is only a smoke test for the distributed training system, not a production language-model workload.

Setup:

```bash
make setup
make doctor
```

This pins the latest PyPI versions checked on 2026-07-28:

```text
torch==2.13.0+cu126
deepspeed==0.19.3
```

PyTorch 2.13 requires Python 3.10 or newer.

## Fresh GCP VM Setup

Recommended: start from a Google Cloud Deep Learning VM image with PyTorch/GPU support. Those images are built for ML workloads and avoid most driver/CUDA setup work.

If you use a plain Linux VM, choose a supported OS image. Use Ubuntu 24.04 LTS, Ubuntu 22.04 LTS, Debian 13, or Debian 12. Avoid newer preview/non-LTS images such as Ubuntu 26.04 because Google's GPU installer may fail with errors like `KeyError: '2604'`.

On the VM:

```bash
sudo apt-get update
sudo apt-get install -y git gh make python3 python3-venv python3-pip build-essential libaio-dev pdsh
lspci | grep -i nvidia
nvidia-smi
```

If `lspci` does not show NVIDIA hardware, the VM does not have a GPU attached.

If `nvidia-smi` is missing or cannot see the GPU, install the GCP NVIDIA driver, reboot, and verify:

```bash
curl -L https://storage.googleapis.com/compute-gpu-installation-us/installer/latest/cuda_installer.pyz --output cuda_installer.pyz
sudo python3 cuda_installer.pyz install_driver --installation-mode=repo
```

Only reboot after the installer finishes successfully:

```bash
sudo reboot
```

After reconnecting:

```bash
nvidia-smi
```

Install the CUDA 12.6 toolkit for DeepSpeed's compiler checks:

```bash
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-6
echo 'export CUDA_HOME=/usr/local/cuda-12.6' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
nvcc --version
```

`nvidia-smi` only proves that the GPU driver is installed. DeepSpeed also checks for the CUDA compiler at `$CUDA_HOME/bin/nvcc`. If training fails with `No such file or directory: '/usr/local/cuda-12.6/bin/nvcc'`, the toolkit is missing or `CUDA_HOME` points to the wrong directory. Check with:

```bash
which nvcc || true
ls -ld /usr/local/cuda*
dpkg -l | grep cuda-toolkit
```

This repo installs `torch==2.13.0+cu126`, which works with GCP T4 driver stacks that report CUDA 12.8 in `nvidia-smi`.

Then clone this repo and run:

```bash
make setup
make doctor
make check
make train-hostfile-local MODEL=tiny_gpt STEPS=2
```

`train-hostfile-local` detects the VM's GPU count and creates the local hostfile automatically.
The setup also installs Ninja, which DeepSpeed uses to compile its CPU Adam optimizer for ZeRO offload.

`make doctor` should show:

```text
cuda_available True
gpu_count 1
```

On a single GCP GPU VM:

```bash
make setup
make doctor
make train-hostfile-local MODEL=tiny_gpt STEPS=2
```

For a normal single-node run:

```bash
make train MODEL=tiny_gpt GPUS=4 STEPS=100
```

`make hostfile-local` writes a hostfile like this, using the GPU count from `nvidia-smi`:

```text
localhost slots=4
```

For multi-node GCP, use the internal IP or hostname for each VM and the GPU count on that VM.
DeepSpeed multi-node launch needs passwordless SSH from the launch VM to every worker and the same project path/environment on each VM.

Create a hostfile from the example:

```bash
cp hostfile.example hostfile
```

Edit it with your real hostnames or IPs:

```text
10.128.0.12 slots=8
10.128.0.13 slots=8
```

Then launch from the first VM:

```bash
make train-multinode HOSTFILE=hostfile MASTER_ADDR=10.128.0.12 MODEL=tiny_gpt STEPS=100
```

Only the first VM runs the `make train-multinode` command. DeepSpeed uses `pdsh` and node-to-node SSH to start one process for every hostfile slot. Verify fan-out from that VM before training:

```bash
PDSH_RCMD_TYPE=ssh pdsh -w 10.128.0.12,10.128.0.13 hostname
```

For GCP, assign one cluster network tag to every training VM and allow TCP, UDP, and ICMP only from that same source tag to the same target tag. This permits SSH, NCCL, and dynamic distributed ports without exposing them publicly. With OS Login enabled, register the coordinator's cluster public key with OS Login so ordinary internal-IP SSH works from rank 0.

The multi-node Make targets generate an ignored `.deepspeed_env` file containing the virtualenv `PATH` and `CUDA_HOME`. DeepSpeed exports it to remote ranks so `.venv/bin/ninja` is discoverable while JIT-loading CPUAdam. If remote ranks report `Ninja is required to load C++ extensions` even though `.venv/bin/ninja --version` works, pull the latest repo version and rerun the Make target; do not reinstall PyTorch. The synchronous and asynchronous wrappers also force the project root to the front of `sys.path`, fixing the multi-node-only `module 'train' has no attribute 'main'` import failure.

Checkpoints are saved under `checkpoints/`.

### Synchronous and asynchronous checkpointing

The synchronous entry point is `checkpointing/synchronous/train.py`: every rank waits for `engine.save_checkpoint()` to finish before the next training iteration starts. The asynchronous entry point is `checkpointing/asynchronous/train.py`. Both use the shared root trainer so model loading and training behavior remain identical.

The asynchronous entry point uses DeepSpeed's ZeRO-3-compatible decoupled checkpoint engine, which stages checkpoint state and persists it in a separate CPU process. Training can continue through data loading, forward, and backward while that process writes; DeepSpeed commits the checkpoint before the next parameter update, preserving a consistent snapshot.

Run and trace the asynchronous version on one GPU:

```bash
make train-async-hostfile-local MODEL=tiny_gpt STEPS=16 SAVE_EVERY=2 TRACE=1
make upload
```

Asynchronous checkpoints are written under `checkpoints/asynchronous/`. The checkpoint data-path timeline breaks each save into `Checkpoint state snapshot to host DRAM`, Kineto-observed `Observed GPU to host DRAM copy` events, `Checkpoint CPU serialization`, `Checkpoint DRAM to SSD write`, and any `Async checkpoint commit wait`. CPU checkpoint work and training use separate half-height strips when they overlap; storage has its own lane.

GPU-to-host bars are emitted only for device-to-host memcpy events actually recorded by Kineto inside a checkpoint staging window. This repo offloads ZeRO-3 parameters and optimizer state to CPU, so a checkpoint may already start in host DRAM and show no GPU-to-host checkpoint copy. The asynchronous writer is a separate process outside the parent Kineto session: its CPU serialization and SSD bars therefore show the measured staging-complete-to-commit in-flight window, not invented active-operation boundaries. The storage detail includes the writer process's Linux `write_bytes` delta when `/proc/<pid>/io` is available. Only one checkpoint is kept in flight because DeepSpeed exposes one pending commit at a time.

`gradient_accumulation_steps` is fixed to `1`: every loop iteration performs forward, backward, and one real optimizer parameter update. DeepSpeed names its per-GPU setting `train_micro_batch_size_per_gpu`, but there is no accumulation in this project. Set it through `BATCH_SIZE`; the effective global batch for each update is `BATCH_SIZE x total GPU count`.

For example, four GPUs with `BATCH_SIZE=8` perform one optimizer update from 32 samples per iteration:

```bash
make train MODEL=my_model GPUS=4 STEPS=10000 BATCH_SIZE=8 SAVE_EVERY=500
```

`STEPS` is the total target optimizer-update count. Training checkpoints preserve model, ZeRO-3 optimizer, learning-rate scheduler, and completed-step state. Resume from the checkpoint referenced by `latest`, or select an explicit tag:

```bash
make train-hostfile-local MODEL=my_model STEPS=10000 BATCH_SIZE=8 RESUME=latest
make train-hostfile-local MODEL=my_model STEPS=10000 BATCH_SIZE=8 RESUME=step-5000
```

Use the corresponding `train-async-*` target and `ASYNC_OUTPUT_DIR` when resuming asynchronous checkpoints. Startup prints the per-GPU batch, global batch, step range, and accumulation value. The trainer also seeds model/data execution, rejects invalid sizes and non-finite losses, records the dataset type in traced hardware metadata, and fails immediately when a requested checkpoint cannot be restored.

## Full Distributed Traces

Tracing is opt-in because collecting every CPU operation, CUDA kernel, tensor shape, stack, FLOP estimate, and memory event adds significant overhead and can produce large files.

By default, training completes three untraced warm-up iterations and starts detailed collection on iteration four. Set `TRACE_WARMUP=0` to trace from the first iteration or choose another warm-up count. Warm-up iterations still perform normal optimizer updates and checkpoints.

Run a traced job and include at least one checkpoint:

```bash
make train-hostfile-local MODEL=tiny_gpt STEPS=10 SAVE_EVERY=1 TRACE=1 TRACE_WARMUP=3
```

`JOB` labels the CPU/GPU lanes and defaults to `MODEL`. Set it when multiple jobs use the same model, for example `JOB=pretrain-a`.

Traced setup runs a 64 MiB per-node hardware probe by default. Change its transfer size with `BENCHMARK_MB`, for example:

```bash
make train-hostfile-local MODEL=tiny_gpt STEPS=10 TRACE=1 BENCHMARK_MB=128
```

The hardware profile records CPU and RAM capacity, GPU model/memory/compute capability, NIC interface and rated link speed, measured SSD sequential read/write, host-to-GPU, GPU-to-host, GPU DRAM copy, and distributed all-reduce payload/bus bandwidth. It also records model parameter bytes, estimated gradient bytes, checkpoint bytes/files/throughput, and per-step GPU/host telemetry.

`make visualize` writes the collected simulator inputs to `visualization/simulator-profile.json`; `make upload` archives that JSON beside the standalone report and raw traces.

Each rank writes wall-clock-aligned job spans, separate wall-clock CPU and CUDA-event GPU resource spans, an operator timeline, graph execution trace, step metrics, GPU memory, and checkpoint events below `visualization/traces/<run-id>/<host>/rank-<rank>/`.

`resource-trace.jsonl` uses a versioned per-phase resource schema. It records explicit CPU/GPU/storage start and end fields, CPU wall and profiler timing, GPU stream and kernel timing, kernel counts, checkpoint size and writer I/O bytes, and whether a resource start is observed or estimated. Perfetto remains the exact source for individual operator, GPU-kernel, and GPU-memcpy placement.

Collect traces from every host in `hostfile` and build the dashboard:

```bash
make collect COLLECT_RUN_ID=20260728T223339Z
make visualize COLLECT_RUN_ID=20260728T223339Z
```

Set `COLLECT_RUN_ID` to the run identifier printed by training. This limits collection,
dashboard parsing, and upload to that run; omitting it includes historical traces from
every host and can consume many gigabytes. If traces have already been collected and the
worker VMs are stopped, build and upload directly from the coordinator's persistent disk:

```bash
make upload-existing COLLECT_RUN_ID=20260728T223339Z
```

For a live cluster, `make upload COLLECT_RUN_ID=<run-id>` performs collection, dashboard
generation, and upload in one command.

`make visualize` selects the newest collected run and creates:

- `visualization/index.html`: a self-contained operations report with aggregate findings, defined metrics, paired CPU/GPU forward-backward-optimizer lanes, absolute UTC alignment, collective occupancy, checkpointing, hardware profiles, searchable slow spans, and tabbed per-node drill-downs.
- `visualization/nodes/<host>.html`: separate CPU and GPU lanes for every logical training phase, detailed Kineto operator/kernel lanes, and NCCL collective timelines for one node, with explicit axis controls and links to the raw Perfetto and execution traces.

It already runs collection, so the shorter equivalent is:

```bash
make visualize
```

Serve the dashboard locally:

```bash
make serve PORT=8000
```

Open `http://127.0.0.1:8000`. From a GCP VM, use an SSH tunnel from your computer and then open the same address:

```bash
gcloud compute ssh atharva-instace --zone=us-central1-b -- -L 8000:localhost:8000
```

The dashboard links operator traces to Perfetto for detailed timeline inspection. Multi-node collection uses `scp`, so the launcher VM must have passwordless SSH access to every host, as required by DeepSpeed launch.

Upload the generated dashboard and all collected rank traces to the private GCS archive:

```bash
make upload
```

Uploads use a new folder named `<vm-name>_<UTC-date>_<UTC-time>`, for example:

```text
gs://gbc-oit-rc-basil-app-bo-training-traces/atharva-instace_20260728_191530/
```

`make upload` runs collection and dashboard generation first. Override the destination when needed:

```bash
make upload BUCKET=my-trace-bucket UPLOAD_PREFIX=my-run
```

The command prints an authenticated `https://storage.cloud.google.com/.../index.html` URL. The uploaded `index.html` is self-contained and is stored with `Content-Type: text/html` and `Content-Disposition: inline`, so its charts do not depend on sibling objects. A private bucket does not provide an anonymous static-website URL; use the printed authenticated URL or a signed URL. Private raw operator traces should be downloaded and opened with Perfetto's **Open trace file** command.

The VM uses its attached service account through Application Default Credentials. A GCE VM created with a read-only Storage OAuth scope must be changed once to the `cloud-platform` scope; GCP requires the VM to be stopped for this operation:

```bash
gcloud compute instances stop atharva-instace --zone=us-central1-b
gcloud compute instances set-service-account atharva-instace \
  --zone=us-central1-b \
  --service-account=790904411643-compute@developer.gserviceaccount.com \
  --scopes=cloud-platform
gcloud compute instances start atharva-instace --zone=us-central1-b
```

The bucket is private with public access prevention. The VM service account has object-creator access, which permits timestamped uploads without allowing it to read or delete existing archives.

Development:

```bash
make setup
make lint
make test
make check
```
