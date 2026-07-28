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
sudo apt-get install -y git gh make python3 python3-venv python3-pip build-essential libaio-dev
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

Checkpoints are saved under `checkpoints/`.

## Full Distributed Traces

Tracing is opt-in because collecting every CPU operation, CUDA kernel, tensor shape, stack, FLOP estimate, and memory event adds significant overhead and can produce large files.

Run a traced job and include at least one checkpoint:

```bash
make train-hostfile-local MODEL=tiny_gpt STEPS=10 SAVE_EVERY=5 TRACE=1
```

Each rank writes an operator timeline, graph execution trace, step metrics, GPU memory, and checkpoint events below `visualization/traces/<run-id>/<host>/rank-<rank>/`.

Collect traces from every host in `hostfile` and build the dashboard:

```bash
make collect
make visualize
```

`make visualize` already runs collection, so the shorter equivalent is:

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
