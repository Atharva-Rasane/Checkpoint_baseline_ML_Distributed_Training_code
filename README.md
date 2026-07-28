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
torch==2.13.0
deepspeed==0.19.3
nvidia-cuda-nvcc
```

PyTorch 2.13 requires Python 3.10 or newer.

## Fresh GCP VM Setup

Recommended: start from a Google Cloud Deep Learning VM image with PyTorch/GPU support. Those images are built for ML workloads and avoid most driver/CUDA setup work.

On the VM:

```bash
sudo apt-get update
sudo apt-get install -y git gh make python3 python3-venv python3-pip build-essential libaio-dev
nvidia-smi
```

If `nvidia-smi` is missing or cannot see the GPU, install/repair the NVIDIA driver first using Google Cloud's GPU driver instructions, then reboot the VM and rerun `nvidia-smi`.

Then clone this repo and run:

```bash
make setup
make doctor
make check
make hostfile-local
make train-hostfile-local MODEL=tiny_gpt STEPS=2
```

`make doctor` should show:

```text
cuda_available True
gpu_count 1
```

On a single GCP GPU VM:

```bash
make setup
make doctor
make hostfile-local
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

Development:

```bash
make setup
make lint
make test
make check
```
