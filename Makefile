.PHONY: setup install cuda-check doctor hostfile-local train train-hostfile-local train-multinode test lint check clean

VENV ?= .venv
PYTHON ?= $(VENV)/bin/python
DEEPSPEED ?= $(VENV)/bin/deepspeed
CUDA_HOME ?= $(shell if [ -n "$$CUDA_HOME" ]; then echo "$$CUDA_HOME"; elif [ -d /usr/local/cuda ]; then echo /usr/local/cuda; elif [ -d /usr/local/cuda-12.6 ]; then echo /usr/local/cuda-12.6; elif [ -d /usr/local/cuda-12.8 ]; then echo /usr/local/cuda-12.8; fi)
CUDA_ENV = $(if $(CUDA_HOME),CUDA_HOME=$(CUDA_HOME) PATH=$(CUDA_HOME)/bin:$$PATH,)
GPUS ?= 1
HOSTFILE ?= hostfile
MODEL ?= tiny_gpt
STEPS ?= 100
MASTER_ADDR ?= localhost

setup:
	python3 -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

install: setup

cuda-check:
	@test -n "$(CUDA_HOME)" || (echo "CUDA_HOME is not set and no /usr/local/cuda* directory was found. Install cuda-toolkit-12-6 or set CUDA_HOME=/path/to/cuda."; exit 1)
	@test -x "$(CUDA_HOME)/bin/nvcc" || (echo "Missing nvcc at $(CUDA_HOME)/bin/nvcc. Install cuda-toolkit-12-6 or set CUDA_HOME to the CUDA toolkit root."; exit 1)

doctor: cuda-check
	$(PYTHON) --version
	$(PYTHON) -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available(), 'cuda_version', torch.version.cuda, 'gpu_count', torch.cuda.device_count())"
	$(DEEPSPEED) --version
	nvidia-smi
	$(CUDA_HOME)/bin/nvcc --version

hostfile-local:
	printf "localhost slots=%s\n" "$$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" > $(HOSTFILE)
	cat $(HOSTFILE)

train: cuda-check
	$(CUDA_ENV) $(DEEPSPEED) --num_gpus $(GPUS) train.py --model_name $(MODEL) --steps $(STEPS)

train-hostfile-local: cuda-check hostfile-local
	$(CUDA_ENV) $(DEEPSPEED) --hostfile $(HOSTFILE) --no_ssh --node_rank 0 --num_nodes 1 --master_addr $(MASTER_ADDR) train.py --model_name $(MODEL) --steps $(STEPS)

train-multinode: cuda-check
	$(CUDA_ENV) $(DEEPSPEED) --hostfile $(HOSTFILE) --master_addr $(MASTER_ADDR) train.py --model_name $(MODEL) --steps $(STEPS)

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .

check: lint test

clean:
	rm -rf .pytest_cache .ruff_cache __pycache__ llm_models/__pycache__ tests/__pycache__ checkpoints
