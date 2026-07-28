.PHONY: setup install doctor hostfile-local train train-hostfile-local train-multinode test lint check clean

VENV ?= .venv
PYTHON ?= $(VENV)/bin/python
DEEPSPEED ?= $(VENV)/bin/deepspeed
CUDA_HOME ?= $(shell if [ -n "$$CUDA_HOME" ]; then echo "$$CUDA_HOME"; elif [ -d /usr/local/cuda ]; then echo /usr/local/cuda; else $(PYTHON) -c "import pathlib, site; print(pathlib.Path(site.getsitepackages()[0]) / 'nvidia' / 'cu13')" 2>/dev/null; fi)
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

doctor:
	$(PYTHON) --version
	$(PYTHON) -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available(), 'cuda_version', torch.version.cuda, 'gpu_count', torch.cuda.device_count())"
	$(DEEPSPEED) --version
	nvidia-smi

hostfile-local:
	printf "localhost slots=%s\n" "$$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" > $(HOSTFILE)
	cat $(HOSTFILE)

train:
	CUDA_HOME=$(CUDA_HOME) PATH=$(CUDA_HOME)/bin:$$PATH $(DEEPSPEED) --num_gpus $(GPUS) train.py --model_name $(MODEL) --steps $(STEPS)

train-hostfile-local:
	CUDA_HOME=$(CUDA_HOME) PATH=$(CUDA_HOME)/bin:$$PATH $(DEEPSPEED) --hostfile $(HOSTFILE) --no_ssh --node_rank 0 --num_nodes 1 --master_addr $(MASTER_ADDR) train.py --model_name $(MODEL) --steps $(STEPS)

train-multinode:
	CUDA_HOME=$(CUDA_HOME) PATH=$(CUDA_HOME)/bin:$$PATH $(DEEPSPEED) --hostfile $(HOSTFILE) --master_addr $(MASTER_ADDR) train.py --model_name $(MODEL) --steps $(STEPS)

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .

check: lint test

clean:
	rm -rf .pytest_cache .ruff_cache __pycache__ llm_models/__pycache__ tests/__pycache__ checkpoints
