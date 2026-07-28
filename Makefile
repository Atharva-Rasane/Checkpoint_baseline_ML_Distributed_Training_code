.PHONY: setup install cuda-check doctor hostfile-local deepspeed-env train train-hostfile-local train-multinode train-async train-async-hostfile-local train-async-multinode collect visualize serve upload upload-existing test lint check clean

VENV ?= .venv
PYTHON ?= $(VENV)/bin/python
DEEPSPEED ?= $(VENV)/bin/deepspeed
CUDA_HOME ?= $(shell if [ -n "$$CUDA_HOME" ]; then echo "$$CUDA_HOME"; elif [ -d /usr/local/cuda ]; then echo /usr/local/cuda; elif [ -d /usr/local/cuda-12.6 ]; then echo /usr/local/cuda-12.6; elif [ -d /usr/local/cuda-12.8 ]; then echo /usr/local/cuda-12.8; fi)
CUDA_ENV = $(if $(CUDA_HOME),CUDA_HOME=$(CUDA_HOME) PATH=$(abspath $(VENV))/bin:$(CUDA_HOME)/bin:$$PATH,)
GPUS ?= 1
HOSTFILE ?= hostfile
MODEL ?= tiny_gpt
JOB ?= $(MODEL)
ASYNC_JOB ?= $(MODEL)-async
STEPS ?= 100
BATCH_SIZE ?= 2
SEED ?= 1234
RESUME ?=
SAVE_EVERY ?= 100
ASYNC_OUTPUT_DIR ?= checkpoints/asynchronous
BENCHMARK_MB ?= 64
MASTER_ADDR ?= localhost
TRACE ?= 0
TRACE_WARMUP ?= 3
TRACE_DIR ?= visualization/traces
RUN_ID ?= $(shell date -u +%Y%m%dT%H%M%SZ)
TRACE_ARGS = $(if $(filter 1 true yes,$(TRACE)),--trace --trace_warmup_steps $(TRACE_WARMUP) --trace_dir $(TRACE_DIR) --run_id $(RUN_ID),)
RESUME_ARGS = $(if $(RESUME),--resume $(RESUME),)
TRAIN_ARGS = --model_name $(MODEL) --job_name $(JOB) --steps $(STEPS) --batch_size $(BATCH_SIZE) --seed $(SEED) --save_every $(SAVE_EVERY) --hardware_benchmark_mb $(BENCHMARK_MB) $(RESUME_ARGS) $(TRACE_ARGS)
ASYNC_TRAIN_ARGS = --model_name $(MODEL) --job_name $(ASYNC_JOB) --output_dir $(ASYNC_OUTPUT_DIR) --steps $(STEPS) --batch_size $(BATCH_SIZE) --seed $(SEED) --save_every $(SAVE_EVERY) --hardware_benchmark_mb $(BENCHMARK_MB) $(RESUME_ARGS) $(TRACE_ARGS)
SYNC_TRAIN = checkpointing/synchronous/train.py
ASYNC_TRAIN = checkpointing/asynchronous/train.py
VIS_DIR ?= visualization
REMOTE_DIR ?= $(CURDIR)
SERVE_HOST ?= 127.0.0.1
PORT ?= 8000
BUCKET ?= gbc-oit-rc-basil-app-bo-training-traces
UPLOAD_PREFIX ?= $(shell hostname -s)_$(shell date -u +%Y%m%d_%H%M%S)
COLLECT_RUN_ID ?=
COLLECT_RUN_ARG = $(if $(COLLECT_RUN_ID),--run-id $(COLLECT_RUN_ID),)

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
	$(CUDA_ENV) ninja --version
	$(CUDA_ENV) $(PYTHON) -c "import deepspeed; print('deepspeed', deepspeed.__version__)"
	nvidia-smi
	$(CUDA_HOME)/bin/nvcc --version

hostfile-local:
	printf "localhost slots=%s\n" "$$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" > $(HOSTFILE)
	cat $(HOSTFILE)

deepspeed-env: cuda-check
	printf 'PATH=%s:%s/bin:$$PATH\nCUDA_HOME=%s\n' "$(abspath $(VENV))/bin" "$(CUDA_HOME)" "$(CUDA_HOME)" > .deepspeed_env

train: cuda-check
	$(CUDA_ENV) $(DEEPSPEED) --num_gpus $(GPUS) $(SYNC_TRAIN) $(TRAIN_ARGS)

train-hostfile-local: cuda-check hostfile-local
	$(CUDA_ENV) $(DEEPSPEED) --hostfile $(HOSTFILE) --no_ssh --node_rank 0 --num_nodes 1 --master_addr $(MASTER_ADDR) $(SYNC_TRAIN) $(TRAIN_ARGS)

train-multinode: deepspeed-env
	$(CUDA_ENV) $(DEEPSPEED) --hostfile $(HOSTFILE) --master_addr $(MASTER_ADDR) $(SYNC_TRAIN) $(TRAIN_ARGS)

train-async: cuda-check
	$(CUDA_ENV) $(DEEPSPEED) --num_gpus $(GPUS) $(ASYNC_TRAIN) $(ASYNC_TRAIN_ARGS)

train-async-hostfile-local: cuda-check hostfile-local
	$(CUDA_ENV) $(DEEPSPEED) --hostfile $(HOSTFILE) --no_ssh --node_rank 0 --num_nodes 1 --master_addr $(MASTER_ADDR) $(ASYNC_TRAIN) $(ASYNC_TRAIN_ARGS)

train-async-multinode: deepspeed-env
	$(CUDA_ENV) $(DEEPSPEED) --hostfile $(HOSTFILE) --master_addr $(MASTER_ADDR) $(ASYNC_TRAIN) $(ASYNC_TRAIN_ARGS)

collect:
	$(PYTHON) $(VIS_DIR)/trace_tools.py collect --hostfile $(HOSTFILE) --source $(TRACE_DIR) --output $(VIS_DIR)/collected --remote-dir $(REMOTE_DIR) $(COLLECT_RUN_ARG)

visualize: collect
	$(PYTHON) $(VIS_DIR)/trace_tools.py build --data $(VIS_DIR)/collected --output $(VIS_DIR)/index.html $(COLLECT_RUN_ARG)

serve: visualize
	$(PYTHON) $(VIS_DIR)/trace_tools.py serve --directory $(VIS_DIR) --host $(SERVE_HOST) --port $(PORT)

upload: visualize
	$(PYTHON) $(VIS_DIR)/trace_tools.py upload --source $(VIS_DIR) --bucket $(BUCKET) --prefix $(UPLOAD_PREFIX) $(COLLECT_RUN_ARG)

upload-existing:
	$(PYTHON) $(VIS_DIR)/trace_tools.py build --data $(VIS_DIR)/collected --output $(VIS_DIR)/index.html $(COLLECT_RUN_ARG)
	$(PYTHON) $(VIS_DIR)/trace_tools.py upload --source $(VIS_DIR) --bucket $(BUCKET) --prefix $(UPLOAD_PREFIX) $(COLLECT_RUN_ARG)

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .

check: lint test

clean:
	rm -rf .pytest_cache .ruff_cache __pycache__ llm_models/__pycache__ tests/__pycache__ checkpoints visualization/traces visualization/collected visualization/nodes visualization/index.html visualization/plotly.min.js visualization/simulator-profile.json
