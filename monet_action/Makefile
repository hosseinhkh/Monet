# Monet-Action developer targets (run from monet_action/). `make help` lists them.
VENV ?= $(HOME)/venvs/monet_action
# Python: an active conda env (other than base) wins, then $(VENV), then whatever `python` is on PATH.
ifneq ($(and $(CONDA_PREFIX),$(filter-out base,$(CONDA_DEFAULT_ENV))),)
PY ?= $(CONDA_PREFIX)/bin/python
endif
PY ?= $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,python)
MUJOCO_GL ?= egl
export PYTHONPATH := $(CURDIR)/src
export TOKENIZERS_PARALLELISM := false
export TRANSFORMERS_VERBOSITY := error
export MUJOCO_GL
export PYOPENGL_PLATFORM := $(MUJOCO_GL)
PYTEST := $(PY) -m pytest -q -p no:warnings
RUN ?= runs/d1_k10
CPU := MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa

.PHONY: help test-fast test-all test-tokens test-monet test-monet7b test-gradients test-dit test-data test-tool \
        test-sim test-train test-dashboard audit-cpu audit lint format dashboard bundle cpu-validation \
        preflight preflight-cpu dry-run train-all evaluate-all

help: ## list targets
	@grep -E '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-16s %s\n", $$1, $$2}'

# ---------------------------------------------------------------- tests, one per component
test-fast: ## all CPU tests except slow / sim / gpu (~30 s)
	$(PYTEST) -m "not slow and not sim and not gpu" tests
test-all: ## every test this machine can run (gpu tests self-skip without CUDA + Monet-7B weights)
	$(PYTEST) -rs tests
test-tokens: ## Monet-7B tokenizer, special tokens, prompt / latent-slot layout
	$(PYTEST) tests/test_tokens.py
test-monet: ## latent recurrence with Monet's real code (tiny config): self-feed, parity, state dependence
	$(PYTEST) tests/test_monet_latents.py
test-monet7b: ## REAL Monet-7B on the GPU (weights required) -> results/audit/monet7b_gpu_tests.json
	$(PYTEST) -rs tests/test_monet7b_gpu.py
test-gradients: ## FM loss -> DiT -> conditioner -> latent loop -> LoRA
	$(PYTEST) tests/test_gradients.py
test-dit: ## flow matching math, DiT, masks, conditioner, interventions
	$(PYTEST) tests/test_flow_matching.py tests/test_masks.py
test-data: ## LeRobot LIBERO data conventions, splits, normalisation
	$(PYTEST) tests/test_data.py tests/test_normalization.py
test-tool: ## CALL_ACTION / TOOL_RESULT runtime, failure tags, bypass diagnostics
	$(PYTEST) tests/test_tool_loop.py tests/test_tool_result.py
test-sim: ## LIBERO / robosuite / MuJoCo: controller, determinism, orientation, demo replay
	$(PYTEST) tests/test_libero_sim.py
test-train: ## overfit + checkpoint reload + logging / dashboard file contract
	$(PYTEST) tests/test_train_smoke.py tests/test_logging.py
test-dashboard: ## dashboard API + debug bundles
	$(PYTEST) tests/test_dashboard.py

# ---------------------------------------------------------------- audits and tools
audit-cpu: ## G0 audit with the tiny Monet (CPU, real data + real simulator)
	$(CPU) $(PY) -m monet_action.audit --config configs/audit.yaml --device cpu --out results/audit \
	  --override monet.model_path=tiny-random monet.dtype=float32 prompt.image_size=112
audit: ## G0 audit with the REAL Monet-7B (GPU)
	bash scripts/audit_upstream.sh
dashboard: ## monitoring dashboard on http://127.0.0.1:8765 (ssh -L 8765:localhost:8765 server)
	$(PY) -m monet_action.dashboard --runs runs --results results
bundle: ## debug zip of a run or eval dir: make bundle RUN=runs/d1_k10
	$(PY) -m monet_action.bundle $(RUN) -o $(notdir $(RUN))_debug_bundle.zip
cpu-validation: ## everything validated without a GPU (tests, audit, overfit, closed-loop trace)
	bash scripts/cpu_validation.sh
lint: ## ruff (style + pyflakes) on the package and tests
	$(PY) -m ruff check src tests
format: ## ruff autofix + import sorting
	$(PY) -m ruff check --fix src tests

# ---------------------------------------------------------------- pipeline
preflight: ## full GPU preflight (env, data, weights, render, tests, Monet-7B, audit, memory, dry run, smoke)
	bash scripts/preflight.sh
preflight-cpu: ## preflight self-test on a CPU machine (tiny model, GPU steps skipped)
	bash scripts/preflight.sh --cpu
dry-run: ## whole pipeline for a few steps (MODEL=real on the GPU server; MODEL=tiny elsewhere)
	MODEL=$(or $(MODEL),real) bash scripts/dry_run.sh
train-all: ## all Phase-1 training runs in dependency order (resumable)
	bash scripts/train_all.sh
evaluate-all: ## full closed-loop evaluation + report
	bash scripts/evaluate_all.sh
