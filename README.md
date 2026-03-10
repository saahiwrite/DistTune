# 🚀 DistTune — Distributed Training Framework for Multi-GPU Fine-Tuning

A scalable distributed training framework for multi-GPU model fine-tuning with automated experiment tracking (MLflow & Weights & Biases) and hyperparameter optimization (Optuna).

> **Status:** 🚧 In Progress

---

## Features

- **Multi-GPU Distributed Training** — PyTorch DistributedDataParallel (DDP) and FSDP support
- **Experiment Tracking** — Unified logging to MLflow and Weights & Biases
- **Hyperparameter Optimization** — Optuna-based search across 50+ configurations
- **Config-Driven Workflows** — YAML-based experiment definitions
- **Checkpoint Management** — Auto-save, resume, and best-model selection
- **Mixed Precision** — Native AMP support for faster training
- **Modular Architecture** — Swap models, datasets, and optimizers via config

## Quick Start

### Installation

```bash
git clone https://github.com/<your-username>/disttune.git
cd disttune
pip install -e ".[dev]"
```

### Single-GPU Training

```bash
python scripts/train.py --config configs/experiments/bert_finetune.yaml
```

### Multi-GPU Training (DDP)

```bash
torchrun --nproc_per_node=4 scripts/train.py --config configs/experiments/bert_finetune.yaml
```

### Hyperparameter Sweep

```bash
python scripts/sweep.py --config configs/sweep.yaml --n-trials 50
```

### Tracking Dashboards

```bash
# MLflow
mlflow ui --port 5000

# Weights & Biases — results auto-sync to your W&B project
```

## Configuration

Experiments are defined in YAML. See `configs/` for examples.

## Development

```bash
make install    # Install with dev dependencies
make test       # Run test suite
make lint       # Ruff + mypy
make format     # Ruff format
```

## Roadmap

- [x] Core DDP training loop
- [x] MLflow + W&B unified tracking
- [x] Optuna hyperparameter sweeps
- [ ] FSDP support for large models
- [ ] DeepSpeed integration
- [ ] Elastic training (fault tolerance)
- [ ] Multi-node training support

## License

MIT
