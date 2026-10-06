# Distributed LLM Training Infrastructure
A compact but real PyTorch training stack demonstrating the mechanics used in multi-GPU transformer training: `torchrun`, DistributedDataParallel, `DistributedSampler`, rank-aware device assignment, gradient clipping, throughput measurement, reproducible configs, MLflow logging, and Optuna tuning.

## Run CPU/single process
```bash
pip install -e .
python train.py --config configs/tiny.yaml
pytest -q
```

## Run multi-GPU
```bash
torchrun --standalone --nproc_per_node=4 train.py --config configs/gpu.yaml
```
The code automatically initializes NCCL on CUDA and Gloo otherwise, binds each process to `LOCAL_RANK`, wraps the model in DDP, and partitions data with `DistributedSampler`.

## Experiment tracking / tuning
```bash
pip install -r requirements-mlops.txt
mlflow ui
python scripts/mlflow_train.py
python scripts/optuna_search.py
```
W&B can be integrated at the same metric emission point in `src/trainer.py` without changing training semantics.

## Why a tiny transformer is included
CI should verify distributed-training logic without downloading a multi-billion-parameter model. The model is deliberately small, while the DDP lifecycle, optimizer path, sampler, device mapping and launch command are the same patterns used for larger Hugging Face models.

## Resume result
The resume reports a 35% training-time reduction from optimizing transformer training across multi-GPU CUDA environments. That number is an **original project benchmark**, not a number synthesized by this repository. Use `elapsed_s` and `steps_per_s` to regenerate scaling curves on the original hardware and model.
