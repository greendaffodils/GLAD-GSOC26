# GLAD Graph Anomaly Detection Benchmark

This repository contains an organized workflow for **graph-based anomaly detection on particle collision events** with:

1. A graph architecture using GNN-style message passing + Isolation Forest.
2. A benchmark dataset generator for synthetic particle-collision graphs.
3. Trial-and-error hyperparameter search over graph model settings.
4. Comparison against a **convolutional autoencoder-inspired** baseline.
5. Structured benchmark outputs in `results/benchmark_results.json`.

## Project structure

- `src/glad_network/data.py` - synthetic particle-collision graph dataset generation.
- `src/glad_network/model.py` - graph anomaly detector and message-passing embedding model.
- `src/glad_network/baseline.py` - CAE-like baseline (detector image projection + reconstruction).
- `src/glad_network/experiment.py` - benchmarking and trial runner.
- `scripts/run_benchmark.py` - full experiment script.
- `tests/test_glad_pipeline.py` - regression test for quality floor + baseline comparison.
- `results/benchmark_results.json` - generated benchmark report.

## Run benchmark

```bash
PYTHONPATH=src python scripts/run_benchmark.py
```

## Run tests

```bash
PYTHONPATH=src pytest -q
```

## Benchmark outputs

The benchmark report includes:

- baseline CAE-like ROC-AUC / AP / F1,
- best graph trial config and metrics,
- metrics for all graph trials,
- graph-vs-baseline AUC improvement.
