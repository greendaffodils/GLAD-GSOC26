# GLAD Network Benchmarking

This repository now contains a small, organized experimentation pipeline to:

1. Build a **GLAD network** (a feed-forward neural model wrapper).
2. Train and evaluate it on a **benchmark dataset** (`sklearn` digits).
3. Run **trial-and-error tuning** over multiple hyperparameter combinations.
4. Save structured results in `results/benchmark_results.json`.

## Project structure

- `src/glad_network/data.py` - benchmark data loading and normalization.
- `src/glad_network/model.py` - GLAD network config + model wrapper.
- `src/glad_network/experiment.py` - trial runner and result objects.
- `src/glad_network/utils.py` - utility functions.
- `scripts/run_benchmark.py` - end-to-end benchmark and tuning run.
- `tests/test_glad_pipeline.py` - regression test for model quality floor.
- `results/benchmark_results.json` - generated benchmark output.

## Run benchmark

```bash
PYTHONPATH=src python scripts/run_benchmark.py
```

## Run tests

```bash
PYTHONPATH=src pytest -q
```

## Current outcome

The script compares a baseline network against tuned candidates and reports:

- baseline test accuracy,
- best tuned config,
- best cross-validation performance,
- test accuracy of the best config,
- improvement over baseline.

Check `results/benchmark_results.json` for all trial metrics.
