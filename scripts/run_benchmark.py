#!/usr/bin/env python3
"""Run graph anomaly detection benchmark and compare against CAE baseline."""

from __future__ import annotations

from pathlib import Path

from glad_network.data import generate_particle_collision_dataset
from glad_network.experiment import run_baseline, to_serializable_graph_trial, to_serializable_metric
from glad_network.experiment import GraphTrialRunner
from glad_network.utils import write_json


def main() -> None:
    split = generate_particle_collision_dataset(
        num_normal=1000,
        num_anomalous=250,
        num_nodes=28,
        test_size=0.3,
        random_state=42,
    )

    baseline_result = run_baseline(split)

    trials, best = GraphTrialRunner(random_state=42).run_trials(
        split,
        message_steps_grid=[1, 2, 3],
        hidden_dim_grid=[8, 16, 24],
        contamination_grid=[0.1, 0.2, 0.25],
    )

    ordered_trials = sorted(trials, key=lambda t: t.metrics.roc_auc, reverse=True)

    report = {
        "dataset": "synthetic_particle_collision_graphs",
        "baseline_conv_autoencoder": to_serializable_metric(baseline_result),
        "best_graph_trial": to_serializable_graph_trial(best),
        "all_graph_trials": [to_serializable_graph_trial(t) for t in ordered_trials],
        "graph_minus_baseline_auc": best.metrics.roc_auc - baseline_result.roc_auc,
    }

    write_json(Path("results") / "benchmark_results.json", report)

    print("Baseline (CAE-like) ROC-AUC:", f"{baseline_result.roc_auc:.4f}")
    print("Best graph config:", best.config)
    print("Best graph ROC-AUC:", f"{best.metrics.roc_auc:.4f}")
    print("Graph improvement vs baseline (AUC):", f"{report['graph_minus_baseline_auc']:.4f}")
    print("Detailed benchmark saved to results/benchmark_results.json")


if __name__ == "__main__":
    main()
