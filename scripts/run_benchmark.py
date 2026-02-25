#!/usr/bin/env python3
"""Run GLAD benchmark experiments with trial-and-error tuning."""

from __future__ import annotations

from pathlib import Path

from glad_network.data import load_benchmark_dataset
from glad_network.experiment import GladTrialRunner, to_serializable
from glad_network.model import GladConfig, GladNetwork
from glad_network.utils import write_json


def run_baseline(dataset):
    baseline_config = GladConfig(hidden_layer_sizes=(64,), learning_rate_init=0.001, alpha=1e-4, max_iter=200)
    baseline_model = GladNetwork(baseline_config).fit(dataset.X_train, dataset.y_train)
    baseline_acc = baseline_model.score(dataset.X_test, dataset.y_test)
    return baseline_config, baseline_acc


def main() -> None:
    dataset = load_benchmark_dataset()

    baseline_config, baseline_test_acc = run_baseline(dataset)

    runner = GladTrialRunner(random_state=42)
    trials, best = runner.run_trials(
        dataset.X_train,
        dataset.y_train,
        dataset.X_test,
        dataset.y_test,
        hidden_layer_grid=[(64,), (128, 64), (256, 128), (128, 128, 64)],
        learning_rate_grid=[0.001, 0.003],
        alpha_grid=[1e-4, 5e-4],
        max_iter=300,
        cv_folds=5,
    )

    ordered = sorted(trials, key=lambda x: x.cv_accuracy_mean, reverse=True)
    report = {
        "dataset": "sklearn_digits",
        "baseline": {
            "config": {
                "hidden_layer_sizes": list(baseline_config.hidden_layer_sizes),
                "learning_rate_init": baseline_config.learning_rate_init,
                "alpha": baseline_config.alpha,
                "max_iter": baseline_config.max_iter,
            },
            "test_accuracy": baseline_test_acc,
        },
        "best_trial": to_serializable(best),
        "all_trials": [to_serializable(trial) for trial in ordered],
        "improvement_over_baseline": best.test_accuracy - baseline_test_acc,
    }

    write_json(Path("results") / "benchmark_results.json", report)

    print("Baseline test accuracy:", f"{baseline_test_acc:.4f}")
    print("Best trial config:", best.config)
    print("Best CV accuracy:", f"{best.cv_accuracy_mean:.4f} ± {best.cv_accuracy_std:.4f}")
    print("Best test accuracy:", f"{best.test_accuracy:.4f}")
    print("Improvement over baseline:", f"{report['improvement_over_baseline']:.4f}")
    print("Detailed report written to results/benchmark_results.json")


if __name__ == "__main__":
    main()
