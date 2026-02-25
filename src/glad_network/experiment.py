"""Benchmark runner and trial-and-error tuning for anomaly detection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from glad_network.baseline import ConvAutoencoderConfig, ConvolutionalAutoencoderBaseline
from glad_network.data import GraphDatasetSplit
from glad_network.model import GraphAnomalyConfig, GraphAnomalyDetector


@dataclass(frozen=True)
class BenchmarkResult:
    model_name: str
    roc_auc: float
    average_precision: float
    f1: float
    threshold: float


@dataclass(frozen=True)
class GraphTrialResult:
    config: GraphAnomalyConfig
    metrics: BenchmarkResult


class BenchmarkRunner:
    """Runs benchmark comparison between graph model and CAE baseline."""

    def evaluate(self, detector, split: GraphDatasetSplit, model_name: str) -> BenchmarkResult:
        train_scores = detector.score_samples(split.train_graphs)
        threshold = float(np.percentile(train_scores, 90))

        test_scores = detector.score_samples(split.test_graphs)
        preds = (test_scores >= threshold).astype(int)
        y_true = split.test_labels

        return BenchmarkResult(
            model_name=model_name,
            roc_auc=float(roc_auc_score(y_true, test_scores)),
            average_precision=float(average_precision_score(y_true, test_scores)),
            f1=float(f1_score(y_true, preds)),
            threshold=threshold,
        )


class GraphTrialRunner:
    """Trial-and-error tuning for graph model hyperparameters."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.benchmark = BenchmarkRunner()

    def run_trials(
        self,
        split: GraphDatasetSplit,
        message_steps_grid: list[int],
        hidden_dim_grid: list[int],
        contamination_grid: list[float],
    ) -> tuple[list[GraphTrialResult], GraphTrialResult]:
        results: list[GraphTrialResult] = []

        for steps, hidden, contamination in product(message_steps_grid, hidden_dim_grid, contamination_grid):
            config = GraphAnomalyConfig(
                message_passing_steps=steps,
                hidden_dim=hidden,
                contamination=contamination,
                random_state=self.random_state,
            )
            detector = GraphAnomalyDetector(config).fit(split.train_graphs, split.train_labels)
            metrics = self.benchmark.evaluate(detector, split, model_name="graph_detector")
            results.append(GraphTrialResult(config=config, metrics=metrics))

        best = max(results, key=lambda r: r.metrics.roc_auc)
        return results, best


def run_baseline(split: GraphDatasetSplit) -> BenchmarkResult:
    baseline = ConvolutionalAutoencoderBaseline(ConvAutoencoderConfig()).fit(split.train_graphs, split.train_labels)
    return BenchmarkRunner().evaluate(baseline, split, model_name="conv_autoencoder_baseline")


def to_serializable_graph_trial(result: GraphTrialResult) -> dict:
    return {
        "config": asdict(result.config),
        "metrics": asdict(result.metrics),
    }


def to_serializable_metric(metric: BenchmarkResult) -> dict:
    return asdict(metric)
