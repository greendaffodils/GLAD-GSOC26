"""Experiment runner for trial-and-error GLAD network tuning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Iterable, Sequence

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score

from glad_network.model import GladConfig, GladNetwork


@dataclass(frozen=True)
class TrialResult:
    config: GladConfig
    cv_accuracy_mean: float
    cv_accuracy_std: float
    test_accuracy: float


class GladTrialRunner:
    """Run multiple GLAD configs and keep the best one by CV accuracy."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def run_trials(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        hidden_layer_grid: Iterable[Sequence[int]],
        learning_rate_grid: Iterable[float],
        alpha_grid: Iterable[float],
        max_iter: int = 250,
        cv_folds: int = 5,
    ) -> tuple[list[TrialResult], TrialResult]:
        results: list[TrialResult] = []
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)

        for hidden_layers, lr, alpha in product(hidden_layer_grid, learning_rate_grid, alpha_grid):
            config = GladConfig(
                hidden_layer_sizes=tuple(hidden_layers),
                learning_rate_init=lr,
                alpha=alpha,
                max_iter=max_iter,
                random_state=self.random_state,
            )
            model = GladNetwork(config).model
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=None)

            fitted = GladNetwork(config).fit(X_train, y_train)
            test_acc = fitted.score(X_test, y_test)

            results.append(
                TrialResult(
                    config=config,
                    cv_accuracy_mean=float(np.mean(scores)),
                    cv_accuracy_std=float(np.std(scores)),
                    test_accuracy=test_acc,
                )
            )

        best = max(results, key=lambda r: r.cv_accuracy_mean)
        return results, best


def to_serializable(result: TrialResult) -> dict:
    payload = asdict(result)
    payload["config"]["hidden_layer_sizes"] = list(payload["config"]["hidden_layer_sizes"])
    return payload
