"""GLAD network model definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from sklearn.neural_network import MLPClassifier


@dataclass(frozen=True)
class GladConfig:
    """Configuration for a GLAD network run."""

    hidden_layer_sizes: Sequence[int] = (128, 64)
    learning_rate_init: float = 0.001
    alpha: float = 1e-4
    max_iter: int = 200
    random_state: int = 42


class GladNetwork:
    """Simple feed-forward network wrapper used in experiments."""

    def __init__(self, config: GladConfig) -> None:
        self.config = config
        self.model = MLPClassifier(
            hidden_layer_sizes=tuple(config.hidden_layer_sizes),
            learning_rate_init=config.learning_rate_init,
            alpha=config.alpha,
            max_iter=config.max_iter,
            random_state=config.random_state,
        )

    def fit(self, X, y) -> "GladNetwork":
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y) -> float:
        return float(self.model.score(X, y))
