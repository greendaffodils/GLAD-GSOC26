"""Graph-based anomaly detector for particle-collision events."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import IsolationForest

from glad_network.data import ParticleCollisionGraph


@dataclass(frozen=True)
class GraphAnomalyConfig:
    """Configuration of the graph anomaly detector."""

    message_passing_steps: int = 2
    hidden_dim: int = 16
    contamination: float = 0.2
    random_state: int = 42


class GraphFeatureExtractor:
    """Simple GNN-style message passing to obtain graph-level embeddings."""

    def __init__(self, message_passing_steps: int = 2, hidden_dim: int = 16, random_state: int = 42):
        self.message_passing_steps = message_passing_steps
        self.hidden_dim = hidden_dim
        rng = np.random.default_rng(random_state)
        self.W_in = rng.normal(scale=0.35, size=(3, hidden_dim))
        self.W_msg = rng.normal(scale=0.35, size=(hidden_dim, hidden_dim))

    def transform(self, graph: ParticleCollisionGraph) -> np.ndarray:
        X = graph.node_features
        A = graph.adjacency
        deg = A.sum(axis=1)
        deg_inv_sqrt = np.diag(1.0 / np.sqrt(deg + 1e-8))
        A_hat = deg_inv_sqrt @ (A + np.eye(A.shape[0])) @ deg_inv_sqrt

        H = np.tanh(X @ self.W_in)
        for _ in range(self.message_passing_steps):
            H = np.tanh(A_hat @ H @ self.W_msg)

        graph_stats = np.array(
            [
                A.sum() / (A.shape[0] ** 2),
                np.mean(deg),
                np.std(deg),
                np.mean(X[:, 2]),
                np.std(X[:, 2]),
            ]
        )

        pooled = np.concatenate([H.mean(axis=0), H.std(axis=0), graph_stats], axis=0)
        return pooled


class GraphAnomalyDetector:
    """Graph-based anomaly detector trained on mostly-normal events."""

    def __init__(self, config: GraphAnomalyConfig):
        self.config = config
        self.extractor = GraphFeatureExtractor(
            message_passing_steps=config.message_passing_steps,
            hidden_dim=config.hidden_dim,
            random_state=config.random_state,
        )
        self.model = IsolationForest(
            contamination=config.contamination,
            random_state=config.random_state,
            n_estimators=250,
        )

    def _embed(self, graphs: list[ParticleCollisionGraph]) -> np.ndarray:
        return np.vstack([self.extractor.transform(g) for g in graphs])

    def fit(self, graphs: list[ParticleCollisionGraph], labels: np.ndarray | None = None) -> "GraphAnomalyDetector":
        if labels is None:
            train_graphs = graphs
        else:
            train_graphs = [g for g, y in zip(graphs, labels) if y == 0]

        X = self._embed(train_graphs)
        self.model.fit(X)
        return self

    def score_samples(self, graphs: list[ParticleCollisionGraph]) -> np.ndarray:
        X = self._embed(graphs)
        # IsolationForest score_samples: higher for normal points -> invert for anomaly score.
        return -self.model.score_samples(X)

    def predict(self, graphs: list[ParticleCollisionGraph], threshold: float) -> np.ndarray:
        scores = self.score_samples(graphs)
        return (scores >= threshold).astype(int)
