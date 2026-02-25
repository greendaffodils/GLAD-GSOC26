"""Convolutional-autoencoder-inspired baseline for graph events."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA

from glad_network.data import ParticleCollisionGraph


@dataclass(frozen=True)
class ConvAutoencoderConfig:
    """Configuration for the baseline model."""

    grid_size: int = 16
    latent_dim: int = 20


class ConvolutionalAutoencoderBaseline:
    """A lightweight baseline using occupancy maps + reconstruction error.

    This mimics a convolutional autoencoder pipeline by projecting each event to a 2D
    detector image, smoothing via local averaging, then learning a low-dimensional
    reconstruction model. Anomaly score = reconstruction error.
    """

    def __init__(self, config: ConvAutoencoderConfig):
        self.config = config
        self.model = PCA(n_components=config.latent_dim, random_state=42)

    def _to_image(self, graph: ParticleCollisionGraph) -> np.ndarray:
        pos = graph.node_features[:, :2]
        e = graph.node_features[:, 2]

        # Bin positions into a detector grid in range [-2,2] x [-2,2].
        bins = np.linspace(-2.0, 2.0, self.config.grid_size + 1)
        img, _, _ = np.histogram2d(pos[:, 0], pos[:, 1], bins=[bins, bins], weights=e)

        # Simple 3x3 smoothing acts as a fixed convolutional filter bank.
        padded = np.pad(img, ((1, 1), (1, 1)), mode="edge")
        smooth = np.zeros_like(img)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                smooth[i, j] = padded[i : i + 3, j : j + 3].mean()

        return smooth

    def _matrix(self, graphs: list[ParticleCollisionGraph]) -> np.ndarray:
        return np.vstack([self._to_image(g).reshape(1, -1) for g in graphs])

    def fit(self, graphs: list[ParticleCollisionGraph], labels: np.ndarray | None = None) -> "ConvolutionalAutoencoderBaseline":
        if labels is None:
            train_graphs = graphs
        else:
            train_graphs = [g for g, y in zip(graphs, labels) if y == 0]
        X = self._matrix(train_graphs)
        self.model.fit(X)
        return self

    def score_samples(self, graphs: list[ParticleCollisionGraph]) -> np.ndarray:
        X = self._matrix(graphs)
        X_hat = self.model.inverse_transform(self.model.transform(X))
        mse = np.mean((X - X_hat) ** 2, axis=1)
        return mse

    def predict(self, graphs: list[ParticleCollisionGraph], threshold: float) -> np.ndarray:
        return (self.score_samples(graphs) >= threshold).astype(int)
