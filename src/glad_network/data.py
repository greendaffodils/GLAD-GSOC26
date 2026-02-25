"""Data generation and preprocessing for particle-collision graph anomaly detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class ParticleCollisionGraph:
    """Single collision event represented as a graph."""

    node_features: np.ndarray  # shape: [num_nodes, 3] => (px, py, energy)
    adjacency: np.ndarray  # shape: [num_nodes, num_nodes], symmetric and binary


@dataclass(frozen=True)
class GraphDatasetSplit:
    """Train/test split for graph anomaly experiments."""

    train_graphs: list[ParticleCollisionGraph]
    train_labels: np.ndarray
    test_graphs: list[ParticleCollisionGraph]
    test_labels: np.ndarray


def _build_radius_graph(positions: np.ndarray, radius: float) -> np.ndarray:
    deltas = positions[:, None, :] - positions[None, :, :]
    distances = np.linalg.norm(deltas, axis=-1)
    adjacency = (distances < radius).astype(float)
    np.fill_diagonal(adjacency, 0.0)
    return adjacency


def _generate_event(rng: np.random.Generator, anomalous: bool, num_nodes: int = 28) -> ParticleCollisionGraph:
    if anomalous:
        # Anomalous events exhibit shifted momentum and concentrated jets.
        centers = np.array([[1.2, 1.2], [-1.0, -1.1]])
        assignments = rng.integers(0, len(centers), size=num_nodes)
        positions = centers[assignments] + rng.normal(scale=0.22, size=(num_nodes, 2))
        energy = rng.lognormal(mean=1.0, sigma=0.45, size=(num_nodes, 1))
        radius = 0.48
    else:
        # Normal events are more diffuse and lower-energy.
        positions = rng.normal(loc=0.0, scale=0.75, size=(num_nodes, 2))
        energy = rng.lognormal(mean=0.25, sigma=0.25, size=(num_nodes, 1))
        radius = 0.62

    adjacency = _build_radius_graph(positions, radius=radius)
    node_features = np.concatenate([positions, energy], axis=1)
    return ParticleCollisionGraph(node_features=node_features, adjacency=adjacency)


def generate_particle_collision_dataset(
    num_normal: int = 1000,
    num_anomalous: int = 250,
    num_nodes: int = 28,
    test_size: float = 0.3,
    random_state: int = 42,
) -> GraphDatasetSplit:
    """Create a synthetic benchmark of particle-collision graph events."""

    rng = np.random.default_rng(random_state)
    normal_events = [_generate_event(rng, anomalous=False, num_nodes=num_nodes) for _ in range(num_normal)]
    anomaly_events = [_generate_event(rng, anomalous=True, num_nodes=num_nodes) for _ in range(num_anomalous)]

    graphs = normal_events + anomaly_events
    labels = np.array([0] * num_normal + [1] * num_anomalous, dtype=int)

    idx = np.arange(len(graphs))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=test_size,
        stratify=labels,
        random_state=random_state,
    )

    train_graphs = [graphs[i] for i in train_idx]
    test_graphs = [graphs[i] for i in test_idx]

    return GraphDatasetSplit(
        train_graphs=train_graphs,
        train_labels=labels[train_idx],
        test_graphs=test_graphs,
        test_labels=labels[test_idx],
    )
