"""Dataset utilities for GLAD network experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class DatasetBundle:
    """Structured container for train/test splits."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


def load_benchmark_dataset(test_size: float = 0.2, random_state: int = 42) -> DatasetBundle:
    """Load the sklearn digits benchmark and produce normalized train/test splits."""

    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data,
        digits.target,
        test_size=test_size,
        stratify=digits.target,
        random_state=random_state,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return DatasetBundle(
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        y_train=y_train,
        y_test=y_test,
    )
