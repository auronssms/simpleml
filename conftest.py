"""Pytest configuration and shared fixtures."""
import sys
import os

# Allow tests to import simpleml from the src directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import numpy as np


# ── Shared fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def simple_regression_data():
    """y = 2*x1 + 3*x2 + 1  (noise-free)."""
    rng = np.random.RandomState(0)
    X = rng.randn(100, 2)
    y = 2 * X[:, 0] + 3 * X[:, 1] + 1.0
    return X, y


@pytest.fixture
def binary_classification_data():
    """Linearly separable 2-class dataset."""
    rng = np.random.RandomState(42)
    X0 = rng.randn(50, 2) - 2
    X1 = rng.randn(50, 2) + 2
    X = np.vstack([X0, X1])
    y = np.array([0] * 50 + [1] * 50)
    return X, y


@pytest.fixture
def multiclass_data():
    """3-class dataset."""
    rng = np.random.RandomState(7)
    X = np.vstack([
        rng.randn(40, 2) + np.array([0, 0]),
        rng.randn(40, 2) + np.array([5, 5]),
        rng.randn(40, 2) + np.array([10, 0]),
    ])
    y = np.array([0] * 40 + [1] * 40 + [2] * 40)
    return X, y


@pytest.fixture
def cluster_data():
    """Three well-separated clusters."""
    rng = np.random.RandomState(0)
    X = np.vstack([
        rng.randn(30, 2) + np.array([0, 0]),
        rng.randn(30, 2) + np.array([8, 0]),
        rng.randn(30, 2) + np.array([4, 8]),
    ])
    return X
