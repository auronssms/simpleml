"""Tests for metrics module."""

import numpy as np
import pytest
from sklearn_lite.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


class TestClassificationMetrics:
    """Test classification metrics."""
    
    def test_accuracy_score_perfect(self):
        """Test accuracy with perfect predictions."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        assert accuracy_score(y_true, y_pred) == 1.0
    
    def test_accuracy_score_half(self):
        """Test accuracy with 50% correct."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 0])
        assert accuracy_score(y_true, y_pred) == 0.0
    
    def test_precision_score(self):
        """Test precision calculation."""
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 0])
        precision = precision_score(y_true, y_pred, pos_label=1)
        assert precision == 1.0  # 1 TP, 0 FP
    
    def test_recall_score(self):
        """Test recall calculation."""
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 0])
        recall = recall_score(y_true, y_pred, pos_label=1)
        assert recall == 0.5  # 1 TP, 2 actual positives
    
    def test_f1_score(self):
        """Test F1 score."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 0])
        f1 = f1_score(y_true, y_pred, pos_label=1)
        assert 0 < f1 < 1
    
    def test_confusion_matrix(self):
        """Test confusion matrix."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1])
        cm = confusion_matrix(y_true, y_pred)
        assert cm.shape == (2, 2)
        assert cm[0, 0] == 1  # True negatives
        assert cm[1, 1] == 1  # True positives


class TestRegressionMetrics:
    """Test regression metrics."""
    
    def test_mean_squared_error_perfect(self):
        """Test MSE with perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        assert mean_squared_error(y_true, y_pred) == 0.0
    
    def test_mean_absolute_error(self):
        """Test MAE."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 4.0])
        mae = mean_absolute_error(y_true, y_pred)
        assert mae == pytest.approx(1.0 / 3)
    
    def test_r2_score_perfect(self):
        """Test R2 with perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        assert r2_score(y_true, y_pred) == 1.0