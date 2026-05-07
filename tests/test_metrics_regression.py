"""Tests for simpleml.metrics.regression — MSE, MAE, MAPE, R²."""
import pytest
import numpy as np

from simpleml.metrics.regression import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)


# ── mean_squared_error ────────────────────────────────────────────────────────

class TestMSE:
    def test_zero_error(self):
        assert mean_squared_error([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)

    def test_simple_value(self):
        # errors = [1, 1] → MSE = 1
        assert mean_squared_error([0, 0], [1, 1]) == pytest.approx(1.0)

    def test_symmetry(self):
        a = [1.0, 2.0, 3.0]
        b = [2.0, 3.0, 4.0]
        assert mean_squared_error(a, b) == pytest.approx(mean_squared_error(b, a))

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            mean_squared_error([1, 2], [1])

    def test_returns_float(self):
        assert isinstance(mean_squared_error([1], [2]), float)

    def test_nonnegative(self):
        assert mean_squared_error([5, 3], [1, 4]) >= 0


# ── mean_absolute_error ───────────────────────────────────────────────────────

class TestMAE:
    def test_zero_error(self):
        assert mean_absolute_error([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)

    def test_simple_value(self):
        assert mean_absolute_error([0, 0], [1, 3]) == pytest.approx(2.0)

    def test_symmetry(self):
        a = [1.0, 2.0]
        b = [3.0, 0.0]
        assert mean_absolute_error(a, b) == pytest.approx(mean_absolute_error(b, a))

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            mean_absolute_error([1, 2, 3], [1, 2])

    def test_returns_float(self):
        assert isinstance(mean_absolute_error([0], [1]), float)

    def test_mae_le_mse_for_large_errors(self):
        y_true = np.zeros(5)
        y_pred = np.array([10.0, 10.0, 10.0, 10.0, 10.0])
        assert mean_absolute_error(y_true, y_pred) <= mean_squared_error(y_true, y_pred)


# ── mean_absolute_percentage_error ───────────────────────────────────────────

class TestMAPE:
    def test_zero_error(self):
        assert mean_absolute_percentage_error([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)

    def test_simple_100_percent(self):
        # predicted = 2 * true → 100 % error
        assert mean_absolute_percentage_error([2, 4], [4, 8]) == pytest.approx(100.0)

    def test_all_zeros_true_returns_zero(self):
        # No non-zero true values → returns 0
        assert mean_absolute_percentage_error([0, 0], [1, 1]) == pytest.approx(0.0)

    def test_returns_float(self):
        assert isinstance(mean_absolute_percentage_error([1], [2]), float)

    def test_nonnegative(self):
        assert mean_absolute_percentage_error([3, 6], [1, 9]) >= 0


# ── r2_score ──────────────────────────────────────────────────────────────────

class TestR2Score:
    def test_perfect_prediction(self):
        y = [1.0, 2.0, 3.0, 4.0]
        assert r2_score(y, y) == pytest.approx(1.0)

    def test_mean_prediction_gives_zero(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.full(4, y_true.mean())
        assert r2_score(y_true, y_pred) == pytest.approx(0.0, abs=1e-10)

    def test_worse_than_mean_negative(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([5.0, 6.0, 7.0])
        assert r2_score(y_true, y_pred) < 0

    def test_constant_target_returns_zero(self):
        y_true = np.array([3.0, 3.0, 3.0])
        y_pred = np.array([3.0, 3.0, 3.0])
        # denominator=0 → returns 0.0 by definition in this implementation
        assert r2_score(y_true, y_pred) == pytest.approx(0.0)

    def test_returns_float(self):
        assert isinstance(r2_score([0, 1], [0, 1]), float)

    def test_r2_bounded_above_by_one(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 3.1])
        assert r2_score(y_true, y_pred) <= 1.0
