"""Tests for simpleml.utils.validation — check_array, check_X_y."""
import pytest
import numpy as np

from simpleml.utils.validation import check_array, check_X_y


# ── check_array ──────────────────────────────────────────────────────────────

class TestCheckArray:
    # --- basic conversion ---

    def test_list_converted_to_ndarray(self):
        result = check_array([[1, 2], [3, 4]])
        assert isinstance(result, np.ndarray)

    def test_nested_list_2d_shape(self):
        result = check_array([[1, 2, 3], [4, 5, 6]])
        assert result.shape == (2, 3)

    def test_existing_ndarray_returned(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = check_array(X)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)

    # --- 1-D input promotion ---

    def test_1d_array_promoted_to_2d_column(self):
        result = check_array([1.0, 2.0, 3.0])
        assert result.shape == (3, 1)

    def test_1d_ndarray_promoted(self):
        result = check_array(np.array([5.0, 6.0]))
        assert result.ndim == 2

    # --- dtype enforcement ---

    def test_integer_array_converted_to_float(self):
        result = check_array([[1, 2], [3, 4]])
        assert np.issubdtype(result.dtype, np.number)

    def test_float_array_unchanged_dtype(self):
        X = np.array([[1.0, 2.0]])
        result = check_array(X)
        assert np.issubdtype(result.dtype, np.floating)

    # --- copy behaviour ---

    def test_copy_true_creates_new_array(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = check_array(X, copy=True)
        result[0, 0] = 999.0
        assert X[0, 0] != 999.0

    # --- error cases ---

    def test_0d_array_raises(self):
        with pytest.raises(ValueError, match="Singleton"):
            check_array(np.array(5))

    def test_non_numeric_raises(self):
        with pytest.raises(ValueError):
            check_array([["a", "b"], ["c", "d"]])

    # --- ensure_2d=False ---

    def test_ensure_2d_false_keeps_1d(self):
        result = check_array([1, 2, 3], ensure_2d=False)
        assert result.ndim == 1


# ── check_X_y ────────────────────────────────────────────────────────────────

class TestCheckXy:
    def test_basic_valid_input(self):
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])
        X_out, y_out = check_X_y(X, y)
        assert X_out.shape == (3, 2)
        assert y_out.shape == (3,)

    def test_list_inputs_converted(self):
        X, y = check_X_y([[1, 2], [3, 4]], [0, 1])
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)

    def test_2d_y_flattened(self):
        X = np.ones((4, 2))
        y = np.array([[0], [1], [0], [1]])
        _, y_out = check_X_y(X, y)
        assert y_out.ndim == 1

    def test_mismatched_lengths_raise(self):
        X = np.ones((5, 2))
        y = np.ones(3)
        with pytest.raises(ValueError, match="inconsistent"):
            check_X_y(X, y)

    def test_X_becomes_2d(self):
        X = [1.0, 2.0, 3.0]
        y = [0, 1, 0]
        X_out, _ = check_X_y(X, y)
        assert X_out.ndim == 2

    def test_returns_numeric_dtype(self):
        X, y = check_X_y([[1, 2], [3, 4]], [0, 1])
        assert np.issubdtype(X.dtype, np.number)

    def test_single_sample_valid(self):
        X = np.array([[1.0, 2.0]])
        y = np.array([0])
        X_out, y_out = check_X_y(X, y)
        assert X_out.shape == (1, 2)
        assert y_out.shape == (1,)
