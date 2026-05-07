import numpy as np
from typing import Union, List

ArrayLike = Union[np.ndarray, List[float], List[List[float]]]

def mean_squared_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean squared error regression loss."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if y_true.shape != y_pred.shape:
        raise ValueError("Shape mismatch between y_true and y_pred")

    return float(np.mean((y_true - y_pred) ** 2))

def mean_absolute_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean absolute error regression loss."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if y_true.shape != y_pred.shape:
        raise ValueError("Shape mismatch between y_true and y_pred")

    return float(np.mean(np.abs(y_true - y_pred)))

def mean_absolute_percentage_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean absolute percentage error (MAPE)."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    mask = y_true != 0
    if not np.any(mask):
        return 0.0

    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def r2_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """R^2 (coefficient of determination) regression score function."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    numerator = ((y_true - y_pred) ** 2).sum()
    denominator = ((y_true - y_true.mean()) ** 2).sum()

    if denominator == 0.0:
        return 0.0

    return float(1 - (numerator / denominator))
