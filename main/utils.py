"""Utility functions and validators for sklearn-lite."""

import numpy as np
from typing import Tuple, Union


def check_X_y(
    X: Union[list, np.ndarray],
    y: Union[list, np.ndarray],
    accept_sparse: bool = False,
    dtype: type = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Input validation for standard estimators.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        Target vector.
    accept_sparse : bool, default=False
        Whether to accept sparse matrices.
    dtype : type, optional
        Data type to convert to.
        
    Returns
    -------
    X_validated : ndarray of shape (n_samples, n_features)
        The validated feature matrix.
    y_validated : ndarray of shape (n_samples,)
        The validated target vector.
    """
    X = check_array(X, accept_sparse=accept_sparse, dtype=dtype, allow_nd=False)
    y = check_array(y, accept_sparse=False, dtype=dtype, ensure_2d=False)
    
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y must have the same number of samples. "
            f"Got X.shape[0]={X.shape[0]} and y.shape[0]={y.shape[0]}"
        )
    
    return X, y


def check_array(
    array: Union[list, np.ndarray],
    accept_sparse: bool = False,
    dtype: type = None,
    allow_nd: bool = True,
    ensure_2d: bool = True,
) -> np.ndarray:
    """
    Input validation on array, list, or similar structures.
    
    Parameters
    ----------
    array : object
        Input object to check and convert.
    accept_sparse : bool, default=False
        Whether to accept sparse matrices.
    dtype : type, optional
        Data type to convert to.
    allow_nd : bool, default=True
        Whether to allow N-dimensional arrays.
    ensure_2d : bool, default=True
        Whether to ensure the array is 2D.
        
    Returns
    -------
    array_converted : ndarray
        The converted and validated array.
    """
    if not isinstance(array, np.ndarray):
        array = np.asarray(array)
    
    if dtype is not None:
        array = array.astype(dtype)
    
    if ensure_2d and array.ndim == 1:
        array = array.reshape(-1, 1)
    
    if not allow_nd and array.ndim > 2:
        raise ValueError(f"Expected 2D array, got {array.ndim}D array instead")
    
    if np.any(np.isnan(array)):
        raise ValueError("Input contains NaN values")
    
    if np.any(np.isinf(array)):
        raise ValueError("Input contains infinity values")
    
    return array


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.25,
    random_state: int = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split arrays into random train and test subsets.
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Target vector.
    test_size : float, default=0.25
        Proportion of the dataset to include in the test split.
    random_state : int, optional
        Random seed for reproducibility.
        
    Returns
    -------
    X_train : ndarray
        Training feature matrix.
    X_test : ndarray
        Test feature matrix.
    y_train : ndarray
        Training target vector.
    y_test : ndarray
        Test target vector.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = X.shape[0]
    n_test = int(np.ceil(n_samples * test_size))
    
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def standardize(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize features by removing mean and scaling to unit variance.
    
    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix to standardize.
        
    Returns
    -------
    X_scaled : ndarray
        Scaled feature matrix.
    mean : ndarray
        Mean values used for scaling.
    std : ndarray
        Standard deviation values used for scaling.
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_scaled = (X - mean) / (std + 1e-8)
    return X_scaled, mean, std