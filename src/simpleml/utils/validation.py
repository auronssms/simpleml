import numpy as np
from typing import Union, Tuple, List

ArrayLike = Union[np.ndarray, List[float], List[List[float]]]

def check_array(X: ArrayLike, ensure_2d: bool = True, copy: bool = False) -> np.ndarray:
    """
    Input validation for standard estimators.

    Parameters
    ----------
    X : array-like
        The input data to check.
    ensure_2d : bool, default=True
        Whether to raise an error if X is not 2D, or reshape 1D arrays.
    copy : bool, default=False
        Whether a forced copy will be triggered. If False, a copy might 
        still be made if the input is not already a numeric numpy array.
    """
    X_arr = np.array(X, dtype=np.float64, copy=True) if copy else np.array(X)
    
    if X_arr.ndim == 0:
        raise ValueError("Singleton array (0-D) is not a valid input.")
        
    if ensure_2d:
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
    
    if not np.issubdtype(X_arr.dtype, np.number):
        try:
            X_arr = X_arr.astype(np.float64)
        except ValueError:
            raise ValueError("Input contains non-numeric data.")

    return X_arr

def check_X_y(X: ArrayLike, y: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    """
    Input validation for standard estimators.
    Checks X and y for consistent length, enforces X to be 2D and y to be 1D.
    """
    X_arr = check_array(X, ensure_2d=True)
    y_arr = np.asarray(y, dtype=np.float64)

    if y_arr.ndim > 1:
        y_arr = y_arr.ravel()
    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(f"Found input variables with inconsistent numbers of samples:[{X_arr.shape[0]}, {y_arr.shape[0]}]")

    return X_arr, y_arr 