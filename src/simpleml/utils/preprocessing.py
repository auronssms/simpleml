import numpy as np
from typing import Optional, Union, Tuple, List
from itertools import combinations_with_replacement, combinations
from ..core.base import BaseEstimator, TransformerMixin
from ..utils.validation import check_array

ArrayLike = Union[np.ndarray, List[float], List[List[float]]]

class MinMaxScaler(BaseEstimator, TransformerMixin):
    """
    Transform features by scaling each feature to a given range, default [0, 1].

    Attributes
    ----------
    min_ : np.ndarray
        Per feature minimum seen in the data.
    scale_ : np.ndarray
        Per feature relative scaling of the data.
    data_min_ : np.ndarray
        Per feature minimum seen in the data.
    data_max_ : np.ndarray
        Per feature maximum seen in the data.
    """

    def __init__(self) -> None:
        self.min_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.data_min_: Optional[np.ndarray] = None
        self.data_max_: Optional[np.ndarray] = None

    def fit(self, X: Union[np.ndarray, list], y: Optional[np.ndarray] = None) -> "MinMaxScaler":
        """
        Compute the minimum and maximum to be used for later scaling.
        """
        X_arr = check_array(X)

        self.data_min_ = np.min(X_arr, axis=0)
        self.data_max_ = np.max(X_arr, axis=0)

        data_range = self.data_max_ - self.data_min_
        data_range[data_range == 0.0] = 1.0

        self.scale_ = 1.0 / data_range
        self.min_ = -self.data_min_ * self.scale_

        return self

    def transform(self, X: Union[np.ndarray, list]) -> np.ndarray:
        """
        Scale features of X according to feature_range.
        """
        if self.scale_ is None or self.min_ is None:
            raise RuntimeError("MinMaxScaler is not fitted yet.")

        X_arr = check_array(X, copy=True)
        X_arr *= self.scale_
        X_arr += self.min_

        return X_arr

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Scale back the data to the original representation.
        """
        if self.scale_ is None or self.min_ is None:
            raise RuntimeError("MinMaxScaler is not fitted yet.")

        X_arr = check_array(X, copy=True)

        X_arr -= self.min_
        X_arr /= self.scale_

        return X_arr

class StandardScaler(BaseEstimator, TransformerMixin):
    """
    Standardize features by removing the mean and scaling to unit variance.

    The standard score of a sample `x` is calculated as:
        z = (x - u) / s

    where `u` is the mean of the training samples and `s` is the
    standard deviation.

    Attributes
    ----------
    mean_ : np.ndarray of shape (n_features,)
        The mean value for each feature in the training set.

    var_ : np.ndarray of shape (n_features,)
        The variance for each feature in the training set.

    scale_ : np.ndarray of shape (n_features,)
        Per feature relative scaling of the data to achieve unit variance.
        (Equivalent to the standard deviation).
    """

    def __init__(self, with_mean: bool = True, with_std: bool = True) -> None:
        self.with_mean = with_mean
        self.with_std = with_std

        self.mean_: Optional[np.ndarray] = None
        self.var_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, X: Union[np.ndarray, list], y: Optional[np.ndarray] = None) -> "StandardScaler":
        """
        Compute the mean and std to be used for later scaling.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute the mean and standard deviation.

        Returns
        -------
        self : object
            Fitted scaler.
        """
        X_arr = check_array(X)

        if self.with_mean:
            self.mean_ = np.mean(X_arr, axis=0)

        if self.with_std:
            self.var_ = np.var(X_arr, axis=0)
            self.scale_ = np.sqrt(self.var_)

            self.scale_[self.scale_ == 0.0] = 1.0

        return self

    def transform(self, X: Union[np.ndarray, list]) -> np.ndarray:
        """
        Perform standardization by centering and scaling.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to scale along the features axis.

        Returns
        -------
        X_tr : np.ndarray of shape (n_samples, n_features)
            Transformed array.
        """
        if self.with_mean and self.mean_ is None:
            raise RuntimeError("StandardScaler is not fitted yet.")
        if self.with_std and self.scale_ is None:
            raise RuntimeError("StandardScaler is not fitted yet.")

        X_arr = check_array(X, copy=True)

        if self.with_mean:
            X_arr -= self.mean_

        if self.with_std:
            X_arr /= self.scale_

        return X_arr

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Scale back the data to the original representation.
        """
        if self.with_mean and self.mean_ is None:
            raise RuntimeError("StandardScaler is not fitted yet.")
        if self.with_std and self.scale_ is None:
            raise RuntimeError("StandardScaler is not fitted yet.")

        X_arr = check_array(X, copy=True)

        if self.with_std:
            X_arr *= self.scale_
        if self.with_mean:
            X_arr += self.mean_

        return X_arr

class PolynomialFeatures(BaseEstimator, TransformerMixin):
    """
    Generate polynomial and interaction features.

    Generate a new feature matrix consisting of all polynomial combinations
    of the features with degree less than or equal to the specified degree.

    For example, if an input sample is two dimensional and of the form
    [a, b], the degree-2 polynomial features are [1, a, b, a^2, ab, b^2].

    Parameters
    ----------
    degree : int, default=2
        The degree of the polynomial features.

    interaction_only : bool, default=False
        If true, only interaction features are produced: features that are
        products of at most ``degree`` *distinct* input features (so not
        ``x[1] ** 2``, just ``x[1] * x[2]``).

    include_bias : bool, default=True
        If True (default), then include a bias column, the feature in which
        all polynomial powers are zero (i.e. a column of ones - acts as an
        intercept term in a linear model).
    """
    def __init__(
        self,
        degree: int = 2,
        interaction_only: bool = False,
        include_bias: bool = True
    ) -> None:
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias

        self.n_input_features_: Optional[int] = None
        self.n_output_features_: Optional[int] = None

    def fit(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> "PolynomialFeatures":
        """
        Compute number of output features.
        """
        X_arr = check_array(X)
        n_samples, n_features = X_arr.shape
        self.n_input_features_ = n_features

        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """
        Transform data to polynomial features.
        """
        X_arr = check_array(X)
        n_samples, n_features = X_arr.shape

        if self.n_input_features_ is not None and n_features != self.n_input_features_:
            raise ValueError(
                f"X shape does not match training shape. "
                f"Expected {self.n_input_features_}, got {n_features}"
            )

        if self.include_bias:
            output_columns: List[np.ndarray] = [np.ones((n_samples, 1))]
        else:
            output_columns = []

        for d in range(1, self.degree + 1):
            if self.interaction_only:
                combination = combinations(range(n_features), d)
            else:
                combination = combinations_with_replacement(range(n_features), d)

            for indices in combination:
                out_col = X_arr[:, indices[0]].copy()

                for i in indices[1:]:
                    out_col *= X_arr[:, i]

                output_columns.append(out_col.reshape(-1, 1))

        if not output_columns:
            return np.empty((n_samples, 0))
        return np.hstack(output_columns)

class OneHotEncoder(BaseEstimator, TransformerMixin):
    """
    Encode categorical features as a one-hot numeric array.

    Attributes
    ----------
    categories_ : list of ndarray
        The categories of each feature.
    """

    def __init__(self) -> None:
        self.categories_: Optional[List[np.ndarray]] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'OneHotEncoder':
        """
        Fit the encoder.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        self : object
            Returns self.
        """
        X = np.asarray(X)
        self.categories_ = [np.unique(X[:, i]) for i in range(X.shape[1])]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform X to one-hot encoding.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        X_encoded : ndarray of shape (n_samples, n_encoded_features)
            Encoded array.
        """
        if self.categories_ is None:
            raise RuntimeError("OneHotEncoder is not fitted yet.")

        X = np.asarray(X)
        encoded: List[np.ndarray] = []

        for i in range(X.shape[1]):
            for category in self.categories_[i]:
                encoded.append((X[:, i] == category).astype(int))

        return np.column_stack(encoded)

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fit to data, then transform it.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        X_tr : ndarray
            Transformed array.
        """
        return self.fit(X, y).transform(X)

def train_test_split(
        X: ArrayLike,
        y: ArrayLike,
        random_seed: int = 42,
        test_size: Union[float, int] = 0.2
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split arrays or matrices into random train and test subsets.

    Parameters
    ----------
    X : ArrayLike
        The input features. Can be a NumPy array or a nested list.
    y : ArrayLike
        The target values. Can be a NumPy array or a list.
    test_size : float, default=0.2
        The proportion of the dataset to include in the test split.
        Must be between 0.0 and 1.0.
    random_seed : int, default=42
        Controls the shuffling applied to the data before applying the split.
        Pass an int for reproducible output across multiple function calls.

    Returns
    -------
    X_train : np.ndarray
        The training subset of the features.
    X_test : np.ndarray
        The testing subset of the features.
    y_train : np.ndarray
        The training subset of the target values.
    y_test : np.ndarray
        The testing subset of the target values.
    """
    if not (0.0 < test_size < 1.0):
        raise ValueError(f"test_size must be between 0 and 1, but got {test_size}")

    X = np.asarray(X)
    y = np.asarray(y)
    num_samples = X.shape[0]
    indices = np.arange(num_samples)

    rng = np.random.RandomState(random_seed)
    rng.shuffle(indices)

    split = int((1 - test_size) * num_samples)

    X_train = X[indices[:split]]
    X_test = X[indices[split:]]

    y_train = y[indices[:split]]
    y_test = y[indices[split:]]

    return X_train, X_test, y_train, y_test
