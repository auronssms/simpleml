"""Tools for model selection and evaluation."""

import numpy as np
from typing import Dict, Optional, Any
from itertools import product


class CrossValidationSplitter:
    """Base class for cross-validation splitters."""

    def split(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Generate indices to split data."""
        raise NotImplementedError


class KFold(CrossValidationSplitter):
    """
    K-Fold cross-validator.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds.
    shuffle : bool, default=False
        Whether to shuffle the data before splitting.
    random_state : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = False,
        random_state: Optional[int] = None
    ) -> None:
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """
        Generate train and test indices.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,), optional
            Target values.

        Yields
        ------
        train_index : ndarray
            Training set indices.
        test_index : ndarray
            Testing set indices.
        """
        X = np.asarray(X)
        n_samples = X.shape[0]

        rng = np.random.RandomState(self.random_state)
        indices = np.arange(n_samples)

        if self.shuffle:
            rng.shuffle(indices)

        fold_size = n_samples // self.n_splits

        for i in range(self.n_splits):
            start = i * fold_size
            end = start + fold_size if i != self.n_splits - 1 else n_samples

            test_idx = indices[start:end]
            train_idx = np.concatenate([indices[:start], indices[end:]])

            yield train_idx, test_idx


def cross_validate(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    scoring: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate metric(s) by cross-validation.

    Parameters
    ----------
    estimator : object
        Estimator object implementing fit and predict/score methods.
    X : ndarray of shape (n_samples, n_features)
        Data to fit.
    y : ndarray of shape (n_samples,)
        Target relative to X.
    cv : int or cross-validation generator, default=5
        Number of folds or CV splitter.
    scoring : str, optional
        Name of scoring metric (default: 'accuracy' for classifiers, 'r2' for regressors).

    Returns
    -------
    scores : dict
        Dictionary with keys 'train_score' and 'test_score' containing lists of scores.
    """
    if isinstance(cv, int):
        cv_splitter = KFold(n_splits=cv)
    else:
        cv_splitter = cv

    train_scores = []
    test_scores = []

    for train_idx, test_idx in cv_splitter.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        estimator_clone = estimator.__class__(**estimator.get_params())
        estimator_clone.fit(X_train, y_train)

        train_score = estimator_clone.score(X_train, y_train)
        test_score = estimator_clone.score(X_test, y_test)

        train_scores.append(train_score)
        test_scores.append(test_score)

    return {
        'train_score': np.array(train_scores),
        'test_score': np.array(test_scores),
        'mean_train_score': float(np.mean(train_scores)),
        'mean_test_score': float(np.mean(test_scores)),
        'std_train_score': float(np.std(train_scores)),
        'std_test_score': float(np.std(test_scores)),
    }


class GridSearchCV:
    """
    Exhaustive search over specified parameter values.

    Parameters
    ----------
    estimator : object
        Estimator object.
    param_grid : dict
        Dictionary with parameters names as keys and lists of parameter settings
        to try as values.
    cv : int or cross-validation generator, default=5
        Number of folds or CV splitter.
    scoring : str, optional
        Metric to optimize.

    Attributes
    ----------
    best_params_ : dict
        Parameter setting that gave the best results.
    best_score_ : float
        Best mean cross-validation score.
    best_estimator_ : object
        Estimator that was chosen by the search.
    """

    def __init__(
        self,
        estimator: Any,
        param_grid: Dict[str, list],
        cv: int = 5,
        scoring: Optional[str] = None
    ) -> None:
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.best_params_: Optional[Dict[str, Any]] = None
        self.best_score_: float = -np.inf
        self.best_estimator_: Optional[Any] = None
        self.cv_results_: list = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GridSearchCV':
        """
        Run fit with all sets of parameters.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data to fit.
        y : ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns self.
        """
        param_names = list(self.param_grid.keys())
        param_values = [self.param_grid[name] for name in param_names]

        for params in product(*param_values):
            param_dict = dict(zip(param_names, params))

            est = self.estimator.__class__(**param_dict)
            cv_results = cross_validate(est, X, y, cv=self.cv)

            mean_score = cv_results['mean_test_score']
            self.cv_results_.append({
                'params': param_dict,
                'mean_test_score': mean_score,
            })

            if mean_score > self.best_score_:
                self.best_score_ = mean_score
                self.best_params_ = param_dict
                self.best_estimator_ = est

        if self.best_estimator_ is not None:
            self.best_estimator_.fit(X, y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the best estimator."""
        if self.best_estimator_ is None:
            raise RuntimeError("GridSearchCV has not been fitted yet")
        return self.best_estimator_.predict(X)
