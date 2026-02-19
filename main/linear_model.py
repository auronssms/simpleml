"""Linear models for regression and classification."""

import numpy as np
from .base import BaseEstimator, ClassifierMixin, RegressorMixin
from .utils import check_X_y, check_array


class LinearRegression(BaseEstimator, RegressorMixin):
    """
    Ordinary least squares Linear Regression.
    
    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model.
    
    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Estimated coefficients.
    intercept_ : float
        Independent term in the linear model.
    """
    
    def __init__(self, fit_intercept: bool = True):
        self.fit_intercept = fit_intercept
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        """
        Fit linear model.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,)
            Target values.
            
        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, dtype=np.float64)
        
        if self.fit_intercept:
            X = np.column_stack([np.ones(X.shape[0]), X])
        
        # Solve using normal equation: (X^T X)^(-1) X^T y
        self.coef_ = np.linalg.lstsq(X, y, rcond=None)[0]
        
        if self.fit_intercept:
            self.intercept_ = self.coef_[0]
            self.coef_ = self.coef_[1:]
        else:
            self.intercept_ = 0.0
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the linear model.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Samples.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        X = check_array(X, dtype=np.float64)
        return X @ self.coef_ + self.intercept_


class LogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Logistic Regression classifier.
    
    Parameters
    ----------
    learning_rate : float, default=0.01
        Learning rate for gradient descent.
    n_iterations : int, default=1000
        Number of iterations for gradient descent.
    fit_intercept : bool, default=True
        Whether to calculate the intercept.
    random_state : int, optional
        Random seed for reproducibility.
    
    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Estimated coefficients.
    intercept_ : float
        Independent term.
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        fit_intercept: bool = True,
        random_state: int = None,
    ):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.fit_intercept = fit_intercept
        self.random_state = random_state
    
    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegression':
        """
        Fit logistic regression model.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,)
            Target values (binary: 0, 1).
            
        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, dtype=np.float64)
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        n_samples, n_features = X.shape
        
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0.0
        
        for _ in range(self.n_iterations):
            z = X @ self.coef_ + self.intercept_
            y_pred = self._sigmoid(z)
            
            # Gradient descent
            dw = (1 / n_samples) * X.T @ (y_pred - y)
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            self.coef_ -= self.learning_rate * dw
            self.intercept_ -= self.learning_rate * db
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Samples.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels (0 or 1).
        """
        X = check_array(X, dtype=np.float64)
        return (self.predict_proba(X) >= 0.5).astype(int).flatten()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Samples.
            
        Returns
        -------
        proba : ndarray of shape (n_samples, 2)
            Class probabilities.
        """
        X = check_array(X, dtype=np.float64)
        z = X @ self.coef_ + self.intercept_
        proba_pos = self._sigmoid(z).reshape(-1, 1)
        proba_neg = 1 - proba_pos
        return np.hstack([proba_neg, proba_pos])