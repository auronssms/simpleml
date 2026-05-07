import numpy as np
from typing import Optional, Union, List
from ..core.base import BaseEstimator, RegressorMixin, ClassifierMixin
from ..utils.validation import check_X_y, check_array
from ..metrics.classification import CrossEntropyLoss, softmax

ArrayLike = Union[np.ndarray, List[float], List[List[float]]]

class LinearRegression(BaseEstimator, RegressorMixin):
    """
    Linear Regression model fitted with Batch Gradient Descent.
    
    Parameters
    ----------
    learning_rate : float, default=0.01
        The learning rate schedule (constant step size).
        
    n_iter : int, default=1000
        The number of passes over the training data (epochs).
        
    alpha : float, default=0.0001
        Constant that multiplies the regularization term. 
        Higher value = stronger regularization (Ridge).
        
    l1_ratio : float, default=0.0
        The ElasticNet mixing parameter, with 0 <= l1_ratio <= 1.
        l1_ratio=0 corresponds to L2 penalty, l1_ratio=1 to L1.
        
    random_state : int, default=None
        The seed of the pseudo random number generator to use when shuffling 
        the data (if we were using SGD) or initializing weights.

    Attributes
    ----------
    coef_ : np.ndarray of shape (n_features,)
        Estimated coefficients for the linear regression problem.
        
    intercept_ : float
        Independent term in the linear model.
    """

    def __init__(
            self, 
            learning_rate: float = 0.01, 
            n_iter: int = 1000, 
            alpha: float = 0.0001,
            l1_ratio: float = 0.0,
            tol: float = 1e-5,
            random_state: Optional[int] = None):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.tol = tol
        self.random_state = random_state
        
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None

    def fit(self, X: ArrayLike, y: ArrayLike) -> "LinearRegression":
        """
        Fit linear model.

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            Training data.

        y : {array-like} of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        Returns
        -------
        self : returns an instance of self.
        """
        X_arr, y_arr = check_X_y(X, y)
        n_samples, n_features = X_arr.shape

        rng = np.random.RandomState(self.random_state)

        self.coef_ = rng.normal(loc=0.0, scale=0.01, size=n_features)
        self.intercept_ = 0.0

        for _ in range(self.n_iter):
            y_pred = X_arr @ self.coef_ + self.intercept_

            error = y_pred - y_arr

            grad_l1 = self.l1_ratio * np.sign(self.coef_)
            grad_l2 = (1 - self.l1_ratio) * self.coef_

            dw = (1 / n_samples) * (X_arr.T @ error) + self.alpha * (grad_l1 + grad_l2)
            db = (1 / n_samples) * np.sum(error)

            self.coef_ -= self.learning_rate * dw
            self.intercept_ -= self.learning_rate * db

            if np.linalg.norm(dw) < self.tol:
                break

        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Predict using the linear model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.
            
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted values.
        """
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("This LinearRegressor instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        
        X_arr = check_array(X)
        return X_arr @ self.coef_ + self.intercept_

class LogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(
            self, 
            learning_rate: float = 0.01, 
            n_iter: int = 1000, 
            alpha: float = 0.0001,
            l1_ratio: float = 0.0,
            tol: float = 1e-5,
            random_state: Optional[int] = None):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.tol = tol
        self.random_state = random_state
        
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: ArrayLike, y: ArrayLike) -> "LogisticRegression":
        X_arr, y_labels = check_X_y(X, y)
        n_samples, n_features = X_arr.shape
        
        self.classes_ = np.unique(y_labels)
        n_classes = len(self.classes_)
        label_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([label_to_idx[c] for c in y_labels])
        y_true = np.eye(n_classes)[y_idx]
        
        rng = np.random.RandomState(self.random_state)
        
        self.coef_ = rng.normal(loc=0.0, scale=0.01, size=(n_features, n_classes))
        self.intercept_ = np.zeros(n_classes)
        
        self.loss_history_ = []
        last_loss = np.inf

        for _ in range(self.n_iter):
            z = X_arr @ self.coef_ + self.intercept_
            y_pred = softmax(z)
            
            current_loss = CrossEntropyLoss(y_true, y_pred)
            self.loss_history_.append(current_loss)
            if np.abs(last_loss - current_loss) < self.tol:
                break
            last_loss = current_loss

            error = y_pred - y_true
            
            grad_l1 = self.l1_ratio * np.sign(self.coef_)
            grad_l2 = (1 - self.l1_ratio) * self.coef_  # type: ignore
            
            dw = (1 / n_samples) * (X_arr.T @ error) + self.alpha * (grad_l1 + grad_l2)
            db = (1 / n_samples) * np.sum(error, axis=0)

            self.coef_ -= self.learning_rate * dw
            self.intercept_ -= self.learning_rate * db
        return self

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Model not fitted.")
            
        X_arr = check_array(X)
        z = X_arr @ self.coef_ + self.intercept_
        return softmax(z)

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict the class with the highest probability."""
        probs = self.predict_proba(X)
        indices = np.argmax(probs, axis=1)
        return self.classes_[indices] # type: ignore