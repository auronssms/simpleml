"""Support Vector Machine models."""

import numpy as np
from .base import BaseEstimator, ClassifierMixin, RegressorMixin


class LinearSVC(BaseEstimator, ClassifierMixin):
    """
    Linear Support Vector Classification (SVM).
    
    Parameters
    ----------
    C : float, default=1.0
        Regularization parameter. The strength of the regularization is
        inversely proportional to C.
    max_iter : int, default=1000
        Maximum number of iterations for the solver.
    learning_rate : float, default=0.01
        Learning rate for gradient descent.
    random_state : int, optional
        Random seed for reproducibility.
    
    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Coefficients of the support vector plane.
    intercept_ : float
        Constants in the decision function.
    """
    
    def __init__(
        self,
        C=1.0,
        max_iter=1000,
        learning_rate=0.01,
        random_state=None,
    ):
        self.C = C
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.random_state = random_state
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearSVC':
        """
        Fit the SVM classifier.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,)
            Target values (binary: -1 or 1).
            
        Returns
        -------
        self : object
            Returns self.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        # Convert to {-1, 1} format
        unique_classes = np.unique(y)
        if len(unique_classes) != 2:
            raise ValueError("Only binary classification is supported")
        
        y[y == unique_classes[0]] = -1
        y[y == unique_classes[1]] = 1
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0.0
        
        for _ in range(self.max_iter):
            for i in range(n_samples):
                margin = y[i] * (np.dot(X[i], self.coef_) + self.intercept_)
                
                if margin < 1:
                    # Misclassified or within margin
                    self.coef_ -= self.learning_rate * (
                        self.coef_ / self.C - y[i] * X[i]
                    )
                    self.intercept_ -= self.learning_rate * (-y[i])
                else:
                    # Correctly classified
                    self.coef_ -= self.learning_rate * (self.coef_ / self.C)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Samples to predict.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        X = np.asarray(X, dtype=np.float64)
        return np.sign(X @ self.coef_ + self.intercept_)


class SVR(BaseEstimator, RegressorMixin):
    """
    Support Vector Regression.
    
    Parameters
    ----------
    C : float, default=1.0
        Regularization parameter.
    epsilon : float, default=0.1
        Epsilon parameter in epsilon-SVR model. It specifies the epsilon-tube
        within which no penalty is associated in the training loss function.
    max_iter : int, default=1000
        Maximum number of iterations for the solver.
    learning_rate : float, default=0.01
        Learning rate for gradient descent.
    random_state : int, optional
        Random seed for reproducibility.
    """
    
    def __init__(
        self,
        C=1.0,
        epsilon=0.1,
        max_iter=1000,
        learning_rate=0.01,
        random_state=None,
    ):
        self.C = C
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.random_state = random_state
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVR':
        """
        Fit the SVR model.
        
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
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0.0
        
        for _ in range(self.max_iter):
            for i in range(n_samples):
                prediction = np.dot(X[i], self.coef_) + self.intercept_
                error = y[i] - prediction
                
                if np.abs(error) > self.epsilon:
                    # Outside epsilon tube
                    if error > 0:
                        self.coef_ += self.learning_rate * X[i]
                        self.intercept_ += self.learning_rate
                    else:
                        self.coef_ -= self.learning_rate * X[i]
                        self.intercept_ -= self.learning_rate
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict values for X.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Samples to predict.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        X = np.asarray(X, dtype=np.float64)
        return X @ self.coef_ + self.intercept_