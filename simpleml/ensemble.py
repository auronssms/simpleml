"""Ensemble methods for machine learning."""

import numpy as np
from .base import BaseEstimator, ClassifierMixin, RegressorMixin
from .tree import DecisionTreeClassifier, DecisionTreeRegressor


class RandomForestClassifier(BaseEstimator, ClassifierMixin):
    """
    Random Forest classifier.
    
    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest.
    max_depth : int, optional
        Maximum depth of the tree.
    min_samples_split : int, default=2
        Minimum samples required to split a node.
    max_features : int or float, default="sqrt"
        Number of features to consider for the best split.
    random_state : int, optional
        Random seed for reproducibility.
    """
    
    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        max_features="sqrt",
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.trees_ = []
    
    def _bootstrap_sample(self, X: np.ndarray, y: np.ndarray):
        """Generate bootstrap sample."""
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestClassifier':
        """
        Build a random forest classifier.
        
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
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        X = np.asarray(X)
        y = np.asarray(y)
        
        self.n_features_ = X.shape[1]
        
        for _ in range(self.n_estimators):
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
            )
            tree.fit(X_sample, y_sample)
            self.trees_.append(tree)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class for X.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Samples.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        X = np.asarray(X)
        predictions = np.array([tree.predict(X) for tree in self.trees_])
        
        y_pred = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            votes = predictions[:, i]
            y_pred[i] = np.bincount(votes.astype(int)).argmax()
        
        return y_pred.astype(int)


class RandomForestRegressor(BaseEstimator, RegressorMixin):
    """
    Random Forest regressor.
    
    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest.
    max_depth : int, optional
        Maximum depth of the tree.
    min_samples_split : int, default=2
        Minimum samples required to split a node.
    random_state : int, optional
        Random seed for reproducibility.
    """
    
    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.trees_ = []
    
    def _bootstrap_sample(self, X: np.ndarray, y: np.ndarray):
        """Generate bootstrap sample."""
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestRegressor':
        """
        Build a random forest regressor.
        
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
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        X = np.asarray(X)
        y = np.asarray(y)
        
        self.n_features_ = X.shape[1]
        
        for _ in range(self.n_estimators):
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
            )
            tree.fit(X_sample, y_sample)
            self.trees_.append(tree)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict values for X.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Samples.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        X = np.asarray(X)
        predictions = np.array([tree.predict(X) for tree in self.trees_])
        return np.mean(predictions, axis=0)