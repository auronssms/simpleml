"""Decision tree algorithms."""

import numpy as np
from collections import Counter
from .base import BaseEstimator, ClassifierMixin, RegressorMixin


class Node:
    """Node in a decision tree."""
    
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature          # Index of feature to split on
        self.threshold = threshold      # Threshold value to split on
        self.left = left               # Left subtree
        self.right = right             # Right subtree
        self.value = value             # Class value if leaf node
        self.samples = 0               # Number of samples
        self.impurity = 0              # Gini impurity or MSE


class DecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    """
    Decision tree classifier.
    
    Parameters
    ----------
    max_depth : int, optional
        Maximum depth of the tree.
    min_samples_split : int, default=2
        Minimum samples required to split a node.
    criterion : str, default="gini"
        Function to measure split quality ('gini' or 'entropy').
    
    Attributes
    ----------
    tree_ : Node
        The underlying tree structure.
    n_classes_ : int
        Number of classes.
    n_features_ : int
        Number of features.
    """
    
    def __init__(self, max_depth=None, min_samples_split=2, criterion="gini"):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTreeClassifier':
        """
        Build a decision tree classifier.
        
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
        X = np.asarray(X)
        y = np.asarray(y)
        
        self.n_classes_ = len(np.unique(y))
        self.n_features_ = X.shape[1]
        
        self.tree_ = self._build_tree(X, y, depth=0)
        return self
    
    def _gini(self, y: np.ndarray) -> float:
        """Calculate Gini impurity."""
        counter = Counter(y)
        gini = 1.0
        for count in counter.values():
            gini -= (count / len(y)) ** 2
        return gini
    
    def _entropy(self, y: np.ndarray) -> float:
        """Calculate entropy."""
        counter = Counter(y)
        entropy = 0.0
        for count in counter.values():
            p = count / len(y)
            if p > 0:
                entropy -= p * np.log2(p)
        return entropy
    
    def _information_gain(self, parent, left_child, right_child) -> float:
        """Calculate information gain."""
        if self.criterion == "gini":
            impurity_fn = self._gini
        else:
            impurity_fn = self._entropy
        
        n = len(parent)
        n_left = len(left_child)
        n_right = len(right_child)
        
        if n_left == 0 or n_right == 0:
            return 0
        
        parent_impurity = impurity_fn(parent)
        left_impurity = impurity_fn(left_child)
        right_impurity = impurity_fn(right_child)
        
        weighted_child_impurity = (n_left / n) * left_impurity + \
                                  (n_right / n) * right_impurity
        
        return parent_impurity - weighted_child_impurity
    
    def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """Find the best split for a node."""
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        for feature in range(self.n_features_):
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) < self.min_samples_split or \
                   np.sum(right_mask) < self.min_samples_split:
                    continue
                
                gain = self._information_gain(y, y[left_mask], y[right_mask])
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        """Recursively build the decision tree."""
        n_samples = len(y)
        n_classes = len(np.unique(y))
        
        node = Node()
        node.samples = n_samples
        node.impurity = self._gini(y) if self.criterion == "gini" else self._entropy(y)
        
        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_classes == 1 or \
           n_samples < self.min_samples_split:
            node.value = Counter(y).most_common(1)[0][0]
            return node
        
        best_feature, best_threshold = self._best_split(X, y)
        
        if best_feature is None:
            node.value = Counter(y).most_common(1)[0][0]
            return node
        
        node.feature = best_feature
        node.threshold = best_threshold
        
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return node
    
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
        return np.array([self._traverse_tree(x, self.tree_) for x in X])
    
    def _traverse_tree(self, x: np.ndarray, node: Node):
        """Traverse the tree to make a prediction."""
        if node.value is not None:
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)


class DecisionTreeRegressor(BaseEstimator, RegressorMixin):
    """
    Decision tree regressor.
    
    Parameters
    ----------
    max_depth : int, optional
        Maximum depth of the tree.
    min_samples_split : int, default=2
        Minimum samples required to split a node.
    """
    
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTreeRegressor':
        """
        Build a decision tree regressor.
        
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
        X = np.asarray(X)
        y = np.asarray(y)
        
        self.n_features_ = X.shape[1]
        self.tree_ = self._build_tree(X, y, depth=0)
        return self
    
    def _mse(self, y: np.ndarray) -> float:
        """Calculate mean squared error."""
        if len(y) == 0:
            return 0
        return np.mean((y - np.mean(y)) ** 2)
    
    def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """Find the best split for a node."""
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        for feature in range(self.n_features_):
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) < self.min_samples_split or \
                   np.sum(right_mask) < self.min_samples_split:
                    continue
                
                mse_parent = self._mse(y)
                mse_left = self._mse(y[left_mask])
                mse_right = self._mse(y[right_mask])
                
                n = len(y)
                weighted_mse = (np.sum(left_mask) / n) * mse_left + \
                               (np.sum(right_mask) / n) * mse_right
                
                gain = mse_parent - weighted_mse
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        """Recursively build the decision tree."""
        n_samples = len(y)
        
        node = Node()
        node.samples = n_samples
        node.impurity = self._mse(y)
        
        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           np.std(y) == 0:
            node.value = np.mean(y)
            return node
        
        best_feature, best_threshold = self._best_split(X, y)
        
        if best_feature is None:
            node.value = np.mean(y)
            return node
        
        node.feature = best_feature
        node.threshold = best_threshold
        
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return node
    
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
        return np.array([self._traverse_tree(x, self.tree_) for x in X])
    
    def _traverse_tree(self, x: np.ndarray, node: Node):
        """Traverse the tree to make a prediction."""
        if node.value is not None:
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)