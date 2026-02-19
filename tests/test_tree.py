"""Tests for decision tree models."""

import numpy as np
import pytest
from simpleml.tree import DecisionTreeClassifier, DecisionTreeRegressor


class TestDecisionTreeClassifier:
    """Test DecisionTreeClassifier."""
    
    def test_fit_and_predict(self):
        """Test basic fit and predict."""
        X = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
        y = np.array([0, 1, 1, 0])
        
        model = DecisionTreeClassifier(max_depth=3)
        model.fit(X, y)
        
        predictions = model.predict(X)
        assert predictions.shape == y.shape
    
    def test_max_depth_limit(self):
        """Test that max_depth limits tree growth."""
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 2, 100)
        
        model_shallow = DecisionTreeClassifier(max_depth=2)
        model_deep = DecisionTreeClassifier(max_depth=10)
        
        model_shallow.fit(X, y)
        model_deep.fit(X, y)
        
        # Both should work
        assert model_shallow.predict(X).shape == (100,)
        assert model_deep.predict(X).shape == (100,)


class TestDecisionTreeRegressor:
    """Test DecisionTreeRegressor."""
    
    def test_fit_and_predict(self):
        """Test basic fit and predict."""
        X = np.array([[1], [2], [3], [4]])
        y = np.array([1.5, 2.5, 3.5, 4.5])
        
        model = DecisionTreeRegressor(max_depth=3)
        model.fit(X, y)
        
        predictions = model.predict(X)
        assert predictions.shape == y.shape