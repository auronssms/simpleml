"""Tests for ensemble methods."""

import numpy as np
import pytest
from simpleml.ensemble import RandomForestClassifier, RandomForestRegressor


class TestRandomForestClassifier:
    """Test RandomForestClassifier."""
    
    def test_fit_and_predict(self):
        """Test basic fit and predict."""
        X = np.array([[0, 0], [1, 1], [0, 1], [1, 0], [0, 0], [1, 1]])
        y = np.array([0, 1, 1, 0, 0, 1])
        
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)
        
        predictions = model.predict(X)
        assert predictions.shape == y.shape
    
    def test_n_estimators(self):
        """Test that n_estimators trees are created."""
        X = np.random.rand(20, 3)
        y = np.random.randint(0, 2, 20)
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        assert len(model.trees_) == 10


class TestRandomForestRegressor:
    """Test RandomForestRegressor."""
    
    def test_fit_and_predict(self):
        """Test basic fit and predict."""
        X = np.array([[1], [2], [3], [4], [5], [6]])
        y = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
        
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit(X, y)
        
        predictions = model.predict(X)
        assert predictions.shape == y.shape