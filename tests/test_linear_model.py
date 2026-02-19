"""Tests for linear models."""

import numpy as np
import pytest
from simpleml.linear_model import LinearRegression, LogisticRegression


class TestLinearRegression:
    """Test LinearRegression model."""
    
    def test_fit_and_predict(self):
        """Test basic fit and predict."""
        X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
        y = np.array([3, 5, 7, 9])
        
        model = LinearRegression()
        model.fit(X, y)

        predictions = model.predict(X)
        assert predictions.shape == y.shape
        assert np.allclose(predictions, y, atol=0.1)
    
    def test_intercept(self):
        """Test intercept parameter."""
        X = np.array([[1], [2], [3]])
        y = np.array([2, 4, 6])
        
        model = LinearRegression(fit_intercept=True)
        model.fit(X, y)
        
        assert hasattr(model, 'intercept_')
        assert hasattr(model, 'coef_')


class TestLogisticRegression:
    """Test LogisticRegression model."""
    
    def test_fit_and_predict(self):
        """Test basic fit and predict."""
        X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        y = np.array([0, 0, 1, 1])
        
        model = LogisticRegression(n_iterations=1000, random_state=42)
        model.fit(X, y)
        
        predictions = model.predict(X)
        assert predictions.shape == y.shape
        assert all(p in [0, 1] for p in predictions)
    
    def test_predict_proba(self):
        """Test probability predictions."""
        X = np.array([[0, 0], [1, 1], [2, 2]])
        y = np.array([0, 0, 1])
        
        model = LogisticRegression(n_iterations=500, random_state=42)
        model.fit(X, y)
        
        proba = model.predict_proba(X)
        assert proba.shape == (3, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)