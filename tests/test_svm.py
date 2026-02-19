"""Tests for SVM models."""

import numpy as np
import pytest
from sklearn_lite.svm import LinearSVC, SVR


class TestLinearSVC:
    """Test LinearSVC."""
    
    def test_fit_and_predict(self):
        """Test basic fit and predict."""
        X = np.array([[0, 0], [1, 1], [2, 2], [8, 8], [9, 9], [10, 10]])
        y = np.array([0, 0, 0, 1, 1, 1])
        
        # Convert to -1, 1
        y = np.where(y == 0, -1, 1)
        
        model = LinearSVC(max_iter=100, random_state=42)
        model.fit(X, y)
        
        predictions = model.predict(X)
        assert predictions.shape == y.shape


class TestSVR:
    """Test SVR."""
    
    def test_fit_and_predict(self):
        """Test basic fit and predict."""
        X = np.array([[1], [2], [3], [4]])
        y = np.array([1.5, 2.5, 3.5, 4.5])
        
        model = SVR(max_iter=100, random_state=42)
        model.fit(X, y)
        
        predictions = model.predict(X)
        assert predictions.shape == y.shape