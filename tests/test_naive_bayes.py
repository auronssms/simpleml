"""Tests for Naive Bayes classifiers."""

import numpy as np
import pytest
from sklearn_lite.naive_bayes import GaussianNaiveBayes, MultinomialNaiveBayes


class TestGaussianNaiveBayes:
    """Test GaussianNaiveBayes."""
    
    def test_fit_and_predict(self):
        """Test basic fit and predict."""
        X = np.array([[0, 0], [1, 1], [2, 2], [8, 8], [9, 9]])
        y = np.array([0, 0, 0, 1, 1])
        
        model = GaussianNaiveBayes()
        model.fit(X, y)
        
        predictions = model.predict(X)
        assert predictions.shape == y.shape


class TestMultinomialNaiveBayes:
    """Test MultinomialNaiveBayes."""
    
    def test_fit_and_predict(self):
        """Test basic fit and predict."""
        X = np.array([[1, 2, 1], [2, 1, 1], [1, 1, 2], [2, 2, 1]])
        y = np.array([0, 0, 1, 1])
        
        model = MultinomialNaiveBayes()
        model.fit(X, y)
        
        predictions = model.predict(X)
        assert predictions.shape == y.shape