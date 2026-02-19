"""Tests for clustering algorithms."""

import numpy as np
import pytest
from simpleml.cluster import KMeans, DBSCAN


class TestKMeans:
    """Test KMeans clustering."""
    
    def test_fit_and_predict(self):
        """Test basic fit and predict."""
        X = np.array([[0, 0], [1, 1], [2, 2], [8, 8], [9, 9]])
        
        model = KMeans(n_clusters=2, random_state=42, n_init=1)
        model.fit(X)
        
        assert model.labels_.shape == (5,)
        assert model.cluster_centers_.shape == (2, 2)
    
    def test_predict(self):
        """Test predict on new data."""
        X_train = np.array([[0, 0], [1, 1], [8, 8], [9, 9]])
        X_test = np.array([[0.5, 0.5], [8.5, 8.5]])
        
        model = KMeans(n_clusters=2, random_state=42, n_init=1)
        model.fit(X_train)
        predictions = model.predict(X_test)
        
        assert predictions.shape == (2,)


class TestDBSCAN:
    """Test DBSCAN clustering."""
    
    def test_fit(self):
        """Test basic fit."""
        X = np.array([[0, 0], [1, 1], [2, 2], [8, 8], [9, 9], [10, 10]])
        
        model = DBSCAN(eps=2, min_samples=2)
        model.fit(X)
        
        assert model.labels_.shape == (6,)