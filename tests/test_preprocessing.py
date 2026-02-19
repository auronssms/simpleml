"""Tests for preprocessing utilities."""

import numpy as np
import pytest
from sklearn_lite.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder


class TestStandardScaler:
    """Test StandardScaler."""
    
    def test_fit_and_transform(self):
        """Test fit and transform."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-7)
        assert np.allclose(X_scaled.std(axis=0), 1, atol=1e-7)
    
    def test_fit_then_transform(self):
        """Test separate fit and transform."""
        X_train = np.array([[0, 0], [1, 1], [2, 2]])
        X_test = np.array([[3, 3]])
        
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        assert X_test_scaled.shape == X_test.shape


class TestMinMaxScaler:
    """Test MinMaxScaler."""
    
    def test_fit_and_transform(self):
        """Test fit and transform."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_scaled = scaler.fit_transform(X)
        
        assert X_scaled.min() >= 0
        assert X_scaled.max() <= 1
    
    def test_feature_range(self):
        """Test custom feature range."""
        X = np.array([[0], [5], [10]])
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_scaled = scaler.fit_transform(X)
        
        assert X_scaled.min() == -1
        assert X_scaled.max() == 1


class TestOneHotEncoder:
    """Test OneHotEncoder."""
    
    def test_fit_and_transform(self):
        """Test fit and transform."""
        X = np.array([[0, 1], [1, 0], [0, 0]])
        encoder = OneHotEncoder()
        X_encoded = encoder.fit_transform(X)
        
        assert X_encoded.shape[0] == X.shape[0]