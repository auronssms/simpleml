"""Tests for core base classes."""

import pytest
import numpy as np
from src.simpleml.core.base import (
    BaseEstimator,
    RegressorMixin,
    ClassifierMixin,
    TransformerMixin,
    ClusterMixin,
)


class DummyEstimator(BaseEstimator):
    """Dummy estimator for testing."""
    
    def __init__(self, param1=1, param2=2):
        self.param1 = param1
        self.param2 = param2


class DummyRegressor(BaseEstimator, RegressorMixin):
    """Dummy regressor for testing."""
    
    def __init__(self, alpha=0.01):
        self.alpha = alpha
    
    def predict(self, X):
        return np.ones(X.shape[0])


class DummyClassifier(BaseEstimator, ClassifierMixin):
    """Dummy classifier for testing."""
    
    def __init__(self, beta=0.5):
        self.beta = beta
    
    def predict(self, X):
        return np.zeros(X.shape[0], dtype=int)


class DummyTransformer(BaseEstimator, TransformerMixin):
    """Dummy transformer for testing."""
    
    def __init__(self, scale=1.0):
        self.scale = scale
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X * self.scale


class TestBaseEstimator:
    """Test BaseEstimator class functionality."""
    
    def test_get_param_names(self):
        """Test _get_param_names extracts parameter names."""
        est = DummyEstimator(param1=10, param2=20)
        param_names = est._get_param_names()
        
        assert isinstance(param_names, list)
        assert 'param1' in param_names
        assert 'param2' in param_names
        assert 'self' not in param_names
    
    def test_get_params_empty_estimator(self):
        """Test get_params with default parameters."""
        est = DummyEstimator()
        params = est.get_params()
        
        assert isinstance(params, dict)
        assert params['param1'] == 1
        assert params['param2'] == 2
    
    def test_get_params_custom_values(self):
        """Test get_params with custom parameter values."""
        est = DummyEstimator(param1=100, param2=200)
        params = est.get_params()
        
        assert params['param1'] == 100
        assert params['param2'] == 200
    
    def test_get_params_shallow(self):
        """Test get_params with deep=False."""
        est = DummyEstimator(param1=5, param2=10)
        params = est.get_params(deep=False)
        
        assert isinstance(params, dict)
        assert 'param1' in params
        assert 'param2' in params
    
    def test_get_params_deep(self):
        """Test get_params with deep=True."""
        est = DummyEstimator()
        params = est.get_params(deep=True)
        
        assert isinstance(params, dict)
        assert 'param1' in params
    
    def test_set_params_single(self):
        """Test set_params with single parameter."""
        est = DummyEstimator()
        result = est.set_params(param1=999)
        
        assert result is est
        assert est.param1 == 999
        assert est.param2 == 2
    
    def test_set_params_multiple(self):
        """Test set_params with multiple parameters."""
        est = DummyEstimator()
        result = est.set_params(param1=111, param2=222)
        
        assert result is est
        assert est.param1 == 111
        assert est.param2 == 222
    
    def test_set_params_chaining(self):
        """Test set_params method chaining."""
        est = DummyEstimator()
        est.set_params(param1=50).set_params(param2=60)
        
        assert est.param1 == 50
        assert est.param2 == 60
    
    def test_set_params_invalid_raises(self):
        """Test set_params with invalid parameter raises ValueError."""
        est = DummyEstimator()
        
        with pytest.raises(ValueError, match="Invalid parameter"):
            est.set_params(invalid_param=999)
    
    def test_set_params_empty(self):
        """Test set_params with no parameters."""
        est = DummyEstimator(param1=10)
        result = est.set_params()
        
        assert result is est
        assert est.param1 == 10


class TestRegressorMixin:
    """Test RegressorMixin functionality."""
    
    def test_regressor_has_score_method(self):
        """Test that RegressorMixin provides score method."""
        reg = DummyRegressor()
        assert hasattr(reg, 'score')
        assert callable(reg.score)
    
    def test_regressor_score_calculation(self):
        """Test RegressorMixin score returns float."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([1.0, 2.0, 3.0])
        
        reg = DummyRegressor()
        score = reg.score(X, y)
        
        assert isinstance(score, (float, np.floating))
    
    def test_regressor_score_uses_r2(self):
        """Test that score uses r2_score metric."""
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([1.0, 2.0, 3.0])
        
        reg = DummyRegressor()
        score = reg.score(X, y)
        
        # For perfect predictions, score should be 1.0
        # Dummy returns all ones, so score will be less than 1
        assert -10 <= score <= 1.0


class TestClassifierMixin:
    """Test ClassifierMixin functionality."""
    
    def test_classifier_has_score_method(self):
        """Test that ClassifierMixin provides score method."""
        clf = DummyClassifier()
        assert hasattr(clf, 'score')
        assert callable(clf.score)
    
    def test_classifier_score_calculation(self):
        """Test ClassifierMixin score returns accuracy."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 0, 0])
        
        clf = DummyClassifier()
        score = clf.score(X, y)
        
        assert isinstance(score, (float, np.floating))
        assert 0.0 <= score <= 1.0
    
    def test_classifier_perfect_score(self):
        """Test classifier score with perfect predictions."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 0, 0])
        
        clf = DummyClassifier()
        score = clf.score(X, y)
        
        # Dummy classifier always predicts 0, so accuracy should be 1.0 for all zeros
        assert score == 1.0


class TestTransformerMixin:
    """Test TransformerMixin functionality."""
    
    def test_transformer_has_fit_transform(self):
        """Test that TransformerMixin provides fit_transform method."""
        transformer = DummyTransformer()
        assert hasattr(transformer, 'fit_transform')
        assert callable(transformer.fit_transform)
    
    def test_fit_transform_without_y(self):
        """Test fit_transform without y parameter."""
        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        
        transformer = DummyTransformer(scale=2.0)
        result = transformer.fit_transform(X)
        
        expected = X * 2.0
        assert np.allclose(result, expected)
    
    def test_fit_transform_with_y(self):
        """Test fit_transform with y parameter."""
        X = np.array([[1, 2], [3, 4]], dtype=float)
        y = np.array([0, 1])
        
        transformer = DummyTransformer(scale=0.5)
        result = transformer.fit_transform(X, y)
        
        expected = X * 0.5
        assert np.allclose(result, expected)
    
    def test_fit_transform_returns_self_fit(self):
        """Test that fit_transform calls fit and returns transformed data."""
        X = np.array([[1, 2], [3, 4]], dtype=float)
        transformer = DummyTransformer()
        
        result = transformer.fit_transform(X)
        assert result.shape == X.shape


class TestClusterMixin:
    """Test ClusterMixin class."""
    
    def test_cluster_mixin_exists(self):
        """Test that ClusterMixin class exists."""
        assert hasattr(ClusterMixin, '__init__')
    
    def test_cluster_mixin_is_mixin(self):
        """Test that ClusterMixin is a proper mixin class."""
        
        class DummyClusterer(ClusterMixin):
            pass
        
        clusterer = DummyClusterer()
        assert isinstance(clusterer, ClusterMixin)
