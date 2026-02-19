"""Tests for model selection utilities."""

import numpy as np
import pytest
from simpleml.model_selection import KFold, cross_validate, GridSearchCV
from simpleml.linear_model import LogisticRegression


class TestKFold:
    """Test KFold cross-validator."""
    
    def test_split(self):
        """Test KFold split."""
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        y = np.array([0, 0, 1, 1, 1])
        
        kf = KFold(n_splits=2)
        splits = list(kf.split(X, y))
        
        assert len(splits) == 2
        assert splits[0][0].shape[0] + splits[0][1].shape[0] == X.shape[0]


class TestCrossValidate:
    """Test cross_validate function."""
    
    def test_cross_validate(self):
        """Test cross validation."""
        X = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
        y = np.array([0, 1, 1, 0])
        
        model = LogisticRegression(n_iterations=100, random_state=42)
        scores = cross_validate(model, X, y, cv=2)
        
        assert 'train_score' in scores
        assert 'test_score' in scores
        assert len(scores['train_score']) == 2


class TestGridSearchCV:
    """Test GridSearchCV."""
    
    def test_grid_search(self):
        """Test grid search."""
        X = np.array([[0, 0], [1, 1], [0, 1], [1, 0], [0, 0], [1, 1]])
        y = np.array([0, 1, 1, 0, 0, 1])
        
        model = LogisticRegression(random_state=42)
        param_grid = {
            'learning_rate': [0.01, 0.1],
            'n_iterations': [100, 200],
        }
        
        gs = GridSearchCV(model, param_grid, cv=2)
        gs.fit(X, y)
        
        assert gs.best_params_ is not None
        assert gs.best_score_ >= -1