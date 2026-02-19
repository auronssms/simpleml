"""Base classes for estimators in sklearn-lite."""

from abc import ABC, abstractmethod
from typing import Union
import numpy as np


class BaseEstimator(ABC):
    """Base class for all estimators in sklearn-lite."""
    
    def get_params(self, deep: bool = True) -> dict:
        """
        Get parameters for this estimator.
        
        Parameters
        ----------
        deep : bool, default=True
            If True, will return parameters for this estimator and
            contained subobjects that are estimators.
            
        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        init_signature = self.__init__.__code__
        args = init_signature.co_varnames[:init_signature.co_argcount]
        params = {arg: getattr(self, arg, None) for arg in args if arg != 'self'}
        return params
    
    def set_params(self, **params) -> 'BaseEstimator':
        """
        Set the parameters of this estimator.
        
        Parameters
        ----------
        **params : dict
            Estimator parameters.
            
        Returns
        -------
        self : object
            Returns self.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self


class ClassifierMixin:
    """Mixin class for classifiers."""
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the mean accuracy on the given test data and labels.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples.
        y : ndarray of shape (n_samples,)
            True labels for X.
            
        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) w.r.t. y.
        """
        from .metrics import accuracy_score
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)


class RegressorMixin:
    """Mixin class for regressors."""
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the R-squared score on the given test data and labels.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples.
        y : ndarray of shape (n_samples,)
            True values for X.
            
        Returns
        -------
        score : float
            R-squared score of self.predict(X) w.r.t. y.
        """
        from .metrics import r2_score
        y_pred = self.predict(X)
        return r2_score(y, y_pred)


class ClusterMixin:
    """Mixin class for clustering algorithms."""
    pass