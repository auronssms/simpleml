"""Base classes for estimators in sklearn-lite."""

import numpy as np
from abc import ABC
from typing import Dict, Any, Optional, Union, List

ArrayLike = Union[np.ndarray, List[float], List[List[float]]]

class BaseEstimator(ABC):
    """
    Base class for all estimators in simpleml.
    
    Notes
    -----
    All estimators should specify all the parameters that can be set
    at the class level in their ``__init__`` as explicit keyword arguments.
    """
    
    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator."""
        import inspect
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            return []
        
        init_signature = inspect.signature(init)
        parameters = [p for p in init_signature.parameters.values() if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        return sorted([p.name for p in parameters])

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Get parameters for this estimator.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key, None)
            out[key] = value
            if deep and hasattr(value, 'get_params') and callable(getattr(value, 'get_params', None)):
                try:
                    deep_items = value.get_params().items()
                    out.update((key + '__' + k, val) for k, val in deep_items)
                except Exception:
                    pass
        return out

    def set_params(self, **params) -> "BaseEstimator":
        """
        Set the parameters of this estimator.
        """
        if not params:
            return self
        valid_params = self.get_params(deep=False)
        for key, value in params.items():
            if key not in valid_params:
                raise ValueError(f"Invalid parameter {key} for estimator {self}. Check the list of available parameters with `estimator.get_params().keys()`.")
            setattr(self, key, value)
        return self

class RegressorMixin:
    """Mixin class for all regression estimators in simpleml."""
    
    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """
        Return the coefficient of determination R^2 of the prediction.
        """
        from ..metrics.regression import r2_score
        y_pred = self.predict(X)    #type: ignore
        return r2_score(y, y_pred)

class ClassifierMixin:
    """Mixin class for all classification estimators in simpleml."""

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return the mean accuracy on the given test data and labels."""
        from ..metrics.classification import accuracy_score
        y = np.array(y)
        y_pred = np.array(self.predict(X))  # type: ignore
        return accuracy_score(y, y_pred)

class TransformerMixin:
    """Mixin class for all transformers in simpleml."""

    def fit_transform(self, X: ArrayLike, y: Optional[ArrayLike] = None, **fit_params) -> np.ndarray:
        """
        Fit to data, then transform it.
        """
        if y is None:
            return self.fit(X, **fit_params).transform(X) #type: ignore
        else:
            return self.fit(X, y, **fit_params).transform(X)  #type: ignore
        
class ClusterMixin:
    """Mixin class for clustering algorithms."""
    pass
