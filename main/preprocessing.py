"""Data preprocessing utilities."""

import numpy as np
from typing import Tuple


class StandardScaler:
    """
    Standardize features by removing mean and scaling to unit variance.
    
    Attributes
    ----------
    mean_ : ndarray of shape (n_features,)
        Per feature relative scaling of the data to achieve zero mean.
    scale_ : ndarray of shape (n_features,)
        Per feature relative scaling of the data to achieve unit variance.
    """
    
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
    
    def fit(self, X: np.ndarray) -> 'StandardScaler':
        """
        Compute the mean and std to be used for later scaling.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The data used to compute the mean and standard deviation.
            
        Returns
        -------
        self : object
            Returns self.
        """
        X = np.asarray(X)
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Perform standardization by centering and scaling.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The data to scale.
            
        Returns
        -------
        X_tr : ndarray of shape (n_samples, n_features)
            Transformed array.
        """
        X = np.asarray(X)
        return (X - self.mean_) / (self.scale_ + 1e-8)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit to data, then transform it.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.
            
        Returns
        -------
        X_tr : ndarray of shape (n_samples, n_features)
            Transformed array.
        """
        return self.fit(X).transform(X)


class MinMaxScaler:
    """
    Transform features by scaling each feature to a given range.
    
    Parameters
    ----------
    feature_range : tuple (min, max), default=(0, 1)
        Desired range of transformed data.
    
    Attributes
    ----------
    data_min_ : ndarray of shape (n_features,)
        Per feature minimum seen in the data.
    data_max_ : ndarray of shape (n_features,)
        Per feature maximum seen in the data.
    """
    
    def __init__(self, feature_range: Tuple[float, float] = (0, 1)):
        self.feature_range = feature_range
        self.data_min_ = None
        self.data_max_ = None
        self.data_range_ = None
    
    def fit(self, X: np.ndarray) -> 'MinMaxScaler':
        """
        Compute the minimum and maximum to be used for later scaling.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The data used to compute the data min and max.
            
        Returns
        -------
        self : object
            Returns self.
        """
        X = np.asarray(X)
        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        self.data_range_ = self.data_max_ - self.data_min_
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Scale features to the configured range.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.
            
        Returns
        -------
        X_tr : ndarray of shape (n_samples, n_features)
            Transformed array.
        """
        X = np.asarray(X)
        X_scaled = (X - self.data_min_) / (self.data_range_ + 1e-8)
        return X_scaled * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit to data, then transform it.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.
            
        Returns
        -------
        X_tr : ndarray of shape (n_samples, n_features)
            Transformed array.
        """
        return self.fit(X).transform(X)


class OneHotEncoder:
    """
    Encode categorical features as a one-hot numeric array.
    
    Attributes
    ----------
    categories_ : list of ndarray
        The categories of each feature.
    """
    
    def __init__(self):
        self.categories_ = None
    
    def fit(self, X: np.ndarray) -> 'OneHotEncoder':
        """
        Fit the encoder.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data.
            
        Returns
        -------
        self : object
            Returns self.
        """
        X = np.asarray(X)
        self.categories_ = [np.unique(X[:, i]) for i in range(X.shape[1])]
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform X to one-hot encoding.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data.
            
        Returns
        -------
        X_encoded : ndarray of shape (n_samples, n_encoded_features)
            Encoded array.
        """
        X = np.asarray(X)
        encoded = []
        
        for i in range(X.shape[1]):
            for category in self.categories_[i]:
                encoded.append((X[:, i] == category).astype(int))
        
        return np.column_stack(encoded)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit to data, then transform it.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.
            
        Returns
        -------
        X_tr : ndarray
            Transformed array.
        """
        return self.fit(X).transform(X)