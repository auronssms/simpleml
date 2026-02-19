"""Naive Bayes classifiers."""

import numpy as np
from collections import defaultdict
from .base import BaseEstimator, ClassifierMixin


class GaussianNaiveBayes(BaseEstimator, ClassifierMixin):
    """
    Gaussian Naive Bayes classifier.
    
    Assumes that features follow a Gaussian distribution.
    
    Attributes
    ----------
    class_prior_ : ndarray of shape (n_classes,)
        Probability of each class.
    theta_ : ndarray of shape (n_classes, n_features)
        Mean of each feature per class.
    sigma_ : ndarray of shape (n_classes, n_features)
        Variance of each feature per class.
    """
    
    def __init__(self):
        pass
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GaussianNaiveBayes':
        """
        Fit Gaussian Naive Bayes classifier.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,)
            Target values.
            
        Returns
        -------
        self : object
            Returns self.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        
        self.theta_ = np.zeros((n_classes, n_features))
        self.sigma_ = np.zeros((n_classes, n_features))
        self.class_prior_ = np.zeros(n_classes)
        
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.theta_[idx, :] = X_c.mean(axis=0)
            self.sigma_[idx, :] = X_c.var(axis=0)
            self.class_prior_[idx] = len(X_c) / len(X)
        
        return self
    
    def _calculate_likelihood(self, x: np.ndarray, theta: np.ndarray, sigma: np.ndarray) -> float:
        """Calculate likelihood of feature given class."""
        numerator = np.exp(-(x - theta) ** 2 / (2 * sigma))
        denominator = np.sqrt(2 * np.pi * sigma)
        return numerator / denominator
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Samples.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        X = np.asarray(X)
        y_pred = np.zeros(X.shape[0])
        
        for idx, x in enumerate(X):
            posteriors = []
            
            for i, c in enumerate(self.classes_):
                prior = np.log(self.class_prior_[i])
                posterior = np.sum(np.log(self._calculate_likelihood(
                    x, self.theta_[i, :], self.sigma_[i, :] + 1e-10
                )))
                posteriors.append(prior + posterior)
            
            y_pred[idx] = self.classes_[np.argmax(posteriors)]
        
        return y_pred


class MultinomialNaiveBayes(BaseEstimator, ClassifierMixin):
    """
    Multinomial Naive Bayes classifier.
    
    Suitable for classification with discrete features (e.g., word counts).
    
    Parameters
    ----------
    alpha : float, default=1.0
        Additive smoothing parameter.
    
    Attributes
    ----------
    feature_log_prob_ : ndarray of shape (n_classes, n_features)
        Log probability of features given a class.
    class_log_prior_ : ndarray of shape (n_classes,)
        Log probability of each class.
    """
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MultinomialNaiveBayes':
        """
        Fit Multinomial Naive Bayes classifier.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,)
            Target values.
            
        Returns
        -------
        self : object
            Returns self.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        
        self.feature_log_prob_ = np.zeros((n_classes, n_features))
        self.class_log_prior_ = np.zeros(n_classes)
        
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.class_log_prior_[idx] = np.log(len(X_c) / len(X))
            
            # Calculate feature probabilities with smoothing
            feature_counts = np.sum(X_c, axis=0)
            smoothed_fc = feature_counts + self.alpha
            self.feature_log_prob_[idx, :] = np.log(
                smoothed_fc / (np.sum(smoothed_fc) + self.alpha * n_features)
            )
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Samples.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        X = np.asarray(X)
        return self.classes_[np.argmax(
            self.class_log_prior_ + np.dot(X, self.feature_log_prob_.T),
            axis=1
        )]