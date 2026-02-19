"""
sklearn-lite: A lightweight machine learning library inspired by scikit-learn.

Modules:
    tree: Decision tree models
    linear_model: Linear regression and classification models
    svm: Support Vector Machines
    ensemble: Ensemble methods
    naive_bayes: Naive Bayes classifiers
    cluster: Clustering algorithms
    preprocessing: Data preprocessing utilities
    model_selection: Model selection and evaluation tools
    metrics: Metrics for model evaluation
"""

__version__ = "0.1.0"
__author__ = "sergeauronss01"

from . import (
    tree,
    linear_model,
    svm,
    ensemble,
    naive_bayes,
    cluster,
    preprocessing,
    model_selection,
    metrics,
)

__all__ = [
    "tree",
    "linear_model",
    "svm",
    "ensemble",
    "naive_bayes",
    "cluster",
    "preprocessing",
    "model_selection",
    "metrics",
]