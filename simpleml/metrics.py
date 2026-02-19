"""Metrics for model evaluation."""

import numpy as np
from typing import Union


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Accuracy classification score.
    
    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        Ground truth labels.
    y_pred : ndarray of shape (n_samples,)
        Predicted labels.
        
    Returns
    -------
    accuracy : float
        Fraction of correct predictions.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    
    return np.mean(y_true == y_pred)


def precision_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "binary",
    pos_label: int = 1,
) -> float:
    """
    Precision classification score.
    
    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        Ground truth labels.
    y_pred : ndarray of shape (n_samples,)
        Predicted labels.
    average : str, default="binary"
        Type of averaging performed on data.
    pos_label : int, default=1
        Label considered as positive.
        
    Returns
    -------
    precision : float
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    true_positives = np.sum((y_pred == pos_label) & (y_true == pos_label))
    predicted_positives = np.sum(y_pred == pos_label)
    
    if predicted_positives == 0:
        return 0.0
    
    return true_positives / predicted_positives


def recall_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "binary",
    pos_label: int = 1,
) -> float:
    """
    Recall classification score.
    
    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        Ground truth labels.
    y_pred : ndarray of shape (n_samples,)
        Predicted labels.
    average : str, default="binary"
        Type of averaging performed on data.
    pos_label : int, default=1
        Label considered as positive.
        
    Returns
    -------
    recall : float
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    true_positives = np.sum((y_pred == pos_label) & (y_true == pos_label))
    actual_positives = np.sum(y_true == pos_label)
    
    if actual_positives == 0:
        return 0.0
    
    return true_positives / actual_positives


def f1_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "binary",
    pos_label: int = 1,
) -> float:
    """
    F1 classification score.
    
    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        Ground truth labels.
    y_pred : ndarray of shape (n_samples,)
        Predicted labels.
    average : str, default="binary"
        Type of averaging performed on data.
    pos_label : int, default=1
        Label considered as positive.
        
    Returns
    -------
    f1 : float
    """
    precision = precision_score(y_true, y_pred, average, pos_label)
    recall = recall_score(y_true, y_pred, average, pos_label)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """
    Compute confusion matrix to evaluate classification accuracy.
    
    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        Ground truth labels.
    y_pred : ndarray of shape (n_samples,)
        Predicted labels.
        
    Returns
    -------
    C : ndarray of shape (n_classes, n_classes)
        Confusion matrix.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    for true, pred in zip(y_true, y_pred):
        cm[class_to_idx[true], class_to_idx[pred]] += 1
    
    return cm


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean squared error regression loss.
    
    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        Ground truth values.
    y_pred : ndarray of shape (n_samples,)
        Predicted values.
        
    Returns
    -------
    mse : float
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean((y_true - y_pred) ** 2)


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean absolute error regression loss.
    
    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        Ground truth values.
    y_pred : ndarray of shape (n_samples,)
        Predicted values.
        
    Returns
    -------
    mae : float
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    R-squared (coefficient of determination) score.
    
    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        Ground truth values.
    y_pred : ndarray of shape (n_samples,)
        Predicted values.
        
    Returns
    -------
    r2 : float
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        return 0.0
    
    return 1 - (ss_res / ss_tot)