# src/simpleml/metrics/__init__.py

from .classification import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    CrossEntropyLoss,
    softmax,
)
from .regression import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)

__all__ = [
    "accuracy_score",
    "precision_score",
    "recall_score",
    "f1_score",
    "roc_auc_score",
    "classification_report",
    "confusion_matrix",
    "CrossEntropyLoss",
    "softmax",
    "mean_squared_error",
    "mean_absolute_error",
    "mean_absolute_percentage_error",
    "r2_score",
]