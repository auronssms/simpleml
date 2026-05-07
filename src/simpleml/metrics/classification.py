import numpy as np
from typing import Union, List, Optional

ArrayLike = Union[np.ndarray, List[float], List[List[float]]]

def accuracy_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Overall accuracy: (TP + TN) / Total."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))

def precision_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """TP / (TP + FP)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

def recall_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """TP / (TP + FN)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

def f1_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """F1 = 2 * (precision * recall) / (precision + recall)."""
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    return float(2 * (p * r) / (p + r)) if (p + r) > 0 else 0.0

def roc_auc_score(y_true: ArrayLike, y_probs: ArrayLike) -> float:
    """
    Compute Area Under the Receiver Operating Characteristic Curve.
    y_probs should be the probabilities of the positive class.
    """
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)

    if y_probs.ndim == 2:
        y_probs = y_probs[:, 1]

    desc_score_indices = np.argsort(y_probs)[::-1]
    y_probs = y_probs[desc_score_indices]
    y_true = y_true[desc_score_indices]

    tps = np.cumsum(y_true)
    fps = np.cumsum(1 - y_true)

    tpr = tps / tps[-1] if tps[-1] > 0 else tps
    fpr = fps / fps[-1] if fps[-1] > 0 else fps

    return float(np.trapezoid(tpr, fpr))

def classification_report(y_true: ArrayLike, y_pred: ArrayLike) -> None:
    """Print classification report with precision, recall, and F1-score."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(y_true)

    report = f"\n{'Class':>10} | {'Precision':>10} | {'Recall':>10} | {'F1-Score':>10}\n"
    report += "-" * 50 + "\n"

    metrics_list: List[tuple] = []

    for cls in classes:
        y_true_cls = (y_true == cls).astype(int)
        y_pred_cls = (y_pred == cls).astype(int)

        p = precision_score(y_true_cls, y_pred_cls)
        r = recall_score(y_true_cls, y_pred_cls)
        f1 = f1_score(y_true_cls, y_pred_cls)

        metrics_list.append((p, r, f1))
        report += f"{str(cls):>10} | {p*100:>10.2f} | {r*100:>10.2f} | {f1*100:>10.2f}\n"

    report += "-" * 50 + "\n"
    acc = accuracy_score(y_true, y_pred)

    macro_p = np.mean([m[0] for m in metrics_list])
    macro_r = np.mean([m[1] for m in metrics_list])
    macro_f1 = np.mean([m[2] for m in metrics_list])

    report += f"{'Accuracy':>10} | {'':>10} | {'':>10} | {acc*100:>10.2f}\n"
    report += f"{'Macro Avg':>10} | {macro_p*100:>10.2f} | {macro_r*100:>10.2f} | {macro_f1*100:>10.2f}\n"

    print(report)

def confusion_matrix(y_true: ArrayLike, y_pred: ArrayLike, labels: Optional[List] = None) -> None:
    """
    Compute confusion matrix to evaluate the accuracy of a classification.
    Prints a confusion matrix.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(y_true)
    n_classes = len(classes)

    matrix = np.zeros((n_classes, n_classes), dtype=int)

    for i in range(n_classes):
        for j in range(n_classes):
            matrix[i, j] = np.sum((y_true == classes[i]) & (y_pred == classes[j]))

    if labels is None:
        labels = list(classes)

    _print_confusion_matrix(matrix, labels=labels)


def CrossEntropyLoss(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Compute the categorical cross entropy loss.
    Automatically handles integer labels (sparse) or one-hot encoded labels.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

    if y_true.ndim == 1:
        num_classes = y_pred.shape[1]
        n_samples = y_pred.shape[0]
        y_true_one_hot = np.zeros((n_samples, num_classes))

        y_true_one_hot[np.arange(len(y_true)), y_true.astype(int)] = 1
        y_true = y_true_one_hot

    return float(-np.sum(y_true * np.log(y_pred)) / y_true.shape[0])


def softmax(z: np.ndarray) -> np.ndarray:
    """Compute softmax activation."""
    shift_z = z - np.max(z, axis=1, keepdims=True)
    return np.exp(shift_z) / np.sum(np.exp(shift_z), axis=1, keepdims=True)

def _print_confusion_matrix(cm: np.ndarray, labels: List) -> None:
    """Print confusion matrix."""
    print(f"\n{'':>15} Predicted {labels[0]:<5} Predicted {labels[1]:<5}")
    print(f"{'Actual ' + str(labels[0]):<15} {cm[0,0]:^15} {cm[0,1]:^15}")
    print(f"{'Actual ' + str(labels[1]):<15} {cm[1,0]:^15} {cm[1,1]:^15}")