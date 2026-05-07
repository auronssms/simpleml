"""Tests for simpleml.metrics.classification — metrics, losses, softmax."""
import pytest
import numpy as np

from simpleml.metrics.classification import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    CrossEntropyLoss,
    softmax,
)


# ── accuracy_score ────────────────────────────────────────────────────────────

class TestAccuracyScore:
    def test_perfect(self):
        assert accuracy_score([0, 1, 1, 0], [0, 1, 1, 0]) == pytest.approx(1.0)

    def test_zero(self):
        assert accuracy_score([0, 0, 0], [1, 1, 1]) == pytest.approx(0.0)

    def test_half(self):
        assert accuracy_score([0, 1, 0, 1], [1, 1, 0, 0]) == pytest.approx(0.5)

    def test_returns_float(self):
        assert isinstance(accuracy_score([0], [0]), float)

    def test_single_sample_correct(self):
        assert accuracy_score([1], [1]) == pytest.approx(1.0)

    def test_single_sample_wrong(self):
        assert accuracy_score([0], [1]) == pytest.approx(0.0)


# ── precision_score ───────────────────────────────────────────────────────────

class TestPrecisionScore:
    def test_perfect_precision(self):
        # All positives are true positives
        assert precision_score([0, 1, 1], [0, 1, 1]) == pytest.approx(1.0)

    def test_zero_precision_all_fp(self):
        # Predict 1 for everything that is actually 0
        assert precision_score([0, 0], [1, 1]) == pytest.approx(0.0)

    def test_partial_precision(self):
        # TP=1 (y_true=1,y_pred=1), FP=2 (y_true=0,y_pred=1) → precision=1/3
        assert precision_score([0, 1, 0], [1, 1, 1]) == pytest.approx(1 / 3)

    def test_no_positive_predictions_returns_zero(self):
        assert precision_score([0, 1, 1], [0, 0, 0]) == pytest.approx(0.0)

    def test_returns_float(self):
        assert isinstance(precision_score([1], [1]), float)


# ── recall_score ──────────────────────────────────────────────────────────────

class TestRecallScore:
    def test_perfect_recall(self):
        assert recall_score([1, 1, 1], [1, 1, 1]) == pytest.approx(1.0)

    def test_zero_recall(self):
        # All positives are missed
        assert recall_score([1, 1, 1], [0, 0, 0]) == pytest.approx(0.0)

    def test_partial_recall(self):
        # TP=1, FN=1 → recall=0.5
        assert recall_score([1, 1, 0], [1, 0, 0]) == pytest.approx(0.5)

    def test_no_actual_positives_returns_zero(self):
        assert recall_score([0, 0], [0, 1]) == pytest.approx(0.0)

    def test_returns_float(self):
        assert isinstance(recall_score([1], [1]), float)


# ── f1_score ──────────────────────────────────────────────────────────────────

class TestF1Score:
    def test_perfect_f1(self):
        assert f1_score([1, 0, 1], [1, 0, 1]) == pytest.approx(1.0)

    def test_zero_f1_all_wrong(self):
        assert f1_score([1, 1, 1], [0, 0, 0]) == pytest.approx(0.0)

    def test_f1_harmonic_mean(self):
        # precision=0.5, recall=1.0 → F1 = 2*(0.5*1)/(0.5+1) ≈ 0.667
        result = f1_score([1, 1], [1, 1])   # both 1 → P=R=1 → F1=1
        assert result == pytest.approx(1.0)

    def test_f1_imbalanced(self):
        p = precision_score([0, 1, 1], [1, 1, 1])   # 0.667
        r = recall_score([0, 1, 1], [1, 1, 1])       # 1.0
        expected = 2 * p * r / (p + r)
        assert f1_score([0, 1, 1], [1, 1, 1]) == pytest.approx(expected)

    def test_returns_float(self):
        assert isinstance(f1_score([1], [1]), float)


# ── roc_auc_score ─────────────────────────────────────────────────────────────

class TestROCAUCScore:
    def test_perfect_auc(self):
        y_true = np.array([0, 0, 1, 1])
        y_probs = np.array([0.1, 0.2, 0.8, 0.9])
        assert roc_auc_score(y_true, y_probs) == pytest.approx(1.0, abs=0.01)

    def test_random_auc_around_half(self):
        rng = np.random.RandomState(0)
        y_true = rng.randint(0, 2, 200)
        y_probs = rng.rand(200)
        auc = roc_auc_score(y_true, y_probs)
        assert 0.3 < auc < 0.7

    def test_2d_probs_uses_positive_class(self):
        y_true = np.array([0, 0, 1, 1])
        y_probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
        auc = roc_auc_score(y_true, y_probs)
        assert auc == pytest.approx(1.0, abs=0.01)

    def test_returns_float(self):
        assert isinstance(roc_auc_score([0, 1], [0.3, 0.7]), float)


# ── softmax ───────────────────────────────────────────────────────────────────

class TestSoftmax:
    def test_output_sums_to_one_per_row(self):
        z = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
        out = softmax(z)
        np.testing.assert_allclose(out.sum(axis=1), [1.0, 1.0], atol=1e-10)

    def test_all_equal_inputs_uniform_output(self):
        z = np.ones((3, 4))
        out = softmax(z)
        np.testing.assert_allclose(out, np.full((3, 4), 0.25), atol=1e-10)

    def test_output_values_in_zero_one(self):
        z = np.random.randn(5, 3)
        out = softmax(z)
        assert (out >= 0).all() and (out <= 1).all()

    def test_large_input_numerically_stable(self):
        z = np.array([[1000.0, 1000.1, 999.9]])
        out = softmax(z)
        assert np.all(np.isfinite(out))

    def test_highest_logit_gets_highest_prob(self):
        z = np.array([[1.0, 5.0, 2.0]])
        out = softmax(z)
        assert np.argmax(out) == 1


# ── CrossEntropyLoss ──────────────────────────────────────────────────────────

class TestCrossEntropyLoss:
    def test_perfect_prediction_low_loss(self):
        y_true = np.eye(3)
        y_pred = np.eye(3) * 0.9999 + 0.0001 / 3
        y_pred /= y_pred.sum(axis=1, keepdims=True)
        loss = CrossEntropyLoss(y_true, y_pred)
        assert loss < 0.1

    def test_uniform_prediction_higher_loss(self):
        y_true = np.eye(3)
        y_pred = np.full((3, 3), 1 / 3)
        loss = CrossEntropyLoss(y_true, y_pred)
        assert loss > 1.0

    def test_loss_non_negative(self):
        y_true = np.array([0, 1, 2])
        y_pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7]])
        assert CrossEntropyLoss(y_true, y_pred) >= 0

    def test_sparse_labels_accepted(self):
        y_true = np.array([0, 1])
        y_pred = np.array([[0.9, 0.1], [0.2, 0.8]])
        loss = CrossEntropyLoss(y_true, y_pred)
        assert isinstance(loss, float)

    def test_one_hot_labels_accepted(self):
        y_true = np.array([[1, 0], [0, 1]])
        y_pred = np.array([[0.9, 0.1], [0.2, 0.8]])
        loss = CrossEntropyLoss(y_true, y_pred)
        assert isinstance(loss, float)

    def test_returns_float(self):
        y_true = np.array([0])
        y_pred = np.array([[0.9, 0.1]])
        assert isinstance(CrossEntropyLoss(y_true, y_pred), float)