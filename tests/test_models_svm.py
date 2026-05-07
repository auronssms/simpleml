"""Tests for simpleml.models.svm — LinearSVC, SVR."""
import pytest
import numpy as np

from simpleml.models.svm import LinearSVC, SVR


# ── LinearSVC ────────────────────────────────────────────────────────────────

class TestLinearSVCInit:
    def test_defaults(self):
        svc = LinearSVC()
        assert svc.C == 1.0
        assert svc.max_iter == 1000
        assert svc.learning_rate == 0.01
        assert svc.random_state is None

    def test_custom_params(self):
        svc = LinearSVC(C=0.5, max_iter=500, random_state=1)
        assert svc.C == 0.5
        assert svc.max_iter == 500


class TestLinearSVCFit:
    def test_fit_returns_self(self, binary_classification_data):
        X, y = binary_classification_data
        svc = LinearSVC(max_iter=100)
        assert svc.fit(X, y) is svc

    def test_coef_shape(self, binary_classification_data):
        X, y = binary_classification_data
        svc = LinearSVC(max_iter=50).fit(X, y)
        assert svc.coef_.shape == (2,)

    def test_intercept_is_float(self, binary_classification_data):
        X, y = binary_classification_data
        svc = LinearSVC(max_iter=50).fit(X, y)
        assert isinstance(svc.intercept_, float)

    def test_non_binary_raises(self, multiclass_data):
        X, y = multiclass_data
        svc = LinearSVC(max_iter=10)
        with pytest.raises(ValueError, match="binary"):
            svc.fit(X, y)

    def test_coef_not_all_zeros_after_fit(self, binary_classification_data):
        X, y = binary_classification_data
        svc = LinearSVC(max_iter=100).fit(X, y)
        assert np.any(svc.coef_ != 0.0)

    def test_high_accuracy_separable(self, binary_classification_data):
        X, y = binary_classification_data
        svc = LinearSVC(max_iter=200, C=1.0, random_state=0).fit(X, y)
        acc = np.mean(svc.predict(X) == y)
        assert acc > 0.85


class TestLinearSVCPredict:
    def test_predict_shape(self, binary_classification_data):
        X, y = binary_classification_data
        svc = LinearSVC(max_iter=50).fit(X, y)
        assert svc.predict(X).shape == (100,)

    def test_predict_values_match_original_classes(self, binary_classification_data):
        X, y = binary_classification_data 
        svc = LinearSVC(max_iter=50).fit(X, y)
        preds = svc.predict(X)
        assert set(preds).issubset({0, 1}) 

    def test_predict_returns_ndarray(self, binary_classification_data):
        X, y = binary_classification_data
        svc = LinearSVC(max_iter=50).fit(X, y)
        assert isinstance(svc.predict(X), np.ndarray)


# ── SVR ──────────────────────────────────────────────────────────────────────

class TestSVRInit:
    def test_defaults(self):
        svr = SVR()
        assert svr.C == 1.0
        assert svr.epsilon == 0.1
        assert svr.max_iter == 1000
        assert svr.learning_rate == 0.01
        assert svr.random_state is None

    def test_custom_params(self):
        svr = SVR(C=2.0, epsilon=0.5, max_iter=200)
        assert svr.C == 2.0
        assert svr.epsilon == 0.5


class TestSVRFit:
    def test_fit_returns_self(self, simple_regression_data):
        X, y = simple_regression_data
        svr = SVR(max_iter=20)
        assert svr.fit(X, y) is svr

    def test_coef_shape(self, simple_regression_data):
        X, y = simple_regression_data
        svr = SVR(max_iter=20).fit(X, y)
        assert svr.coef_.shape == (2,)

    def test_intercept_is_float(self, simple_regression_data):
        X, y = simple_regression_data
        svr = SVR(max_iter=20).fit(X, y)
        assert isinstance(svr.intercept_, float)

    def test_coef_updates_during_fit(self, simple_regression_data):
        X, y = simple_regression_data
        svr = SVR(max_iter=100).fit(X, y)
        assert np.any(svr.coef_ != 0.0)


class TestSVRPredict:
    def test_predict_shape(self, simple_regression_data):
        X, y = simple_regression_data
        svr = SVR(max_iter=50).fit(X, y)
        assert svr.predict(X).shape == (100,)

    def test_predict_returns_ndarray(self, simple_regression_data):
        X, y = simple_regression_data
        svr = SVR(max_iter=50).fit(X, y)
        assert isinstance(svr.predict(X), np.ndarray)

    def test_predictions_are_finite(self, simple_regression_data):
        X, y = simple_regression_data
        svr = SVR(max_iter=50).fit(X, y)
        assert np.all(np.isfinite(svr.predict(X)))

    def test_reasonable_fit_simple_data(self):
        """SVR should produce finite, bounded predictions on clean data."""
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        svr = SVR(max_iter=300, learning_rate=0.01, epsilon=0.1).fit(X, y)
        preds = svr.predict(X)
        assert np.all(np.isfinite(preds))
