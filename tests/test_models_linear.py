"""Tests for simpleml.models.linear — LinearRegression, LogisticRegression."""
import pytest
import numpy as np

from simpleml.models.linear import LinearRegression, LogisticRegression


# ── LinearRegression ─────────────────────────────────────────────────────────

class TestLinearRegressionInit:
    def test_default_params(self):
        lr = LinearRegression()
        assert lr.learning_rate == 0.01
        assert lr.n_iter == 1000
        assert lr.alpha == 0.0001
        assert lr.l1_ratio == 0.0
        assert lr.random_state is None

    def test_custom_params(self):
        lr = LinearRegression(learning_rate=0.1, n_iter=500, alpha=0.01, random_state=7)
        assert lr.learning_rate == 0.1
        assert lr.n_iter == 500
        assert lr.random_state == 7

    def test_coef_none_before_fit(self):
        lr = LinearRegression()
        assert lr.coef_ is None
        assert lr.intercept_ is None


class TestLinearRegressionFit:
    def test_fit_returns_self(self, simple_regression_data):
        X, y = simple_regression_data
        lr = LinearRegression(n_iter=100)
        result = lr.fit(X, y)
        assert result is lr

    def test_coef_shape_after_fit(self, simple_regression_data):
        X, y = simple_regression_data
        lr = LinearRegression(n_iter=100).fit(X, y)
        assert lr.coef_.shape == (2,)

    def test_intercept_is_scalar(self, simple_regression_data):
        X, y = simple_regression_data
        lr = LinearRegression(n_iter=100).fit(X, y)
        assert isinstance(lr.intercept_, float)

    def test_fit_noiseless_linear_data(self):
        """Coefficients should converge close to ground truth for clean data."""
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        y = 3.0 * X[:, 0] + 1.0
        lr = LinearRegression(learning_rate=0.05, n_iter=3000, alpha=0.0, random_state=0)
        lr.fit(X, y)
        assert abs(lr.coef_[0] - 3.0) < 0.3
        assert abs(lr.intercept_ - 1.0) < 0.3

    def test_fit_accepts_lists(self):
        lr = LinearRegression(n_iter=10)
        lr.fit([[1, 2], [3, 4]], [1.0, 2.0])
        assert lr.coef_ is not None

    def test_reproducible_with_seed(self, simple_regression_data):
        X, y = simple_regression_data
        lr1 = LinearRegression(n_iter=50, random_state=0).fit(X, y)
        lr2 = LinearRegression(n_iter=50, random_state=0).fit(X, y)
        np.testing.assert_array_equal(lr1.coef_, lr2.coef_)

    def test_l2_regularisation_shrinks_coef(self, simple_regression_data):
        X, y = simple_regression_data
        lr_no_reg = LinearRegression(n_iter=500, alpha=0.0, random_state=0).fit(X, y)
        lr_reg    = LinearRegression(n_iter=500, alpha=10.0, random_state=0).fit(X, y)
        assert np.linalg.norm(lr_reg.coef_) < np.linalg.norm(lr_no_reg.coef_)

    def test_l1_regularisation(self, simple_regression_data):
        X, y = simple_regression_data
        lr = LinearRegression(n_iter=200, alpha=0.1, l1_ratio=1.0, random_state=0)
        lr.fit(X, y)
        assert lr.coef_ is not None

    def test_elastic_net(self, simple_regression_data):
        X, y = simple_regression_data
        lr = LinearRegression(n_iter=200, alpha=0.1, l1_ratio=0.5, random_state=0)
        lr.fit(X, y)
        assert lr.coef_ is not None


class TestLinearRegressionPredict:
    def test_predict_shape(self, simple_regression_data):
        X, y = simple_regression_data
        lr = LinearRegression(n_iter=50).fit(X, y)
        preds = lr.predict(X)
        assert preds.shape == (100,)

    def test_predict_returns_ndarray(self, simple_regression_data):
        X, y = simple_regression_data
        lr = LinearRegression(n_iter=50).fit(X, y)
        assert isinstance(lr.predict(X), np.ndarray)

    def test_predict_before_fit_raises(self):
        lr = LinearRegression()
        with pytest.raises(RuntimeError, match="not fitted"):
            lr.predict(np.ones((5, 2)))

    def test_predict_close_to_truth(self):
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([2.0, 4.0, 6.0])
        lr = LinearRegression(learning_rate=0.1, n_iter=5000, alpha=0.0, random_state=0)
        lr.fit(X, y)
        preds = lr.predict(X)
        np.testing.assert_allclose(preds, y, atol=0.5)

    def test_score_increases_with_more_iterations(self, simple_regression_data):
        X, y = simple_regression_data
        lr_few  = LinearRegression(n_iter=10,   random_state=0).fit(X, y)
        lr_many = LinearRegression(n_iter=1000, random_state=0).fit(X, y)
        assert lr_many.score(X, y) >= lr_few.score(X, y)


# ── LogisticRegression ───────────────────────────────────────────────────────

class TestLogisticRegressionInit:
    def test_default_params(self):
        clf = LogisticRegression()
        assert clf.learning_rate == 0.01
        assert clf.n_iter == 1000
        assert clf.alpha == 0.0001
        assert clf.l1_ratio == 0.0
        assert clf.random_state is None

    def test_coef_none_before_fit(self):
        clf = LogisticRegression()
        assert clf.coef_ is None
        assert clf.intercept_ is None
        assert clf.classes_ is None


class TestLogisticRegressionFit:
    def test_fit_returns_self(self, binary_classification_data):
        X, y = binary_classification_data
        clf = LogisticRegression(n_iter=50)
        assert clf.fit(X, y) is clf

    def test_classes_identified(self, binary_classification_data):
        X, y = binary_classification_data
        clf = LogisticRegression(n_iter=50).fit(X, y)
        np.testing.assert_array_equal(clf.classes_, [0, 1])

    def test_coef_shape_binary(self, binary_classification_data):
        X, y = binary_classification_data
        clf = LogisticRegression(n_iter=50).fit(X, y)
        assert clf.coef_.shape == (2, 2)   # (n_features, n_classes)

    def test_intercept_shape(self, binary_classification_data):
        X, y = binary_classification_data
        clf = LogisticRegression(n_iter=50).fit(X, y)
        assert clf.intercept_.shape == (2,)

    def test_loss_history_recorded(self, binary_classification_data):
        X, y = binary_classification_data
        clf = LogisticRegression(n_iter=50).fit(X, y)
        assert len(clf.loss_history_) > 0

    def test_loss_generally_decreases(self, binary_classification_data):
        X, y = binary_classification_data
        clf = LogisticRegression(n_iter=200, learning_rate=0.1).fit(X, y)
        # First loss should be higher than final loss
        assert clf.loss_history_[0] > clf.loss_history_[-1]

    def test_multiclass_fit(self, multiclass_data):
        X, y = multiclass_data
        clf = LogisticRegression(n_iter=100).fit(X, y)
        np.testing.assert_array_equal(clf.classes_, [0, 1, 2])
        assert clf.coef_.shape == (2, 3)

    def test_reproducible_with_seed(self, binary_classification_data):
        X, y = binary_classification_data
        clf1 = LogisticRegression(n_iter=30, random_state=0).fit(X, y)
        clf2 = LogisticRegression(n_iter=30, random_state=0).fit(X, y)
        np.testing.assert_array_equal(clf1.coef_, clf2.coef_)


class TestLogisticRegressionPredict:
    def test_predict_shape(self, binary_classification_data):
        X, y = binary_classification_data
        clf = LogisticRegression(n_iter=100).fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (100,)

    def test_predict_only_known_classes(self, binary_classification_data):
        X, y = binary_classification_data
        clf = LogisticRegression(n_iter=100).fit(X, y)
        preds = clf.predict(X)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_sums_to_one(self, binary_classification_data):
        X, y = binary_classification_data
        clf = LogisticRegression(n_iter=100).fit(X, y)
        proba = clf.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(100), atol=1e-6)

    def test_predict_proba_shape(self, binary_classification_data):
        X, y = binary_classification_data
        clf = LogisticRegression(n_iter=100).fit(X, y)
        assert clf.predict_proba(X).shape == (100, 2)

    def test_predict_unfitted_raises(self):
        with pytest.raises(RuntimeError):
            LogisticRegression().predict(np.ones((5, 2)))

    def test_high_accuracy_on_separable_data(self, binary_classification_data):
        X, y = binary_classification_data
        clf = LogisticRegression(n_iter=500, learning_rate=0.1, random_state=0).fit(X, y)
        acc = clf.score(X, y)
        assert acc > 0.90

    def test_multiclass_predictions(self, multiclass_data):
        X, y = multiclass_data
        clf = LogisticRegression(n_iter=300, learning_rate=0.1, random_state=0).fit(X, y)
        preds = clf.predict(X)
        assert set(preds).issubset({0, 1, 2})
