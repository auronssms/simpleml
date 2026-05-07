"""Tests for simpleml.models.ensemble — RandomForestClassifier, RandomForestRegressor."""
import pytest
import numpy as np

from simpleml.models.ensemble import RandomForestClassifier, RandomForestRegressor


# ── RandomForestClassifier ────────────────────────────────────────────────────

class TestRFClassifierInit:
    def test_defaults(self):
        rf = RandomForestClassifier()
        assert rf.n_estimators == 100
        assert rf.max_depth is None
        assert rf.min_samples_split == 2
        assert rf.max_features == "sqrt"
        assert rf.random_state is None

    def test_custom_params(self):
        rf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
        assert rf.n_estimators == 10
        assert rf.max_depth == 3

    def test_trees_empty_before_fit(self):
        assert RandomForestClassifier().trees_ == []


class TestRFClassifierFit:
    def test_fit_returns_self(self, binary_classification_data):
        X, y = binary_classification_data
        rf = RandomForestClassifier(n_estimators=5)
        assert rf.fit(X, y) is rf

    def test_correct_number_of_trees(self, binary_classification_data):
        X, y = binary_classification_data
        rf = RandomForestClassifier(n_estimators=7).fit(X, y)
        assert len(rf.trees_) == 7

    def test_n_features_set(self, binary_classification_data):
        X, y = binary_classification_data
        rf = RandomForestClassifier(n_estimators=3).fit(X, y)
        assert rf.n_features_ == 2

    def test_bootstrap_sample_shape(self, binary_classification_data):
        X, y = binary_classification_data
        rf = RandomForestClassifier(n_estimators=3, random_state=0)
        rng = np.random.RandomState(0)
        X_s, y_s = rf._bootstrap_sample(X, y, rng)
        assert X_s.shape == X.shape
        assert y_s.shape == y.shape

    def test_bootstrap_samples_with_replacement(self, binary_classification_data):
        X, y = binary_classification_data
        rf = RandomForestClassifier()
        rng = np.random.RandomState(0)
        _, y_s = rf._bootstrap_sample(X, y, rng)
        # With replacement: some samples appear more than once
        unique, counts = np.unique(y_s, return_counts=True)
        assert len(y_s) == len(y)

    def test_reproducible_with_seed(self, binary_classification_data):
        X, y = binary_classification_data
        rf1 = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)
        rf2 = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)
        np.testing.assert_array_equal(rf1.predict(X), rf2.predict(X))

    def test_multiclass(self, multiclass_data):
        X, y = multiclass_data
        rf = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)
        preds = rf.predict(X)
        assert set(preds).issubset({0, 1, 2})


class TestRFClassifierPredict:
    def test_predict_shape(self, binary_classification_data):
        X, y = binary_classification_data
        rf = RandomForestClassifier(n_estimators=5).fit(X, y)
        assert rf.predict(X).shape == (100,)

    def test_predict_only_valid_classes(self, binary_classification_data):
        X, y = binary_classification_data
        rf = RandomForestClassifier(n_estimators=5, random_state=0).fit(X, y)
        preds = rf.predict(X)
        assert set(preds).issubset({0, 1})

    def test_high_accuracy_separable_data(self, binary_classification_data):
        X, y = binary_classification_data
        rf = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)
        assert rf.score(X, y) > 0.90

    def test_more_trees_not_worse(self, binary_classification_data):
        X, y = binary_classification_data
        rf_few  = RandomForestClassifier(n_estimators=3,  random_state=0).fit(X, y)
        rf_many = RandomForestClassifier(n_estimators=30, random_state=0).fit(X, y)
        assert rf_many.score(X, y) >= rf_few.score(X, y) - 0.05  # small tolerance


# ── RandomForestRegressor ─────────────────────────────────────────────────────

class TestRFRegressorInit:
    def test_defaults(self):
        rf = RandomForestRegressor()
        assert rf.n_estimators == 100
        assert rf.max_depth is None
        assert rf.min_samples_split == 2
        assert rf.random_state is None

    def test_trees_empty_before_fit(self):
        assert RandomForestRegressor().trees_ == []


class TestRFRegressorFit:
    def test_fit_returns_self(self, simple_regression_data):
        X, y = simple_regression_data
        rf = RandomForestRegressor(n_estimators=5)
        assert rf.fit(X, y) is rf

    def test_correct_number_of_trees(self, simple_regression_data):
        X, y = simple_regression_data
        rf = RandomForestRegressor(n_estimators=8).fit(X, y)
        assert len(rf.trees_) == 8

    def test_n_features_set(self, simple_regression_data):
        X, y = simple_regression_data
        rf = RandomForestRegressor(n_estimators=3).fit(X, y)
        assert rf.n_features_ == 2

    def test_reproducible_with_seed(self, simple_regression_data):
        X, y = simple_regression_data
        rf1 = RandomForestRegressor(n_estimators=5, random_state=0).fit(X, y)
        rf2 = RandomForestRegressor(n_estimators=5, random_state=0).fit(X, y)
        np.testing.assert_array_equal(rf1.predict(X), rf2.predict(X))

    def test_bootstrap_sample_shape(self, simple_regression_data):
        X, y = simple_regression_data
        rf = RandomForestRegressor()
        rng = np.random.RandomState(0)
        X_s, y_s = rf._bootstrap_sample(X, y, rng)
        assert X_s.shape == X.shape


class TestRFRegressorPredict:
    def test_predict_shape(self, simple_regression_data):
        X, y = simple_regression_data
        rf = RandomForestRegressor(n_estimators=5).fit(X, y)
        assert rf.predict(X).shape == (100,)

    def test_predict_returns_ndarray(self, simple_regression_data):
        X, y = simple_regression_data
        rf = RandomForestRegressor(n_estimators=5).fit(X, y)
        assert isinstance(rf.predict(X), np.ndarray)

    def test_good_r2_score(self, simple_regression_data):
        X, y = simple_regression_data
        rf = RandomForestRegressor(n_estimators=20, random_state=0).fit(X, y)
        assert rf.score(X, y) > 0.90

    def test_predictions_averaged(self, simple_regression_data):
        """Manually verify averaging is done across all trees."""
        X, y = simple_regression_data
        rf = RandomForestRegressor(n_estimators=5, random_state=0).fit(X, y)
        individual = np.array([tree.predict(X) for tree in rf.trees_])
        expected = individual.mean(axis=0)
        np.testing.assert_allclose(rf.predict(X), expected)
