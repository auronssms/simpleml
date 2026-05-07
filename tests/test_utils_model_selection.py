"""Tests for simpleml.utils.model_selection — KFold, cross_validate, GridSearchCV."""
import pytest
import numpy as np

from simpleml.utils.model_selection import KFold, cross_validate, GridSearchCV
from simpleml.models.linear import LogisticRegression, LinearRegression


# ── KFold ─────────────────────────────────────────────────────────────────────

class TestKFoldInit:
    def test_defaults(self):
        kf = KFold()
        assert kf.n_splits == 5
        assert kf.shuffle is False
        assert kf.random_state is None

    def test_custom_params(self):
        kf = KFold(n_splits=10, shuffle=True, random_state=0)
        assert kf.n_splits == 10
        assert kf.shuffle is True


class TestKFoldSplit:
    @pytest.fixture
    def data(self):
        X = np.arange(50).reshape(25, 2).astype(float)
        y = np.arange(25)
        return X, y

    def test_correct_number_of_splits(self, data):
        X, y = data
        splits = list(KFold(n_splits=5).split(X, y))
        assert len(splits) == 5

    def test_no_overlap_train_test(self, data):
        X, y = data
        for train_idx, test_idx in KFold(n_splits=5).split(X, y):
            assert len(set(train_idx) & set(test_idx)) == 0

    def test_all_indices_covered(self, data):
        X, y = data
        all_test_indices = []
        for _, test_idx in KFold(n_splits=5).split(X, y):
            all_test_indices.extend(test_idx)
        assert sorted(all_test_indices) == list(range(25))

    def test_shuffle_changes_order(self, data):
        X, y = data
        splits_no_shuffle  = list(KFold(n_splits=5, shuffle=False).split(X))
        splits_shuffled    = list(KFold(n_splits=5, shuffle=True, random_state=1).split(X))
        # At least one fold should differ
        any_diff = any(
            not np.array_equal(a[1], b[1])
            for a, b in zip(splits_no_shuffle, splits_shuffled)
        )
        assert any_diff

    def test_shuffle_reproducible(self, data):
        X, y = data
        splits1 = list(KFold(n_splits=5, shuffle=True, random_state=7).split(X))
        splits2 = list(KFold(n_splits=5, shuffle=True, random_state=7).split(X))
        for (tr1, te1), (tr2, te2) in zip(splits1, splits2):
            np.testing.assert_array_equal(te1, te2)

    def test_train_larger_than_test(self, data):
        X, y = data
        for train_idx, test_idx in KFold(n_splits=5).split(X):
            assert len(train_idx) > len(test_idx)


# ── cross_validate ────────────────────────────────────────────────────────────

class TestCrossValidate:
    @pytest.fixture
    def clf_and_data(self, binary_classification_data):
        X, y = binary_classification_data
        clf = LogisticRegression(n_iter=100, random_state=0)
        return clf, X, y

    def test_returns_dict(self, clf_and_data):
        clf, X, y = clf_and_data
        result = cross_validate(clf, X, y, cv=3)
        assert isinstance(result, dict)

    def test_correct_keys(self, clf_and_data):
        clf, X, y = clf_and_data
        result = cross_validate(clf, X, y, cv=3)
        for key in ("train_score", "test_score", "mean_train_score", "mean_test_score"):
            assert key in result

    def test_score_arrays_length_matches_cv(self, clf_and_data):
        clf, X, y = clf_and_data
        result = cross_validate(clf, X, y, cv=4)
        assert len(result["train_score"]) == 4
        assert len(result["test_score"]) == 4

    def test_mean_scores_in_range(self, clf_and_data):
        clf, X, y = clf_and_data
        result = cross_validate(clf, X, y, cv=3)
        assert 0.0 <= result["mean_test_score"] <= 1.0

    def test_std_scores_non_negative(self, clf_and_data):
        clf, X, y = clf_and_data
        result = cross_validate(clf, X, y, cv=3)
        assert result["std_test_score"] >= 0.0

    def test_regression_cv(self, simple_regression_data):
        X, y = simple_regression_data
        reg = LinearRegression(n_iter=200, random_state=0)
        result = cross_validate(reg, X, y, cv=3)
        assert "mean_test_score" in result

    def test_integer_cv_creates_kfold(self, clf_and_data):
        clf, X, y = clf_and_data
        result = cross_validate(clf, X, y, cv=5)
        assert len(result["test_score"]) == 5

    def test_kfold_object_as_cv(self, clf_and_data):
        clf, X, y = clf_and_data
        kf = KFold(n_splits=3)
        result = cross_validate(clf, X, y, cv=kf)
        assert len(result["test_score"]) == 3


# ── GridSearchCV ──────────────────────────────────────────────────────────────

class TestGridSearchCV:
    @pytest.fixture
    def gs_setup(self, binary_classification_data):
        X, y = binary_classification_data
        clf = LogisticRegression(random_state=0)
        param_grid = {"n_iter": [50, 100], "learning_rate": [0.01, 0.1]}
        return clf, param_grid, X, y

    def test_fit_returns_self(self, gs_setup):
        clf, param_grid, X, y = gs_setup
        gs = GridSearchCV(clf, param_grid, cv=2)
        assert gs.fit(X, y) is gs

    def test_best_params_set(self, gs_setup):
        clf, param_grid, X, y = gs_setup
        gs = GridSearchCV(clf, param_grid, cv=2).fit(X, y)
        assert gs.best_params_ is not None
        assert "n_iter" in gs.best_params_
        assert "learning_rate" in gs.best_params_

    def test_best_score_is_float(self, gs_setup):
        clf, param_grid, X, y = gs_setup
        gs = GridSearchCV(clf, param_grid, cv=2).fit(X, y)
        assert isinstance(gs.best_score_, float)

    def test_best_estimator_set(self, gs_setup):
        clf, param_grid, X, y = gs_setup
        gs = GridSearchCV(clf, param_grid, cv=2).fit(X, y)
        assert gs.best_estimator_ is not None

    def test_cv_results_length(self, gs_setup):
        clf, param_grid, X, y = gs_setup
        gs = GridSearchCV(clf, param_grid, cv=2).fit(X, y)
        # 2 n_iter values × 2 learning_rate values = 4 combinations
        assert len(gs.cv_results_) == 4

    def test_predict_uses_best_estimator(self, gs_setup):
        clf, param_grid, X, y = gs_setup
        gs = GridSearchCV(clf, param_grid, cv=2).fit(X, y)
        preds = gs.predict(X)
        assert preds.shape == (100,)

    def test_predict_before_fit_raises(self, gs_setup):
        clf, param_grid, X, y = gs_setup
        gs = GridSearchCV(clf, param_grid, cv=2)
        with pytest.raises(RuntimeError):
            gs.predict(X)

    def test_best_score_is_highest_in_results(self, gs_setup):
        clf, param_grid, X, y = gs_setup
        gs = GridSearchCV(clf, param_grid, cv=2).fit(X, y)
        all_scores = [r["mean_test_score"] for r in gs.cv_results_]
        assert gs.best_score_ == pytest.approx(max(all_scores))
