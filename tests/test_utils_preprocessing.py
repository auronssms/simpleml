"""Tests for simpleml.utils.preprocessing — scalers, encoders, train_test_split."""
import pytest
import numpy as np

from simpleml.utils.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    PolynomialFeatures,
    OneHotEncoder,
    train_test_split,
)


X_SAMPLE = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])


# ── StandardScaler ────────────────────────────────────────────────────────────

class TestStandardScalerFit:
    def test_fit_returns_self(self):
        ss = StandardScaler()
        assert ss.fit(X_SAMPLE) is ss

    def test_mean_computed(self):
        ss = StandardScaler().fit(X_SAMPLE)
        np.testing.assert_allclose(ss.mean_, [4.0, 5.0])

    def test_scale_computed(self):
        ss = StandardScaler().fit(X_SAMPLE)
        assert ss.scale_ is not None
        assert ss.scale_.shape == (2,)

    def test_without_mean(self):
        ss = StandardScaler(with_mean=False).fit(X_SAMPLE)
        assert ss.mean_ is None

    def test_without_std(self):
        ss = StandardScaler(with_std=False).fit(X_SAMPLE)
        assert ss.scale_ is None


class TestStandardScalerTransform:
    def test_transform_shape(self):
        ss = StandardScaler().fit(X_SAMPLE)
        assert ss.transform(X_SAMPLE).shape == X_SAMPLE.shape

    def test_transform_zero_mean(self):
        ss = StandardScaler().fit(X_SAMPLE)
        X_tr = ss.transform(X_SAMPLE)
        np.testing.assert_allclose(X_tr.mean(axis=0), [0.0, 0.0], atol=1e-10)

    def test_transform_unit_variance(self):
        ss = StandardScaler().fit(X_SAMPLE)
        X_tr = ss.transform(X_SAMPLE)
        np.testing.assert_allclose(X_tr.std(axis=0), [1.0, 1.0], atol=1e-10)

    def test_unfitted_raises(self):
        with pytest.raises(RuntimeError):
            StandardScaler().transform(X_SAMPLE)

    def test_transform_does_not_modify_original(self):
        X = X_SAMPLE.copy()
        ss = StandardScaler().fit(X)
        ss.transform(X)
        np.testing.assert_array_equal(X, X_SAMPLE)


class TestStandardScalerInverseTransform:
    def test_inverse_recovers_original(self):
        ss = StandardScaler().fit(X_SAMPLE)
        X_tr = ss.transform(X_SAMPLE)
        X_recovered = ss.inverse_transform(X_tr)
        np.testing.assert_allclose(X_recovered, X_SAMPLE, atol=1e-10)


# ── MinMaxScaler ──────────────────────────────────────────────────────────────

class TestMinMaxScalerFit:
    def test_fit_returns_self(self):
        mms = MinMaxScaler()
        assert mms.fit(X_SAMPLE) is mms

    def test_data_min_computed(self):
        mms = MinMaxScaler().fit(X_SAMPLE)
        np.testing.assert_allclose(mms.data_min_, [1.0, 2.0])

    def test_data_max_computed(self):
        mms = MinMaxScaler().fit(X_SAMPLE)
        np.testing.assert_allclose(mms.data_max_, [7.0, 8.0])


class TestMinMaxScalerTransform:
    def test_min_becomes_zero(self):
        mms = MinMaxScaler().fit(X_SAMPLE)
        X_tr = mms.transform(X_SAMPLE)
        np.testing.assert_allclose(X_tr.min(axis=0), [0.0, 0.0], atol=1e-10)

    def test_max_becomes_one(self):
        mms = MinMaxScaler().fit(X_SAMPLE)
        X_tr = mms.transform(X_SAMPLE)
        np.testing.assert_allclose(X_tr.max(axis=0), [1.0, 1.0], atol=1e-10)

    def test_transform_shape(self):
        mms = MinMaxScaler().fit(X_SAMPLE)
        assert mms.transform(X_SAMPLE).shape == X_SAMPLE.shape

    def test_unfitted_raises(self):
        with pytest.raises(RuntimeError):
            MinMaxScaler().transform(X_SAMPLE)

    def test_constant_feature_not_divide_by_zero(self):
        X = np.array([[1.0, 5.0], [1.0, 10.0]])  # first feature is constant
        mms = MinMaxScaler().fit(X)
        X_tr = mms.transform(X)
        assert np.all(np.isfinite(X_tr))


class TestMinMaxScalerInverseTransform:
    def test_inverse_recovers_original(self):
        mms = MinMaxScaler().fit(X_SAMPLE)
        X_tr = mms.transform(X_SAMPLE)
        X_rec = mms.inverse_transform(X_tr)
        np.testing.assert_allclose(X_rec, X_SAMPLE, atol=1e-10)


# ── PolynomialFeatures ────────────────────────────────────────────────────────

class TestPolynomialFeatures:
    def test_degree_1_with_bias(self):
        pf = PolynomialFeatures(degree=1, include_bias=True).fit(X_SAMPLE)
        out = pf.transform(X_SAMPLE)
        # [1, x1, x2] → 3 columns
        assert out.shape == (4, 3)

    def test_degree_2_with_bias_two_features(self):
        pf = PolynomialFeatures(degree=2, include_bias=True).fit(X_SAMPLE)
        out = pf.transform(X_SAMPLE)
        # [1, x1, x2, x1², x1x2, x2²] → 6 columns
        assert out.shape == (4, 6)

    def test_bias_column_is_ones(self):
        pf = PolynomialFeatures(degree=1, include_bias=True).fit(X_SAMPLE)
        out = pf.transform(X_SAMPLE)
        np.testing.assert_array_equal(out[:, 0], np.ones(4))

    def test_no_bias(self):
        pf = PolynomialFeatures(degree=1, include_bias=False).fit(X_SAMPLE)
        out = pf.transform(X_SAMPLE)
        assert out.shape[1] == 2

    def test_interaction_only(self):
        pf = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False).fit(X_SAMPLE)
        out = pf.transform(X_SAMPLE)
        # degree-1: x1, x2; degree-2 interaction only: x1*x2 → 3 cols
        assert out.shape == (4, 3)

    def test_fit_transform_consistent(self):
        pf = PolynomialFeatures(degree=2)
        out_ft = pf.fit_transform(X_SAMPLE)
        out_t  = pf.transform(X_SAMPLE)
        np.testing.assert_array_equal(out_ft, out_t)

    def test_feature_mismatch_raises(self):
        pf = PolynomialFeatures(degree=2).fit(X_SAMPLE)
        X_bad = np.ones((3, 5))
        with pytest.raises(ValueError, match="shape"):
            pf.transform(X_bad)


# ── OneHotEncoder ─────────────────────────────────────────────────────────────

class TestOneHotEncoder:
    @pytest.fixture
    def cat_data(self):
        return np.array([[0, "a"], [1, "b"], [0, "a"], [1, "c"]], dtype=object)

    def test_fit_identifies_categories(self):
        X = np.array([[0], [1], [2]])
        enc = OneHotEncoder().fit(X)
        assert enc.categories_ is not None
        assert len(enc.categories_) == 1

    def test_transform_shape(self):
        X = np.array([[0], [1], [2]])
        enc = OneHotEncoder().fit(X)
        out = enc.transform(X)
        assert out.shape == (3, 3)

    def test_transform_binary_values(self):
        X = np.array([[0], [1], [0]])
        enc = OneHotEncoder().fit(X)
        out = enc.transform(X)
        assert set(out.flatten()).issubset({0, 1})

    def test_each_row_sums_to_n_features(self):
        X = np.array([[0], [1], [2]])
        enc = OneHotEncoder().fit(X)
        out = enc.transform(X)
        np.testing.assert_array_equal(out.sum(axis=1), np.ones(3))

    def test_fit_transform_consistent(self):
        X = np.array([[0], [1], [0]])
        enc = OneHotEncoder()
        np.testing.assert_array_equal(enc.fit_transform(X), enc.fit(X).transform(X))

    def test_unfitted_raises(self):
        with pytest.raises(RuntimeError):
            OneHotEncoder().transform(np.array([[0]]))


# ── train_test_split ──────────────────────────────────────────────────────────

class TestTrainTestSplit:
    def test_default_split_sizes(self):
        X = np.arange(100).reshape(50, 2).astype(float)
        y = np.arange(50)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y)
        assert X_tr.shape[0] == 40
        assert X_te.shape[0] == 10

    def test_custom_test_size(self):
        X = np.ones((100, 3))
        y = np.zeros(100)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3)
        assert X_te.shape[0] == 30

    def test_total_samples_preserved(self):
        X = np.ones((80, 2))
        y = np.zeros(80)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y)
        assert X_tr.shape[0] + X_te.shape[0] == 80

    def test_reproducible_with_seed(self):
        X = np.random.randn(50, 2)
        y = np.random.randint(0, 2, 50)
        splits1 = train_test_split(X, y, random_seed=99)
        splits2 = train_test_split(X, y, random_seed=99)
        for a, b in zip(splits1, splits2):
            np.testing.assert_array_equal(a, b)

    def test_different_seeds_different_splits(self):
        X = np.arange(40).reshape(20, 2).astype(float)
        y = np.arange(20)
        _, X_te1, _, _ = train_test_split(X, y, random_seed=1)
        _, X_te2, _, _ = train_test_split(X, y, random_seed=2)
        assert not np.array_equal(X_te1, X_te2)

    def test_invalid_test_size_raises(self):
        with pytest.raises(ValueError):
            train_test_split(np.ones((10, 2)), np.zeros(10), test_size=1.5)

    def test_zero_test_size_raises(self):
        with pytest.raises(ValueError):
            train_test_split(np.ones((10, 2)), np.zeros(10), test_size=0.0)

    def test_y_shape_preserved(self):
        X = np.ones((20, 2))
        y = np.zeros(20)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y)
        assert y_tr.ndim == 1
        assert y_te.ndim == 1
