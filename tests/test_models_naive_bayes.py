"""Tests for simpleml.models.naive_bayes — GaussianNaiveBayes, MultinomialNaiveBayes."""
import pytest
import numpy as np

from simpleml.models.naive_bayes import GaussianNaiveBayes, MultinomialNaiveBayes


# ── GaussianNaiveBayes ────────────────────────────────────────────────────────

class TestGaussianNBInit:
    def test_attributes_none_before_fit(self):
        gnb = GaussianNaiveBayes()
        assert gnb.theta_ is None
        assert gnb.sigma_ is None
        assert gnb.class_prior_ is None
        assert gnb.classes_ is None


class TestGaussianNBFit:
    def test_fit_returns_self(self, binary_classification_data):
        X, y = binary_classification_data
        gnb = GaussianNaiveBayes()
        assert gnb.fit(X, y) is gnb

    def test_classes_identified(self, binary_classification_data):
        X, y = binary_classification_data
        gnb = GaussianNaiveBayes().fit(X, y)
        np.testing.assert_array_equal(gnb.classes_, [0, 1])

    def test_theta_shape(self, binary_classification_data):
        X, y = binary_classification_data
        gnb = GaussianNaiveBayes().fit(X, y)
        assert gnb.theta_.shape == (2, 2)

    def test_sigma_shape(self, binary_classification_data):
        X, y = binary_classification_data
        gnb = GaussianNaiveBayes().fit(X, y)
        assert gnb.sigma_.shape == (2, 2)

    def test_class_prior_sums_to_one(self, binary_classification_data):
        X, y = binary_classification_data
        gnb = GaussianNaiveBayes().fit(X, y)
        assert gnb.class_prior_.sum() == pytest.approx(1.0)

    def test_balanced_classes_equal_prior(self, binary_classification_data):
        X, y = binary_classification_data
        gnb = GaussianNaiveBayes().fit(X, y)
        np.testing.assert_allclose(gnb.class_prior_, [0.5, 0.5])

    def test_theta_are_class_means(self):
        X = np.array([[1.0, 0.0], [3.0, 0.0], [5.0, 0.0]], )
        y = np.array([0, 0, 0])
        gnb = GaussianNaiveBayes().fit(X, y)
        np.testing.assert_allclose(gnb.theta_[0], [3.0, 0.0])

    def test_multiclass_three_classes(self, multiclass_data):
        X, y = multiclass_data
        gnb = GaussianNaiveBayes().fit(X, y)
        assert len(gnb.classes_) == 3
        assert gnb.theta_.shape == (3, 2)

    def test_sigma_positive(self, binary_classification_data):
        X, y = binary_classification_data
        gnb = GaussianNaiveBayes().fit(X, y)
        assert (gnb.sigma_ >= 0).all()


class TestGaussianNBPredict:
    def test_predict_shape(self, binary_classification_data):
        X, y = binary_classification_data
        gnb = GaussianNaiveBayes().fit(X, y)
        assert gnb.predict(X).shape == (100,)

    def test_predict_only_valid_classes(self, binary_classification_data):
        X, y = binary_classification_data
        gnb = GaussianNaiveBayes().fit(X, y)
        assert set(gnb.predict(X)).issubset({0.0, 1.0})

    def test_high_accuracy_separable(self, binary_classification_data):
        X, y = binary_classification_data
        gnb = GaussianNaiveBayes().fit(X, y)
        assert gnb.score(X, y) > 0.85

    def test_multiclass_predictions(self, multiclass_data):
        X, y = multiclass_data
        gnb = GaussianNaiveBayes().fit(X, y)
        preds = gnb.predict(X)
        assert set(preds).issubset({0.0, 1.0, 2.0})
        assert gnb.score(X, y) > 0.85

    def test_likelihood_positive(self):
        gnb = GaussianNaiveBayes()
        x = np.array([0.0])
        theta = np.array([0.0])
        sigma = np.array([1.0])
        likelihood = gnb._calculate_likelihood(x, theta, sigma)
        assert likelihood > 0

    def test_likelihood_peaks_at_mean(self):
        gnb = GaussianNaiveBayes()
        theta = np.array([2.0])
        sigma = np.array([1.0])
        at_mean = gnb._calculate_likelihood(np.array([2.0]), theta, sigma)
        away    = gnb._calculate_likelihood(np.array([5.0]), theta, sigma)
        assert at_mean > away


# ── MultinomialNaiveBayes ─────────────────────────────────────────────────────

class TestMultinomialNBInit:
    def test_default_alpha(self):
        mnb = MultinomialNaiveBayes()
        assert mnb.alpha == 1.0

    def test_custom_alpha(self):
        mnb = MultinomialNaiveBayes(alpha=0.5)
        assert mnb.alpha == 0.5

    def test_attributes_none_before_fit(self):
        mnb = MultinomialNaiveBayes()
        assert mnb.feature_log_prob_ is None
        assert mnb.class_log_prior_ is None
        assert mnb.classes_ is None


class TestMultinomialNBFit:
    @pytest.fixture
    def count_data(self):
        rng = np.random.RandomState(0)
        X0 = rng.randint(0, 10, (30, 5)).astype(float)
        X1 = rng.randint(5, 15, (30, 5)).astype(float)
        X = np.vstack([X0, X1])
        y = np.array([0] * 30 + [1] * 30)
        return X, y

    def test_fit_returns_self(self, count_data):
        X, y = count_data
        mnb = MultinomialNaiveBayes()
        assert mnb.fit(X, y) is mnb


    def test_classes_identified(self, count_data):
        X, y = count_data
        mnb = MultinomialNaiveBayes().fit(X, y)
        np.testing.assert_array_equal(mnb.classes_, [0, 1])

    def test_feature_log_prob_shape(self, count_data):
        X, y = count_data
        mnb = MultinomialNaiveBayes().fit(X, y)
        assert mnb.feature_log_prob_.shape == (2, 5)

    def test_class_log_prior_shape(self, count_data):
        X, y = count_data
        mnb = MultinomialNaiveBayes().fit(X, y)
        assert mnb.class_log_prior_.shape == (2,)

    def test_class_log_prior_balanced(self, count_data):
        X, y = count_data
        mnb = MultinomialNaiveBayes().fit(X, y)
        np.testing.assert_allclose(mnb.class_log_prior_, [np.log(0.5), np.log(0.5)])

    def test_laplace_smoothing_zero_count(self):
        """Feature that never appears in class should still have a finite log-prob."""
        X = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=float)
        y = np.array([0, 0, 1, 1])
        mnb = MultinomialNaiveBayes(alpha=1.0).fit(X, y)
        # class 0 has zero count for feature 1 — log-prob must be finite
        assert np.isfinite(mnb.feature_log_prob_[0, 1])


class TestMultinomialNBPredict:
    @pytest.fixture
    def count_data(self):
        rng = np.random.RandomState(1)
        # Class 0: high counts in features 0-1, near-zero in features 2-3
        X0 = np.hstack([
            rng.randint(10, 20, (40, 2)).astype(float),
            rng.randint(0,   2, (40, 2)).astype(float),
        ])
        # Class 1: near-zero in features 0-1, high counts in features 2-3
        X1 = np.hstack([
            rng.randint(0,   2, (40, 2)).astype(float),
            rng.randint(10, 20, (40, 2)).astype(float),
        ])
        X = np.vstack([X0, X1])
        y = np.array([0] * 40 + [1] * 40)
        return X, y

    def test_predict_shape(self, count_data):
        X, y = count_data
        mnb = MultinomialNaiveBayes().fit(X, y)
        assert mnb.predict(X).shape == (80,)

    def test_predict_only_valid_classes(self, count_data):
        X, y = count_data
        mnb = MultinomialNaiveBayes().fit(X, y)
        assert set(mnb.predict(X)).issubset({0, 1})

    def test_high_accuracy(self, count_data):
        X, y = count_data
        mnb = MultinomialNaiveBayes().fit(X, y)
        assert mnb.score(X, y) > 0.90
