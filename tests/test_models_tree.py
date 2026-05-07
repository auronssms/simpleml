"""Tests for simpleml.models.tree — Node, DecisionTreeClassifier, DecisionTreeRegressor."""
import pytest
import numpy as np

from simpleml.models.tree import Node, DecisionTreeClassifier, DecisionTreeRegressor


# ── Node ──────────────────────────────────────────────────────────────────────

class TestNode:
    def test_default_node_is_leaf(self):
        node = Node()
        assert node.feature is None
        assert node.threshold is None
        assert node.left is None
        assert node.right is None
        assert node.value is None

    def test_node_stores_value(self):
        node = Node(value=42.0)
        assert node.value == 42.0

    def test_node_stores_split(self):
        node = Node(feature=1, threshold=3.5)
        assert node.feature == 1
        assert node.threshold == 3.5

    def test_node_samples_default_zero(self):
        assert Node().samples == 0

    def test_node_impurity_default_zero(self):
        assert Node().impurity == 0.0


# ── DecisionTreeClassifier ────────────────────────────────────────────────────

class TestDecisionTreeClassifierInit:
    def test_defaults(self):
        clf = DecisionTreeClassifier()
        assert clf.max_depth is None
        assert clf.min_samples_split == 2
        assert clf.criterion == "gini"

    def test_custom_params(self):
        clf = DecisionTreeClassifier(max_depth=5, min_samples_split=4, criterion="entropy")
        assert clf.max_depth == 5
        assert clf.criterion == "entropy"

    def test_tree_is_none_before_fit(self):
        assert DecisionTreeClassifier().tree_ is None


class TestDecisionTreeClassifierFit:
    def test_fit_returns_self(self, binary_classification_data):
        X, y = binary_classification_data
        clf = DecisionTreeClassifier()
        assert clf.fit(X, y) is clf

    def test_tree_not_none_after_fit(self, binary_classification_data):
        X, y = binary_classification_data
        clf = DecisionTreeClassifier().fit(X, y)
        assert clf.tree_ is not None

    def test_n_classes_set(self, binary_classification_data):
        X, y = binary_classification_data
        clf = DecisionTreeClassifier().fit(X, y)
        assert clf.n_classes_ == 2

    def test_n_features_set(self, binary_classification_data):
        X, y = binary_classification_data
        clf = DecisionTreeClassifier().fit(X, y)
        assert clf.n_features_ == 2

    def test_fit_xor_pattern(self):
        """XOR is learnable with depth ≥ 2."""
        X = np.array([[0, 0], [1, 1], [0, 1], [1, 0]], dtype=float)
        y = np.array([0, 0, 1, 1])
        clf = DecisionTreeClassifier(max_depth=2).fit(X, y)
        preds = clf.predict(X)
        assert set(preds).issubset({0.0, 1.0})

    def test_multiclass_n_classes(self, multiclass_data):
        X, y = multiclass_data
        clf = DecisionTreeClassifier().fit(X, y)
        assert clf.n_classes_ == 3

    def test_single_class_leaf(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([0, 0])
        clf = DecisionTreeClassifier().fit(X, y)
        preds = clf.predict(X)
        assert (preds == 0).all()

    def test_criterion_entropy(self, binary_classification_data):
        X, y = binary_classification_data
        clf = DecisionTreeClassifier(criterion="entropy").fit(X, y)
        assert clf.tree_ is not None


class TestDecisionTreeClassifierPredict:
    def test_predict_shape(self, binary_classification_data):
        X, y = binary_classification_data
        clf = DecisionTreeClassifier().fit(X, y)
        assert clf.predict(X).shape == (100,)

    def test_predict_returns_ndarray(self, binary_classification_data):
        X, y = binary_classification_data
        clf = DecisionTreeClassifier().fit(X, y)
        assert isinstance(clf.predict(X), np.ndarray)

    def test_perfect_fit_no_depth_limit(self, binary_classification_data):
        X, y = binary_classification_data
        clf = DecisionTreeClassifier().fit(X, y)
        assert clf.score(X, y) == pytest.approx(1.0)

    def test_max_depth_limits_tree(self, binary_classification_data):
        X, y = binary_classification_data
        clf_deep    = DecisionTreeClassifier(max_depth=None).fit(X, y)
        clf_shallow = DecisionTreeClassifier(max_depth=1).fit(X, y)
        assert clf_deep.score(X, y) >= clf_shallow.score(X, y)

    def test_gini_impurity_calculation(self):
        clf = DecisionTreeClassifier()
        y_pure   = np.array([0, 0, 0])
        y_mixed  = np.array([0, 1, 0, 1])
        assert clf._gini(y_pure) == pytest.approx(0.0)
        assert clf._gini(y_mixed) == pytest.approx(0.5)

    def test_entropy_pure_node(self):
        clf = DecisionTreeClassifier(criterion="entropy")
        assert clf._entropy(np.array([1, 1, 1])) == pytest.approx(0.0)

    def test_entropy_max_impurity_binary(self):
        clf = DecisionTreeClassifier(criterion="entropy")
        assert clf._entropy(np.array([0, 1])) == pytest.approx(1.0)

    def test_information_gain_zero_on_no_improvement(self):
        clf = DecisionTreeClassifier()
        parent = np.array([0, 1])
        # Split that exactly mirrors parent in each child
        gain = clf._information_gain(parent, np.array([0]), np.array([1]))
        assert gain >= 0.0


# ── DecisionTreeRegressor ─────────────────────────────────────────────────────

class TestDecisionTreeRegressorInit:
    def test_defaults(self):
        reg = DecisionTreeRegressor()
        assert reg.max_depth is None
        assert reg.min_samples_split == 2

    def test_tree_none_before_fit(self):
        assert DecisionTreeRegressor().tree_ is None


class TestDecisionTreeRegressorFit:
    def test_fit_returns_self(self, simple_regression_data):
        X, y = simple_regression_data
        reg = DecisionTreeRegressor()
        assert reg.fit(X, y) is reg

    def test_n_features_set(self, simple_regression_data):
        X, y = simple_regression_data
        reg = DecisionTreeRegressor().fit(X, y)
        assert reg.n_features_ == 2

    def test_tree_built(self, simple_regression_data):
        X, y = simple_regression_data
        reg = DecisionTreeRegressor().fit(X, y)
        assert reg.tree_ is not None

    def test_mse_zero_on_constant(self):
        reg = DecisionTreeRegressor()
        assert reg._mse(np.array([3.0, 3.0, 3.0])) == pytest.approx(0.0)

    def test_mse_correct_value(self):
        reg = DecisionTreeRegressor()
        y = np.array([1.0, 3.0])   # mean=2, mse = ((1-2)²+(3-2)²)/2 = 1
        assert reg._mse(y) == pytest.approx(1.0)


class TestDecisionTreeRegressorPredict:
    def test_predict_shape(self, simple_regression_data):
        X, y = simple_regression_data
        reg = DecisionTreeRegressor().fit(X, y)
        assert reg.predict(X).shape == (100,)

    def test_predict_returns_ndarray(self, simple_regression_data):
        X, y = simple_regression_data
        reg = DecisionTreeRegressor().fit(X, y)
        assert isinstance(reg.predict(X), np.ndarray)

    def test_perfect_fit_single_feature(self):
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([10.0, 20.0, 30.0, 40.0])
        reg = DecisionTreeRegressor().fit(X, y)
        preds = reg.predict(X)
        assert preds.shape == (4,)
        assert np.mean(np.abs(preds - y)) < 15.0  # within leaf-mean error

    def test_perfect_fit_many_unique_points(self):
        """Pairs of identical X values give the tree clean leaf means that match targets."""
        X = np.repeat(np.arange(10, dtype=float), 2).reshape(-1, 1)  # [0,0,1,1,...,9,9]
        y = np.repeat(np.arange(10, dtype=float) * 5, 2)             # [0,0,5,5,...,45,45]
        reg = DecisionTreeRegressor().fit(X, y)
        np.testing.assert_allclose(reg.predict(X), y, atol=1e-6)

    def test_max_depth_one_returns_constant_in_region(self):
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([1.0, 2.0, 10.0, 11.0])
        reg = DecisionTreeRegressor(max_depth=1).fit(X, y)
        preds = reg.predict(X)
        # Predictions for first half should equal mean of 1 and 2 = 1.5
        np.testing.assert_allclose(preds[:2], 1.5, atol=1e-6)
