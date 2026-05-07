"""Tests for simpleml.models.clustering — KMeans, DBSCAN."""
import pytest
import numpy as np

from simpleml.models.clustering import KMeans, DBSCAN


# ── KMeans ────────────────────────────────────────────────────────────────────

class TestKMeansInit:
    def test_defaults(self):
        km = KMeans()
        assert km.n_clusters == 8
        assert km.n_init == 10
        assert km.max_iter == 300
        assert km.random_state is None
        assert km.tol == 1e-4

    def test_custom_params(self):
        km = KMeans(n_clusters=3, n_init=5, random_state=42)
        assert km.n_clusters == 3
        assert km.n_init == 5
        assert km.random_state == 42


class TestKMeansFit:
    def test_fit_returns_self(self, cluster_data):
        km = KMeans(n_clusters=3, n_init=2, random_state=0)
        assert km.fit(cluster_data) is km

    def test_cluster_centers_shape(self, cluster_data):
        km = KMeans(n_clusters=3, n_init=2, random_state=0).fit(cluster_data)
        assert km.cluster_centers_.shape == (3, 2)

    def test_labels_shape(self, cluster_data):
        km = KMeans(n_clusters=3, n_init=2, random_state=0).fit(cluster_data)
        assert km.labels_.shape == (90,)

    def test_labels_only_valid_cluster_ids(self, cluster_data):
        km = KMeans(n_clusters=3, n_init=2, random_state=0).fit(cluster_data)
        assert set(km.labels_).issubset({0, 1, 2})

    def test_inertia_is_positive(self, cluster_data):
        km = KMeans(n_clusters=3, n_init=2, random_state=0).fit(cluster_data)
        assert km.inertia_ > 0

    def test_more_clusters_lower_inertia(self, cluster_data):
        km3 = KMeans(n_clusters=3, n_init=2, random_state=0).fit(cluster_data)
        km9 = KMeans(n_clusters=9, n_init=2, random_state=0).fit(cluster_data)
        assert km9.inertia_ <= km3.inertia_

    def test_n_clusters_greater_than_samples_raises(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        km = KMeans(n_clusters=5, n_init=1)
        with pytest.raises(ValueError):
            km.fit(X)

    def test_reproducible_with_seed(self, cluster_data):
        km1 = KMeans(n_clusters=3, n_init=3, random_state=7).fit(cluster_data)
        km2 = KMeans(n_clusters=3, n_init=3, random_state=7).fit(cluster_data)
        np.testing.assert_array_equal(km1.labels_, km2.labels_)

    def test_well_separated_clusters_found(self, cluster_data):
        km = KMeans(n_clusters=3, n_init=5, random_state=0).fit(cluster_data)
        # Each of the 3 natural clusters of 30 points should form a distinct predicted cluster
        unique_labels = np.unique(km.labels_)
        assert len(unique_labels) == 3

    def test_assign_clusters_shape(self, cluster_data):
        km = KMeans(n_clusters=3, n_init=2, random_state=0).fit(cluster_data)
        labels = km._assign_clusters(cluster_data, km.cluster_centers_)
        assert labels.shape == (90,)

    def test_assign_clusters_valid_ids(self, cluster_data):
        km = KMeans(n_clusters=3, n_init=2, random_state=0).fit(cluster_data)
        labels = km._assign_clusters(cluster_data, km.cluster_centers_)
        assert set(labels).issubset({0, 1, 2})

    def test_initialize_centroids_shape(self, cluster_data):
        km = KMeans(n_clusters=3)
        rng = np.random.RandomState(0)
        centroids = km._initialize_centroids(cluster_data, rng)
        assert centroids.shape == (3, 2)

    def test_initialize_centroids_are_from_data(self, cluster_data):
        km = KMeans(n_clusters=3)
        rng = np.random.RandomState(0)
        centroids = km._initialize_centroids(cluster_data, rng)
        for c in centroids:
            assert any(np.allclose(c, x) for x in cluster_data)


class TestKMeansPredict:
    def test_predict_shape(self, cluster_data):
        km = KMeans(n_clusters=3, n_init=2, random_state=0).fit(cluster_data)
        preds = km.predict(cluster_data[:10])
        assert preds.shape == (10,)

    def test_predict_only_valid_ids(self, cluster_data):
        km = KMeans(n_clusters=3, n_init=2, random_state=0).fit(cluster_data)
        preds = km.predict(cluster_data)
        assert set(preds).issubset({0, 1, 2})

    def test_predict_consistent_with_labels(self, cluster_data):
        km = KMeans(n_clusters=3, n_init=2, random_state=0).fit(cluster_data)
        np.testing.assert_array_equal(km.predict(cluster_data), km.labels_)


# ── DBSCAN ────────────────────────────────────────────────────────────────────

class TestDBSCANInit:
    def test_defaults(self):
        db = DBSCAN()
        assert db.eps == 0.5
        assert db.min_samples == 5

    def test_custom_params(self):
        db = DBSCAN(eps=1.0, min_samples=3)
        assert db.eps == 1.0
        assert db.min_samples == 3


class TestDBSCANFit:
    def test_fit_returns_self(self, cluster_data):
        db = DBSCAN(eps=1.5, min_samples=3)
        assert db.fit(cluster_data) is db

    def test_labels_shape(self, cluster_data):
        db = DBSCAN(eps=1.5, min_samples=3).fit(cluster_data)
        assert db.labels_.shape == (90,)

    def test_outliers_labeled_minus_one(self):
        """Single isolated point should be an outlier."""
        X = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1],
                      [100.0, 100.0]])        # clear outlier
        db = DBSCAN(eps=0.5, min_samples=2).fit(X)
        assert db.labels_[3] == -1

    def test_dense_cluster_not_outlier(self):
        X = np.zeros((10, 2)) + np.random.RandomState(0).randn(10, 2) * 0.05
        db = DBSCAN(eps=0.5, min_samples=3).fit(X)
        assert all(db.labels_ != -1)

    def test_separated_clusters_get_different_ids(self, cluster_data):
        db = DBSCAN(eps=1.5, min_samples=3).fit(cluster_data)
        unique = set(db.labels_) - {-1}
        assert len(unique) >= 2

    def test_very_small_eps_all_outliers(self, cluster_data):
        db = DBSCAN(eps=1e-10, min_samples=5).fit(cluster_data)
        assert all(db.labels_ == -1)

    def test_get_neighbors_includes_self(self):
        X = np.array([[0.0, 0.0], [1.0, 0.0]])
        db = DBSCAN(eps=0.5)
        neighbours = db._get_neighbors(X, 0)
        assert 0 in neighbours

    def test_get_neighbors_excludes_far_point(self):
        X = np.array([[0.0, 0.0], [100.0, 0.0]])
        db = DBSCAN(eps=1.0)
        assert 1 not in db._get_neighbors(X, 0)


class TestDBSCANPredict:
    def test_predict_shape(self, cluster_data):
        db = DBSCAN(eps=1.5, min_samples=3).fit(cluster_data)
        assert db.predict(cluster_data[:10]).shape == (10,)

    def test_predict_known_points_match_fit_labels(self, cluster_data):
        """Points seen during fit should get the same label from predict."""
        db = DBSCAN(eps=1.5, min_samples=3).fit(cluster_data)
        preds = db.predict(cluster_data)
        # Training points that are not noise should match their fit labels
        non_noise = db.labels_ != -1
        np.testing.assert_array_equal(preds[non_noise], db.labels_[non_noise])

    def test_predict_isolated_point_is_noise(self):
        """A point far from all clusters should be labelled -1."""
        X = np.zeros((10, 2)) + np.random.RandomState(0).randn(10, 2) * 0.05
        db = DBSCAN(eps=0.5, min_samples=3).fit(X)
        far_point = np.array([[100.0, 100.0]])
        assert db.predict(far_point)[0] == -1
