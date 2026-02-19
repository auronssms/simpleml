"""Clustering algorithms."""

import numpy as np
from .base import BaseEstimator, ClusterMixin


class KMeans(BaseEstimator, ClusterMixin):
    """
    K-Means clustering algorithm.
    
    Parameters
    ----------
    n_clusters : int, default=8
        Number of clusters to form.
    n_init : int, default=10
        Number of time the k-means algorithm will be run with different
        centroid seeds.
    max_iter : int, default=300
        Maximum number of iterations for a single run.
    random_state : int, optional
        Random seed for reproducibility.
    tol : float, default=1e-4
        Relative tolerance with regards to inertia to declare convergence.
    
    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers.
    labels_ : ndarray of shape (n_samples,)
        Labels of each point.
    inertia_ : float
        Sum of squared distances of samples to their nearest cluster center.
    """
    
    def __init__(
        self,
        n_clusters=8,
        n_init=10,
        max_iter=300,
        random_state=None,
        tol=1e-4,
    ):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.tol = tol
    
    def _initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        """Initialize cluster centroids."""
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        return X[indices]
    
    def _assign_clusters(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Assign each sample to the nearest centroid."""
        distances = np.zeros((X.shape[0], self.n_clusters))
        
        for i, centroid in enumerate(centroids):
            distances[:, i] = np.sum((X - centroid) ** 2, axis=1)
        
        return np.argmin(distances, axis=1)
    
    def _update_centroids(
        self,
        X: np.ndarray,
        labels: np.ndarray,
    ) -> np.ndarray:
        """Update centroids based on cluster assignments."""
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                centroids[k] = np.mean(cluster_points, axis=0)
            else:
                centroids[k] = X[np.random.choice(X.shape[0])]
        
        return centroids
    
    def _calculate_inertia(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Calculate inertia (sum of squared distances)."""
        inertia = 0
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - self.cluster_centers_[k]) ** 2)
        return inertia
    
    def fit(self, X: np.ndarray, y=None) -> 'KMeans':
        """
        Compute k-means clustering.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present here for API consistency by convention.
            
        Returns
        -------
        self : object
            Returns self.
        """
        X = np.asarray(X)
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        best_inertia = np.inf
        best_centroids = None
        best_labels = None
        
        for _ in range(self.n_init):
            centroids = self._initialize_centroids(X)
            
            for _ in range(self.max_iter):
                labels = self._assign_clusters(X, centroids)
                new_centroids = self._update_centroids(X, labels)
                
                if np.sum((new_centroids - centroids) ** 2) < self.tol:
                    break
                
                centroids = new_centroids
            
            self.cluster_centers_ = centroids
            inertia = self._calculate_inertia(X, labels)
            
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids
                best_labels = labels
        
        self.cluster_centers_ = best_centroids
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the closest cluster each sample in X belongs to.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            New data to predict.
            
        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        X = np.asarray(X)
        return self._assign_clusters(X, self.cluster_centers_)


class DBSCAN(BaseEstimator, ClusterMixin):
    """
    Density-Based Spatial Clustering of Applications with Noise.
    
    Parameters
    ----------
    eps : float, default=0.5
        The maximum distance between two samples for one to be considered
        as in the neighborhood of the other.
    min_samples : int, default=5
        The number of samples in a neighborhood for a point to be considered
        as a core point.
    
    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels for each point in the dataset. Points labeled as
        -1 are considered outliers.
    """
    
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
    
    def _get_neighbors(self, X: np.ndarray, point_idx: int) -> list:
        """Get indices of neighbors for a point."""
        distances = np.sqrt(np.sum((X - X[point_idx]) ** 2, axis=1))
        return np.where(distances <= self.eps)[0].tolist()
    
    def fit(self, X: np.ndarray, y=None) -> 'DBSCAN':
        """
        Perform DBSCAN clustering.
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present here for API consistency by convention.
            
        Returns
        -------
        self : object
            Returns self.
        """
        X = np.asarray(X)
        n_samples = X.shape[0]
        
        self.labels_ = np.full(n_samples, -1, dtype=int)
        cluster_id = 0
        visited = np.zeros(n_samples, dtype=bool)
        
        for i in range(n_samples):
            if visited[i]:
                continue
            
            neighbors = self._get_neighbors(X, i)
            
            if len(neighbors) < self.min_samples:
                self.labels_[i] = -1
                visited[i] = True
                continue
            
            cluster_id += 1
            self._expand_cluster(X, i, neighbors, cluster_id, visited)
        
        return self
    
    def _expand_cluster(
        self,
        X: np.ndarray,
        point_idx: int,
        neighbors: list,
        cluster_id: int,
        visited: np.ndarray,
    ):
        """Expand cluster from a core point."""
        self.labels_[point_idx] = cluster_id
        visited[point_idx] = True
        
        seed = 0
        while seed < len(neighbors):
            neighbor_idx = neighbors[seed]
            seed += 1
            
            if visited[neighbor_idx]:
                continue
            
            visited[neighbor_idx] = True
            neighbor_neighbors = self._get_neighbors(X, neighbor_idx)
            
            if len(neighbor_neighbors) >= self.min_samples:
                neighbors.extend(neighbor_neighbors)
            
            if self.labels_[neighbor_idx] == -1:
                self.labels_[neighbor_idx] = cluster_id
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data (not supported for DBSCAN).
        
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            New data to predict.
            
        Returns
        -------
        labels : ndarray
            This method is not supported. Returns -1 for all samples.
        """
        return np.full(X.shape[0], -1, dtype=int)