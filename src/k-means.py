"""
Naive K-Means Clustering from Scratch (NumPy only)
===================================================

The 4-Step Algorithm
--------------------
1. **Initialization:** Pick K random points from your dataset to be the
   initial centroids.
2. **Assignment:** Assign every data point to the nearest centroid
   (using Euclidean distance).
3. **Update:** Calculate the mean of all points assigned to each cluster.
   This mean becomes the new centroid.
4. **Repeat:** Keep doing steps 2 and 3 until the centroids stop moving
   (convergence).
"""

import numpy as np


def kmeans(X, k, max_iters=100, seed=None):
    """
    Naive K-Means clustering from scratch (NumPy only).

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Data points to cluster.
    k : int
        Number of clusters.
    max_iters : int
        Maximum number of iterations.
    seed : int or None
        Random seed for reproducible centroid initialisation.

    Returns
    -------
    labels : np.ndarray, shape (n_samples,)
        Cluster assignment for each point (0 to k-1).
    centroids : np.ndarray, shape (k, n_features)
        Final centroid positions.
    """
    rng = np.random.default_rng(seed)

    # --- 1. Initialise centroids: pick k random data points ---
    indices = rng.choice(len(X), size=k, replace=False)
    centroids = X[indices].copy()

    for _ in range(max_iters):
        # --- 2. Assign each point to the nearest centroid ---
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)  # (n, k)
        labels = np.argmin(distances, axis=1)

        # --- 3. Recompute centroids as the mean of assigned points ---
        new_centroids = np.array([
            X[labels == i].mean(axis=0) if np.any(labels == i)
            else X[rng.integers(len(X))]          # empty cluster → reinitialise
            for i in range(k)
        ])

        # --- 4. Check for convergence ---
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return labels, centroids


# ── Quick demo ──────────────────────────────────────────────────
if __name__ == "__main__":
    # Generate 3 blobs manually (no sklearn)
    rng = np.random.default_rng(42)
    blob1 = rng.normal(loc=[0, 0], scale=0.5, size=(50, 2))
    blob2 = rng.normal(loc=[5, 5], scale=0.5, size=(50, 2))
    blob3 = rng.normal(loc=[0, 5], scale=0.5, size=(50, 2))
    X = np.vstack([blob1, blob2, blob3])

    labels, centroids = kmeans(X, k=3, seed=0)

    print("Centroids:")
    print(centroids)
    print(f"\nLabel counts: {np.bincount(labels)}")
    print("Done.")
