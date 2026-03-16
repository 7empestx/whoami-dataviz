"""HDBSCAN clustering for repo embeddings."""
import numpy as np
import hdbscan

MIN_CLUSTER_SIZE = 5
MIN_SAMPLES = 3


def cluster_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """Cluster embeddings using HDBSCAN.

    Args:
        embeddings: (N, D) array of embedding vectors

    Returns:
        Array of cluster labels. -1 indicates noise points.
    """
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        metric='euclidean',
        cluster_selection_method='eom',
    )

    labels = clusterer.fit_predict(embeddings)
    return labels


def get_cluster_stats(labels: np.ndarray) -> dict:
    """Get statistics about clustering results."""
    unique_labels = set(labels)
    n_clusters = len(unique_labels - {-1})
    n_noise = int(np.sum(labels == -1))

    cluster_sizes = {}
    for label in unique_labels:
        if label != -1:
            cluster_sizes[int(label)] = int(np.sum(labels == label))

    return {
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'cluster_sizes': cluster_sizes,
    }
