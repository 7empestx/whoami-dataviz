"""Tests for HDBSCAN clusterer."""
import pytest
import numpy as np


def test_cluster_embeddings():
    """Clusterer should assign cluster IDs to embeddings."""
    from pipeline.clusterer import cluster_embeddings

    np.random.seed(42)
    cluster1 = np.random.randn(20, 10) + np.array([5, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    cluster2 = np.random.randn(20, 10) + np.array([0, 5, 0, 0, 0, 0, 0, 0, 0, 0])
    cluster3 = np.random.randn(20, 10) + np.array([0, 0, 5, 0, 0, 0, 0, 0, 0, 0])
    embeddings = np.vstack([cluster1, cluster2, cluster3])

    labels = cluster_embeddings(embeddings)

    assert len(labels) == 60
    assert isinstance(labels, np.ndarray)
    unique_labels = set(labels) - {-1}
    assert len(unique_labels) >= 2


def test_get_cluster_stats():
    """Should return cluster statistics."""
    from pipeline.clusterer import get_cluster_stats

    labels = np.array([0, 0, 0, 1, 1, -1, -1])
    stats = get_cluster_stats(labels)

    assert stats['n_clusters'] == 2
    assert stats['n_noise'] == 2
    assert stats['cluster_sizes'][0] == 3
    assert stats['cluster_sizes'][1] == 2
