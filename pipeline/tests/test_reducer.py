"""Tests for dimension reducers."""
import pytest
import numpy as np


def test_reduce_umap():
    """UMAP reducer should produce 3D coordinates."""
    from pipeline.reducer import reduce_umap

    np.random.seed(42)
    embeddings = np.random.randn(50, 100)

    coords = reduce_umap(embeddings)

    assert coords.shape == (50, 3)


def test_reduce_tsne():
    """t-SNE reducer should produce 3D coordinates."""
    from pipeline.reducer import reduce_tsne

    np.random.seed(42)
    embeddings = np.random.randn(50, 100)

    coords = reduce_tsne(embeddings)

    assert coords.shape == (50, 3)


def test_reduce_both():
    """Should return both UMAP and t-SNE coordinates."""
    from pipeline.reducer import reduce_both

    np.random.seed(42)
    embeddings = np.random.randn(50, 100)

    umap_coords, tsne_coords = reduce_both(embeddings)

    assert umap_coords.shape == (50, 3)
    assert tsne_coords.shape == (50, 3)
