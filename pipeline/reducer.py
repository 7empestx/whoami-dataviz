"""Dimension reduction using UMAP and t-SNE."""
import numpy as np
from typing import Tuple
import umap
from sklearn.manifold import TSNE


def reduce_umap(embeddings: np.ndarray, n_components: int = 3) -> np.ndarray:
    """Reduce embeddings to n_components dimensions using UMAP."""
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=min(15, len(embeddings) - 1),
        min_dist=0.1,
        metric='cosine',
        random_state=42,
    )
    return reducer.fit_transform(embeddings)


def reduce_tsne(embeddings: np.ndarray, n_components: int = 3) -> np.ndarray:
    """Reduce embeddings to n_components dimensions using t-SNE."""
    reducer = TSNE(
        n_components=n_components,
        perplexity=min(30, len(embeddings) - 1),
        random_state=42,
        n_iter=1000,
    )
    return reducer.fit_transform(embeddings)


def reduce_both(embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Reduce embeddings using both UMAP and t-SNE."""
    umap_coords = reduce_umap(embeddings)
    tsne_coords = reduce_tsne(embeddings)
    return umap_coords, tsne_coords
