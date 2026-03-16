"""Main pipeline orchestrator."""
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm

from pipeline.s3_reader import list_repos, get_repo_files
from pipeline.embedder import embed_repo
from pipeline.ast_features import extract_repo_features
from pipeline.clusterer import cluster_embeddings, get_cluster_stats
from pipeline.reducer import reduce_both
from pipeline.labeler import label_all_clusters


def process_repo(repo_name: str) -> Tuple[str, np.ndarray, np.ndarray, List[str], str]:
    """Process a single repo, returning embeddings and metadata."""
    files = list(get_repo_files(repo_name))

    readme = ""
    for filename, content in files:
        if filename.lower() in ('readme.md', 'readme', 'readme.txt', 'readme.rst'):
            readme = content
            break

    file_list = [f[0] for f in files]

    semantic_emb = embed_repo(files)
    structural_emb = extract_repo_features(files)

    return repo_name, semantic_emb, structural_emb, file_list, readme


def run_pipeline(output_path: str = "coords.json") -> None:
    """Run the full pipeline."""
    print("Listing repos from S3...")
    repo_names = list_repos()
    print(f"Found {len(repo_names)} repos")

    print("Processing repos...")
    repos_data = []
    semantic_embeddings = []
    structural_embeddings = []

    for repo_name in tqdm(repo_names):
        try:
            name, sem_emb, struct_emb, files, readme = process_repo(repo_name)
            repos_data.append({
                'name': name,
                'files': files,
                'readme': readme,
            })
            semantic_embeddings.append(sem_emb)
            structural_embeddings.append(struct_emb)
        except Exception as e:
            print(f"Error processing {repo_name}: {e}")
            continue

    if not repos_data:
        print("No repos processed!")
        return

    print("Concatenating embeddings...")
    semantic_arr = np.array(semantic_embeddings)
    structural_arr = np.array(structural_embeddings)
    combined = np.concatenate([semantic_arr, structural_arr], axis=1)

    print("Clustering...")
    labels = cluster_embeddings(combined)
    stats = get_cluster_stats(labels)
    print(f"Found {stats['n_clusters']} clusters, {stats['n_noise']} noise points")

    print("Labeling clusters...")
    cluster_labels = label_all_clusters(repos_data, labels.tolist())

    print("Reducing dimensions...")
    umap_coords, tsne_coords = reduce_both(combined)

    print("Building output...")
    output = {
        'repos': [],
        'clusters': [],
    }

    for i, repo in enumerate(repos_data):
        label = int(labels[i])
        cluster_info = cluster_labels.get(label, {'name': 'unknown', 'confidence': 0.0})

        output['repos'].append({
            'name': repo['name'],
            'umap': umap_coords[i].tolist(),
            'tsne': tsne_coords[i].tolist(),
            'cluster_id': label,
            'cluster_name': cluster_info['name'],
            'confidence': cluster_info['confidence'],
        })

    for cluster_id, info in cluster_labels.items():
        if cluster_id != -1:
            output['clusters'].append({
                'id': cluster_id,
                'name': info['name'],
                'size': stats['cluster_sizes'].get(cluster_id, 0),
            })

    print(f"Writing to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print("Done!")


if __name__ == "__main__":
    run_pipeline()
