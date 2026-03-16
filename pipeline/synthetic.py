"""Synthetic data generator for public HF Space demo.

Generates realistic coords.json without exposing real repo data.
"""
import json
import argparse
import numpy as np
from typing import List, Dict

# Cluster definitions for realistic synthetic data
CLUSTERS = [
    {
        'id': 0,
        'name': 'ML experimentation',
        'size': 35,
        'prefixes': ['bert-', 'gpt-', 'rl-', 'transformer-', 'ml-', 'pytorch-', 'tf-'],
        'suffixes': ['-trainer', '-finetuner', '-playground', '-experiments', '-benchmark'],
        'center': np.array([4.0, 2.0, 1.0]),
        'spread': 0.8,
    },
    {
        'id': 1,
        'name': 'React / frontend',
        'size': 28,
        'prefixes': ['react-', 'next-', 'vue-', 'dashboard-', 'ui-', 'web-'],
        'suffixes': ['-ui', '-app', '-frontend', '-portal', '-components'],
        'center': np.array([-3.0, 3.0, 0.5]),
        'spread': 1.2,
    },
    {
        'id': 2,
        'name': 'Data pipelines',
        'size': 24,
        'prefixes': ['etl-', 'dbt-', 'airflow-', 'spark-', 'data-', 'pipeline-'],
        'suffixes': ['-transforms', '-scheduler', '-loader', '-pipeline', '-ingestion'],
        'center': np.array([1.0, -3.0, 2.0]),
        'spread': 0.9,
    },
    {
        'id': 3,
        'name': 'CLI tooling',
        'size': 20,
        'prefixes': ['cli-', 'gh-', 'git-', 'dev-', ''],
        'suffixes': ['-tool', '-cli', '-utils', '-helper', 'dotfiles'],
        'center': np.array([-2.0, -2.0, -1.5]),
        'spread': 1.0,
    },
    {
        'id': 4,
        'name': 'Backend / API',
        'size': 32,
        'prefixes': ['api-', 'fastapi-', 'flask-', 'service-', 'auth-', 'gateway-'],
        'suffixes': ['-service', '-api', '-server', '-backend', '-gateway'],
        'center': np.array([0.0, 0.0, -3.0]),
        'spread': 1.1,
    },
    {
        'id': 5,
        'name': 'DevOps / infra',
        'size': 18,
        'prefixes': ['terraform-', 'k8s-', 'docker-', 'ansible-', 'infra-', 'deploy-'],
        'suffixes': ['-modules', '-configs', '-templates', '-scripts', '-automation'],
        'center': np.array([3.0, -1.0, -2.0]),
        'spread': 0.7,
    },
    {
        'id': 6,
        'name': 'LLM / agents',
        'size': 22,
        'prefixes': ['rag-', 'agent-', 'llm-', 'chat-', 'langchain-', 'claude-'],
        'suffixes': ['-pipeline', '-framework', '-bot', '-assistant', '-chain'],
        'center': np.array([2.5, 2.5, 2.5]),
        'spread': 0.85,
    },
]

NOISE_NAMES = [
    'scratch-pad', 'old-experiments', 'temp-project', 'learning-rust',
    'advent-of-code', 'interview-prep', 'blog-drafts', 'config-backup',
    'random-scripts', 'test-repo', 'playground', 'sandbox',
    'legacy-app', 'deprecated-api', 'archive-2019', 'misc-utils',
    'quick-hack', 'prototype-v1', 'demo-project', 'poc-idea',
    'notes', 'snippets', 'templates', 'boilerplate',
]


def generate_repo_name(cluster: Dict, idx: int) -> str:
    """Generate a realistic repo name for a cluster."""
    prefix = np.random.choice(cluster['prefixes'])
    suffix = np.random.choice(cluster['suffixes'])

    # Sometimes add a number
    if np.random.random() < 0.3:
        suffix = suffix + f"-{np.random.randint(1, 5)}"

    name = f"{prefix}{suffix}".strip('-').replace('--', '-')

    # Ensure unique by adding index if needed
    if np.random.random() < 0.2:
        name = f"{name}-{idx}"

    return name


def generate_coords(center: np.ndarray, spread: float, n: int) -> np.ndarray:
    """Generate 3D coordinates around a center point."""
    return center + np.random.randn(n, 3) * spread


def add_tsne_distortion(umap_coords: np.ndarray) -> np.ndarray:
    """Create t-SNE-like coords by distorting UMAP coords.

    t-SNE typically has more separation between clusters but less
    preservation of global structure.
    """
    # Scale clusters apart more
    tsne = umap_coords.copy()

    # Add some rotation
    angle = np.pi / 6
    rotation = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    tsne = tsne @ rotation.T

    # Stretch
    tsne *= np.array([1.3, 0.9, 1.1])

    # Add noise
    tsne += np.random.randn(*tsne.shape) * 0.15

    return tsne


def generate_synthetic_data(n_repos: int = 237, seed: int = 42) -> Dict:
    """Generate synthetic coords.json data."""
    np.random.seed(seed)

    repos = []
    all_umap = []

    # Calculate cluster sizes proportionally
    total_cluster_size = sum(c['size'] for c in CLUSTERS)
    noise_count = int(n_repos * 0.1)  # 10% noise
    remaining = n_repos - noise_count

    # Generate clustered repos
    for cluster in CLUSTERS:
        size = int(remaining * cluster['size'] / total_cluster_size)
        coords = generate_coords(cluster['center'], cluster['spread'], size)

        for i in range(size):
            name = generate_repo_name(cluster, i)
            repos.append({
                'name': name,
                'cluster_id': cluster['id'],
                'cluster_name': cluster['name'],
                'confidence': round(0.7 + np.random.random() * 0.25, 2),
            })
            all_umap.append(coords[i])

    # Generate noise points
    noise_coords = np.random.randn(noise_count, 3) * 2.5  # Scattered
    for i in range(noise_count):
        name = np.random.choice(NOISE_NAMES)
        if np.random.random() < 0.5:
            name = f"{name}-{np.random.randint(1, 100)}"

        repos.append({
            'name': name,
            'cluster_id': -1,
            'cluster_name': 'unclustered',
            'confidence': round(0.3 + np.random.random() * 0.3, 2),
        })
        all_umap.append(noise_coords[i])

    # Convert to arrays
    umap_coords = np.array(all_umap)
    tsne_coords = add_tsne_distortion(umap_coords)

    # Add coordinates to repos
    for i, repo in enumerate(repos):
        repo['umap'] = umap_coords[i].tolist()
        repo['tsne'] = tsne_coords[i].tolist()

    # Build cluster list
    clusters = []
    for cluster in CLUSTERS:
        count = sum(1 for r in repos if r['cluster_id'] == cluster['id'])
        clusters.append({
            'id': cluster['id'],
            'name': cluster['name'],
            'size': count,
        })

    return {
        'repos': repos,
        'clusters': clusters,
    }


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic coords.json')
    parser.add_argument('--output', '-o', default='coords.json', help='Output file path')
    parser.add_argument('--repos', '-n', type=int, default=237, help='Number of repos')
    parser.add_argument('--seed', '-s', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    data = generate_synthetic_data(n_repos=args.repos, seed=args.seed)

    with open(args.output, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Generated {len(data['repos'])} repos across {len(data['clusters'])} clusters")
    print(f"Noise points: {sum(1 for r in data['repos'] if r['cluster_id'] == -1)}")
    print(f"Saved to {args.output}")


if __name__ == '__main__':
    main()
