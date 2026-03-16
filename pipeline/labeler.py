"""Claude API cluster labeling."""
import json
from typing import Dict, List
from anthropic import Anthropic

_client = None


def get_client() -> Anthropic:
    """Get or create Anthropic client."""
    global _client
    if _client is None:
        _client = Anthropic()
    return _client


PROMPT_TEMPLATE = """Given these repo READMEs from the same cluster, give this cluster a short descriptive name (3-5 words max).

Repos in this cluster:
{repos}

Respond with JSON only: {{"name": "cluster name here", "confidence": 0.0-1.0}}"""


def label_cluster(repos: List[Dict]) -> Dict:
    """Label a cluster using Claude API.

    Args:
        repos: List of dicts with 'name', 'readme', 'files' keys

    Returns:
        Dict with 'name' and 'confidence' keys
    """
    repos_text = ""
    for repo in repos[:10]:
        repos_text += f"\n## {repo['name']}\n"
        repos_text += f"Files: {', '.join(repo['files'][:20])}\n"
        readme = repo.get('readme', '')[:1000]
        if readme:
            repos_text += f"README:\n{readme}\n"

    prompt = PROMPT_TEMPLATE.format(repos=repos_text)

    client = get_client()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}]
    )

    try:
        result = json.loads(response.content[0].text)
        return {
            'name': result.get('name', 'Unknown'),
            'confidence': result.get('confidence', 0.0),
        }
    except (json.JSONDecodeError, IndexError, KeyError):
        return {'name': 'Unknown', 'confidence': 0.0}


def label_all_clusters(
    repos: List[Dict],
    labels: List[int],
) -> Dict[int, Dict]:
    """Label all clusters.

    Args:
        repos: List of repo dicts
        labels: Cluster label for each repo (-1 = noise)

    Returns:
        Dict mapping cluster_id to {'name', 'confidence'}
    """
    cluster_labels = {}

    clusters = {}
    for repo, label in zip(repos, labels):
        if label == -1:
            continue
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(repo)

    for cluster_id, cluster_repos in clusters.items():
        cluster_labels[cluster_id] = label_cluster(cluster_repos)

    cluster_labels[-1] = {'name': 'unclustered', 'confidence': 1.0}

    return cluster_labels
