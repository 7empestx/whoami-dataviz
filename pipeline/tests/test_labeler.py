"""Tests for Claude cluster labeler."""
import pytest
from unittest.mock import MagicMock, patch


def test_label_cluster():
    """Labeler should call Claude API and return cluster name."""
    from pipeline.labeler import label_cluster

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text='{"name": "ML experimentation tools", "confidence": 0.92}')]

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    repos = [
        {'name': 'pytorch-exp', 'readme': '# PyTorch Experiments', 'files': ['train.py', 'model.py']},
        {'name': 'ml-utils', 'readme': '# ML Utilities', 'files': ['utils.py', 'data.py']},
    ]

    with patch('pipeline.labeler.get_client', return_value=mock_client):
        result = label_cluster(repos)

    assert result['name'] == 'ML experimentation tools'
    assert result['confidence'] == 0.92


def test_label_all_clusters():
    """Should label all clusters including noise."""
    from pipeline.labeler import label_all_clusters

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text='{"name": "Test Cluster", "confidence": 0.8}')]

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    repos = [
        {'name': 'repo1', 'readme': 'readme1', 'files': ['a.py']},
        {'name': 'repo2', 'readme': 'readme2', 'files': ['b.py']},
        {'name': 'repo3', 'readme': 'readme3', 'files': ['c.py']},
    ]
    labels = [0, 0, -1]

    with patch('pipeline.labeler.get_client', return_value=mock_client):
        result = label_all_clusters(repos, labels)

    assert 0 in result
    assert -1 in result
    assert result[-1]['name'] == 'unclustered'
