"""Tests for S3 reader."""
import pytest
from unittest.mock import MagicMock, patch


def test_list_repos_returns_repo_names():
    """S3 reader should return list of repo names from bucket."""
    from pipeline.s3_reader import list_repos

    mock_paginator = MagicMock()
    mock_paginator.paginate.return_value = [{
        'CommonPrefixes': [
            {'Prefix': 'repos/repo-one/'},
            {'Prefix': 'repos/repo-two/'},
        ]
    }]

    mock_s3 = MagicMock()
    mock_s3.get_paginator.return_value = mock_paginator

    with patch('pipeline.s3_reader.get_s3_client', return_value=mock_s3):
        repos = list_repos()

    assert repos == ['repo-one', 'repo-two']


def test_get_repo_files_returns_file_contents():
    """S3 reader should stream file contents for a repo."""
    from pipeline.s3_reader import get_repo_files

    mock_paginator = MagicMock()
    mock_paginator.paginate.return_value = [{
        'Contents': [
            {'Key': 'repos/my-repo/README.md'},
            {'Key': 'repos/my-repo/main.py'},
        ]
    }]

    mock_s3 = MagicMock()
    mock_s3.get_paginator.return_value = mock_paginator
    mock_s3.get_object.side_effect = [
        {'Body': MagicMock(read=lambda: b'# My Repo')},
        {'Body': MagicMock(read=lambda: b'print("hello")')},
    ]

    with patch('pipeline.s3_reader.get_s3_client', return_value=mock_s3):
        files = list(get_repo_files('my-repo'))

    assert len(files) == 2
    assert files[0] == ('README.md', '# My Repo')
    assert files[1] == ('main.py', 'print("hello")')


def test_skips_binary_files():
    """S3 reader should skip binary files."""
    from pipeline.s3_reader import _is_binary_file

    assert _is_binary_file('image.png') is True
    assert _is_binary_file('data.pkl') is True
    assert _is_binary_file('main.py') is False
    assert _is_binary_file('README.md') is False
