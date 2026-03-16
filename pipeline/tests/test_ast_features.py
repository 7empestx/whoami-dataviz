"""Tests for AST feature extractor."""
import pytest
import numpy as np


def test_extract_features_python():
    """AST extractor should return feature dict for Python code."""
    from pipeline.ast_features import extract_features

    code = '''
def hello():
    if True:
        for i in range(10):
            print(i)

async def fetch():
    await something()
'''
    features = extract_features('main.py', code)

    assert isinstance(features, dict)
    assert 'function_count' in features
    assert 'max_nesting_depth' in features
    assert 'async_usage_ratio' in features
    assert features['function_count'] == 2
    assert features['max_nesting_depth'] >= 3


def test_extract_features_javascript():
    """AST extractor should handle JavaScript."""
    from pipeline.ast_features import extract_features

    code = '''
function hello() {
    console.log("hi");
}

const arrow = () => {
    return 42;
};
'''
    features = extract_features('app.js', code)

    assert features['function_count'] >= 2
    assert features['language'] == 'javascript'


def test_extract_repo_features_returns_vector():
    """Should return 8-dimensional feature vector for a repo."""
    from pipeline.ast_features import extract_repo_features

    files = [
        ('main.py', 'def hello():\n    pass'),
        ('test_main.py', 'def test_hello():\n    assert True'),
    ]

    vector = extract_repo_features(files)

    assert isinstance(vector, np.ndarray)
    assert len(vector) == 8


def test_extract_repo_features_detects_tests():
    """Should detect test files and compute test ratio."""
    from pipeline.ast_features import extract_repo_features

    files = [
        ('src/main.py', 'def main(): pass'),
        ('tests/test_main.py', 'def test_main(): pass'),
        ('tests/test_utils.py', 'def test_utils(): pass'),
    ]

    vector = extract_repo_features(files)
    test_ratio = vector[4]

    assert test_ratio > 0.5


def test_unsupported_extension_returns_empty():
    """Should return empty dict for unsupported file types."""
    from pipeline.ast_features import extract_features

    features = extract_features('data.json', '{"key": "value"}')
    assert features == {}
