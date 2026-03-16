# whoami-dataviz Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a 3D interactive visualization of 237 repos mapped in vector space using dual embeddings (semantic + structural), with HDBSCAN clustering and Claude-generated cluster names.

**Architecture:** Pull repos from S3, embed with StarEncoder (semantic) + tree-sitter (structural), cluster with HDBSCAN, name clusters via Claude API, reduce to 3D with UMAP/t-SNE, serve via FastAPI, render with Deck.gl.

**Tech Stack:** Python (boto3, transformers, tree-sitter, umap-learn, hdbscan, anthropic), FastAPI, Vanilla JS + Deck.gl, Docker, ECS Fargate

---

## Task 1: Project Setup

**Files:**
- Create: `pipeline/__init__.py`
- Create: `pipeline/requirements.txt`
- Create: `api/requirements.txt`
- Create: `README.md`
- Create: `.gitignore`

**Step 1: Create .gitignore**

```gitignore
__pycache__/
*.pyc
.env
venv/
.venv/
*.egg-info/
dist/
build/
.DS_Store
coords.json
embeddings/
*.pkl
```

**Step 2: Create pipeline requirements.txt**

```txt
boto3>=1.34.0
transformers>=4.36.0
torch>=2.1.0
tree-sitter>=0.21.0
tree-sitter-python>=0.21.0
tree-sitter-javascript>=0.21.0
tree-sitter-go>=0.21.0
tree-sitter-java>=0.21.0
tree-sitter-typescript>=0.21.0
tree-sitter-rust>=0.21.0
umap-learn>=0.5.5
hdbscan>=0.8.33
scikit-learn>=1.3.0
anthropic>=0.18.0
numpy>=1.26.0
tqdm>=4.66.0
```

**Step 3: Create api requirements.txt**

```txt
fastapi>=0.109.0
uvicorn>=0.27.0
```

**Step 4: Create pipeline __init__.py**

```python
"""whoami-dataviz pipeline - dual embedding for repo visualization."""
```

**Step 5: Create README.md**

Copy the existing README spec into the repo.

**Step 6: Commit**

```bash
git add -A
git commit -m "chore: initial project setup with requirements"
```

---

## Task 2: S3 Reader

**Files:**
- Create: `pipeline/s3_reader.py`
- Create: `pipeline/tests/__init__.py`
- Create: `pipeline/tests/test_s3_reader.py`

**Step 1: Write the failing test**

```python
# pipeline/tests/test_s3_reader.py
import pytest
from unittest.mock import MagicMock, patch

def test_list_repos_returns_repo_names():
    """S3 reader should return list of repo names from bucket."""
    from pipeline.s3_reader import list_repos

    mock_s3 = MagicMock()
    mock_s3.list_objects_v2.return_value = {
        'CommonPrefixes': [
            {'Prefix': 'repos/repo-one/'},
            {'Prefix': 'repos/repo-two/'},
        ]
    }

    with patch('pipeline.s3_reader.get_s3_client', return_value=mock_s3):
        repos = list_repos()

    assert repos == ['repo-one', 'repo-two']


def test_get_repo_files_returns_file_contents():
    """S3 reader should stream file contents for a repo."""
    from pipeline.s3_reader import get_repo_files

    mock_s3 = MagicMock()
    mock_s3.list_objects_v2.return_value = {
        'Contents': [
            {'Key': 'repos/my-repo/README.md'},
            {'Key': 'repos/my-repo/main.py'},
        ]
    }
    mock_s3.get_object.side_effect = [
        {'Body': MagicMock(read=lambda: b'# My Repo')},
        {'Body': MagicMock(read=lambda: b'print("hello")')},
    ]

    with patch('pipeline.s3_reader.get_s3_client', return_value=mock_s3):
        files = list(get_repo_files('my-repo'))

    assert len(files) == 2
    assert files[0] == ('README.md', '# My Repo')
    assert files[1] == ('main.py', 'print("hello")')
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/gstarkman/Mine/whoami-dataviz && python -m pytest pipeline/tests/test_s3_reader.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# pipeline/s3_reader.py
"""Stream repo files from S3 without local download."""
import boto3
from typing import Generator, List, Tuple

BUCKET = "nexio-code-kb-dev-720154970215"
PREFIX = "repos/"


def get_s3_client():
    """Return boto3 S3 client."""
    return boto3.client('s3')


def list_repos() -> List[str]:
    """List all repo names in the bucket."""
    s3 = get_s3_client()
    response = s3.list_objects_v2(
        Bucket=BUCKET,
        Prefix=PREFIX,
        Delimiter='/'
    )
    repos = []
    for prefix in response.get('CommonPrefixes', []):
        # Extract repo name from 'repos/repo-name/'
        repo_name = prefix['Prefix'].replace(PREFIX, '').rstrip('/')
        repos.append(repo_name)
    return repos


def get_repo_files(repo_name: str) -> Generator[Tuple[str, str], None, None]:
    """Stream all files from a repo as (filename, content) tuples."""
    s3 = get_s3_client()
    repo_prefix = f"{PREFIX}{repo_name}/"

    response = s3.list_objects_v2(Bucket=BUCKET, Prefix=repo_prefix)

    for obj in response.get('Contents', []):
        key = obj['Key']
        filename = key.replace(repo_prefix, '')

        # Skip directories and binary files
        if filename.endswith('/') or _is_binary_file(filename):
            continue

        try:
            file_obj = s3.get_object(Bucket=BUCKET, Key=key)
            content = file_obj['Body'].read().decode('utf-8', errors='ignore')
            yield (filename, content)
        except Exception:
            continue


def _is_binary_file(filename: str) -> bool:
    """Check if file is likely binary based on extension."""
    binary_extensions = {
        '.png', '.jpg', '.jpeg', '.gif', '.ico', '.pdf',
        '.zip', '.tar', '.gz', '.whl', '.pyc', '.so',
        '.exe', '.dll', '.bin', '.dat', '.pkl', '.npy'
    }
    return any(filename.lower().endswith(ext) for ext in binary_extensions)
```

**Step 4: Create tests __init__.py**

```python
# pipeline/tests/__init__.py
"""Tests for whoami-dataviz pipeline."""
```

**Step 5: Run test to verify it passes**

Run: `cd /Users/gstarkman/Mine/whoami-dataviz && python -m pytest pipeline/tests/test_s3_reader.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add pipeline/s3_reader.py pipeline/tests/
git commit -m "feat: add S3 reader for streaming repo files"
```

---

## Task 3: StarEncoder Semantic Embedder

**Files:**
- Create: `pipeline/embedder.py`
- Create: `pipeline/tests/test_embedder.py`

**Step 1: Write the failing test**

```python
# pipeline/tests/test_embedder.py
import pytest
import numpy as np

def test_embed_code_returns_vector():
    """Embedder should return a numpy vector for code input."""
    from pipeline.embedder import embed_code

    code = "def hello():\n    print('world')"
    vector = embed_code(code)

    assert isinstance(vector, np.ndarray)
    assert len(vector.shape) == 1
    assert vector.shape[0] > 0


def test_embed_repo_averages_chunks():
    """Embedder should chunk large content and average embeddings."""
    from pipeline.embedder import embed_repo

    files = [
        ('README.md', '# Test Repo\nThis is a test.'),
        ('main.py', 'def main():\n    pass'),
    ]

    vector = embed_repo(files)

    assert isinstance(vector, np.ndarray)
    assert len(vector.shape) == 1
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/gstarkman/Mine/whoami-dataviz && python -m pytest pipeline/tests/test_embedder.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# pipeline/embedder.py
"""StarEncoder-based semantic embeddings for code repositories."""
import numpy as np
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModel
import torch

MODEL_NAME = "bigcode/starencoder"
MAX_TOKENS = 1024
CHUNK_OVERLAP = 100

_tokenizer = None
_model = None


def _get_model():
    """Lazy load the StarEncoder model."""
    global _tokenizer, _model
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModel.from_pretrained(MODEL_NAME)
        _model.eval()
    return _tokenizer, _model


def embed_code(code: str) -> np.ndarray:
    """Embed a code snippet, returning the mean pooled vector."""
    tokenizer, model = _get_model()

    # Truncate to max tokens
    inputs = tokenizer(
        code,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_TOKENS,
        padding=True
    )

    with torch.no_grad():
        outputs = model(**inputs)
        # Mean pooling over sequence dimension
        embeddings = outputs.last_hidden_state.mean(dim=1)

    return embeddings.squeeze().numpy()


def chunk_text(text: str, tokenizer, max_tokens: int = MAX_TOKENS) -> List[str]:
    """Split text into overlapping chunks that fit within token limit."""
    tokens = tokenizer.encode(text)
    chunks = []

    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
        start = end - CHUNK_OVERLAP

    return chunks if chunks else [text]


def embed_repo(files: List[Tuple[str, str]]) -> np.ndarray:
    """Embed all files in a repo, chunking and averaging."""
    tokenizer, _ = _get_model()

    all_embeddings = []

    for filename, content in files:
        # Prepend filename as context
        full_content = f"# File: {filename}\n{content}"

        chunks = chunk_text(full_content, tokenizer)
        for chunk in chunks:
            if chunk.strip():
                embedding = embed_code(chunk)
                all_embeddings.append(embedding)

    if not all_embeddings:
        # Return zero vector if no content
        return np.zeros(768)  # StarEncoder hidden size

    # Average all chunk embeddings
    return np.mean(all_embeddings, axis=0)
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/gstarkman/Mine/whoami-dataviz && python -m pytest pipeline/tests/test_embedder.py -v`
Expected: PASS (may take time to download model on first run)

**Step 5: Commit**

```bash
git add pipeline/embedder.py pipeline/tests/test_embedder.py
git commit -m "feat: add StarEncoder semantic embedder with chunking"
```

---

## Task 4: Tree-sitter AST Feature Extractor

**Files:**
- Create: `pipeline/ast_features.py`
- Create: `pipeline/tests/test_ast_features.py`

**Step 1: Write the failing test**

```python
# pipeline/tests/test_ast_features.py
import pytest
import numpy as np

def test_extract_features_python():
    """AST extractor should return feature vector for Python code."""
    from pipeline.ast_features import extract_features

    code = '''
def hello():
    if True:
        for i in range(10):
            print(i)

async def fetch():
    await something()

class MyClass:
    pass
'''
    features = extract_features('main.py', code)

    assert isinstance(features, dict)
    assert 'function_count' in features
    assert 'max_nesting_depth' in features
    assert 'async_usage_ratio' in features
    assert features['function_count'] == 2
    assert features['max_nesting_depth'] >= 3


def test_extract_repo_features_returns_vector():
    """Should return normalized feature vector for a repo."""
    from pipeline.ast_features import extract_repo_features

    files = [
        ('main.py', 'def hello():\n    pass'),
        ('test_main.py', 'def test_hello():\n    assert True'),
    ]

    vector = extract_repo_features(files)

    assert isinstance(vector, np.ndarray)
    assert len(vector) == 8  # 8 structural features
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/gstarkman/Mine/whoami-dataviz && python -m pytest pipeline/tests/test_ast_features.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# pipeline/ast_features.py
"""Tree-sitter based structural feature extraction."""
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path

import tree_sitter_python
import tree_sitter_javascript
import tree_sitter_go
import tree_sitter_java
import tree_sitter_typescript
import tree_sitter_rust
from tree_sitter import Language, Parser

# Language mapping
LANGUAGES = {
    '.py': ('python', tree_sitter_python.language()),
    '.js': ('javascript', tree_sitter_javascript.language()),
    '.jsx': ('javascript', tree_sitter_javascript.language()),
    '.ts': ('typescript', tree_sitter_typescript.language_typescript()),
    '.tsx': ('typescript', tree_sitter_typescript.language_tsx()),
    '.go': ('go', tree_sitter_go.language()),
    '.java': ('java', tree_sitter_java.language()),
    '.rs': ('rust', tree_sitter_rust.language()),
}

# Function node types by language
FUNCTION_TYPES = {
    'python': ['function_definition', 'async_function_definition'],
    'javascript': ['function_declaration', 'arrow_function', 'function_expression'],
    'typescript': ['function_declaration', 'arrow_function', 'function_expression'],
    'go': ['function_declaration', 'method_declaration'],
    'java': ['method_declaration'],
    'rust': ['function_item'],
}

ASYNC_TYPES = {
    'python': ['async_function_definition', 'await_expression'],
    'javascript': ['async', 'await_expression'],
    'typescript': ['async', 'await_expression'],
    'go': [],  # Go uses goroutines differently
    'java': [],
    'rust': ['async', 'await_expression'],
}

CLASS_TYPES = {
    'python': ['class_definition'],
    'javascript': ['class_declaration'],
    'typescript': ['class_declaration'],
    'go': ['type_declaration'],
    'java': ['class_declaration'],
    'rust': ['struct_item', 'impl_item'],
}

IMPORT_TYPES = {
    'python': ['import_statement', 'import_from_statement'],
    'javascript': ['import_statement'],
    'typescript': ['import_statement'],
    'go': ['import_declaration'],
    'java': ['import_declaration'],
    'rust': ['use_declaration'],
}


def get_parser(ext: str) -> Tuple[Parser, str]:
    """Get parser for file extension."""
    if ext not in LANGUAGES:
        return None, None

    lang_name, lang = LANGUAGES[ext]
    parser = Parser(Language(lang))
    return parser, lang_name


def count_nodes(tree, node_types: List[str]) -> int:
    """Count nodes of given types in tree."""
    count = 0

    def visit(node):
        nonlocal count
        if node.type in node_types:
            count += 1
        for child in node.children:
            visit(child)

    visit(tree.root_node)
    return count


def max_depth(tree) -> int:
    """Find maximum nesting depth in tree."""
    def depth(node, current=0):
        if not node.children:
            return current
        return max(depth(child, current + 1) for child in node.children)

    return depth(tree.root_node)


def cyclomatic_complexity(tree, lang: str) -> int:
    """Estimate cyclomatic complexity by counting decision points."""
    decision_types = {
        'if_statement', 'elif_clause', 'else_clause',
        'for_statement', 'while_statement',
        'try_statement', 'except_clause',
        'case_clause', 'switch_statement',
        'conditional_expression', 'ternary_expression',
        'and', 'or', '&&', '||',
    }
    return count_nodes(tree, list(decision_types)) + 1


def extract_features(filename: str, content: str) -> Dict[str, float]:
    """Extract structural features from a single file."""
    ext = Path(filename).suffix.lower()
    parser, lang = get_parser(ext)

    if parser is None:
        return {}

    try:
        tree = parser.parse(bytes(content, 'utf8'))
    except Exception:
        return {}

    func_types = FUNCTION_TYPES.get(lang, [])
    async_types = ASYNC_TYPES.get(lang, [])
    class_types = CLASS_TYPES.get(lang, [])
    import_types = IMPORT_TYPES.get(lang, [])

    func_count = count_nodes(tree, func_types)
    async_count = count_nodes(tree, async_types)

    return {
        'function_count': func_count,
        'max_nesting_depth': max_depth(tree),
        'async_usage_ratio': async_count / max(func_count, 1),
        'import_count': count_nodes(tree, import_types),
        'class_count': count_nodes(tree, class_types),
        'cyclomatic_complexity': cyclomatic_complexity(tree, lang),
        'language': lang,
    }


def extract_repo_features(files: List[Tuple[str, str]]) -> np.ndarray:
    """Extract aggregated structural features for a repo.

    Returns 8-dimensional vector:
    - function_count (total)
    - max_nesting_depth (max across files)
    - async_usage_ratio (average)
    - import_count (total)
    - test_file_ratio
    - cyclomatic_complexity (average)
    - language_mix (entropy-like measure)
    - entry_point_pattern (1 if has main/index/cli, 0 otherwise)
    """
    all_features = []
    lang_counts = {}
    test_files = 0
    has_entry_point = False

    entry_patterns = ['main.py', 'index.js', 'index.ts', 'main.go', 'Main.java', 'main.rs', 'cli.py', 'app.py']

    for filename, content in files:
        features = extract_features(filename, content)
        if features:
            all_features.append(features)
            lang = features.get('language', 'unknown')
            lang_counts[lang] = lang_counts.get(lang, 0) + 1

        # Check for test files
        if 'test' in filename.lower() or filename.startswith('test_'):
            test_files += 1

        # Check for entry points
        if any(filename.endswith(ep) or filename == ep for ep in entry_patterns):
            has_entry_point = True

    if not all_features:
        return np.zeros(8)

    # Aggregate features
    total_files = len(files)
    total_func = sum(f['function_count'] for f in all_features)
    max_depth = max(f['max_nesting_depth'] for f in all_features)
    avg_async = np.mean([f['async_usage_ratio'] for f in all_features])
    total_imports = sum(f['import_count'] for f in all_features)
    test_ratio = test_files / max(total_files, 1)
    avg_complexity = np.mean([f['cyclomatic_complexity'] for f in all_features])

    # Language mix: normalized entropy
    total_lang_files = sum(lang_counts.values())
    if total_lang_files > 0:
        probs = np.array(list(lang_counts.values())) / total_lang_files
        lang_entropy = -np.sum(probs * np.log(probs + 1e-10))
        lang_mix = lang_entropy / np.log(max(len(lang_counts), 2))  # Normalize
    else:
        lang_mix = 0

    return np.array([
        np.log1p(total_func),  # Log scale for counts
        max_depth,
        avg_async,
        np.log1p(total_imports),
        test_ratio,
        np.log1p(avg_complexity),
        lang_mix,
        float(has_entry_point),
    ])
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/gstarkman/Mine/whoami-dataviz && python -m pytest pipeline/tests/test_ast_features.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add pipeline/ast_features.py pipeline/tests/test_ast_features.py
git commit -m "feat: add tree-sitter AST feature extractor"
```

---

## Task 5: HDBSCAN Clusterer

**Files:**
- Create: `pipeline/clusterer.py`
- Create: `pipeline/tests/test_clusterer.py`

**Step 1: Write the failing test**

```python
# pipeline/tests/test_clusterer.py
import pytest
import numpy as np

def test_cluster_embeddings():
    """Clusterer should assign cluster IDs to embeddings."""
    from pipeline.clusterer import cluster_embeddings

    # Create synthetic data with 3 clear clusters
    np.random.seed(42)
    cluster1 = np.random.randn(20, 10) + np.array([5, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    cluster2 = np.random.randn(20, 10) + np.array([0, 5, 0, 0, 0, 0, 0, 0, 0, 0])
    cluster3 = np.random.randn(20, 10) + np.array([0, 0, 5, 0, 0, 0, 0, 0, 0, 0])
    embeddings = np.vstack([cluster1, cluster2, cluster3])

    labels = cluster_embeddings(embeddings)

    assert len(labels) == 60
    assert isinstance(labels, np.ndarray)
    # Should have found at least 2 clusters (HDBSCAN may merge some)
    unique_labels = set(labels) - {-1}  # Exclude noise
    assert len(unique_labels) >= 2
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/gstarkman/Mine/whoami-dataviz && python -m pytest pipeline/tests/test_clusterer.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# pipeline/clusterer.py
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
    n_noise = np.sum(labels == -1)

    cluster_sizes = {}
    for label in unique_labels:
        if label != -1:
            cluster_sizes[int(label)] = int(np.sum(labels == label))

    return {
        'n_clusters': n_clusters,
        'n_noise': int(n_noise),
        'cluster_sizes': cluster_sizes,
    }
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/gstarkman/Mine/whoami-dataviz && python -m pytest pipeline/tests/test_clusterer.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add pipeline/clusterer.py pipeline/tests/test_clusterer.py
git commit -m "feat: add HDBSCAN clusterer"
```

---

## Task 6: Dimension Reducer (UMAP + t-SNE)

**Files:**
- Create: `pipeline/reducer.py`
- Create: `pipeline/tests/test_reducer.py`

**Step 1: Write the failing test**

```python
# pipeline/tests/test_reducer.py
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
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/gstarkman/Mine/whoami-dataviz && python -m pytest pipeline/tests/test_reducer.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# pipeline/reducer.py
"""Dimension reduction using UMAP and t-SNE."""
import numpy as np
from typing import Tuple
import umap
from sklearn.manifold import TSNE


def reduce_umap(embeddings: np.ndarray, n_components: int = 3) -> np.ndarray:
    """Reduce embeddings to n_components dimensions using UMAP."""
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=15,
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
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/gstarkman/Mine/whoami-dataviz && python -m pytest pipeline/tests/test_reducer.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add pipeline/reducer.py pipeline/tests/test_reducer.py
git commit -m "feat: add UMAP and t-SNE dimension reducers"
```

---

## Task 7: Claude API Cluster Labeler

**Files:**
- Create: `pipeline/labeler.py`
- Create: `pipeline/tests/test_labeler.py`

**Step 1: Write the failing test**

```python
# pipeline/tests/test_labeler.py
import pytest
from unittest.mock import MagicMock, patch
import json

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
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/gstarkman/Mine/whoami-dataviz && python -m pytest pipeline/tests/test_labeler.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# pipeline/labeler.py
"""Claude API cluster labeling."""
import json
import os
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
    # Format repos for prompt
    repos_text = ""
    for repo in repos[:10]:  # Limit to 10 repos
        repos_text += f"\n## {repo['name']}\n"
        repos_text += f"Files: {', '.join(repo['files'][:20])}\n"  # Limit files
        readme = repo.get('readme', '')[:1000]  # Limit README length
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

    # Group repos by cluster
    clusters = {}
    for repo, label in zip(repos, labels):
        if label == -1:
            continue
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(repo)

    # Label each cluster
    for cluster_id, cluster_repos in clusters.items():
        cluster_labels[cluster_id] = label_cluster(cluster_repos)

    # Add noise label
    cluster_labels[-1] = {'name': 'unclustered', 'confidence': 1.0}

    return cluster_labels
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/gstarkman/Mine/whoami-dataviz && python -m pytest pipeline/tests/test_labeler.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add pipeline/labeler.py pipeline/tests/test_labeler.py
git commit -m "feat: add Claude API cluster labeler"
```

---

## Task 8: Pipeline Orchestrator

**Files:**
- Create: `pipeline/main.py`

**Step 1: Write the orchestrator**

```python
# pipeline/main.py
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

    # Get README
    readme = ""
    for filename, content in files:
        if filename.lower() == 'readme.md' or filename.lower() == 'readme':
            readme = content
            break

    # Get file list
    file_list = [f[0] for f in files]

    # Compute embeddings
    semantic_emb = embed_repo(files)
    structural_emb = extract_repo_features(files)

    return repo_name, semantic_emb, structural_emb, file_list, readme


def run_pipeline(output_path: str = "coords.json") -> None:
    """Run the full pipeline."""
    print("Listing repos from S3...")
    repo_names = list_repos()
    print(f"Found {len(repo_names)} repos")

    # Process all repos
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

    # Concatenate embeddings
    print("Concatenating embeddings...")
    semantic_arr = np.array(semantic_embeddings)
    structural_arr = np.array(structural_embeddings)
    combined = np.concatenate([semantic_arr, structural_arr], axis=1)

    # Cluster on high-dimensional embeddings
    print("Clustering...")
    labels = cluster_embeddings(combined)
    stats = get_cluster_stats(labels)
    print(f"Found {stats['n_clusters']} clusters, {stats['n_noise']} noise points")

    # Label clusters with Claude
    print("Labeling clusters...")
    cluster_labels = label_all_clusters(repos_data, labels.tolist())

    # Reduce to 3D
    print("Reducing dimensions...")
    umap_coords, tsne_coords = reduce_both(combined)

    # Build output JSON
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

    # Write output
    print(f"Writing to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print("Done!")


if __name__ == "__main__":
    run_pipeline()
```

**Step 2: Commit**

```bash
git add pipeline/main.py
git commit -m "feat: add pipeline orchestrator"
```

---

## Task 9: FastAPI Server

**Files:**
- Create: `api/main.py`

**Step 1: Write the API server**

```python
# api/main.py
"""FastAPI server to serve coords.json."""
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="whoami-dataviz API")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

COORDS_PATH = Path("coords.json")


@app.get("/coords")
async def get_coords():
    """Return the coordinates JSON."""
    if not COORDS_PATH.exists():
        raise HTTPException(status_code=404, detail="coords.json not found")

    with open(COORDS_PATH) as f:
        return json.load(f)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}
```

**Step 2: Commit**

```bash
git add api/main.py
git commit -m "feat: add FastAPI server"
```

---

## Task 10: Deck.gl Frontend

**Files:**
- Create: `frontend/index.html`
- Create: `frontend/app.js`
- Create: `frontend/style.css`

**Step 1: Write index.html**

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>whoami.dataviz</title>
    <link rel="stylesheet" href="style.css">
    <script src="https://unpkg.com/deck.gl@8.9.35/dist.min.js"></script>
</head>
<body>
    <div id="container">
        <div id="controls">
            <h1>whoami.dataviz</h1>
            <div class="control-group">
                <label>Projection:</label>
                <button id="btn-umap" class="active">UMAP</button>
                <button id="btn-tsne">t-SNE</button>
            </div>
            <div class="control-group">
                <label>Search:</label>
                <input type="text" id="search" placeholder="repo name...">
            </div>
            <div id="stats"></div>
        </div>
        <div id="map"></div>
        <div id="tooltip"></div>
    </div>
    <script src="app.js"></script>
</body>
</html>
```

**Step 2: Write style.css**

```css
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #0a0a0a;
    color: #fff;
    overflow: hidden;
}

#container {
    display: flex;
    height: 100vh;
}

#controls {
    width: 280px;
    padding: 20px;
    background: #111;
    border-right: 1px solid #222;
    z-index: 10;
}

#controls h1 {
    font-size: 18px;
    margin-bottom: 20px;
    color: #fff;
}

.control-group {
    margin-bottom: 16px;
}

.control-group label {
    display: block;
    font-size: 12px;
    color: #888;
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

button {
    padding: 8px 16px;
    border: 1px solid #333;
    background: #1a1a1a;
    color: #888;
    cursor: pointer;
    font-size: 13px;
    margin-right: 8px;
    border-radius: 4px;
    transition: all 0.15s;
}

button:hover {
    background: #222;
    color: #fff;
}

button.active {
    background: #2563eb;
    border-color: #2563eb;
    color: #fff;
}

input[type="text"] {
    width: 100%;
    padding: 10px;
    border: 1px solid #333;
    background: #1a1a1a;
    color: #fff;
    font-size: 13px;
    border-radius: 4px;
}

input[type="text"]:focus {
    outline: none;
    border-color: #2563eb;
}

#stats {
    margin-top: 20px;
    font-size: 12px;
    color: #666;
    line-height: 1.6;
}

#map {
    flex: 1;
    position: relative;
}

#tooltip {
    position: absolute;
    z-index: 100;
    pointer-events: none;
    background: rgba(0, 0, 0, 0.9);
    border: 1px solid #333;
    border-radius: 4px;
    padding: 10px 14px;
    font-size: 13px;
    max-width: 300px;
    display: none;
}

#tooltip .name {
    font-weight: 600;
    margin-bottom: 4px;
}

#tooltip .cluster {
    color: #888;
    font-size: 12px;
}
```

**Step 3: Write app.js**

```javascript
// whoami.dataviz - Deck.gl visualization

const API_URL = '/coords';
const COLORS = {
    clusters: [
        [59, 130, 246],   // blue
        [16, 185, 129],   // green
        [245, 158, 11],   // amber
        [239, 68, 68],    // red
        [139, 92, 246],   // purple
        [236, 72, 153],   // pink
        [20, 184, 166],   // teal
        [249, 115, 22],   // orange
        [132, 204, 22],   // lime
        [6, 182, 212],    // cyan
    ],
    noise: [128, 128, 128],  // gray for unclustered
};

let data = null;
let currentProjection = 'umap';
let searchTerm = '';
let deckgl = null;

async function loadData() {
    const response = await fetch(API_URL);
    data = await response.json();
    updateStats();
    render();
}

function getColor(repo) {
    if (searchTerm && !repo.name.toLowerCase().includes(searchTerm.toLowerCase())) {
        return [50, 50, 50, 100];  // Dim non-matching
    }

    if (repo.cluster_id === -1) {
        return [...COLORS.noise, 102];  // 40% opacity for noise
    }

    const colorIndex = repo.cluster_id % COLORS.clusters.length;
    return [...COLORS.clusters[colorIndex], 255];
}

function getPosition(repo) {
    const coords = currentProjection === 'umap' ? repo.umap : repo.tsne;
    return coords;
}

function getRadius(repo) {
    if (searchTerm && repo.name.toLowerCase().includes(searchTerm.toLowerCase())) {
        return 12;  // Highlight matches
    }
    return 6;
}

function render() {
    if (!data) return;

    const layer = new deck.ScatterplotLayer({
        id: 'repos',
        data: data.repos,
        getPosition,
        getRadius,
        getFillColor: getColor,
        radiusMinPixels: 3,
        radiusMaxPixels: 20,
        pickable: true,
        onHover: handleHover,
        transitions: {
            getPosition: 500,
        },
    });

    if (!deckgl) {
        deckgl = new deck.DeckGL({
            container: 'map',
            initialViewState: {
                target: [0, 0, 0],
                zoom: 3,
                rotationX: 45,
                rotationOrbit: 30,
            },
            controller: {
                type: deck.OrbitController,
            },
            layers: [layer],
        });
    } else {
        deckgl.setProps({ layers: [layer] });
    }
}

function handleHover({ object, x, y }) {
    const tooltip = document.getElementById('tooltip');

    if (object) {
        tooltip.style.display = 'block';
        tooltip.style.left = x + 10 + 'px';
        tooltip.style.top = y + 10 + 'px';
        tooltip.innerHTML = `
            <div class="name">${object.name}</div>
            <div class="cluster">${object.cluster_name}</div>
        `;
    } else {
        tooltip.style.display = 'none';
    }
}

function updateStats() {
    const stats = document.getElementById('stats');
    const clusters = data.clusters.length;
    const repos = data.repos.length;
    const noise = data.repos.filter(r => r.cluster_id === -1).length;

    stats.innerHTML = `
        ${repos} repos<br>
        ${clusters} clusters<br>
        ${noise} unclustered
    `;
}

function setProjection(proj) {
    currentProjection = proj;
    document.getElementById('btn-umap').classList.toggle('active', proj === 'umap');
    document.getElementById('btn-tsne').classList.toggle('active', proj === 'tsne');
    render();
}

// Event listeners
document.getElementById('btn-umap').addEventListener('click', () => setProjection('umap'));
document.getElementById('btn-tsne').addEventListener('click', () => setProjection('tsne'));
document.getElementById('search').addEventListener('input', (e) => {
    searchTerm = e.target.value;
    render();
});

// Initialize
loadData();
```

**Step 4: Commit**

```bash
git add frontend/
git commit -m "feat: add Deck.gl frontend"
```

---

## Task 11: Dockerfiles

**Files:**
- Create: `docker/Dockerfile.pipeline`
- Create: `docker/Dockerfile.api`
- Create: `docker/Dockerfile.frontend`

**Step 1: Write Dockerfile.pipeline**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for tree-sitter
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pipeline/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pipeline/ ./pipeline/

CMD ["python", "-m", "pipeline.main"]
```

**Step 2: Write Dockerfile.api**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/
COPY coords.json .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Step 3: Write Dockerfile.frontend**

```dockerfile
FROM nginx:alpine

COPY frontend/ /usr/share/nginx/html/

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
```

**Step 4: Commit**

```bash
git add docker/
git commit -m "feat: add Dockerfiles"
```

---

## Task 12: GitHub Setup and Push

**Step 1: Create GitHub repo**

```bash
gh repo create whoami-dataviz --public --description "3D visualization of 237 repos mapped in vector space" --source=. --remote=origin
```

**Step 2: Push to GitHub**

```bash
git push -u origin main
```

---

## Task 13: Hugging Face Space Setup

**Files:**
- Create: `app.py` (Gradio wrapper for HF Spaces)
- Create: `requirements.txt` (root level for HF)

**Step 1: Create Gradio app for HF Spaces**

```python
# app.py
"""Hugging Face Spaces app - serves the visualization."""
import gradio as gr
from pathlib import Path

# Read the frontend files
css_content = Path("frontend/style.css").read_text()
js_content = Path("frontend/app.js").read_text()

# Filter out the API_URL line from app.js and use hardcoded path
js_lines = [line for line in js_content.split('\n') if not line.startswith('const API_URL')]
js_filtered = '\n'.join(js_lines)

# Inline everything for HF Spaces
full_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>whoami.dataviz</title>
    <style>{css_content}</style>
    <script src="https://unpkg.com/deck.gl@8.9.35/dist.min.js"></script>
</head>
<body>
    <div id="container">
        <div id="controls">
            <h1>whoami.dataviz</h1>
            <div class="control-group">
                <label>Projection:</label>
                <button id="btn-umap" class="active">UMAP</button>
                <button id="btn-tsne">t-SNE</button>
            </div>
            <div class="control-group">
                <label>Search:</label>
                <input type="text" id="search" placeholder="repo name...">
            </div>
            <div id="stats"></div>
        </div>
        <div id="map"></div>
        <div id="tooltip"></div>
    </div>
    <script>
        const API_URL = 'coords.json';
        {js_filtered}
    </script>
</body>
</html>
"""

with gr.Blocks() as demo:
    gr.HTML(full_html)

demo.launch()
```

**Note:** coords.json must be generated locally first (run pipeline), then copied to HF Space repo before deploying.

**Step 2: Create root requirements.txt for HF**

```txt
gradio>=4.0.0
```

**Step 3: Commit**

```bash
git add app.py requirements.txt
git commit -m "feat: add Hugging Face Spaces app"
git push
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Project setup | requirements, gitignore, README |
| 2 | S3 reader | s3_reader.py + tests |
| 3 | StarEncoder embedder | embedder.py + tests |
| 4 | Tree-sitter AST features | ast_features.py + tests |
| 5 | HDBSCAN clusterer | clusterer.py + tests |
| 6 | UMAP/t-SNE reducer | reducer.py + tests |
| 7 | Claude cluster labeler | labeler.py + tests |
| 8 | Pipeline orchestrator | main.py |
| 9 | FastAPI server | api/main.py |
| 10 | Deck.gl frontend | HTML, CSS, JS |
| 11 | Dockerfiles | 3 Dockerfiles |
| 12 | GitHub push | gh repo create + push |
| 13 | Hugging Face Space | app.py + requirements |
