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

FUNCTION_TYPES = {
    'python': ['function_definition', 'async_function_definition'],
    'javascript': ['function_declaration', 'arrow_function', 'function_expression', 'method_definition'],
    'typescript': ['function_declaration', 'arrow_function', 'function_expression', 'method_definition'],
    'go': ['function_declaration', 'method_declaration'],
    'java': ['method_declaration'],
    'rust': ['function_item'],
}

ASYNC_TYPES = {
    'python': ['await'],  # tree-sitter-python 0.25+ uses 'await' node
    'javascript': ['await_expression'],
    'typescript': ['await_expression'],
    'go': [],
    'java': [],
    'rust': ['await_expression'],
}

IMPORT_TYPES = {
    'python': ['import_statement', 'import_from_statement'],
    'javascript': ['import_statement', 'import_declaration'],
    'typescript': ['import_statement', 'import_declaration'],
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


def count_nodes(node, node_types: List[str]) -> int:
    """Count nodes of given types in tree."""
    count = 0

    def visit(n):
        nonlocal count
        if n.type in node_types:
            count += 1
        for child in n.children:
            visit(child)

    visit(node)
    return count


def max_depth(node, current: int = 0) -> int:
    """Find maximum nesting depth in tree."""
    if not node.children:
        return current
    return max(max_depth(child, current + 1) for child in node.children)


def cyclomatic_complexity(node) -> int:
    """Estimate cyclomatic complexity by counting decision points."""
    decision_types = {
        'if_statement', 'elif_clause', 'else_clause',
        'for_statement', 'while_statement', 'for_in_statement',
        'try_statement', 'except_clause', 'catch_clause',
        'case_clause', 'switch_statement', 'match_statement',
        'conditional_expression', 'ternary_expression',
        'boolean_operator', 'binary_expression',
    }
    return count_nodes(node, list(decision_types)) + 1


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

    root = tree.root_node
    func_types = FUNCTION_TYPES.get(lang, [])
    async_types = ASYNC_TYPES.get(lang, [])
    import_types = IMPORT_TYPES.get(lang, [])

    func_count = count_nodes(root, func_types)
    async_count = count_nodes(root, async_types)

    return {
        'function_count': func_count,
        'max_nesting_depth': max_depth(root),
        'async_usage_ratio': async_count / max(func_count, 1),
        'import_count': count_nodes(root, import_types),
        'cyclomatic_complexity': cyclomatic_complexity(root),
        'language': lang,
    }


def extract_repo_features(files: List[Tuple[str, str]]) -> np.ndarray:
    """Extract aggregated structural features for a repo.

    Returns 8-dimensional vector:
    - function_count (total, log scaled)
    - max_nesting_depth (max across files)
    - async_usage_ratio (average)
    - import_count (total, log scaled)
    - test_file_ratio
    - cyclomatic_complexity (average, log scaled)
    - language_mix (normalized entropy)
    - entry_point_pattern (1 if has main/index/cli, 0 otherwise)
    """
    all_features = []
    lang_counts = {}
    test_files = 0
    has_entry_point = False

    entry_patterns = ['main.py', 'index.js', 'index.ts', 'main.go', 'Main.java', 'main.rs', 'cli.py', 'app.py', '__main__.py']

    for filename, content in files:
        features = extract_features(filename, content)
        if features:
            all_features.append(features)
            lang = features.get('language', 'unknown')
            lang_counts[lang] = lang_counts.get(lang, 0) + 1

        if 'test' in filename.lower() or filename.startswith('test_') or '_test.' in filename:
            test_files += 1

        base_name = Path(filename).name
        if base_name in entry_patterns or any(filename.endswith(ep) for ep in entry_patterns):
            has_entry_point = True

    if not all_features:
        return np.zeros(8)

    total_files = len(files)
    total_func = sum(f['function_count'] for f in all_features)
    max_nest = max(f['max_nesting_depth'] for f in all_features)
    avg_async = np.mean([f['async_usage_ratio'] for f in all_features])
    total_imports = sum(f['import_count'] for f in all_features)
    test_ratio = test_files / max(total_files, 1)
    avg_complexity = np.mean([f['cyclomatic_complexity'] for f in all_features])

    total_lang_files = sum(lang_counts.values())
    if total_lang_files > 0 and len(lang_counts) > 1:
        probs = np.array(list(lang_counts.values())) / total_lang_files
        lang_entropy = -np.sum(probs * np.log(probs + 1e-10))
        lang_mix = lang_entropy / np.log(len(lang_counts))
    else:
        lang_mix = 0

    return np.array([
        np.log1p(total_func),
        max_nest,
        avg_async,
        np.log1p(total_imports),
        test_ratio,
        np.log1p(avg_complexity),
        lang_mix,
        float(has_entry_point),
    ])
