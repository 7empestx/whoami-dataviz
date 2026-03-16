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
        # StarEncoder doesn't have a pad token by default
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
        _model = AutoModel.from_pretrained(MODEL_NAME)
        _model.eval()
        if torch.cuda.is_available():
            _model = _model.cuda()
    return _tokenizer, _model


def embed_code(code: str) -> np.ndarray:
    """Embed a code snippet, returning the mean pooled vector."""
    tokenizer, model = _get_model()
    device = next(model.parameters()).device

    inputs = tokenizer(
        code,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_TOKENS,
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)

    return embeddings.squeeze().cpu().numpy()


def chunk_text(text: str, tokenizer, max_tokens: int = MAX_TOKENS) -> List[str]:
    """Split text into overlapping chunks that fit within token limit."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
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
    tokenizer, model = _get_model()

    all_embeddings = []

    for filename, content in files:
        full_content = f"# File: {filename}\n{content}"

        chunks = chunk_text(full_content, tokenizer)
        for chunk in chunks:
            if chunk.strip():
                embedding = embed_code(chunk)
                all_embeddings.append(embedding)

    if not all_embeddings:
        return np.zeros(768)

    return np.mean(all_embeddings, axis=0)
