"""Tests for StarEncoder embedder."""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch


def test_embed_code_returns_vector():
    """Embedder should return a numpy vector for code input."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {'input_ids': MagicMock(), 'attention_mask': MagicMock()}

    mock_output = MagicMock()
    mock_output.last_hidden_state = MagicMock()
    mock_output.last_hidden_state.mean.return_value.squeeze.return_value.cpu.return_value.numpy.return_value = np.random.randn(768)

    mock_model = MagicMock()
    mock_model.return_value = mock_output
    mock_model.parameters.return_value = iter([MagicMock(device='cpu')])

    with patch('pipeline.embedder._get_model', return_value=(mock_tokenizer, mock_model)):
        from pipeline.embedder import embed_code
        code = "def hello():\n    print('world')"
        vector = embed_code(code)

    assert isinstance(vector, np.ndarray)
    assert len(vector.shape) == 1
    assert vector.shape[0] == 768


def test_chunk_text_splits_long_content():
    """Chunker should split text that exceeds token limit."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.return_value = list(range(2000))
    mock_tokenizer.decode.side_effect = lambda x: f"chunk_{len(x)}"

    from pipeline.embedder import chunk_text
    chunks = chunk_text("x" * 10000, mock_tokenizer, max_tokens=1024)

    assert len(chunks) > 1
