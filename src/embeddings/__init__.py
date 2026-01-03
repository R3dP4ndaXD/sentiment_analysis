"""Embedding utilities."""
from .fasttext_loader import (
    load_fasttext_model,
    create_embedding_matrix,
    create_embedding_layer,
    get_word_vector,
)

__all__ = [
    "load_fasttext_model",
    "create_embedding_matrix",
    "create_embedding_layer",
    "get_word_vector",
]
