"""FastText embedding loader for Romanian text.

Download the Romanian fastText model from:
https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ro.300.bin.gz

After downloading, extract and place in: embeddings/cc.ro.300.bin
"""
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import fasttext.util


from ..data.vocab import Vocabulary


_FASTTEXT_MODEL = None


def load_fasttext_model(
    model_path: Optional[Path | str] = None,
    download_if_missing: bool = True,
    language: str = "ro",
) -> "fasttext.FastText._FastText":
    """Load a fastText model.
    
    Args:
        model_path: Path to .bin file. If None, will try to download.
        download_if_missing: If True and model not found, download it.
        language: Language code for downloading (default: "ro" for Romanian)
    
    Returns:
        Loaded fastText model
    
    Raises:
        RuntimeError: If fasttext not installed or model not found
    """
    global _FASTTEXT_MODEL
    
    if _FASTTEXT_MODEL is not None:
        return _FASTTEXT_MODEL
    
    if model_path is not None:
        model_path = Path(model_path)
        if model_path.exists():
            print(f"Loading fastText model from {model_path}...")
            _FASTTEXT_MODEL = fasttext.load_model(str(model_path))
            print(f"Loaded. Embedding dimension: {_FASTTEXT_MODEL.get_dimension()}")
            return _FASTTEXT_MODEL
        elif not download_if_missing:
            raise FileNotFoundError(f"FastText model not found at {model_path}")
    
    # Download model if not found
    print(f"Downloading fastText model for '{language}'...")
    print("This may take a while (Romanian model is ~1.2GB compressed)...")
    fasttext.util.download_model(language, if_exists="ignore")
    
    # Default download location
    default_path = Path(f"cc.{language}.300.bin")
    if default_path.exists():
        _FASTTEXT_MODEL = fasttext.load_model(str(default_path))
        print(f"Loaded. Embedding dimension: {_FASTTEXT_MODEL.get_dimension()}")
        return _FASTTEXT_MODEL
    
    raise RuntimeError(
        f"Failed to load fastText model. Please download manually from:\n"
        f"https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.{language}.300.bin.gz\n"
        f"Extract and place at: {model_path or default_path}"
    )


def get_word_vector(
    word: str,
    model: Optional["fasttext.FastText._FastText"] = None,
) -> np.ndarray:
    """Get the embedding vector for a word.
    
    FastText can generate vectors for OOV words using subword information.
    
    Args:
        word: The word to get embedding for
        model: FastText model (uses cached model if None)
    
    Returns:
        Embedding vector as numpy array
    """
    if model is None:
        model = load_fasttext_model()
    return model.get_word_vector(word)


def create_embedding_matrix(
    vocab: Vocabulary,
    model: Optional["fasttext.FastText._FastText"] = None,
    embedding_dim: int = 300,
    init_special: str = "zero",
) -> torch.Tensor:
    """Create an embedding matrix from vocabulary using fastText.
    
    Args:
        vocab: Vocabulary instance with token2idx mapping
        model: FastText model (loads default if None)
        embedding_dim: Expected embedding dimension (300 for pretrained)
        init_special: How to initialize special tokens:
            - "zero": Zero vectors for PAD/UNK
            - "random": Random normal initialization
            - "mean": Mean of all word vectors (for UNK only, PAD stays zero)
    
    Returns:
        Embedding matrix of shape (vocab_size, embedding_dim)
    """
    if model is None:
        model = load_fasttext_model()
    
    actual_dim = model.get_dimension()
    if actual_dim != embedding_dim:
        print(f"Warning: Model dimension ({actual_dim}) != expected ({embedding_dim})")
        embedding_dim = actual_dim
    
    vocab_size = len(vocab)
    embedding_matrix = np.zeros((vocab_size, embedding_dim), dtype=np.float32)
    
    # Track statistics
    found = 0
    oov_subword = 0  # OOV but fastText can generate from subwords
    
    # Collect vectors for mean calculation (for UNK initialization)
    word_vectors = []
    
    for token, idx in vocab.token2idx.items():
        # Skip special tokens for now
        if token in (vocab.pad_token, vocab.unk_token):
            continue
        
        vec = model.get_word_vector(token)
        embedding_matrix[idx] = vec
        word_vectors.append(vec)
        
        # Check if word is in vocabulary or generated from subwords
        if token in model.words:
            found += 1
        else:
            oov_subword += 1
    
    # Initialize special tokens
    if init_special == "zero":
        # PAD and UNK stay as zeros (already initialized)
        pass
    elif init_special == "random":
        std = np.std(word_vectors) if word_vectors else 0.1
        embedding_matrix[vocab.pad_idx] = np.random.normal(0, std, embedding_dim)
        embedding_matrix[vocab.unk_idx] = np.random.normal(0, std, embedding_dim)
    elif init_special == "mean":
        if word_vectors:
            mean_vec = np.mean(word_vectors, axis=0)
            embedding_matrix[vocab.unk_idx] = mean_vec
        # PAD stays zero
    
    total = vocab_size - 2  # Exclude special tokens
    print(f"Embedding matrix created:")
    print(f"  - Vocabulary size: {vocab_size}")
    print(f"  - Found in fastText vocab: {found}/{total} ({100*found/total:.1f}%)")
    print(f"  - Generated from subwords: {oov_subword}/{total} ({100*oov_subword/total:.1f}%)")
    
    return torch.from_numpy(embedding_matrix)


def create_embedding_layer(
    vocab: Vocabulary,
    model: Optional["fasttext.FastText._FastText"] = None,
    embedding_dim: int = 300,
    freeze: bool = False,
    init_special: str = "zero",
) -> nn.Embedding:
    """Create a PyTorch Embedding layer initialized with fastText vectors.
    
    Args:
        vocab: Vocabulary instance
        model: FastText model
        embedding_dim: Embedding dimension
        freeze: If True, embedding weights won't be updated during training
        init_special: How to initialize special tokens
    
    Returns:
        nn.Embedding layer with pretrained weights
    """
    embedding_matrix = create_embedding_matrix(
        vocab=vocab,
        model=model,
        embedding_dim=embedding_dim,
        init_special=init_special,
    )
    
    embedding = nn.Embedding.from_pretrained(
        embedding_matrix,
        freeze=freeze,
        padding_idx=vocab.pad_idx,
    )
    
    return embedding
