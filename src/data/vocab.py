"""Vocabulary management for text-to-index mapping."""
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd


class Vocabulary:
    """Maps tokens to indices and vice versa.
    
    Handles special tokens <PAD> (index 0) and <UNK> (index 1).
    """
    
    def __init__(
        self,
        pad_token: str = "<PAD>",
        unk_token: str = "<UNK>",
    ):
        self.pad_token = pad_token
        self.unk_token = unk_token
        
        # Initialize with special tokens
        self.token2idx: Dict[str, int] = {
            pad_token: 0,
            unk_token: 1,
        }
        self.idx2token: Dict[int, str] = {
            0: pad_token,
            1: unk_token,
        }
        self._frozen = False
    
    @property
    def pad_idx(self) -> int:
        return self.token2idx[self.pad_token]
    
    @property
    def unk_idx(self) -> int:
        return self.token2idx[self.unk_token]
    
    def __len__(self) -> int:
        return len(self.token2idx)
    
    def add_token(self, token: str) -> int:
        """Add a token to vocabulary. Returns its index."""
        if self._frozen:
            raise RuntimeError("Cannot add tokens to frozen vocabulary")
        if token not in self.token2idx:
            idx = len(self.token2idx)
            self.token2idx[token] = idx
            self.idx2token[idx] = token
        return self.token2idx[token]
    
    def get_idx(self, token: str) -> int:
        """Get index for token, returns UNK index if not found."""
        return self.token2idx.get(token, self.unk_idx)
    
    def get_token(self, idx: int) -> str:
        """Get token for index."""
        return self.idx2token.get(idx, self.unk_token)
    
    def encode(self, tokens: List[str]) -> List[int]:
        """Convert list of tokens to list of indices."""
        return [self.get_idx(t) for t in tokens]
    
    def decode(self, indices: List[int], skip_special: bool = True) -> List[str]:
        """Convert list of indices back to tokens."""
        tokens = [self.get_token(i) for i in indices]
        if skip_special:
            tokens = [t for t in tokens if t not in (self.pad_token, self.unk_token)]
        return tokens
    
    def freeze(self):
        """Freeze vocabulary to prevent further additions."""
        self._frozen = True
    
    def save(self, path: Path | str):
        """Save vocabulary to JSON file."""
        path = Path(path)
        data = {
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "token2idx": self.token2idx,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: Path | str) -> "Vocabulary":
        """Load vocabulary from JSON file."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        vocab = cls(
            pad_token=data["pad_token"],
            unk_token=data["unk_token"],
        )
        vocab.token2idx = data["token2idx"]
        # vocab.idx2token = {int(k): v for k, v in enumerate(vocab.token2idx.keys())}
        # Rebuild idx2token properly
        vocab.idx2token = {v: k for k, v in vocab.token2idx.items()}
        vocab._frozen = True
        return vocab


def build_vocab_from_texts(
    texts: Iterable[str],
    tokenizer,
    min_freq: int = 2,
    max_size: Optional[int] = None,
    pad_token: str = "<PAD>",
    unk_token: str = "<UNK>",
) -> Vocabulary:
    """Build vocabulary from an iterable of texts.
    
    Args:
        texts: Iterable of raw text strings
        tokenizer: Callable that takes text and returns list of tokens
        min_freq: Minimum frequency for a token to be included
        max_size: Maximum vocabulary size (excluding special tokens)
        pad_token: Padding token string
        unk_token: Unknown token string
    
    Returns:
        Vocabulary instance
    """
    # Count all tokens
    counter: Counter = Counter()
    for text in texts:
        tokens = tokenizer(text)
        counter.update(tokens)
    
    # Create vocabulary
    vocab = Vocabulary(pad_token=pad_token, unk_token=unk_token)
    
    # Sort by frequency (descending) and add to vocab
    sorted_tokens = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    
    for token, freq in sorted_tokens:
        if freq < min_freq:
            continue
        if max_size is not None and len(vocab) >= max_size + 2:  # +2 for special tokens
            break
        vocab.add_token(token)
    
    vocab.freeze()
    return vocab


def build_vocab_from_csv(
    csv_path: Path | str,
    text_col: str,
    tokenizer,
    min_freq: int = 2,
    max_size: Optional[int] = None,
    pad_token: str = "<PAD>",
    unk_token: str = "<UNK>",
) -> Vocabulary:
    """Build vocabulary from a CSV file.
    
    Args:
        csv_path: Path to CSV file
        text_col: Name of the text column
        tokenizer: Callable that takes text and returns list of tokens
        min_freq: Minimum frequency for a token to be included
        max_size: Maximum vocabulary size
        pad_token: Padding token string
        unk_token: Unknown token string
    
    Returns:
        Vocabulary instance
    """
    df = pd.read_csv(csv_path)
    texts = df[text_col].astype(str).tolist()
    return build_vocab_from_texts(
        texts=texts,
        tokenizer=tokenizer,
        min_freq=min_freq,
        max_size=max_size,
        pad_token=pad_token,
        unk_token=unk_token,
    )
