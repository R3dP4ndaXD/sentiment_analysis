"""PyTorch Dataset and DataLoader utilities for sentiment analysis."""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple
import random

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from .vocab import Vocabulary


@dataclass
class Sample:
    """Raw sample with text and label."""
    text: str
    label: int


@dataclass
class EncodedSample:
    """Encoded sample with token indices and label."""
    token_ids: List[int]
    label: int
    length: int  # Original sequence length before padding


class TextCsvDataset(Dataset):
    """Dataset that loads text from CSV and optionally encodes to indices.
    
    Args:
        csv_path: Path to CSV file
        vocab: Vocabulary for encoding (optional, returns raw text if None)
        tokenizer: Function to tokenize text (required if vocab is provided)
        text_col: Name of text column in CSV
        label_col: Name of label column in CSV
        max_seq_len: Maximum sequence length (truncate longer sequences)
        augment_fn: Optional augmentation function applied to token list
        augment_prob: Probability of applying augmentation
    """
    
    def __init__(
        self,
        csv_path: Path | str,
        vocab: Optional[Vocabulary] = None,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
        text_col: str = "text",
        label_col: str = "label",
        max_seq_len: int = 128,
        augment_fn: Optional[Callable[[List[str]], List[str]]] = None,
        augment_prob: float = 0.0,
    ):
        self.csv_path = Path(csv_path)
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.text_col = text_col
        self.label_col = label_col
        self.max_seq_len = max_seq_len
        self.augment_fn = augment_fn
        self.augment_prob = augment_prob
        
        self.df = pd.read_csv(self.csv_path)
        if text_col not in self.df.columns or label_col not in self.df.columns:
            raise ValueError(
                f"CSV must have columns '{text_col}' and '{label_col}'. "
                f"Found: {self.df.columns.tolist()}"
            )
        
        # Validate configuration
        if vocab is not None and tokenizer is None:
            raise ValueError("tokenizer is required when vocab is provided")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> EncodedSample | Sample:
        row = self.df.iloc[idx]
        text = str(row[self.text_col])
        label = int(row[self.label_col])
        
        # If no vocab, return raw sample
        if self.vocab is None:
            return Sample(text=text, label=label)
        
        # Tokenize
        tokens = self.tokenizer(text)
        
        # Apply augmentation with probability
        if self.augment_fn and random.random() < self.augment_prob:
            tokens = self.augment_fn(tokens)
        
        # Truncate if needed
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        
        # Ensure at least one token (use UNK if empty)
        if len(tokens) == 0:
            tokens = ["<UNK>"]
        
        # Encode to indices
        token_ids = self.vocab.encode(tokens)
        
        # Ensure at least length 1
        if len(token_ids) == 0:
            token_ids = [self.vocab.unk_idx]
        
        return EncodedSample(
            token_ids=token_ids,
            label=label,
            length=len(token_ids),
        )


def collate_encoded(
    batch: List[EncodedSample],
    pad_idx: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate function for encoded samples.
    
    Returns:
        token_ids: Padded tensor of shape (batch_size, max_len)
        labels: Tensor of shape (batch_size,)
        lengths: Tensor of original lengths, shape (batch_size,)
    """
    # Convert to tensors
    sequences = [torch.tensor(s.token_ids, dtype=torch.long) for s in batch]
    labels = torch.tensor([s.label for s in batch], dtype=torch.long)
    lengths = torch.tensor([s.length for s in batch], dtype=torch.long)
    
    # Pad sequences (batch_first=True by default in pad_sequence is False, we want True)
    padded = pad_sequence(sequences, batch_first=True, padding_value=pad_idx)
    
    return padded, labels, lengths


def collate_raw(batch: List[Sample]) -> Tuple[List[str], torch.Tensor]:
    """Collate function for raw samples (text strings).
    
    Returns:
        texts: List of raw text strings
        labels: Tensor of shape (batch_size,)
    """
    texts = [s.text for s in batch]
    labels = torch.tensor([s.label for s in batch], dtype=torch.long)
    return texts, labels


def create_dataloaders(
    train_path: Path | str,
    val_path: Path | str,
    test_path: Path | str,
    vocab: Vocabulary,
    tokenizer: Callable[[str], List[str]],
    batch_size: int = 64,
    max_seq_len: int = 128,
    augment_fn: Optional[Callable[[List[str]], List[str]]] = None,
    augment_prob: float = 0.0,
    num_workers: int = 0,
    text_col: str = "text",
    label_col: str = "label",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test DataLoaders.
    
    Args:
        train_path: Path to training CSV
        val_path: Path to validation CSV
        test_path: Path to test CSV
        vocab: Vocabulary instance
        tokenizer: Tokenization function
        batch_size: Batch size for all loaders
        max_seq_len: Maximum sequence length
        augment_fn: Augmentation function (applied only to training)
        augment_prob: Probability of augmentation (training only)
        num_workers: Number of data loading workers
        text_col: Text column name
        label_col: Label column name
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Training dataset with augmentation
    train_dataset = TextCsvDataset(
        csv_path=train_path,
        vocab=vocab,
        tokenizer=tokenizer,
        text_col=text_col,
        label_col=label_col,
        max_seq_len=max_seq_len,
        augment_fn=augment_fn,
        augment_prob=augment_prob,
    )
    
    # Validation and test without augmentation
    val_dataset = TextCsvDataset(
        csv_path=val_path,
        vocab=vocab,
        tokenizer=tokenizer,
        text_col=text_col,
        label_col=label_col,
        max_seq_len=max_seq_len,
    )
    
    test_dataset = TextCsvDataset(
        csv_path=test_path,
        vocab=vocab,
        tokenizer=tokenizer,
        text_col=text_col,
        label_col=label_col,
        max_seq_len=max_seq_len,
    )
    
    # Collate function with correct pad_idx
    def collate_fn(batch):
        return collate_encoded(batch, pad_idx=vocab.pad_idx)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader
