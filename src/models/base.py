"""Base model class for sentiment classification."""
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SentimentClassifier(nn.Module, ABC):
    """Abstract base class for sentiment classification models.
    
    All models follow the architecture:
    1. Embedding layer (pretrained or trainable)
    2. Recurrent encoder (RNN/LSTM/GRU)
    3. Pooling/aggregation
    4. Classification head (FC layers)
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        fc_hidden: Optional[int] = 128,
        pad_idx: int = 0,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        freeze_embeddings: bool = False,
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Hidden dimension of recurrent layer
            num_layers: Number of recurrent layers
            num_classes: Number of output classes (2 for binary sentiment)
            dropout: Dropout probability
            bidirectional: Use bidirectional recurrent layers
            fc_hidden: Hidden dimension of FC classifier (None to skip)
            pad_idx: Padding token index for embedding layer
            pretrained_embeddings: Pretrained embedding matrix
            freeze_embeddings: Whether to freeze embedding weights
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout_p = dropout
        self.bidirectional = bidirectional
        self.fc_hidden = fc_hidden
        self.pad_idx = pad_idx
        
        # Embedding layer
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(
                pretrained_embeddings,
                freeze=freeze_embeddings,
                padding_idx=pad_idx,
            )
            # Override embedding_dim if pretrained has different size
            self.embedding_dim = pretrained_embeddings.shape[1]
        else:
            self.embedding = nn.Embedding(
                vocab_size, embedding_dim, padding_idx=pad_idx
            )
        
        # Dropout after embedding
        self.embed_dropout = nn.Dropout(dropout)
        
        # Calculate output dimension of recurrent layer
        self.rnn_output_dim = hidden_dim * (2 if bidirectional else 1)
        
        # Classification head (to be built after RNN in subclass)
        self.classifier = None
    
    def _build_classifier(self):
        """Build the classification head."""
        layers = []
        
        input_dim = self.rnn_output_dim
        
        if self.fc_hidden is not None:
            layers.extend([
                nn.Linear(input_dim, self.fc_hidden),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
            ])
            input_dim = self.fc_hidden
        
        layers.append(nn.Linear(input_dim, self.num_classes))
        
        self.classifier = nn.Sequential(*layers)
    
    def _pack_embeddings(
        self, 
        embedded: torch.Tensor, 
        lengths: torch.Tensor,
    ) -> nn.utils.rnn.PackedSequence:
        """Pack padded embeddings for efficient RNN processing.
        
        Args:
            embedded: Embedded sequences (batch, seq_len, embed_dim)
            lengths: Original sequence lengths (batch,)
        
        Returns:
            PackedSequence for RNN input
        """
        # Ensure lengths are on CPU for pack_padded_sequence
        lengths_cpu = lengths.cpu()
        
        # Sort by length (descending) for packing
        sorted_lengths, sort_idx = lengths_cpu.sort(descending=True)
        sorted_embedded = embedded[sort_idx]
        
        # Pack the sequences
        packed = pack_padded_sequence(
            sorted_embedded, 
            sorted_lengths.clamp(min=1),  # Ensure minimum length of 1
            batch_first=True,
            enforce_sorted=True,
        )
        
        return packed, sort_idx
    
    def _unpack_and_unsort(
        self,
        packed_output: nn.utils.rnn.PackedSequence,
        sort_idx: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Unpack RNN output and restore original order.
        
        Args:
            packed_output: Packed RNN output
            sort_idx: Indices used for sorting
            batch_size: Original batch size
        
        Returns:
            Unpacked tensor in original order (batch, seq_len, hidden)
        """
        # Unpack
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        # Unsort to restore original order
        unsort_idx = sort_idx.argsort()
        output = output[unsort_idx]
        
        return output
    
    @abstractmethod
    def forward(
        self, 
        x: torch.Tensor, 
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input token indices (batch, seq_len)
            lengths: Original sequence lengths (batch,)
        
        Returns:
            Logits of shape (batch, num_classes)
        """
        pass
    
    def get_pooled_output(
        self,
        rnn_output: torch.Tensor,
        lengths: torch.Tensor,
        pooling: str = "last",
    ) -> torch.Tensor:
        """Pool RNN outputs to fixed-size representation.
        
        Args:
            rnn_output: RNN output (batch, seq_len, hidden_dim)
            lengths: Original sequence lengths
            pooling: Pooling strategy - "last", "max", "mean", "attention"
        
        Returns:
            Pooled representation (batch, hidden_dim)
        """
        if pooling == "last":
            # Get the last non-padded hidden state
            batch_size = rnn_output.size(0)
            # lengths - 1 for 0-indexed, clamp to avoid negative indices
            last_indices = (lengths - 1).clamp(min=0).long()
            
            # Gather last hidden states
            pooled = rnn_output[
                torch.arange(batch_size, device=rnn_output.device),
                last_indices,
            ]
        
        elif pooling == "max":
            # Max pooling over sequence
            # Create mask for padding
            max_len = rnn_output.size(1)
            mask = torch.arange(max_len, device=rnn_output.device).unsqueeze(0)
            mask = mask < lengths.unsqueeze(1)
            
            # Set padded positions to very negative value
            rnn_output_masked = rnn_output.masked_fill(~mask.unsqueeze(-1), float('-inf'))
            pooled, _ = rnn_output_masked.max(dim=1)
        
        elif pooling == "mean":
            # Mean pooling over non-padded positions
            max_len = rnn_output.size(1)
            mask = torch.arange(max_len, device=rnn_output.device).unsqueeze(0)
            mask = mask < lengths.unsqueeze(1)
            
            # Sum and divide by actual length
            rnn_output_masked = rnn_output * mask.unsqueeze(-1).float()
            pooled = rnn_output_masked.sum(dim=1) / lengths.unsqueeze(1).float().clamp(min=1)
        
        else:
            raise ValueError(f"Unknown pooling strategy: {pooling}")
        
        return pooled
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def freeze_embeddings(self):
        """Freeze embedding layer weights."""
        self.embedding.weight.requires_grad = False
    
    def unfreeze_embeddings(self):
        """Unfreeze embedding layer weights."""
        self.embedding.weight.requires_grad = True
