"""Simple RNN models for sentiment classification."""
from typing import Optional

import torch
import torch.nn as nn

from .base import SentimentClassifier


class SimpleRNN(SentimentClassifier):
    """Simple RNN (Elman RNN) for sentiment classification.
    
    Architecture:
        Embedding → Dropout → RNN → Pooling → FC → Output
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
        pooling: str = "last",
        nonlinearity: str = "tanh",
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Hidden dimension of RNN
            num_layers: Number of RNN layers
            num_classes: Number of output classes
            dropout: Dropout probability
            bidirectional: Use bidirectional RNN
            fc_hidden: Hidden dimension of FC classifier
            pad_idx: Padding token index
            pretrained_embeddings: Pretrained embedding matrix
            freeze_embeddings: Freeze embedding weights
            pooling: Pooling strategy ("last", "max", "mean")
            nonlinearity: RNN nonlinearity ("tanh" or "relu")
        """
        super().__init__(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
            bidirectional=bidirectional,
            fc_hidden=fc_hidden,
            pad_idx=pad_idx,
            pretrained_embeddings=pretrained_embeddings,
            freeze_embeddings=freeze_embeddings,
        )
        
        self.pooling = pooling
        
        # RNN layer
        self.rnn = nn.RNN(
            input_size=self.embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            nonlinearity=nonlinearity,
        )
        
        # Dropout after RNN
        self.rnn_dropout = nn.Dropout(dropout)
        
        # Build classifier head
        self._build_classifier()
    
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
        # Embed tokens: (batch, seq_len) -> (batch, seq_len, embed_dim)
        embedded = self.embedding(x)
        embedded = self.embed_dropout(embedded)
        
        # Pack for efficient processing
        packed, sort_idx = self._pack_embeddings(embedded, lengths)
        
        # RNN forward pass
        packed_output, hidden = self.rnn(packed)
        
        # Unpack and restore order
        rnn_output = self._unpack_and_unsort(packed_output, sort_idx, x.size(0))
        
        # Also unsort lengths for pooling
        unsort_idx = sort_idx.argsort()
        
        # Apply dropout
        rnn_output = self.rnn_dropout(rnn_output)
        
        # Pool to fixed representation
        pooled = self.get_pooled_output(rnn_output, lengths, self.pooling)
        
        # Classify
        logits = self.classifier(pooled)
        
        return logits


class StackedRNN(SentimentClassifier):
    """Stacked RNN with optional layer normalization and residual connections.
    
    Architecture:
        Embedding → Dropout → [RNN → LayerNorm → Dropout]×N → Pooling → FC → Output
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
        pooling: str = "last",
        use_layer_norm: bool = True,
        use_residual: bool = False,
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Hidden dimension of RNN
            num_layers: Number of stacked RNN layers
            num_classes: Number of output classes
            dropout: Dropout probability
            bidirectional: Use bidirectional RNN
            fc_hidden: Hidden dimension of FC classifier
            pad_idx: Padding token index
            pretrained_embeddings: Pretrained embedding matrix
            freeze_embeddings: Freeze embedding weights
            pooling: Pooling strategy
            use_layer_norm: Apply layer normalization between layers
            use_residual: Add residual connections (requires matching dimensions)
        """
        super().__init__(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
            bidirectional=bidirectional,
            fc_hidden=fc_hidden,
            pad_idx=pad_idx,
            pretrained_embeddings=pretrained_embeddings,
            freeze_embeddings=freeze_embeddings,
        )
        
        self.pooling = pooling
        self.use_layer_norm = use_layer_norm
        self.use_residual = use_residual
        
        # Input projection if embedding dim != rnn input dim
        rnn_input_dim = hidden_dim * (2 if bidirectional else 1)
        if self.embedding_dim != rnn_input_dim:
            self.input_proj = nn.Linear(self.embedding_dim, rnn_input_dim)
        else:
            self.input_proj = None
        
        # Stack of RNN layers
        self.rnn_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList() if use_layer_norm else None
        self.dropouts = nn.ModuleList()
        
        for i in range(num_layers):
            input_size = rnn_input_dim if i > 0 or self.input_proj else self.embedding_dim
            
            self.rnn_layers.append(nn.RNN(
                input_size=input_size,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=bidirectional,
            ))
            
            if use_layer_norm:
                self.layer_norms.append(nn.LayerNorm(rnn_input_dim))
            
            self.dropouts.append(nn.Dropout(dropout))
        
        # Build classifier head
        self._build_classifier()
    
    def forward(
        self, 
        x: torch.Tensor, 
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with optional residual connections."""
        # Embed
        embedded = self.embedding(x)
        embedded = self.embed_dropout(embedded)
        
        # Project if needed
        if self.input_proj is not None:
            hidden = self.input_proj(embedded)
        else:
            hidden = embedded
        
        # Process through stacked layers
        for i, rnn in enumerate(self.rnn_layers):
            residual = hidden if self.use_residual else None
            
            # Pack sequences
            packed, sort_idx = self._pack_embeddings(hidden, lengths)
            packed_output, _ = rnn(packed)
            hidden = self._unpack_and_unsort(packed_output, sort_idx, x.size(0))
            
            # Layer norm
            if self.use_layer_norm:
                hidden = self.layer_norms[i](hidden)
            
            # Residual connection
            if residual is not None and hidden.shape == residual.shape:
                hidden = hidden + residual
            
            # Dropout
            hidden = self.dropouts[i](hidden)
        
        # Pool and classify
        pooled = self.get_pooled_output(hidden, lengths, self.pooling)
        logits = self.classifier(pooled)
        
        return logits
