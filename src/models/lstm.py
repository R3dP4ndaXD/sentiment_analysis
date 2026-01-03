"""LSTM models for sentiment classification."""
from typing import Optional

import torch
import torch.nn as nn

from .base import SentimentClassifier


class LSTMClassifier(SentimentClassifier):
    """LSTM-based sentiment classifier.
    
    Architecture:
        Embedding → Dropout → LSTM → Pooling → FC → Output
    
    Supports:
        - Unidirectional and bidirectional LSTM
        - Multiple pooling strategies
        - Pretrained embeddings
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
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Hidden dimension of LSTM
            num_layers: Number of LSTM layers
            num_classes: Number of output classes
            dropout: Dropout probability
            bidirectional: Use bidirectional LSTM
            fc_hidden: Hidden dimension of FC classifier
            pad_idx: Padding token index
            pretrained_embeddings: Pretrained embedding matrix
            freeze_embeddings: Freeze embedding weights
            pooling: Pooling strategy ("last", "max", "mean")
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
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        
        # Dropout after LSTM
        self.lstm_dropout = nn.Dropout(dropout)
        
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
        # Embed tokens
        embedded = self.embedding(x)
        embedded = self.embed_dropout(embedded)
        
        # Pack for efficient processing
        packed, sort_idx = self._pack_embeddings(embedded, lengths)
        
        # LSTM forward pass
        packed_output, (hidden, cell) = self.lstm(packed)
        
        # Unpack and restore order
        lstm_output = self._unpack_and_unsort(packed_output, sort_idx, x.size(0))
        
        # Apply dropout
        lstm_output = self.lstm_dropout(lstm_output)
        
        # Pool to fixed representation
        pooled = self.get_pooled_output(lstm_output, lengths, self.pooling)
        
        # Classify
        logits = self.classifier(pooled)
        
        return logits


class BiLSTMWithAttention(SentimentClassifier):
    """Bidirectional LSTM with self-attention mechanism.
    
    Architecture:
        Embedding → Dropout → BiLSTM → Attention → FC → Output
    
    The attention mechanism learns to weight different timesteps,
    allowing the model to focus on sentiment-bearing words.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3,
        fc_hidden: Optional[int] = 128,
        pad_idx: int = 0,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        freeze_embeddings: bool = False,
        attention_dim: int = 128,
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Hidden dimension of LSTM (will be bidirectional)
            num_layers: Number of LSTM layers
            num_classes: Number of output classes
            dropout: Dropout probability
            fc_hidden: Hidden dimension of FC classifier
            pad_idx: Padding token index
            pretrained_embeddings: Pretrained embedding matrix
            freeze_embeddings: Freeze embedding weights
            attention_dim: Dimension of attention layer
        """
        super().__init__(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
            bidirectional=True,  # Always bidirectional for attention
            fc_hidden=fc_hidden,
            pad_idx=pad_idx,
            pretrained_embeddings=pretrained_embeddings,
            freeze_embeddings=freeze_embeddings,
        )
        
        self.attention_dim = attention_dim
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(self.rnn_output_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1, bias=False),
        )
        
        # Dropout
        self.lstm_dropout = nn.Dropout(dropout)
        
        # Build classifier head
        self._build_classifier()
    
    def forward(
        self, 
        x: torch.Tensor, 
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with attention.
        
        Args:
            x: Input token indices (batch, seq_len)
            lengths: Original sequence lengths (batch,)
        
        Returns:
            Logits of shape (batch, num_classes)
        """
        batch_size, max_len = x.size()
        
        # Embed tokens
        embedded = self.embedding(x)
        embedded = self.embed_dropout(embedded)
        
        # Pack for efficient processing
        packed, sort_idx = self._pack_embeddings(embedded, lengths)
        
        # LSTM forward pass
        packed_output, _ = self.lstm(packed)
        
        # Unpack and restore order
        lstm_output = self._unpack_and_unsort(packed_output, sort_idx, batch_size)
        lstm_output = self.lstm_dropout(lstm_output)
        
        # Compute attention scores
        attention_scores = self.attention(lstm_output).squeeze(-1)  # (batch, seq_len)
        
        # Create mask for padding
        mask = torch.arange(max_len, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        
        # Mask out padding positions with large negative value
        attention_scores = attention_scores.masked_fill(~mask, float('-inf'))
        
        # Softmax to get attention weights
        attention_weights = torch.softmax(attention_scores, dim=1)  # (batch, seq_len)
        
        # Weighted sum of LSTM outputs
        context = torch.bmm(attention_weights.unsqueeze(1), lstm_output).squeeze(1)
        # context shape: (batch, hidden_dim * 2)
        
        # Classify
        logits = self.classifier(context)
        
        return logits
    
    def get_attention_weights(
        self, 
        x: torch.Tensor, 
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Get attention weights for interpretability.
        
        Args:
            x: Input token indices (batch, seq_len)
            lengths: Original sequence lengths (batch,)
        
        Returns:
            Attention weights (batch, seq_len)
        """
        batch_size, max_len = x.size()
        
        # Embed tokens
        embedded = self.embedding(x)
        
        # Pack and process
        packed, sort_idx = self._pack_embeddings(embedded, lengths)
        packed_output, _ = self.lstm(packed)
        lstm_output = self._unpack_and_unsort(packed_output, sort_idx, batch_size)
        
        # Compute attention
        attention_scores = self.attention(lstm_output).squeeze(-1)
        
        # Mask padding
        mask = torch.arange(max_len, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        attention_scores = attention_scores.masked_fill(~mask, float('-inf'))
        
        return torch.softmax(attention_scores, dim=1)


class LSTMWithBatchNorm(SentimentClassifier):
    """LSTM with batch normalization for improved training stability.
    
    Architecture:
        Embedding → Dropout → LSTM → BatchNorm → Pooling → FC → Output
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
    ):
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
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        
        # Batch normalization after LSTM output
        self.batch_norm = nn.BatchNorm1d(self.rnn_output_dim)
        
        # Dropout
        self.lstm_dropout = nn.Dropout(dropout)
        
        # Build classifier head
        self._build_classifier()
    
    def forward(
        self, 
        x: torch.Tensor, 
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with batch normalization."""
        # Embed tokens
        embedded = self.embedding(x)
        embedded = self.embed_dropout(embedded)
        
        # Pack for efficient processing
        packed, sort_idx = self._pack_embeddings(embedded, lengths)
        
        # LSTM forward pass
        packed_output, _ = self.lstm(packed)
        
        # Unpack and restore order
        lstm_output = self._unpack_and_unsort(packed_output, sort_idx, x.size(0))
        
        # Pool first, then batch norm (BatchNorm1d expects (batch, features))
        pooled = self.get_pooled_output(lstm_output, lengths, self.pooling)
        
        # Batch normalization
        pooled = self.batch_norm(pooled)
        pooled = self.lstm_dropout(pooled)
        
        # Classify
        logits = self.classifier(pooled)
        
        return logits


class StackedLSTM(SentimentClassifier):
    """Stacked LSTM with layer normalization and residual connections.
    
    More flexible than standard multi-layer LSTM, allowing for:
    - Layer normalization between layers
    - Residual connections for better gradient flow
    - Different configurations per layer
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
        use_residual: bool = True,
    ):
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
        
        # Input projection to match LSTM output dim for residuals
        if self.embedding_dim != self.rnn_output_dim:
            self.input_proj = nn.Linear(self.embedding_dim, self.rnn_output_dim)
        else:
            self.input_proj = None
        
        # Stacked LSTM layers
        self.lstm_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList() if use_layer_norm else None
        self.dropouts = nn.ModuleList()
        
        for i in range(num_layers):
            input_size = self.rnn_output_dim if i > 0 or self.input_proj else self.embedding_dim
            
            self.lstm_layers.append(nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=bidirectional,
            ))
            
            if use_layer_norm:
                self.layer_norms.append(nn.LayerNorm(self.rnn_output_dim))
            
            self.dropouts.append(nn.Dropout(dropout))
        
        # Build classifier head
        self._build_classifier()
    
    def forward(
        self, 
        x: torch.Tensor, 
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with residual connections."""
        # Embed
        embedded = self.embedding(x)
        embedded = self.embed_dropout(embedded)
        
        # Project if needed
        if self.input_proj is not None:
            hidden = self.input_proj(embedded)
        else:
            hidden = embedded
        
        # Process through stacked layers
        for i, lstm in enumerate(self.lstm_layers):
            residual = hidden if self.use_residual else None
            
            # Pack sequences
            packed, sort_idx = self._pack_embeddings(hidden, lengths)
            packed_output, _ = lstm(packed)
            hidden = self._unpack_and_unsort(packed_output, sort_idx, x.size(0))
            
            # Layer norm
            if self.use_layer_norm:
                hidden = self.layer_norms[i](hidden)
            
            # Residual connection (only if dimensions match)
            if residual is not None and hidden.shape == residual.shape:
                hidden = hidden + residual
            
            # Dropout (except last layer for pooling)
            hidden = self.dropouts[i](hidden)
        
        # Pool and classify
        pooled = self.get_pooled_output(hidden, lengths, self.pooling)
        logits = self.classifier(pooled)
        
        return logits
