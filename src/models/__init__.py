"""Sentiment classification models."""
from .base import SentimentClassifier
from .rnn import SimpleRNN, StackedRNN
from .lstm import (
    LSTMClassifier,
    BiLSTMWithAttention,
    LSTMWithBatchNorm,
    StackedLSTM,
)

__all__ = [
    # Base
    "SentimentClassifier",
    # RNN variants
    "SimpleRNN",
    "StackedRNN",
    # LSTM variants
    "LSTMClassifier",
    "BiLSTMWithAttention",
    "LSTMWithBatchNorm",
    "StackedLSTM",
]


def get_model(
    model_name: str,
    vocab_size: int,
    pretrained_embeddings=None,
    **kwargs,
) -> SentimentClassifier:
    """Factory function to create models by name.
    
    Args:
        model_name: One of "rnn", "stacked_rnn", "lstm", "bilstm_attention", 
                    "lstm_batchnorm", "stacked_lstm"
        vocab_size: Size of vocabulary
        pretrained_embeddings: Pretrained embedding matrix (optional)
        **kwargs: Additional model arguments
    
    Returns:
        Instantiated model
    """
    models = {
        "rnn": SimpleRNN,
        "stacked_rnn": StackedRNN,
        "lstm": LSTMClassifier,
        "bilstm_attention": BiLSTMWithAttention,
        "lstm_batchnorm": LSTMWithBatchNorm,
        "stacked_lstm": StackedLSTM,
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name](
        vocab_size=vocab_size,
        pretrained_embeddings=pretrained_embeddings,
        **kwargs,
    )