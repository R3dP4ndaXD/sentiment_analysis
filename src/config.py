"""Centralized configuration for the sentiment analysis project."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class Paths:
    """Data and output paths."""
    root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    
    @property
    def data_raw(self) -> Path:
        return self.root / "data" / "raw"
    
    @property
    def data_processed(self) -> Path:
        return self.root / "data" / "processed"
    
    @property
    def train_csv(self) -> Path:
        return self.data_processed / "train.csv"
    
    @property
    def val_csv(self) -> Path:
        return self.data_processed / "val.csv"
    
    @property
    def test_csv(self) -> Path:
        return self.data_processed / "test.csv"
    
    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"
    
    @property
    def plots(self) -> Path:
        return self.root / "plots"
    
    @property
    def vocab_path(self) -> Path:
        return self.data_processed / "vocab.json"
    
    @property
    def fasttext_model(self) -> Path:
        """Path to fastText Romanian model (download separately)."""
        return self.root / "embeddings" / "cc.ro.300.bin"


@dataclass
class DataConfig:
    """Data processing configuration."""
    max_seq_len: int = 160
    min_freq: int = 2  # Minimum word frequency to include in vocab
    max_vocab_size: int = 30000
    
    # Special tokens
    pad_token: str = "<PAD>"
    unk_token: str = "<UNK>"
    
    # Column names in CSV
    text_col: str = "text"
    label_col: str = "label"


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    model_type: str = "lstm"  # "rnn" or "lstm"
    embedding_dim: int = 300  # fastText dimension
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = True
    
    # Classifier head
    fc_hidden: Optional[int] = 128  # None to skip intermediate FC layer
    num_classes: int = 2


@dataclass
class TrainConfig:
    """Training configuration."""
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 20
    
    # Optimizer: "adam", "adamw", "sgd"
    optimizer: str = "adamw"
    
    # LR scheduler
    scheduler: Optional[str] = "plateau"  # "plateau", "cosine", None
    scheduler_patience: int = 3
    scheduler_factor: float = 0.5
    
    # Early stopping
    early_stopping_patience: int = 5
    
    # Reproducibility
    seed: int = 42
    
    # Augmentation: list of augmentation names to apply
    augmentations: List[str] = field(default_factory=list)
    augment_prob: float = 0.3  # Probability of applying augmentation


@dataclass
class Config:
    """Main configuration combining all sub-configs."""
    paths: Paths = field(default_factory=Paths)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    
    def __post_init__(self):
        # Ensure output directories exist
        self.paths.checkpoints.mkdir(parents=True, exist_ok=True)
        self.paths.plots.mkdir(parents=True, exist_ok=True)


# Default configuration instance
def get_config(**overrides) -> Config:
    """Get config with optional overrides."""
    config = Config()
    
    # Apply any overrides (flat dict like {"model.hidden_dim": 512})
    for key, value in overrides.items():
        parts = key.split(".")
        obj = config
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)
    
    return config
