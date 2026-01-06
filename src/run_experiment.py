#!/usr/bin/env python3
"""
Main entry point for Romanian Sentiment Analysis experiments.

Usage examples:
    # Train LSTM with default settings
    python -m src.run_experiment --model lstm
    
    # Train BiLSTM with attention and augmentation
    python -m src.run_experiment --model bilstm_attention --augment random_swap --epochs 30
    
    # Train with custom hyperparameters
    python -m src.run_experiment --model lstm --hidden_dim 256 --num_layers 2 --lr 0.001 --batch_size 64
    
    # Resume training from checkpoint
    python -m src.run_experiment --model lstm --resume checkpoints/best_model.pt
    
    # Evaluate only (no training)
    python -m src.run_experiment --model lstm --evaluate_only --checkpoint checkpoints/best_model.pt
"""
import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Paths, DataConfig, ModelConfig, TrainConfig
from src.data.vocab import Vocabulary, build_vocab_from_csv
from src.data.dataloader import create_dataloaders, TextCsvDataset, collate_encoded
from src.data.augmentations import (
    random_swap, random_delete, contextual_insert,
    random_crop, synonym_replacement_fasttext, contextual_word_replacement, 
    TextAugmenter, get_romanian_stopwords,
    get_eda_augmenter, get_light_augmenter, get_semantic_augmenter, 
    get_heavy_augmenter, get_noise_augmenter,
    back_translate, keyboard_typo, ocr_error,
)
from src.preprocessing.text import clean_text, tokenize
from src.models import get_model, SimpleRNN, StackedRNN, LSTMClassifier, BiLSTMWithAttention, LSTMWithBatchNorm, StackedLSTM
from src.train import Trainer, EarlyStopping, ModelCheckpoint, HistoryLogger
from src.evaluate import (
    compute_metrics, compute_detailed_metrics, print_classification_report,
    MetricsTracker, plot_training_curves, plot_metrics_curves,
    plot_confusion_matrix, plot_comparison, create_results_summary,
)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Romanian Sentiment Analysis with RNN/LSTM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Model architecture
    parser.add_argument(
        "--model", type=str, default="lstm",
        choices=["simple_rnn", "stacked_rnn", "lstm", "bilstm_attention", "lstm_batchnorm", "stacked_lstm"],
        help="Model architecture to use"
    )
    parser.add_argument("--embedding_dim", type=int, default=300, help="Embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden state dimension")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of RNN/LSTM layers")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout probability")
    parser.add_argument("--bidirectional", action="store_true", help="Use bidirectional LSTM")
    parser.add_argument("--pooling", type=str, default="last", choices=["last", "max", "mean"], help="Pooling strategy")
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="L2 regularization")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw", "sgd"], help="Optimizer")
    parser.add_argument("--scheduler", type=str, default="plateau", choices=["plateau", "cosine", "none"], help="LR scheduler")
    parser.add_argument("--gradient_clip", type=float, default=1.0, help="Gradient clipping norm (0 to disable)")
    
    # Data configuration
    parser.add_argument("--max_seq_len", type=int, default=160, help="Maximum sequence length")
    parser.add_argument("--min_freq", type=int, default=2, help="Minimum word frequency for vocabulary")
    parser.add_argument("--max_vocab_size", type=int, default=50000, help="Maximum vocabulary size")
    parser.add_argument("--remove_stopwords", action="store_true", help="Remove stopwords (keeps sentiment words)")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers (0=main thread, 2-4 recommended)")
    
    # Augmentation
    parser.add_argument(
        "--augment", type=str, default=None,
        choices=[
            # Single augmentations (online - fast)
            "random_swap", "random_delete", "random_crop", "insert",
            "synonym", "keyboard", "ocr",
            # Composite augmentations (nlpaug flows - online)
            "eda", "eda_light", "eda_noise",
            "none"
        ],
        help="Data augmentation technique."
    )
    parser.add_argument("--aug_prob", type=float, default=0.2, help="Augmentation probability/intensity")
    parser.add_argument(
        "--aug_mode", type=str, default="sometimes",
        choices=["sometimes", "sequential", "one_of"],
        help="How to apply multiple augmentations (sometimes=probabilistic, sequential=all, one_of=pick one)"
    )
    
    # Dataset expansion and balancing
    parser.add_argument(
        "--expand_factor", type=float, default=1.0,
        help="Multiply training set size with augmented samples (1.0 = no expansion)"
    )
    parser.add_argument(
        "--balance_classes", action="store_true",
        help="Balance classes by oversampling minority class with augmentation"
    )
    parser.add_argument(
        "--weighted_sampler", action="store_true",
        help="Use WeightedRandomSampler to balance batch class distribution"
    )
    
    # Embeddings
    parser.add_argument("--pretrained_embeddings", type=str, default=None, help="Path to pretrained fastText embeddings (.bin)")
    parser.add_argument("--freeze_embeddings", action="store_true", help="Freeze pretrained embeddings")
    
    # Callbacks
    parser.add_argument("--early_stopping", type=int, default=5, help="Early stopping patience (0 to disable)")
    parser.add_argument("--checkpoint_metric", type=str, default="val_f1", help="Metric to monitor for checkpointing")
    
    # Experiment management
    parser.add_argument("--experiment_name", type=str, default=None, help="Experiment name (auto-generated if not provided)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu, auto if not specified)")
    
    # Paths
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Data directory")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--plots_dir", type=str, default="plots", help="Plots directory")
    parser.add_argument("--results_dir", type=str, default="results", help="Results directory")
    
    # Modes
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training")
    parser.add_argument("--evaluate_only", action="store_true", help="Only evaluate, no training")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint to load for evaluation")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--no_plots", action="store_true", help="Skip generating plots")
    
    return parser.parse_args()


def get_augmentation_fn(aug_name: str, aug_prob: float, aug_mode: str = "sometimes", 
                        fasttext_path: str = None, device: str = "cpu"):
    """Get augmentation function by name using nlpaug.
    
    Args:
        aug_name: Name of the augmentation to use
        aug_prob: Probability/intensity parameter
        aug_mode: How to chain augmentations ("sometimes", "sequential", "one_of")
        fasttext_path: Path to fastText embeddings (for synonym augmentation)
        device: Device for BERT models ("cpu" or "cuda")
    
    Returns:
        Callable that takes tokens and returns augmented tokens
    """
    if aug_name is None or aug_name == "none":
        return None
    
    # Convert probability to discrete count for swap/insert (1-5 operations)
    n_ops = max(1, int(aug_prob * 10)) if aug_prob else 1
    
    # Load Romanian stopwords
    stopwords = get_romanian_stopwords()
    
    # Single augmentation functions
    single_augs = {
        "random_swap": lambda tokens: random_swap(tokens, n_swaps=n_ops),
        "random_delete": lambda tokens: random_delete(tokens, p=aug_prob),
        "random_crop": lambda tokens: random_crop(tokens, min_ratio=0.7, max_ratio=0.9),
        "inser": lambda tokens: contextual_insert(tokens, n_inserts=n_ops, device=device),
        "synonym": lambda tokens: synonym_replacement_fasttext(
            tokens, n_replacements=n_ops, model_path=fasttext_path, stopwords=stopwords
        ),
        "contextual": lambda tokens: contextual_word_replacement(
            tokens, n_replacements=n_ops, stopwords=stopwords, device=device
        ),
        "back_translate": lambda tokens: back_translate(tokens, device=device),
        "keyboard": lambda tokens: keyboard_typo(tokens, aug_p=aug_prob),
        "ocr": lambda tokens: ocr_error(tokens, aug_p=aug_prob),
    }
    
    if aug_name in single_augs:
        return single_augs[aug_name]
    
    # Composite augmentations using TextAugmenter with nlpaug flows
    composite_augs = {
        # Standard EDA: swap, delete, insert, synonym (probabilistic)
        "eda": lambda tokens: TextAugmenter(
            strategies=["random_swap", "random_delete", "contextual_insert", "synonym_wordnet"],
            p=0.3,
            mode=aug_mode,
            fasttext_path=fasttext_path,
            device=device,
            aug_p=aug_prob,
            n_ops=n_ops,
            stopwords=stopwords,
        )(tokens),
        
        # Light: structural only (swap + delete)
        "eda_light": lambda tokens: TextAugmenter(
            strategies=["random_swap", "random_delete"],
            p=0.3,
            mode=aug_mode,
            aug_p=aug_prob,
            n_ops=max(1, n_ops // 2),
        )(tokens),
        
        # Semantic: synonym + contextual (pick ONE)
        "eda_semantic": lambda tokens: TextAugmenter(
            strategies=["synonym_wordnet", "contextual_substitute"],
            p=0.5,
            mode="one_of",  # Apply ONE semantic change to avoid drift
            fasttext_path=fasttext_path,
            device=device,
            aug_p=aug_prob,
            n_ops=n_ops,
            stopwords=stopwords,
        )(tokens),
        
        # Heavy: all techniques (use carefully)
        "eda_heavy": lambda tokens: TextAugmenter(
            strategies=["random_swap", "random_delete", "random_crop", 
                       "synonym_wordnet", "contextual_substitute"],
            p=0.2,  # Lower probability to reduce compounding
            mode=aug_mode,
            fasttext_path=fasttext_path,
            device=device,
            aug_p=aug_prob,
            n_ops=n_ops,
            stopwords=stopwords,
        )(tokens),
        
        # Noise: keyboard typos + OCR errors
        "eda_noise": lambda tokens: TextAugmenter(
            strategies=["keyboard", "ocr"],
            p=0.3,
            mode="one_of",
            aug_p=aug_prob,
        )(tokens),
    }
    
    return composite_augs.get(aug_name)


def create_experiment_name(args: argparse.Namespace) -> str:
    """Generate experiment name from args."""
    if args.experiment_name:
        return args.experiment_name
    
    parts = [args.model]
    parts.append(f"h{args.hidden_dim}")
    parts.append(f"l{args.num_layers}")
    parts.append(f"lr{args.lr}")
    
    if args.augment and args.augment != "none":
        parts.append(args.augment)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts.append(timestamp)
    
    return "_".join(parts)


def load_or_build_vocab(
    args: argparse.Namespace,
    vocab_path: Path,
    train_path: Path,
) -> Vocabulary:
    """Load existing vocabulary or build new one."""
    if vocab_path.exists():
        print(f"Loading vocabulary from {vocab_path}")
        vocab = Vocabulary.load(vocab_path)
    else:
        print(f"Building vocabulary from {train_path}")
        vocab = build_vocab_from_csv(
            csv_path=train_path,
            text_col="text",
            tokenizer=tokenize,
            min_freq=args.min_freq,
            max_size=args.max_vocab_size,
        )
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        vocab.save(vocab_path)
        print(f"Vocabulary saved to {vocab_path}")
    
    print(f"Vocabulary size: {len(vocab)}")
    return vocab


def load_pretrained_embeddings(
    args: argparse.Namespace,
    vocab: Vocabulary,
) -> Optional[torch.Tensor]:
    """Load pretrained embeddings if specified."""
    if args.pretrained_embeddings is None:
        return None
    
    try:
        from src.embeddings.fasttext_loader import create_embedding_matrix, load_fasttext_model
        
        print(f"Loading pretrained embeddings from {args.pretrained_embeddings}")
        ft_model = load_fasttext_model(args.pretrained_embeddings)
        embedding_matrix = create_embedding_matrix(vocab, ft_model, args.embedding_dim)
        print(f"Loaded embeddings: {embedding_matrix.shape}")
        return embedding_matrix
    except Exception as e:
        print(f"Warning: Could not load pretrained embeddings: {e}")
        return None


def create_model(
    args: argparse.Namespace,
    vocab_size: int,
    pretrained_embeddings: Optional[torch.Tensor] = None,
) -> nn.Module:
    """Create model based on arguments."""
    common_kwargs = {
        "vocab_size": vocab_size,
        "embedding_dim": args.embedding_dim,
        "hidden_dim": args.hidden_dim,
        "num_classes": 2,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "pad_idx": 0,  # PAD token
        "pretrained_embeddings": pretrained_embeddings,
        "freeze_embeddings": args.freeze_embeddings,
    }
    
    model_map = {
        "simple_rnn": lambda: SimpleRNN(
            pooling=args.pooling,
            **common_kwargs,
        ),
        "stacked_rnn": lambda: StackedRNN(
            pooling=args.pooling,
            **common_kwargs,
        ),
        "lstm": lambda: LSTMClassifier(
            bidirectional=args.bidirectional,
            pooling=args.pooling,
            **common_kwargs,
        ),
        "bilstm_attention": lambda: BiLSTMWithAttention(
            **common_kwargs,
        ),
        "lstm_batchnorm": lambda: LSTMWithBatchNorm(
            bidirectional=args.bidirectional,
            pooling=args.pooling,
            **common_kwargs,
        ),
        "stacked_lstm": lambda: StackedLSTM(
            bidirectional=args.bidirectional,
            pooling=args.pooling,
            **common_kwargs,
        ),
    }
    
    if args.model not in model_map:
        raise ValueError(f"Unknown model: {args.model}")
    
    model = model_map[args.model]()
    
    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {args.model}")
    print(f"  Total parameters: {num_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    return model


def save_experiment_config(args: argparse.Namespace, save_path: Path):
    """Save experiment configuration to JSON."""
    config = vars(args).copy()
    config["timestamp"] = datetime.now().isoformat()
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(config, f, indent=2, default=str)


def save_results(
    args: argparse.Namespace,
    history: Dict,
    metrics: Dict,
    save_dir: Path,
):
    """Save training results."""
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save history
    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    # Save final metrics
    with open(save_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Save summary
    summary = {
        "model": args.model,
        "best_val_f1": max(history.get("val_f1", [0])),
        "best_val_acc": max(history.get("val_acc", [0])),
        "best_val_loss": min(history.get("val_loss", [float("inf")])),
        "epochs_trained": len(history.get("train_loss", [])),
        "final_test_metrics": metrics,
    }
    with open(save_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def generate_plots(
    args: argparse.Namespace,
    history: Dict,
    y_true: List[int],
    y_pred: List[int],
    save_dir: Path,
):
    """Generate and save training plots."""
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Loss curves
    plot_training_curves(
        history,
        title=f"{args.model} - Training Curves",
        save_path=save_dir / "loss_curves.png",
        show=False,
    )
    
    # Metrics curves
    plot_metrics_curves(
        history,
        metrics=["acc", "f1"],
        title=f"{args.model} - Metrics",
        save_path=save_dir / "metrics_curves.png",
        show=False,
    )
    
    # Confusion matrix
    plot_confusion_matrix(
        y_true, y_pred,
        class_names=["Negative", "Positive"],
        title=f"{args.model} - Confusion Matrix",
        save_path=save_dir / "confusion_matrix.png",
        show=False,
    )
    
    # Normalized confusion matrix
    plot_confusion_matrix(
        y_true, y_pred,
        class_names=["Negative", "Positive"],
        title=f"{args.model} - Confusion Matrix (Normalized)",
        normalize=True,
        save_path=save_dir / "confusion_matrix_normalized.png",
        show=False,
    )
    
    print(f"Plots saved to {save_dir}")


def main():
    """Main entry point."""
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    print(f"Random seed: {args.seed}")
    
    # Generate experiment name
    experiment_name = create_experiment_name(args)
    print(f"\n{'='*60}")
    print(f"Experiment: {experiment_name}")
    print(f"{'='*60}")
    
    # Setup paths
    data_dir = Path(args.data_dir)
    checkpoint_dir = Path(args.checkpoint_dir) / experiment_name
    plots_dir = Path(args.plots_dir) / experiment_name
    results_dir = Path(args.results_dir) / experiment_name
    vocab_path = data_dir / "vocab.json"
    
    # Save experiment config
    save_experiment_config(args, results_dir / "config.json")
    
    # Device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load/build vocabulary
    vocab = load_or_build_vocab(args, vocab_path, data_dir / "train.csv")
    
    # Load pretrained embeddings
    pretrained_embeddings = load_pretrained_embeddings(args, vocab)
    
    # Create augmentation function
    aug_fn = get_augmentation_fn(
        aug_name=args.augment,
        aug_prob=args.aug_prob,
        aug_mode=args.aug_mode,
        fasttext_path=args.pretrained_embeddings,  # Use same path for synonym aug
        device=device,
    )
    if aug_fn:
        print(f"Augmentation: {args.augment} (aug_prob={args.aug_prob}, mode={args.aug_mode})")
    
    # Create tokenizer with optional stopword removal
    if args.remove_stopwords:
        print("Stopword removal: ENABLED (preserving sentiment words)")
        tokenizer_fn = lambda text: tokenize(text, remove_stopwords=True, keep_sentiment_words=True)
    else:
        tokenizer_fn = tokenize
    
    # Create data loaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_path=data_dir / "train.csv",
        val_path=data_dir / "val.csv",
        test_path=data_dir / "test.csv",
        vocab=vocab,
        tokenizer=tokenizer_fn,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        augment_fn=aug_fn,
        augment_prob=args.aug_prob,
        num_workers=args.num_workers,  # 2-4 workers recommended for parallel loading
        # Dataset expansion and balancing options
        expand_factor=args.expand_factor,
        balance_classes=args.balance_classes,
        use_weighted_sampler=args.weighted_sampler,
    )
    
    # Print dataset statistics
    train_ds = train_loader.dataset
    print(f"  Train samples: {len(train_ds)}")
    if hasattr(train_ds, 'class_counts'):
        print(f"    Class distribution: {train_ds.class_counts}")
        total = sum(train_ds.class_counts.values())
        for label, count in train_ds.class_counts.items():
            print(f"      Label {label}: {count} ({100*count/total:.1f}%)")
    print(f"  Val samples: {len(val_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")
    
    # Create model
    model = create_model(args, len(vocab), pretrained_embeddings)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        scheduler=args.scheduler if args.scheduler != "none" else None,
        device=device,
        gradient_clip=args.gradient_clip if args.gradient_clip > 0 else None,
    )
    
    # Setup callbacks
    callbacks = []
    
    if args.early_stopping > 0:
        # Use same metric as checkpointing for consistency
        es_mode = "max" if "f1" in args.checkpoint_metric or "acc" in args.checkpoint_metric else "min"
        callbacks.append(EarlyStopping(
            monitor=args.checkpoint_metric,
            patience=args.early_stopping,
            mode=es_mode,
            verbose=True,
        ))
    
    # Checkpoint callback
    checkpoint_mode = "max" if "f1" in args.checkpoint_metric or "acc" in args.checkpoint_metric else "min"
    callbacks.append(ModelCheckpoint(
        filepath=checkpoint_dir / f"best_{args.checkpoint_metric}.pt",
        monitor=args.checkpoint_metric,
        mode=checkpoint_mode,
        save_best_only=True,
        verbose=True,
    ))
    
    # History logger
    callbacks.append(HistoryLogger(
        filepath=results_dir / "history.json",
        save_every=1,
    ))
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Training
    if not args.evaluate_only:
        print(f"\n{'='*60}")
        print("Starting training...")
        print(f"{'='*60}")
        
        history = trainer.train(
            epochs=args.epochs,
            callbacks=callbacks,
            verbose=True,
        )
        
        history_dict = history.to_dict()
        print(f"\nTraining completed!")
        print(f"  Best val loss: {trainer.best_val_loss:.4f}")
        print(f"  Best val F1: {trainer.best_val_f1:.4f}")
    else:
        # Load checkpoint for evaluation
        if args.checkpoint:
            print(f"\nLoading checkpoint: {args.checkpoint}")
            trainer.load_checkpoint(args.checkpoint, load_optimizer=False)
        history_dict = trainer.history.to_dict()
    
    # Evaluation on test set
    print(f"\n{'='*60}")
    print("Evaluating on test set...")
    print(f"{'='*60}")
    
    y_pred, y_true, y_probs = trainer.predict(test_loader)
    
    # Print detailed metrics
    print_classification_report(y_true, y_pred, class_names=["Negative", "Positive"])
    
    # Get metrics dict
    test_metrics = compute_detailed_metrics(y_true, y_pred, class_names=["Negative", "Positive"])
    
    # Save results
    save_results(args, history_dict, test_metrics, results_dir)
    print(f"\nResults saved to {results_dir}")
    
    # Generate plots
    if not args.no_plots:
        generate_plots(args, history_dict, y_true, y_pred, plots_dir)
    
    # Final summary
    print(f"\n{'='*60}")
    print("Experiment Summary")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Test Accuracy: {test_metrics['overall']['accuracy']:.4f}")
    print(f"Test F1: {test_metrics['overall']['f1']:.4f}")
    print(f"Test Precision: {test_metrics['overall']['precision']:.4f}")
    print(f"Test Recall: {test_metrics['overall']['recall']:.4f}")
    print(f"\nCheckpoints: {checkpoint_dir}")
    print(f"Plots: {plots_dir}")
    print(f"Results: {results_dir}")
    
    return test_metrics


if __name__ == "__main__":
    main()
