"""Visualization utilities for training analysis and model evaluation."""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def set_plot_style():
    """Set consistent plot style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12


def plot_training_curves(
    history: Dict[str, List[float]],
    title: str = "Training Curves",
    save_path: Optional[Path | str] = None,
    show: bool = True,
) -> plt.Figure:
    """Plot training and validation loss curves.
    
    Args:
        history: Dictionary with train_loss, val_loss keys
        title: Plot title
        save_path: Path to save figure
        show: Whether to display the plot
    
    Returns:
        matplotlib Figure
    """
    set_plot_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(history["train_loss"]) + 1)
    
    ax.plot(epochs, history["train_loss"], 'b-', label='Training Loss', linewidth=2)
    ax.plot(epochs, history["val_loss"], 'r-', label='Validation Loss', linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # Force integer epochs
    
    # Mark best validation loss
    best_epoch = np.argmin(history["val_loss"]) + 1
    best_loss = min(history["val_loss"])
    ax.axvline(x=best_epoch, color='gray', linestyle='--', alpha=0.5)
    ax.annotate(
        f'Best: {best_loss:.4f}\n(Epoch {best_epoch})',
        xy=(best_epoch, best_loss),
        xytext=(best_epoch + 1, best_loss + 0.1),
        fontsize=10,
        arrowprops=dict(arrowstyle='->', color='gray'),
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_metrics_curves(
    history: Dict[str, List[float]],
    metrics: List[str] = ["acc", "f1"],
    title: str = "Training Metrics",
    save_path: Optional[Path | str] = None,
    show: bool = True,
) -> plt.Figure:
    """Plot training and validation metric curves.
    
    Args:
        history: Dictionary with train_acc, val_acc, train_f1, val_f1, etc.
        metrics: List of metric names to plot (without train_/val_ prefix)
        title: Plot title
        save_path: Path to save figure
        show: Whether to display the plot
    
    Returns:
        matplotlib Figure
    """
    set_plot_style()
    
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))
    
    if n_metrics == 1:
        axes = [axes]
    
    epochs = range(1, len(history.get("train_loss", [])) + 1)
    colors = {'train': 'blue', 'val': 'red'}
    
    for ax, metric in zip(axes, metrics):
        train_key = f"train_{metric}"
        val_key = f"val_{metric}"
        
        if train_key in history:
            ax.plot(epochs, history[train_key], 'b-', label=f'Training {metric.upper()}', linewidth=2)
        if val_key in history:
            ax.plot(epochs, history[val_key], 'r-', label=f'Validation {metric.upper()}', linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.upper())
        ax.set_title(f'{metric.upper()} over Training')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # Force integer epochs
        
        # Mark best validation metric
        if val_key in history and history[val_key]:
            best_epoch = np.argmax(history[val_key]) + 1
            best_val = max(history[val_key])
            ax.axvline(x=best_epoch, color='gray', linestyle='--', alpha=0.5)
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    normalize: bool = False,
    save_path: Optional[Path | str] = None,
    show: bool = True,
    cmap: str = "Blues",
) -> plt.Figure:
    """Plot confusion matrix heatmap.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Names for each class
        title: Plot title
        normalize: Normalize values (show percentages)
        save_path: Path to save figure
        show: Whether to display the plot
        cmap: Colormap name
    
    Returns:
        matplotlib Figure
    """
    set_plot_style()
    
    if class_names is None:
        class_names = ["Negative", "Positive"]
    
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
    else:
        fmt = 'd'
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        annot_kws={'size': 14},
    )
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_comparison(
    histories: Dict[str, Dict[str, List[float]]],
    metric: str = "val_loss",
    title: str = "Model Comparison",
    save_path: Optional[Path | str] = None,
    show: bool = True,
) -> plt.Figure:
    """Compare multiple experiments on the same plot.
    
    Args:
        histories: Dict mapping experiment names to their histories
        metric: Metric to compare
        title: Plot title
        save_path: Path to save figure
        show: Whether to display the plot
    
    Returns:
        matplotlib Figure
    """
    set_plot_style()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))
    
    for (name, history), color in zip(histories.items(), colors):
        if metric in history:
            epochs = range(1, len(history[metric]) + 1)
            ax.plot(epochs, history[metric], label=name, linewidth=2, color=color)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # Force integer epochs
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_augmentation_comparison(
    history_without_aug: Dict[str, List[float]],
    history_with_aug: Dict[str, List[float]],
    aug_name: str = "Augmentation",
    metric: str = "val_loss",
    title: Optional[str] = None,
    save_path: Optional[Path | str] = None,
    show: bool = True,
) -> plt.Figure:
    """Compare training with and without augmentation.
    
    Args:
        history_without_aug: Training history without augmentation
        history_with_aug: Training history with augmentation
        aug_name: Name of the augmentation technique
        metric: Metric to compare
        title: Plot title (auto-generated if None)
        save_path: Path to save figure
        show: Whether to display the plot
    
    Returns:
        matplotlib Figure
    """
    if title is None:
        title = f"Effect of {aug_name} on {metric.replace('_', ' ').title()}"
    
    return plot_comparison(
        {
            "Without Augmentation": history_without_aug,
            f"With {aug_name}": history_with_aug,
        },
        metric=metric,
        title=title,
        save_path=save_path,
        show=show,
    )


def plot_learning_rate(
    history: Dict[str, List[float]],
    title: str = "Learning Rate Schedule",
    save_path: Optional[Path | str] = None,
    show: bool = True,
) -> plt.Figure:
    """Plot learning rate over training.
    
    Args:
        history: Dictionary with learning_rates key
        title: Plot title
        save_path: Path to save figure
        show: Whether to display the plot
    
    Returns:
        matplotlib Figure
    """
    set_plot_style()
    
    if "learning_rates" not in history:
        raise ValueError("history must contain 'learning_rates' key")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    epochs = range(1, len(history["learning_rates"]) + 1)
    ax.plot(epochs, history["learning_rates"], 'g-', linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title(title)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # Force integer epochs
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_attention_weights(
    tokens: List[str],
    attention_weights: List[float],
    title: str = "Attention Weights",
    save_path: Optional[Path | str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (14, 3),
) -> plt.Figure:
    """Visualize attention weights over tokens.
    
    Args:
        tokens: List of tokens
        attention_weights: Attention weight for each token
        title: Plot title
        save_path: Path to save figure
        show: Whether to display the plot
        figsize: Figure size
    
    Returns:
        matplotlib Figure
    """
    set_plot_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap-style visualization
    weights = np.array(attention_weights).reshape(1, -1)
    
    im = ax.imshow(weights, cmap='YlOrRd', aspect='auto')
    
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha='right')
    ax.set_yticks([])
    ax.set_title(title)
    
    # Add weight values on top
    for i, (token, weight) in enumerate(zip(tokens, attention_weights)):
        ax.text(i, 0, f'{weight:.2f}', ha='center', va='center', fontsize=9)
    
    plt.colorbar(im, ax=ax, orientation='vertical', label='Attention Weight')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def create_results_summary(
    experiment_results: Dict[str, Dict],
    save_dir: Path | str,
    show: bool = False,
):
    """Create comprehensive visualization summary for all experiments.
    
    Args:
        experiment_results: Dict mapping experiment names to their results
            Each result should have: "history", "y_true", "y_pred", "metadata"
        save_dir: Directory to save all plots
        show: Whether to display plots
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Training curves comparison
    histories = {name: res["history"] for name, res in experiment_results.items()}
    
    plot_comparison(
        histories,
        metric="val_loss",
        title="Validation Loss Comparison",
        save_path=save_dir / "comparison_val_loss.png",
        show=show,
    )
    
    plot_comparison(
        histories,
        metric="val_f1",
        title="Validation F1 Comparison",
        save_path=save_dir / "comparison_val_f1.png",
        show=show,
    )
    
    # 2. Individual experiment plots
    for name, result in experiment_results.items():
        exp_dir = save_dir / name.replace(" ", "_").lower()
        exp_dir.mkdir(exist_ok=True)
        
        # Loss curves
        plot_training_curves(
            result["history"],
            title=f"{name} - Loss Curves",
            save_path=exp_dir / "loss_curves.png",
            show=show,
        )
        
        # Metrics curves
        plot_metrics_curves(
            result["history"],
            metrics=["acc", "f1"],
            title=f"{name} - Metrics",
            save_path=exp_dir / "metrics_curves.png",
            show=show,
        )
        
        # Confusion matrix
        if "y_true" in result and "y_pred" in result:
            plot_confusion_matrix(
                result["y_true"],
                result["y_pred"],
                title=f"{name} - Confusion Matrix",
                save_path=exp_dir / "confusion_matrix.png",
                show=show,
            )
    
    print(f"Results saved to {save_dir}")
