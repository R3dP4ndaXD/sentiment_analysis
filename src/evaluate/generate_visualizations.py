#!/usr/bin/env python3
"""
Generate all required visualizations for the ML assignment report.

This script produces:
1. Data Exploration visualizations:
   - Class balance bar plots for train/val/test
   - Text length distributions per sentiment class
   - Most frequent words per class (word clouds / bar charts)

2. Training & Evaluation visualizations:
   - Training vs validation loss curves
   - Training vs validation accuracy/F1 curves
   - Confusion matrices for best models

3. Comparative visualizations:
   - Model comparison plots (different architectures)
   - Augmentation impact comparison (with vs without)
   - Results summary table

Usage:
    # Generate EDA plots only
    python -m src.evaluate.generate_visualizations --eda
    
    # Generate comparison plots from experiment results
    python -m src.evaluate.generate_visualizations --compare --results_dir results/
    
    # Generate all visualizations
    python -m src.evaluate.generate_visualizations --all --results_dir results/
"""
import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style globally
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12


# =============================================================================
# Data Exploration Visualizations
# =============================================================================

def plot_class_distribution(
    data_dir: Path,
    save_dir: Path,
    show: bool = False,
):
    """Plot class distribution for train/val/test sets."""
    save_dir.mkdir(parents=True, exist_ok=True)
    
    splits = ["train", "val", "test"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    colors = ["#e74c3c", "#2ecc71"]  # Red for negative, Green for positive
    
    for ax, split in zip(axes, splits):
        df = pd.read_csv(data_dir / f"{split}.csv")
        counts = df["label"].value_counts().sort_index()
        
        bars = ax.bar(
            ["Negative (0)", "Positive (1)"],
            [counts.get(0, 0), counts.get(1, 0)],
            color=colors,
            edgecolor="black",
            linewidth=1.5,
        )
        
        # Add count labels on bars
        for bar, count in zip(bars, [counts.get(0, 0), counts.get(1, 0)]):
            height = bar.get_height()
            ax.annotate(
                f'{count:,}\n({100*count/len(df):.1f}%)',
                xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=11, fontweight='bold',
            )
        
        ax.set_title(f"{split.capitalize()} Set (n={len(df):,})", fontsize=14, fontweight='bold')
        ax.set_ylabel("Count" if ax == axes[0] else "")
        ax.set_ylim(0, max(counts) * 1.2)
    
    fig.suptitle("Class Distribution Across Splits", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_dir / "class_distribution.png", dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    plt.close()
    
    print(f"✓ Saved class distribution plot to {save_dir / 'class_distribution.png'}")


def plot_text_length_distribution(
    data_dir: Path,
    save_dir: Path,
    show: bool = False,
):
    """Plot text length distributions (words and characters) per sentiment class."""
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load train data
    df = pd.read_csv(data_dir / "train.csv")
    df = df.dropna(subset=["text"])
    
    # Compute lengths
    df["word_count"] = df["text"].apply(lambda x: len(str(x).split()))
    df["char_count"] = df["text"].apply(lambda x: len(str(x)))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = {0: "#e74c3c", 1: "#2ecc71"}
    labels = {0: "Negative", 1: "Positive"}
    
    # Word count distribution - histogram
    for label in [0, 1]:
        subset = df[df["label"] == label]
        axes[0, 0].hist(
            subset["word_count"], bins=50, alpha=0.6,
            label=labels[label], color=colors[label], edgecolor="white"
        )
    axes[0, 0].set_xlabel("Word Count")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("Word Count Distribution by Sentiment")
    axes[0, 0].legend()
    axes[0, 0].axvline(df["word_count"].median(), color="black", linestyle="--", label="Median")
    
    # Word count - boxplot
    df_melted = df[["label", "word_count"]].copy()
    df_melted["Sentiment"] = df_melted["label"].map(labels)
    sns.boxplot(x="Sentiment", y="word_count", data=df_melted, ax=axes[0, 1], palette=colors.values())
    axes[0, 1].set_title("Word Count by Sentiment Class")
    axes[0, 1].set_ylabel("Word Count")
    
    # Character count distribution - histogram
    for label in [0, 1]:
        subset = df[df["label"] == label]
        axes[1, 0].hist(
            subset["char_count"], bins=50, alpha=0.6,
            label=labels[label], color=colors[label], edgecolor="white"
        )
    axes[1, 0].set_xlabel("Character Count")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].set_title("Character Count Distribution by Sentiment")
    axes[1, 0].legend()
    
    # Character count - boxplot
    df_melted = df[["label", "char_count"]].copy()
    df_melted["Sentiment"] = df_melted["label"].map(labels)
    sns.boxplot(x="Sentiment", y="char_count", data=df_melted, ax=axes[1, 1], palette=colors.values())
    axes[1, 1].set_title("Character Count by Sentiment Class")
    axes[1, 1].set_ylabel("Character Count")
    
    fig.suptitle("Text Length Analysis", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_dir / "text_length_distribution.png", dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    plt.close()
    
    # Print statistics
    print("\nText Length Statistics:")
    print("-" * 50)
    for label, name in labels.items():
        subset = df[df["label"] == label]
        print(f"{name}:")
        print(f"  Words: mean={subset['word_count'].mean():.1f}, median={subset['word_count'].median():.0f}, "
              f"std={subset['word_count'].std():.1f}")
        print(f"  Chars: mean={subset['char_count'].mean():.1f}, median={subset['char_count'].median():.0f}")
    
    print(f"\n✓ Saved text length plots to {save_dir / 'text_length_distribution.png'}")


def plot_most_frequent_words(
    data_dir: Path,
    save_dir: Path,
    top_k: int = 20,
    show: bool = False,
):
    """Plot most frequent words per sentiment class."""
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Import tokenizer - use proper spaCy tokenization for consistency
    try:
        from src.preprocessing.text import tokenize
    except ImportError:
        import warnings
        warnings.warn("Could not import tokenize from src.preprocessing.text, using simple split")
        tokenize = lambda x: str(x).lower().split()
    
    # Load train data
    df = pd.read_csv(data_dir / "train.csv")
    df = df.dropna(subset=["text"])
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    colors = {0: "#e74c3c", 1: "#2ecc71"}
    labels = {0: "Negative", 1: "Positive"}
    
    for ax, label in zip(axes, [0, 1]):
        subset = df[df["label"] == label]
        
        # Tokenize and count
        all_tokens = []
        for text in subset["text"]:
            all_tokens.extend(tokenize(str(text)))
        
        # Get most common
        word_counts = Counter(all_tokens)
        most_common = word_counts.most_common(top_k)
        
        words = [w for w, c in most_common]
        counts = [c for w, c in most_common]
        
        # Horizontal bar chart
        y_pos = np.arange(len(words))
        ax.barh(y_pos, counts, color=colors[label], edgecolor="white")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words)
        ax.invert_yaxis()
        ax.set_xlabel("Frequency")
        ax.set_title(f"Top {top_k} Words - {labels[label]} Reviews", fontsize=14, fontweight='bold')
        
        # Add count labels
        for i, count in enumerate(counts):
            ax.text(count + max(counts)*0.01, i, f"{count:,}", va='center', fontsize=9)
    
    fig.suptitle("Most Frequent Words by Sentiment Class", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_dir / "most_frequent_words.png", dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    plt.close()
    
    print(f"✓ Saved word frequency plots to {save_dir / 'most_frequent_words.png'}")


def generate_eda_visualizations(data_dir: Path, save_dir: Path, show: bool = False):
    """Generate all EDA visualizations."""
    print("\n" + "="*60)
    print("Generating EDA Visualizations")
    print("="*60)
    
    eda_dir = save_dir / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)
    
    plot_class_distribution(data_dir, eda_dir, show)
    plot_text_length_distribution(data_dir, eda_dir, show)
    plot_most_frequent_words(data_dir, eda_dir, show=show)
    
    print(f"\n✓ All EDA visualizations saved to {eda_dir}")


# =============================================================================
# Comparative Visualizations
# =============================================================================

def load_experiment_results(results_dir: Path) -> Dict:
    """Load experiment results from results directory.
    
    Expects structure:
        results/
            experiment_name_1/
                history.json
                metrics.json
                config.json
            experiment_name_2/
                ...
    """
    results = {}
    
    for exp_dir in results_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        
        history_path = exp_dir / "history.json"
        metrics_path = exp_dir / "metrics.json"
        config_path = exp_dir / "config.json"
        
        if not history_path.exists():
            continue
        
        exp_name = exp_dir.name
        results[exp_name] = {}
        
        # Load history
        with open(history_path) as f:
            results[exp_name]["history"] = json.load(f)
        
        # Load metrics if exists
        if metrics_path.exists():
            with open(metrics_path) as f:
                results[exp_name]["metrics"] = json.load(f)
        
        # Load config if exists
        if config_path.exists():
            with open(config_path) as f:
                results[exp_name]["config"] = json.load(f)
    
    return results


def plot_model_comparison(
    results: Dict,
    metric: str,
    title: str,
    save_path: Path,
    show: bool = False,
):
    """Plot comparison of multiple models on a metric."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    for (name, data), color in zip(results.items(), colors):
        history = data.get("history", {})
        if metric not in history:
            continue
        
        epochs = range(1, len(history[metric]) + 1)
        
        # Clean up name for legend
        display_name = name.replace("_", " ").replace("-", " ")
        if len(display_name) > 40:
            display_name = display_name[:37] + "..."
        
        ax.plot(epochs, history[metric], label=display_name, linewidth=2, color=color)
    
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    plt.close()


def plot_augmentation_comparison(
    results: Dict,
    save_dir: Path,
    show: bool = False,
):
    """Compare same experiment with and without augmentation on the same plot.
    
    Expects naming convention:
        - Base experiment: `expname_noaug`
        - With augmentation: `expname_aug_augtype`
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse experiment names to find pairs
    # Pattern: base experiments contain "_noaug"
    base_experiments = {}
    aug_experiments = {}

    # 1. First pass: Identify all base experiments
    for name, data in results.items():
        if "_noaug" in name:
            # Base experiment: base_name_noaug
            base_name = name.split("_noaug")[0].rstrip("_")
            base_experiments[base_name] = data

    # 2. Second pass: Identify augmented experiments based on known base names
    for name, data in results.items():
        # Skip if it is a base experiment
        if "_noaug" in name:
            continue
            
        if "_p0." in name:
            # Candidate structure: base_name_augtype_p0.x
            # We try to match the start of the string with known base names
            matched_base = None
            # Find longest matching base name to avoid partial matches
            # (e.g. valid match 'bilstm_attention_1layer' vs 'bilstm_attention')
            sorted_base_names = sorted(base_experiments.keys(), key=len, reverse=True)
            
            for base_name in sorted_base_names:
                # Check if this name starts with base_name + "_"
                if name.startswith(base_name + "_"):
                    matched_base = base_name
                    break
            
            if matched_base:
                # successfully identified base model
                # name is like: {base_name}_{aug_type}_p0.x
                # aug_part is: {aug_type}_p0.x
                aug_part = name[len(matched_base)+1:]
                
                # split off the prob part
                if "_p0." in aug_part:
                    aug_type = aug_part.split("_p0.")[0]
                    
                    if matched_base not in aug_experiments:
                        aug_experiments[matched_base] = []
                    aug_experiments[matched_base].append((name, aug_type, data))
    
    # Find matching pairs
    pairs = []
    for base_name, base_data in base_experiments.items():
        if base_name in aug_experiments:
            for aug_name, aug_type, aug_data in aug_experiments[base_name]:
                pairs.append({
                    "base_name": base_name,
                    "base_data": base_data,
                    "aug_name": aug_name,
                    "aug_type": aug_type,
                    "aug_data": aug_data,
                })
    
    if not pairs:
        print("⚠️  No matching experiment pairs found (base vs augmented)")
        print("   Expected naming: 'expname' (base) and 'expname_aug_augtype' (augmented)")
        print(f"   Found base experiments: {list(base_experiments.keys())}")
        print(f"   Found aug experiments: {list(aug_experiments.keys())}")
        return
    
    print(f"Found {len(pairs)} experiment pairs to compare:")
    for pair in pairs:
        print(f"  - {pair['base_name']} vs {pair['aug_name']} ({pair['aug_type']})")
    
    # Plot each pair on the same plot
    for metric in ["val_loss", "val_f1", "val_acc"]:
        metric_label = metric.replace("_", " ").title()
        
        # Determine grid size
        n_pairs = len(pairs)
        n_cols = min(2, n_pairs)
        n_rows = (n_pairs + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows), squeeze=False)
        axes = axes.flatten()
        
        for idx, pair in enumerate(pairs):
            ax = axes[idx]
            
            base_history = pair["base_data"].get("history", {})
            aug_history = pair["aug_data"].get("history", {})
            
            # Plot base experiment
            if metric in base_history:
                epochs = range(1, len(base_history[metric]) + 1)
                ax.plot(epochs, base_history[metric], 
                       label="Without Augmentation", linewidth=2, color="#3498db", linestyle="-")
            
            # Plot augmented experiment
            if metric in aug_history:
                epochs = range(1, len(aug_history[metric]) + 1)
                ax.plot(epochs, aug_history[metric],
                       label=f"With {pair['aug_type']}", linewidth=2, color="#e74c3c", linestyle="--")
            
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric_label)
            
            # Create readable title from base name
            title_name = pair["base_name"].replace("_", " ")
            if len(title_name) > 30:
                title_name = title_name[:27] + "..."
            ax.set_title(title_name, fontsize=11, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for idx in range(len(pairs), len(axes)):
            axes[idx].set_visible(False)
        
        fig.suptitle(f"Augmentation Impact: {metric_label}", fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(save_dir / f"augmentation_comparison_{metric}.png", dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        plt.close()
    
    # Also create a summary plot with all pairs on one figure (for val_f1)
    if len(pairs) > 1:
        fig, ax = plt.subplots(figsize=(12, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, len(pairs)))
        
        for pair, color in zip(pairs, colors):
            base_history = pair["base_data"].get("history", {})
            aug_history = pair["aug_data"].get("history", {})
            
            base_label = pair["base_name"].split("_")[0]  # Short name (model type)
            
            if "val_f1" in base_history:
                epochs = range(1, len(base_history["val_f1"]) + 1)
                ax.plot(epochs, base_history["val_f1"], 
                       label=f"{base_label} (no aug)", linewidth=2, color=color, linestyle="-")
            
            if "val_f1" in aug_history:
                epochs = range(1, len(aug_history["val_f1"]) + 1)
                ax.plot(epochs, aug_history["val_f1"],
                       label=f"{base_label} ({pair['aug_type']})", linewidth=2, color=color, linestyle="--")
        
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Validation F1", fontsize=12)
        ax.set_title("All Experiments: With vs Without Augmentation", fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / "augmentation_comparison_all_val_f1.png", dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        plt.close()
    
    print(f"✓ Saved augmentation comparison plots to {save_dir}")


def create_results_table(
    results: Dict,
    save_path: Path,
    show: bool = True,
):
    """Create summary table of all experiment results."""
    rows = []
    
    for name, data in results.items():
        config = data.get("config", {})
        metrics = data.get("metrics", {})
        history = data.get("history", {})
        
        # Extract overall metrics
        overall = metrics.get("overall", {})
        
        row = {
            "Experiment": name,
            "Model": config.get("model", "N/A"),
            "Hidden": config.get("hidden_dim", "N/A"),
            "Layers": config.get("num_layers", "N/A"),
            "Augment": config.get("augment", "none") or "none",
            "Test Acc": overall.get("accuracy", "N/A"),
            "Test F1": overall.get("f1", "N/A"),
            "Test Prec": overall.get("precision", "N/A"),
            "Test Rec": overall.get("recall", "N/A"),
            "Best Val Loss": min(history.get("val_loss", [float("inf")])) if history.get("val_loss") else "N/A",
            "Best Val F1": max(history.get("val_f1", [0])) if history.get("val_f1") else "N/A",
            "Epochs": len(history.get("train_loss", [])),
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Sort by Test F1 descending
    if "Test F1" in df.columns:
        df = df.sort_values("Test F1", ascending=False, key=lambda x: pd.to_numeric(x, errors='coerce'))
    
    # Format numeric columns
    for col in ["Test Acc", "Test F1", "Test Prec", "Test Rec", "Best Val Loss", "Best Val F1"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.4f}" if isinstance(x, float) else x)
    
    # Save to CSV
    df.to_csv(save_path.with_suffix(".csv"), index=False)
    
    # Save formatted markdown table
    try:
        with open(save_path.with_suffix(".md"), "w") as f:
            f.write("# Experiment Results Summary\n\n")
            f.write(df.to_markdown(index=False))
            f.write("\n")
    except ImportError:
        print("⚠️  Could not generate markdown table (tabulate not installed/updated). Skipped.")
    except Exception as e:
        print(f"⚠️  Could not generate markdown table: {e}")
    
    if show:
        print("\n" + "="*80)
        print("RESULTS SUMMARY")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80)
    
    print(f"\n✓ Saved results table to {save_path.with_suffix('.csv')}")
    
    return df


def plot_final_metrics_comparison(
    results: Dict,
    save_dir: Path,
    show: bool = False,
):
    """Bar chart comparing final test metrics across experiments."""
    save_dir.mkdir(parents=True, exist_ok=True)
    
    experiments = []
    accuracies = []
    f1_scores = []
    precisions = []
    recalls = []
    
    for name, data in results.items():
        metrics = data.get("metrics", {}).get("overall", {})
        if not metrics:
            continue
        
        experiments.append(name.split("_")[0])  # Short name
        accuracies.append(metrics.get("accuracy", 0))
        f1_scores.append(metrics.get("f1", 0))
        precisions.append(metrics.get("precision", 0))
        recalls.append(metrics.get("recall", 0))
    
    if not experiments:
        print("⚠️  No experiments with test metrics found")
        return
    
    x = np.arange(len(experiments))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(max(12, len(experiments) * 2), 6))
    
    bars1 = ax.bar(x - 1.5*width, accuracies, width, label='Accuracy', color='#3498db')
    bars2 = ax.bar(x - 0.5*width, f1_scores, width, label='F1 Score', color='#2ecc71')
    bars3 = ax.bar(x + 0.5*width, precisions, width, label='Precision', color='#e74c3c')
    bars4 = ax.bar(x + 1.5*width, recalls, width, label='Recall', color='#9b59b6')
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Test Metrics Comparison Across Models', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(experiments, rotation=45, ha='right')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8, rotation=90)
    
    plt.tight_layout()
    plt.savefig(save_dir / "metrics_comparison_bar.png", dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    plt.close()
    
    print(f"✓ Saved metrics comparison bar chart to {save_dir / 'metrics_comparison_bar.png'}")


def generate_comparison_visualizations(
    results_dir: Path,
    save_dir: Path,
    show: bool = False,
):
    """Generate all comparison visualizations from experiment results."""
    print("\n" + "="*60)
    print("Generating Comparison Visualizations")
    print("="*60)
    
    # Load results
    results = load_experiment_results(results_dir)
    
    if not results:
        print(f"⚠️  No experiment results found in {results_dir}")
        return
    
    print(f"Found {len(results)} experiments:")
    for name in results:
        print(f"  - {name}")
    
    comparison_dir = save_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Model comparison curves
    for metric in ["val_loss", "val_f1", "val_acc"]:
        plot_model_comparison(
            results, metric,
            title=f"Model Comparison - {metric.replace('_', ' ').title()}",
            save_path=comparison_dir / f"model_comparison_{metric}.png",
            show=show,
        )
    print(f"✓ Saved model comparison plots")
    
    # 2. Augmentation comparison
    plot_augmentation_comparison(results, comparison_dir / "augmentation", show)
    
    # 3. Final metrics bar chart
    plot_final_metrics_comparison(results, comparison_dir, show)
    
    # 4. Results summary table
    create_results_table(results, comparison_dir / "results_summary", show=True)
    
    print(f"\n✓ All comparison visualizations saved to {comparison_dir}")


# =============================================================================
# Main Entry Point
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate visualizations for ML assignment report",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Mode selection
    parser.add_argument("--eda", action="store_true", help="Generate EDA visualizations")
    parser.add_argument("--compare", action="store_true", help="Generate comparison visualizations")
    parser.add_argument("--all", action="store_true", help="Generate all visualizations")
    
    # Paths
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Data directory")
    parser.add_argument("--results_dir", type=str, default="results", help="Results directory")
    parser.add_argument("--save_dir", type=str, default="plots", help="Output directory for plots")
    
    # Options
    parser.add_argument("--show", action="store_true", help="Display plots interactively")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    save_dir = Path(args.save_dir)
    
    if args.all:
        args.eda = True
        args.compare = True
    
    if not args.eda and not args.compare:
        print("No mode selected. Use --eda, --compare, or --all")
        print("Run with --help for more information")
        return
    
    if args.eda:
        generate_eda_visualizations(data_dir, save_dir, args.show)
    
    if args.compare:
        generate_comparison_visualizations(results_dir, save_dir, args.show)
    
    print("\n" + "="*60)
    print("✓ Visualization generation complete!")
    print(f"  Output directory: {save_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
