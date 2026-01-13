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
from matplotlib.ticker import MaxNLocator
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
        print("⚠️  WARNING: Could not import tokenize from src.preprocessing.text!")
        print("   Using simple .split() fallback - results may be inaccurate.")
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
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # Force integer epochs
    
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
    
    Expects naming conventions:
        - Base experiment (no aug): `expname_noaug`
        - Offline balanced augmentation: `expname_balanced`
        - Offline expanded augmentation: `expname_expanded`
        - Online augmentation: `expname_aug_augtype_p0.x`
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse experiment names to find pairs
    base_experiments = {}  # base_name -> data
    aug_experiments = {}   # base_name -> [(full_name, aug_type, data), ...]

    # 1. First pass: Identify all base experiments (contain "_noaug")
    for name, data in results.items():
        if "_noaug" in name:
            # Base experiment: base_name_noaug
            base_name = name.replace("_noaug", "")
            base_experiments[base_name] = data

    # 2. Second pass: Identify augmented experiments based on known base names
    for name, data in results.items():
        # Skip if it is a base experiment
        if "_noaug" in name:
            continue
        
        # Find longest matching base name to avoid partial matches
        sorted_base_names = sorted(base_experiments.keys(), key=len, reverse=True)
        matched_base = None
        
        for base_name in sorted_base_names:
            # Check if this name starts with base_name + "_"
            if name.startswith(base_name + "_"):
                matched_base = base_name
                break
        
        if not matched_base:
            continue
        
        # Get the suffix after base name
        suffix = name[len(matched_base)+1:]  # e.g., "balanced", "expanded", "random_swap_p0.3"
        
        # Determine augmentation type
        aug_type = None
        
        # Check for offline augmentation patterns
        if suffix in ("balanced", "expanded"):
            aug_type = f"offline_{suffix}"
        # Check for online augmentation pattern (_p0.x)
        elif "_p0." in suffix:
            aug_type = suffix.split("_p0.")[0]  # e.g., "random_swap"
        
        if aug_type:
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
        print("   Expected naming patterns:")
        print("     - Base: 'expname_noaug'")
        print("     - Offline: 'expname_balanced' or 'expname_expanded'")
        print("     - Online: 'expname_augtype_p0.x'")
        print(f"   Found base experiments: {list(base_experiments.keys())}")
        print(f"   Found aug experiments: {list(aug_experiments.keys())}")
        return
    
    print(f"Found {len(pairs)} experiment pairs to compare:")
    for pair in pairs:
        print(f"  - {pair['base_name']} vs {pair['aug_name']} ({pair['aug_type']})")
    
    # Group pairs by base experiment for multi-augmentation comparison
    pairs_by_base = {}
    for pair in pairs:
        base = pair["base_name"]
        if base not in pairs_by_base:
            pairs_by_base[base] = []
        pairs_by_base[base].append(pair)
    
    # Plot each base experiment with ALL its augmentation variants
    for metric in ["val_loss", "val_f1", "val_acc"]:
        metric_label = metric.replace("_", " ").title()
        
        # Determine grid size based on number of base experiments
        n_bases = len(pairs_by_base)
        n_cols = min(2, n_bases)
        n_rows = (n_bases + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows), squeeze=False)
        axes = axes.flatten()
        
        # Color palette for different augmentation types
        aug_colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        for idx, (base_name, base_pairs) in enumerate(pairs_by_base.items()):
            ax = axes[idx]
            
            # Plot base experiment (no augmentation)
            base_history = base_pairs[0]["base_data"].get("history", {})
            if metric in base_history:
                epochs = range(1, len(base_history[metric]) + 1)
                ax.plot(epochs, base_history[metric], 
                       label="No Augmentation", linewidth=2, color="black", linestyle="-")
            
            # Plot all augmented variants
            for i, pair in enumerate(base_pairs):
                aug_history = pair["aug_data"].get("history", {})
                if metric in aug_history:
                    epochs = range(1, len(aug_history[metric]) + 1)
                    
                    # Use different line styles for offline vs online
                    if pair["aug_type"].startswith("offline_"):
                        linestyle = "--"
                        label = pair["aug_type"].replace("offline_", "").title()
                    else:
                        linestyle = "-."
                        label = pair["aug_type"].replace("_", " ").title()
                    
                    ax.plot(epochs, aug_history[metric],
                           label=label, linewidth=2, color=aug_colors[i % 10], linestyle=linestyle)
            
            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric_label)
            
            # Create readable title from base name
            title_name = base_name.replace("_", " ")
            if len(title_name) > 30:
                title_name = title_name[:27] + "..."
            ax.set_title(title_name, fontsize=11, fontweight='bold')
            ax.legend(fontsize=8, loc='best')
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # Force integer epochs
        
        # Hide empty subplots
        for idx in range(len(pairs_by_base), len(axes)):
            axes[idx].set_visible(False)
        
        fig.suptitle(f"Augmentation Impact: {metric_label}", fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(save_dir / f"augmentation_comparison_{metric}.png", dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        plt.close()
    
    # Also create a summary plot with all experiments on one figure (for val_f1)
    if len(pairs_by_base) > 0:
        fig, ax = plt.subplots(figsize=(14, 7))
        
        line_idx = 0
        colors = plt.cm.tab20(np.linspace(0, 1, 20))
        
        for base_name, base_pairs in pairs_by_base.items():
            base_label = base_name.replace("_", " ")[:15]  # Short name
            
            # Plot base (no aug)
            base_history = base_pairs[0]["base_data"].get("history", {})
            if "val_f1" in base_history:
                epochs = range(1, len(base_history["val_f1"]) + 1)
                ax.plot(epochs, base_history["val_f1"], 
                       label=f"{base_label} (no aug)", linewidth=2, 
                       color=colors[line_idx % 20], linestyle="-")
                line_idx += 1
            
            # Plot all augmented variants
            for pair in base_pairs:
                aug_history = pair["aug_data"].get("history", {})
                if "val_f1" in aug_history:
                    epochs = range(1, len(aug_history["val_f1"]) + 1)
                    aug_label = pair["aug_type"].replace("offline_", "").replace("_", " ")
                    
                    linestyle = "--" if pair["aug_type"].startswith("offline_") else "-."
                    
                    ax.plot(epochs, aug_history["val_f1"],
                           label=f"{base_label} ({aug_label})", linewidth=2, 
                           color=colors[line_idx % 20], linestyle=linestyle)
                    line_idx += 1
        
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Validation F1", fontsize=12)
        ax.set_title("All Experiments: Augmentation Comparison", fontsize=14, fontweight='bold')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # Force integer epochs
        
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
    """Generate multiple comparison visualizations for final test metrics."""
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect data
    rows = []
    for name, data in results.items():
        metrics = data.get("metrics", {}).get("overall", {})
        if not metrics:
            continue
        rows.append({
            "Experiment": name,
            "Accuracy": metrics.get("accuracy", 0),
            "F1": metrics.get("f1", 0),
            "Precision": metrics.get("precision", 0),
            "Recall": metrics.get("recall", 0),
        })
    
    if not rows:
        print("⚠️  No experiments with test metrics found")
        return
    
    df = pd.DataFrame(rows)
    
    # ==========================================================================
    # 1. Sorted horizontal bar charts - one per metric (most useful!)
    # ==========================================================================
    for metric in ["F1", "Accuracy", "Precision", "Recall"]:
        fig, ax = plt.subplots(figsize=(12, max(6, len(df) * 0.4)))
        
        # Sort by metric value
        df_sorted = df.sort_values(metric, ascending=True)
        
        # Color gradient based on value
        colors = plt.cm.RdYlGn(df_sorted[metric])
        
        y_pos = np.arange(len(df_sorted))
        bars = ax.barh(y_pos, df_sorted[metric], color=colors, edgecolor='black', linewidth=0.5)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_sorted["Experiment"], fontsize=9)
        ax.set_xlabel(metric, fontsize=12)
        ax.set_title(f"Test {metric} Comparison (sorted)", fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1.05)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar, val in zip(bars, df_sorted[metric]):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{val:.4f}', va='center', fontsize=9)
        
        # Add vertical line at mean
        mean_val = df_sorted[metric].mean()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.text(mean_val + 0.01, len(df_sorted) - 0.5, f'Mean: {mean_val:.4f}', 
               color='red', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_dir / f"comparison_{metric.lower()}_sorted.png", dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
    
    # ==========================================================================
    # 2. Heatmap - experiments vs metrics
    # ==========================================================================
    fig, ax = plt.subplots(figsize=(8, max(8, len(df) * 0.35)))
    
    # Sort by F1 for consistent ordering
    df_sorted = df.sort_values("F1", ascending=False)
    
    heatmap_data = df_sorted[["Accuracy", "F1", "Precision", "Recall"]].values
    
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.4f',
        cmap='RdYlGn',
        vmin=0.5,
        vmax=1.0,
        xticklabels=["Accuracy", "F1", "Precision", "Recall"],
        yticklabels=df_sorted["Experiment"],
        ax=ax,
        cbar_kws={'label': 'Score'},
        linewidths=0.5,
    )
    
    ax.set_title("Test Metrics Heatmap (sorted by F1)", fontsize=14, fontweight='bold')
    ax.set_xlabel("")
    ax.set_ylabel("")
    
    plt.tight_layout()
    plt.savefig(save_dir / "comparison_heatmap.png", dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
    
    # ==========================================================================
    # 3. Augmentation effect comparison (grouped by base model)
    # ==========================================================================
    # Parse experiment names to group by base model
    groups = {}
    for _, row in df.iterrows():
        name = row["Experiment"]
        
        # Determine base name and augmentation type
        if "_noaug" in name:
            base = name.replace("_noaug", "")
            aug_type = "No Aug"
        elif "_balanced" in name:
            base = name.replace("_balanced", "")
            aug_type = "Balanced"
        elif "_expanded" in name:
            base = name.replace("_expanded", "")
            aug_type = "Expanded"
        elif "_p0." in name:
            # Online augmentation: base_augtype_p0.x
            parts = name.rsplit("_p0.", 1)
            prefix = parts[0]
            # Find the augmentation type - check from longest to shortest to avoid partial matches
            known_augs = [
                "eda_plus", "eda",  # EDA variants (check eda_plus before eda)
                "random_swap", "random_delete", "random_insert", 
                "synonym", "back_translate", "contextual",
                "bert_insert", "bert_substitute",  # BERT-based
            ]
            for aug in known_augs:
                if f"_{aug}" in prefix:
                    base = prefix.replace(f"_{aug}", "")
                    aug_type = aug.replace("_", " ").title()
                    break
            else:
                # Unknown augmentation type - try to extract it anyway
                # Pattern: base_AUGTYPE_p0.x -> extract AUGTYPE
                parts_underscore = prefix.rsplit("_", 1)
                if len(parts_underscore) == 2:
                    base = parts_underscore[0]
                    aug_type = parts_underscore[1].replace("_", " ").title()
                else:
                    continue
        else:
            continue
        
        if base not in groups:
            groups[base] = {}
        groups[base][aug_type] = row["F1"]
    
    if groups:
        fig, ax = plt.subplots(figsize=(12, max(6, len(groups) * 1.2)))
        
        # Prepare data for grouped bar chart
        base_models = list(groups.keys())
        aug_types = sorted(set(aug for g in groups.values() for aug in g.keys()))
        
        x = np.arange(len(base_models))
        width = 0.8 / len(aug_types)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(aug_types)))
        
        for i, aug_type in enumerate(aug_types):
            values = [groups[base].get(aug_type, 0) for base in base_models]
            offset = (i - len(aug_types)/2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=aug_type, color=colors[i], edgecolor='black', linewidth=0.5)
            
            # Add value labels
            for bar, val in zip(bars, values):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                           f'{val:.3f}', ha='center', va='bottom', fontsize=8, rotation=90)
        
        ax.set_xlabel('Base Model', fontsize=12)
        ax.set_ylabel('Test F1 Score', fontsize=12)
        ax.set_title('Augmentation Effect on F1 Score by Model', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([b.replace("_", " ") for b in base_models], rotation=45, ha='right', fontsize=10)
        ax.legend(title="Augmentation", loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)
        ax.set_ylim(0, 1.15)  # Extra space for rotated labels
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_dir / "comparison_augmentation_effect.png", dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
    
    # ==========================================================================
    # 4. Top N models comparison (radar/spider chart for top 5)
    # ==========================================================================
    top_n = min(5, len(df))
    df_top = df.nlargest(top_n, "F1")
    
    metrics = ["Accuracy", "F1", "Precision", "Recall"]
    num_vars = len(metrics)
    
    # Compute angle for each metric
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    colors = plt.cm.tab10(np.linspace(0, 1, top_n))
    
    for idx, (_, row) in enumerate(df_top.iterrows()):
        values = [row[m] for m in metrics]
        values += values[:1]  # Complete the loop
        
        label = row["Experiment"]
        if len(label) > 25:
            label = label[:22] + "..."
        
        ax.plot(angles, values, 'o-', linewidth=2, label=label, color=colors[idx])
        ax.fill(angles, values, alpha=0.1, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylim(0.5, 1.0)
    ax.set_title(f"Top {top_n} Models Comparison (by F1)", fontsize=14, fontweight='bold', y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_dir / "comparison_top_models_radar.png", dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
    
    print(f"✓ Saved comparison plots:")
    print(f"   - comparison_<metric>_sorted.png (4 files)")
    print(f"   - comparison_heatmap.png")
    print(f"   - comparison_augmentation_effect.png")
    print(f"   - comparison_top_models_radar.png")


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
