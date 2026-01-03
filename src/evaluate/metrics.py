"""Evaluation metrics for sentiment classification."""
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    average: str = "binary",
) -> Dict[str, float]:
    """Compute classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        average: Averaging method for F1 ("binary", "macro", "micro", "weighted")
    
    Returns:
        Dictionary with accuracy, precision, recall, F1
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average, zero_division=0),
    }


def compute_detailed_metrics(
    y_true: List[int],
    y_pred: List[int],
    class_names: Optional[List[str]] = None,
) -> Dict:
    """Compute detailed per-class metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Names for each class (default: ["Negative", "Positive"])
    
    Returns:
        Dictionary with overall and per-class metrics
    """
    if class_names is None:
        class_names = ["Negative", "Positive"]
    
    # Overall metrics
    overall = compute_metrics(y_true, y_pred)
    
    # Per-class metrics
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        "overall": overall,
        "per_class": {
            name: {
                "precision": report[name]["precision"],
                "recall": report[name]["recall"],
                "f1": report[name]["f1-score"],
                "support": report[name]["support"],
            }
            for name in class_names
        },
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=class_names, zero_division=0
        ),
    }


def print_classification_report(
    y_true: List[int],
    y_pred: List[int],
    class_names: Optional[List[str]] = None,
):
    """Print formatted classification report.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: Names for each class
    """
    if class_names is None:
        class_names = ["Negative", "Positive"]
    
    print("\n" + "=" * 60)
    print("Classification Report")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    
    print("\nConfusion Matrix:")
    print("-" * 30)
    cm = confusion_matrix(y_true, y_pred)
    
    # Pretty print confusion matrix
    header = "".join(f"{name:>12}" for name in class_names)
    print(f"{'Predicted:':>12}{header}")
    print(f"{'Actual:':<12}")
    for i, name in enumerate(class_names):
        row = "".join(f"{cm[i, j]:>12}" for j in range(len(class_names)))
        print(f"{name:>12}{row}")
    print("=" * 60)


class MetricsTracker:
    """Track and compare metrics across multiple experiments.
    
    Example:
        tracker = MetricsTracker()
        tracker.add_experiment("LSTM_baseline", y_true, y_pred_lstm)
        tracker.add_experiment("BiLSTM_attention", y_true, y_pred_bilstm)
        tracker.print_comparison()
        tracker.to_dataframe().to_csv("results.csv")
    """
    
    def __init__(self):
        self.experiments: Dict[str, Dict] = {}
    
    def add_experiment(
        self,
        name: str,
        y_true: List[int],
        y_pred: List[int],
        metadata: Optional[Dict] = None,
    ):
        """Add experiment results.
        
        Args:
            name: Experiment name
            y_true: Ground truth labels
            y_pred: Predicted labels
            metadata: Optional metadata (hyperparameters, etc.)
        """
        metrics = compute_detailed_metrics(y_true, y_pred)
        self.experiments[name] = {
            "metrics": metrics,
            "metadata": metadata or {},
            "predictions": y_pred,
            "labels": y_true,
        }
    
    def get_comparison_table(self) -> List[Dict]:
        """Get comparison table as list of dicts.
        
        Returns:
            List of dicts with experiment name and metrics
        """
        rows = []
        for name, data in self.experiments.items():
            row = {
                "Experiment": name,
                "Accuracy": data["metrics"]["overall"]["accuracy"],
                "Precision": data["metrics"]["overall"]["precision"],
                "Recall": data["metrics"]["overall"]["recall"],
                "F1": data["metrics"]["overall"]["f1"],
            }
            # Add metadata
            row.update(data["metadata"])
            rows.append(row)
        return rows
    
    def print_comparison(self):
        """Print comparison table."""
        rows = self.get_comparison_table()
        if not rows:
            print("No experiments added yet.")
            return
        
        # Header
        print("\n" + "=" * 80)
        print("Experiment Comparison")
        print("=" * 80)
        print(f"{'Experiment':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print("-" * 80)
        
        # Sort by F1 descending
        rows_sorted = sorted(rows, key=lambda x: x["F1"], reverse=True)
        
        for row in rows_sorted:
            print(
                f"{row['Experiment']:<25} "
                f"{row['Accuracy']:>10.4f} "
                f"{row['Precision']:>10.4f} "
                f"{row['Recall']:>10.4f} "
                f"{row['F1']:>10.4f}"
            )
        print("=" * 80)
    
    def to_dataframe(self):
        """Convert to pandas DataFrame.
        
        Returns:
            DataFrame with experiment comparison
        """
        import pandas as pd
        return pd.DataFrame(self.get_comparison_table())
    
    def save_to_csv(self, path: str):
        """Save comparison table to CSV.
        
        Args:
            path: Path to save CSV file
        """
        self.to_dataframe().to_csv(path, index=False)
