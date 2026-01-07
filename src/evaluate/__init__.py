"""Evaluation utilities for sentiment classification."""
from .metrics import (
    compute_metrics,
    compute_detailed_metrics,
    print_classification_report,
    MetricsTracker,
)
from .visualize import (
    set_plot_style,
    plot_training_curves,
    plot_metrics_curves,
    plot_confusion_matrix,
    plot_comparison,
    plot_augmentation_comparison,
    plot_learning_rate,
    plot_attention_weights,
    create_results_summary,
)

# Note: generate_visualizations is a standalone script, run with:
#   python -m src.evaluate.generate_visualizations --help

__all__ = [
    # Metrics
    "compute_metrics",
    "compute_detailed_metrics",
    "print_classification_report",
    "MetricsTracker",
    # Visualization
    "set_plot_style",
    "plot_training_curves",
    "plot_metrics_curves",
    "plot_confusion_matrix",
    "plot_comparison",
    "plot_augmentation_comparison",
    "plot_learning_rate",
    "plot_attention_weights",
    "create_results_summary",
]
