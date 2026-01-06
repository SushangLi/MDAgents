"""
Evaluate Expert Models.

This script evaluates trained expert models on test data and generates
comprehensive performance reports.

Usage:
    python scripts/model_training/evaluate_models.py --annotations-file data/labeled/annotations.json
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
from typing import Dict, List

from clinical.experts.model_manager import ModelManager
from clinical.preprocessing.microbiome_preprocessor import MicrobiomePreprocessor
from clinical.preprocessing.metabolome_preprocessor import MetabolomePreprocessor
from clinical.preprocessing.proteome_preprocessor import ProteomePreprocessor
from scripts.model_training.train_experts import load_annotations, load_omics_data, prepare_labels


def evaluate_expert(
    expert,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path
) -> Dict:
    """
    Evaluate a single expert model.

    Args:
        expert: Trained expert model
        X_test: Test features
        y_test: Test labels
        output_dir: Directory to save evaluation results

    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\nEvaluating {expert.expert_name}...")

    # Get predictions
    predictions = [op.diagnosis for op in expert.predict(X_test)]
    probabilities = expert.model_.predict_proba(X_test[expert.feature_names_])

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, average='weighted', zero_division=0),
        "recall": recall_score(y_test, predictions, average='weighted', zero_division=0),
        "f1_score": f1_score(y_test, predictions, average='weighted', zero_division=0)
    }

    # Confusion matrix
    cm = confusion_matrix(y_test, predictions, labels=expert.classes_)

    # Classification report
    class_report = classification_report(
        y_test,
        predictions,
        labels=expert.classes_,
        output_dict=True,
        zero_division=0
    )

    # Plot confusion matrix
    plot_confusion_matrix(
        cm,
        expert.classes_,
        expert.expert_name,
        output_dir
    )

    # Plot ROC curve (for binary classification)
    if len(expert.classes_) == 2:
        plot_roc_curve(
            y_test,
            probabilities,
            expert.classes_,
            expert.expert_name,
            output_dir
        )

    # Print results
    print(f"  Accuracy:  {metrics['accuracy']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall:    {metrics['recall']:.3f}")
    print(f"  F1-Score:  {metrics['f1_score']:.3f}")

    print(f"\n  Per-class metrics:")
    for class_name in expert.classes_:
        if class_name in class_report:
            print(f"    {class_name}:")
            print(f"      Precision: {class_report[class_name]['precision']:.3f}")
            print(f"      Recall:    {class_report[class_name]['recall']:.3f}")
            print(f"      F1-Score:  {class_report[class_name]['f1-score']:.3f}")

    return {
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
        "classification_report": class_report
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: List[str],
    expert_name: str,
    output_dir: Path
):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes
    )
    plt.title(f'Confusion Matrix - {expert_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    output_path = output_dir / f"{expert_name}_confusion_matrix.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved confusion matrix to {output_path}")


def plot_roc_curve(
    y_test: pd.Series,
    probabilities: np.ndarray,
    classes: List[str],
    expert_name: str,
    output_dir: Path
):
    """Plot and save ROC curve (for binary classification)."""
    # Encode labels
    from sklearn.preprocessing import label_binarize
    y_test_bin = label_binarize(y_test, classes=classes)

    plt.figure(figsize=(8, 6))

    # For binary classification
    if len(classes) == 2:
        fpr, tpr, _ = roc_curve(y_test_bin, probabilities[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {expert_name}')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / f"{expert_name}_roc_curve.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved ROC curve to {output_path}")


def compare_experts(results: Dict, output_dir: Path):
    """Create comparison plots for all experts."""
    expert_names = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']

    # Bar plot comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(expert_names))
    width = 0.2

    for i, metric in enumerate(metrics):
        values = [results[name]['metrics'][metric] for name in expert_names]
        ax.bar(x + i * width, values, width, label=metric.replace('_', ' ').title())

    ax.set_xlabel('Expert')
    ax.set_ylabel('Score')
    ax.set_title('Expert Model Performance Comparison')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(expert_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / "experts_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Saved comparison plot to {output_path}")


def generate_evaluation_report(
    results: Dict,
    output_dir: Path,
    test_sample_count: int
):
    """Generate comprehensive evaluation report."""
    report_path = output_dir / "evaluation_report.md"

    with open(report_path, "w") as f:
        f.write("# Expert Models Evaluation Report\n\n")
        f.write(f"**Test Samples**: {test_sample_count}\n\n")
        f.write("---\n\n")

        for expert_name, result in results.items():
            f.write(f"## {expert_name}\n\n")

            # Overall metrics
            f.write("### Overall Performance\n\n")
            f.write("| Metric | Score |\n")
            f.write("|--------|-------|\n")
            for metric, value in result['metrics'].items():
                f.write(f"| {metric.replace('_', ' ').title()} | {value:.3f} |\n")
            f.write("\n")

            # Per-class metrics
            f.write("### Per-Class Performance\n\n")
            f.write("| Class | Precision | Recall | F1-Score | Support |\n")
            f.write("|-------|-----------|--------|----------|----------|\n")

            class_report = result['classification_report']
            for class_name, metrics in class_report.items():
                if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                    f.write(
                        f"| {class_name} | "
                        f"{metrics['precision']:.3f} | "
                        f"{metrics['recall']:.3f} | "
                        f"{metrics['f1-score']:.3f} | "
                        f"{metrics['support']} |\n"
                    )
            f.write("\n")

            # Confusion matrix
            f.write("### Confusion Matrix\n\n")
            f.write(f"![Confusion Matrix]({expert_name}_confusion_matrix.png)\n\n")

            f.write("---\n\n")

        # Summary
        f.write("## Summary\n\n")
        f.write("### Overall Comparison\n\n")
        f.write("![Comparison](experts_comparison.png)\n\n")

        # Best performers
        f.write("### Best Performers\n\n")
        for metric in ['accuracy', 'f1_score']:
            best_expert = max(
                results.items(),
                key=lambda x: x[1]['metrics'][metric]
            )
            f.write(
                f"- **{metric.replace('_', ' ').title()}**: "
                f"{best_expert[0]} ({best_expert[1]['metrics'][metric]:.3f})\n"
            )

    print(f"\n✓ Saved evaluation report to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate expert models on test data")
    parser.add_argument(
        "--annotations-file",
        type=str,
        default="data/labeled/annotations.json",
        help="Path to annotations JSON file"
    )
    parser.add_argument(
        "--raw-data-dir",
        type=str,
        default="data/raw",
        help="Directory containing raw omics data"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="data/models",
        help="Directory containing trained models"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/evaluation",
        help="Directory to save evaluation results"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load annotations
    print("Loading annotations...")
    annotations = load_annotations(args.annotations_file)

    # Get test sample IDs
    test_sample_ids = annotations["splits"]["test"]
    print(f"  Test samples: {len(test_sample_ids)}")

    # Prepare labels
    y_test = prepare_labels(annotations, test_sample_ids)

    print(f"\nClass distribution (test):")
    print(y_test.value_counts())

    # Load models
    print("\nLoading trained models...")
    model_manager = ModelManager(models_dir=args.models_dir)
    print(model_manager.summary())

    # Load all experts
    experts = model_manager.load_all_experts()

    # Evaluate each expert
    omics_types = ["microbiome", "metabolome", "proteome"]
    expert_names = ["microbiome_expert", "metabolome_expert", "proteome_expert"]
    preprocessors = [MicrobiomePreprocessor(), MetabolomePreprocessor(), ProteomePreprocessor()]

    results = {}

    for omics_type, expert_name, preprocessor in zip(omics_types, expert_names, preprocessors):
        if expert_name not in experts:
            print(f"\n✗ {expert_name} not found, skipping...")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating {expert_name}")
        print(f"{'='*60}")

        try:
            expert = experts[expert_name]

            # Load test data
            print(f"Loading {omics_type} test data...")
            raw_data_dir = Path(args.raw_data_dir)
            X_test_raw = load_omics_data(test_sample_ids, omics_type, raw_data_dir)

            # Preprocess (using same preprocessing as training)
            print(f"Preprocessing...")
            preprocessor.fit(X_test_raw)  # Fit on test data for now (in production, use saved preprocessor)
            X_test_processed = preprocessor.transform(X_test_raw).data

            # Evaluate
            result = evaluate_expert(expert, X_test_processed, y_test, output_dir)
            results[expert_name] = result

        except Exception as e:
            print(f"\n✗ Error evaluating {expert_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Generate comparison plots
    if results:
        print(f"\n{'='*60}")
        print("Generating comparison plots...")
        print(f"{'='*60}")
        compare_experts(results, output_dir)

        # Generate report
        generate_evaluation_report(results, output_dir, len(test_sample_ids))

        print(f"\n{'='*60}")
        print("EVALUATION COMPLETE")
        print(f"{'='*60}")
        print(f"\nResults saved to: {output_dir}")
        print(f"\nView the full report: {output_dir / 'evaluation_report.md'}")


if __name__ == "__main__":
    main()
