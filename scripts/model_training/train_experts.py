"""
Train Expert Models.

This script trains all three expert models (microbiome, metabolome, proteome)
on labeled multi-omics data.

Usage:
    python scripts/model_training/train_experts.py --data-dir data/labeled --output-dir data/models
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple

from clinical.experts.microbiome_expert import MicrobiomeExpert
from clinical.experts.metabolome_expert import MetabolomeExpert
from clinical.experts.proteome_expert import ProteomeExpert
from clinical.experts.model_manager import ModelManager
from clinical.preprocessing.microbiome_preprocessor import MicrobiomePreprocessor
from clinical.preprocessing.metabolome_preprocessor import MetabolomePreprocessor
from clinical.preprocessing.proteome_preprocessor import ProteomePreprocessor
from clinical.preprocessing.feature_engineering import FeatureEngineer
from clinical.preprocessing.quality_control import QualityController


def load_annotations(annotations_file: str) -> Dict:
    """Load annotations JSON file."""
    with open(annotations_file, "r") as f:
        return json.load(f)


def load_omics_data(
    sample_ids: list,
    omics_type: str,
    data_dir: Path
) -> pd.DataFrame:
    """
    Load omics data for multiple samples.

    Args:
        sample_ids: List of sample IDs
        omics_type: Type of omics ("microbiome", "metabolome", "proteome")
        data_dir: Directory containing omics data

    Returns:
        DataFrame with samples as rows and features as columns
    """
    omics_dir = data_dir / omics_type
    samples_data = []

    for sample_id in sample_ids:
        # Try CSV first
        file_path = omics_dir / f"{sample_id}.csv"
        if not file_path.exists():
            # Try Excel
            file_path = omics_dir / f"{sample_id}.xlsx"

        if file_path.exists():
            if file_path.suffix == ".csv":
                df = pd.read_csv(file_path, index_col=0)
            else:
                df = pd.read_excel(file_path, index_col=0)

            # Flatten to single row (assuming data is already in correct format)
            if len(df) == 1:
                sample_data = df.iloc[0]
            else:
                # If multiple rows, take mean
                sample_data = df.mean(axis=0)

            samples_data.append(sample_data)
        else:
            print(f"Warning: Data not found for {sample_id} in {omics_type}")

    # Combine all samples
    combined_df = pd.DataFrame(samples_data, index=sample_ids)
    return combined_df


def prepare_labels(annotations: Dict, sample_ids: list) -> pd.Series:
    """
    Extract labels from annotations.

    Args:
        annotations: Annotations dictionary
        sample_ids: List of sample IDs

    Returns:
        Series with sample IDs as index and diagnosis as values
    """
    labels = {}
    for sample_id in sample_ids:
        if sample_id in annotations["annotations"]:
            labels[sample_id] = annotations["annotations"][sample_id]["diagnosis"]

    return pd.Series(labels)


def train_expert(
    expert_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    use_grid_search: bool = False
) -> Tuple:
    """
    Train a single expert model.

    Args:
        expert_name: Name of expert ("microbiome_expert", "metabolome_expert", "proteome_expert")
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        use_grid_search: Whether to use grid search

    Returns:
        Tuple of (trained_expert, train_performance, val_performance)
    """
    print(f"\n{'='*60}")
    print(f"Training {expert_name}...")
    print(f"{'='*60}")

    # Create expert instance
    if expert_name == "microbiome_expert":
        expert = MicrobiomeExpert()
    elif expert_name == "metabolome_expert":
        expert = MetabolomeExpert()
    elif expert_name == "proteome_expert":
        expert = ProteomeExpert()
    else:
        raise ValueError(f"Unknown expert: {expert_name}")

    # Train
    print(f"Training with {len(X_train)} samples, {len(X_train.columns)} features...")
    expert.fit(X_train, y_train, use_grid_search=use_grid_search)

    # Evaluate on training set
    print("\nEvaluating on training set...")
    train_perf = expert.get_model_performance(X_train, y_train)
    print(f"  Training Accuracy: {train_perf['accuracy']:.3f}")
    print(f"  Training F1-Score: {train_perf['f1_score']:.3f}")

    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_perf = expert.get_model_performance(X_val, y_val)
    print(f"  Validation Accuracy: {val_perf['accuracy']:.3f}")
    print(f"  Validation F1-Score: {val_perf['f1_score']:.3f}")

    return expert, train_perf, val_perf


def main():
    parser = argparse.ArgumentParser(description="Train expert models on multi-omics data")
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
        "--output-dir",
        type=str,
        default="data/models",
        help="Directory to save trained models"
    )
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="Use grid search for hyperparameter tuning (slower but better)"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data to use for validation (default: 0.2)"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Load annotations
    print("Loading annotations...")
    annotations = load_annotations(args.annotations_file)
    print(f"  Found {len(annotations['annotations'])} annotated samples")

    # Get sample IDs from training split
    train_sample_ids = annotations["splits"]["train"]
    val_sample_ids = annotations["splits"]["validation"]

    print(f"  Training samples: {len(train_sample_ids)}")
    print(f"  Validation samples: {len(val_sample_ids)}")

    # Prepare labels
    y_train = prepare_labels(annotations, train_sample_ids)
    y_val = prepare_labels(annotations, val_sample_ids)

    print(f"\nClass distribution (training):")
    print(y_train.value_counts())

    # Initialize model manager
    model_manager = ModelManager(models_dir=args.output_dir)

    # Train each expert
    omics_types = ["microbiome", "metabolome", "proteome"]
    expert_names = ["microbiome_expert", "metabolome_expert", "proteome_expert"]
    preprocessors = [MicrobiomePreprocessor(), MetabolomePreprocessor(), ProteomePreprocessor()]

    results = {}

    for omics_type, expert_name, preprocessor in zip(omics_types, expert_names, preprocessors):
        print(f"\n{'#'*70}")
        print(f"# {omics_type.upper()} EXPERT")
        print(f"{'#'*70}")

        try:
            # Load data
            print(f"\nLoading {omics_type} data...")
            raw_data_dir = Path(args.raw_data_dir)

            X_train_raw = load_omics_data(train_sample_ids, omics_type, raw_data_dir)
            X_val_raw = load_omics_data(val_sample_ids, omics_type, raw_data_dir)

            print(f"  Training data: {X_train_raw.shape}")
            print(f"  Validation data: {X_val_raw.shape}")

            # Preprocessing
            print(f"\nPreprocessing {omics_type} data...")
            preprocessor.fit(X_train_raw)
            X_train_processed = preprocessor.transform(X_train_raw).data
            X_val_processed = preprocessor.transform(X_val_raw).data

            print(f"  After preprocessing: {X_train_processed.shape[1]} features")

            # Quality control
            print(f"\nPerforming quality control...")
            qc = QualityController(omics_type=omics_type)
            qc_report = qc.assess_dataset_quality(X_train_processed)

            print(f"  Samples passed QC: {qc_report.samples_passed}/{qc_report.n_samples}")
            print(f"  Detection rate: {qc_report.overall_detection_rate:.2%}")

            # Filter failed samples
            X_train_qc = qc.filter_samples(X_train_processed, qc_report)
            y_train_qc = y_train.loc[X_train_qc.index]

            # Feature engineering
            print(f"\nPerforming feature engineering...")
            feature_engineer = FeatureEngineer(method="anova", max_features=100)
            feature_engineer.fit(X_train_qc, y_train_qc)

            X_train_final = feature_engineer.transform(X_train_qc)
            X_val_final = feature_engineer.transform(X_val_processed)

            print(f"  Selected {len(X_train_final.columns)} important features")

            # Train expert
            expert, train_perf, val_perf = train_expert(
                expert_name,
                X_train_final,
                y_train_qc,
                X_val_final,
                y_val,
                use_grid_search=args.grid_search
            )

            # Save model
            print(f"\nSaving {expert_name}...")
            model_manager.save_expert(
                expert,
                notes=f"Trained on {len(X_train_final)} samples with {len(X_train_final.columns)} features"
            )

            # Store results
            results[expert_name] = {
                "train_performance": train_perf,
                "val_performance": val_perf,
                "n_features": len(X_train_final.columns),
                "n_train_samples": len(X_train_final),
                "n_val_samples": len(X_val_final)
            }

        except Exception as e:
            print(f"\n✗ Error training {expert_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Summary
    print(f"\n{'='*70}")
    print("TRAINING SUMMARY")
    print(f"{'='*70}")

    for expert_name, result in results.items():
        print(f"\n{expert_name}:")
        print(f"  Train Accuracy: {result['train_performance']['accuracy']:.3f}")
        print(f"  Val Accuracy: {result['val_performance']['accuracy']:.3f}")
        print(f"  Train F1-Score: {result['train_performance']['f1_score']:.3f}")
        print(f"  Val F1-Score: {result['val_performance']['f1_score']:.3f}")
        print(f"  Features: {result['n_features']}")

    print(f"\n✓ Training completed! Models saved to {args.output_dir}")
    print(f"\n{model_manager.summary()}")


if __name__ == "__main__":
    main()
