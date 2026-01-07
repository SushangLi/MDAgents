#!/usr/bin/env python3
"""Debug SHAP values shape issue"""

import sys
import numpy as np
import pandas as pd
import json
from clinical.preprocessing.microbiome_preprocessor import MicrobiomePreprocessor
from clinical.experts.microbiome_expert import MicrobiomeExpert

# Load training data
microbiome_data = pd.read_csv("data/training/microbiome_raw.csv", index_col=0)
labels = pd.read_csv("data/training/labels.csv", index_col=0)
with open("data/training/splits.json", "r") as f:
    splits = json.load(f)

# Get training samples - ensure all 3 classes
train_ids = splits['train']
X_train = microbiome_data.loc[train_ids]
y_train = labels.loc[train_ids, 'diagnosis']

print("Data shapes:")
print(f"  X_train: {X_train.shape}")
print(f"  y_train: {y_train.shape}")
print(f"  Classes: {y_train.unique()}")
print(f"  Class counts: {y_train.value_counts().to_dict()}")

# Initialize expert
preprocessor = MicrobiomePreprocessor()
expert = MicrobiomeExpert()

# Preprocess
X_processed_result = preprocessor.fit_transform(X_train)
X_processed = X_processed_result.data
print(f"  X_processed: {X_processed.shape}")

# Train expert
print("\nTraining expert...")
expert.fit(X_processed, y_train)
print(f"  Model classes: {expert.classes_}")
print(f"  Number of classes: {len(expert.classes_)}")

# Test SHAP values
print("\nTesting SHAP values...")
shap_values = expert.explainer_.shap_values(X_processed)

print(f"Type of shap_values: {type(shap_values)}")
if isinstance(shap_values, list):
    print(f"Number of classes: {len(shap_values)}")
    for i, sv in enumerate(shap_values):
        print(f"  Class {i} SHAP shape: {sv.shape}")
        print(f"  Class {i} SHAP dtype: {sv.dtype}")
else:
    print(f"SHAP values shape: {shap_values.shape}")
    print(f"SHAP values dtype: {shap_values.dtype}")

# Test prediction
print("\nTesting prediction...")
try:
    predictions = expert.predict(X_processed)
    print(f"Success! Got {len(predictions)} predictions")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

    # Try to debug further
    print("\n--- Debugging _get_top_features ---")
    X_aligned = X_processed[expert.feature_names_]
    probabilities = expert.model_.predict_proba(X_aligned)
    predictions_raw = expert.model_.predict(X_aligned)

    # Get first sample
    i = 0
    diagnosis = predictions_raw[i]
    class_idx = list(expert.classes_).index(diagnosis)

    print(f"Sample {i}:")
    print(f"  Diagnosis: {diagnosis}")
    print(f"  Class index: {class_idx}")

    if isinstance(shap_values, list):
        sample_shap = shap_values[class_idx][i]
    else:
        sample_shap = shap_values[i]

    print(f"  sample_shap type: {type(sample_shap)}")
    print(f"  sample_shap shape: {sample_shap.shape if hasattr(sample_shap, 'shape') else 'N/A'}")
    print(f"  sample_shap dtype: {sample_shap.dtype if hasattr(sample_shap, 'dtype') else 'N/A'}")
    print(f"  sample_shap content: {sample_shap}")

    # Test argsort
    abs_shap = np.abs(sample_shap)
    print(f"  abs_shap shape: {abs_shap.shape}")
    top_indices = np.argsort(abs_shap)[-10:][::-1]
    print(f"  top_indices type: {type(top_indices)}")
    print(f"  top_indices shape: {top_indices.shape if hasattr(top_indices, 'shape') else 'N/A'}")
    print(f"  top_indices: {top_indices}")

    # Try iteration
    print(f"\n  Testing iteration:")
    for idx in top_indices:
        print(f"    idx type: {type(idx)}, value: {idx}")
        print(f"    idx is scalar: {np.isscalar(idx)}")
        if not np.isscalar(idx):
            print(f"    idx shape: {idx.shape}")
            print(f"    Converting to int: {int(idx)}")
