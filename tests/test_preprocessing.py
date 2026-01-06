"""
Test Preprocessing Modules.
"""

import pytest
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from clinical.preprocessing.microbiome_preprocessor import MicrobiomePreprocessor
from clinical.preprocessing.metabolome_preprocessor import MetabolomePreprocessor
from clinical.preprocessing.proteome_preprocessor import ProteomePreprocessor


def test_microbiome_preprocessor():
    """Test microbiome preprocessing."""
    # Load test data
    data_path = Path("data/test/microbiome_raw.csv")
    if not data_path.exists():
        pytest.skip("Test data not generated yet")

    df = pd.read_csv(data_path, index_col=0)

    # Initialize preprocessor
    preprocessor = MicrobiomePreprocessor()

    # Fit and transform
    X_transformed = preprocessor.fit_transform(df)

    assert X_transformed is not None
    assert X_transformed.shape[0] == df.shape[0]
    assert X_transformed.shape[1] <= df.shape[1]  # May filter features

    print(f"\nMicrobiome Preprocessing:")
    print(f"  Original shape: {df.shape}")
    print(f"  Transformed shape: {X_transformed.shape}")
    print(f"  Transformation: {preprocessor.normalization}")


def test_metabolome_preprocessor():
    """Test metabolome preprocessing."""
    # Load test data
    data_path = Path("data/test/metabolome_raw.csv")
    if not data_path.exists():
        pytest.skip("Test data not generated yet")

    df = pd.read_csv(data_path, index_col=0)

    # Initialize preprocessor
    preprocessor = MetabolomePreprocessor()

    # Fit and transform
    X_transformed = preprocessor.fit_transform(df)

    assert X_transformed is not None
    assert X_transformed.shape[0] == df.shape[0]

    print(f"\nMetabolome Preprocessing:")
    print(f"  Original shape: {df.shape}")
    print(f"  Transformed shape: {X_transformed.shape}")
    print(f"  Log transform: {preprocessor.log_transform}")


def test_proteome_preprocessor():
    """Test proteome preprocessing."""
    # Load test data
    data_path = Path("data/test/proteome_raw.csv")
    if not data_path.exists():
        pytest.skip("Test data not generated yet")

    df = pd.read_csv(data_path, index_col=0)

    # Initialize preprocessor
    preprocessor = ProteomePreprocessor()

    # Fit and transform
    X_transformed = preprocessor.fit_transform(df)

    assert X_transformed is not None
    assert X_transformed.shape[0] == df.shape[0]

    print(f"\nProteome Preprocessing:")
    print(f"  Original shape: {df.shape}")
    print(f"  Transformed shape: {X_transformed.shape}")
    print(f"  Normalization: {preprocessor.normalization}")


def test_missing_value_handling():
    """Test missing value handling."""
    # Create data with missing values
    df = pd.DataFrame({
        'feature1': [1.0, 2.0, np.nan, 4.0],
        'feature2': [5.0, np.nan, 7.0, 8.0],
        'feature3': [9.0, 10.0, 11.0, 12.0]
    })

    preprocessor = MicrobiomePreprocessor(missing_value_strategy='median')

    X_transformed = preprocessor.fit_transform(df)

    # Check no missing values
    assert not X_transformed.isnull().any().any()

    print(f"\nMissing Value Handling:")
    print(f"  Original missing: {df.isnull().sum().sum()}")
    print(f"  After processing: {X_transformed.isnull().sum().sum()}")


def test_feature_filtering():
    """Test low variance feature filtering."""
    # Create data with low variance feature
    df = pd.DataFrame({
        'high_var': np.random.randn(50),
        'low_var': [0.001] * 50,  # Very low variance
        'medium_var': np.random.randn(50) * 0.1
    })

    preprocessor = MicrobiomePreprocessor(
        filter_low_variance=True,
        variance_threshold=0.01
    )

    X_transformed = preprocessor.fit_transform(df)

    # low_var should be filtered out
    assert 'low_var' not in X_transformed.columns

    print(f"\nFeature Filtering:")
    print(f"  Original features: {df.shape[1]}")
    print(f"  After filtering: {X_transformed.shape[1]}")
    print(f"  Removed: {set(df.columns) - set(X_transformed.columns)}")


def test_preprocessing_pipeline():
    """Test full preprocessing pipeline."""
    data_paths = {
        'microbiome': Path("data/test/microbiome_raw.csv"),
        'metabolome': Path("data/test/metabolome_raw.csv"),
        'proteome': Path("data/test/proteome_raw.csv")
    }

    preprocessors = {
        'microbiome': MicrobiomePreprocessor(),
        'metabolome': MetabolomePreprocessor(),
        'proteome': ProteomePreprocessor()
    }

    results = {}

    for omics_type, data_path in data_paths.items():
        if not data_path.exists():
            continue

        df = pd.read_csv(data_path, index_col=0)
        preprocessor = preprocessors[omics_type]

        X_transformed = preprocessor.fit_transform(df)
        results[omics_type] = X_transformed

        print(f"\n{omics_type.title()} Pipeline:")
        print(f"  Input: {df.shape}")
        print(f"  Output: {X_transformed.shape}")
        print(f"  Data range: [{X_transformed.min().min():.3f}, {X_transformed.max().max():.3f}]")

    assert len(results) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
