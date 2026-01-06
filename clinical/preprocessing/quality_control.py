"""
Quality Control Module.

This module provides quality control checks for preprocessed omics data,
including sample quality assessment, feature coverage checks, and QC reporting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import json


@dataclass
class QCMetrics:
    """Quality control metrics for a sample or dataset."""

    sample_id: Optional[str] = None
    total_features: int = 0
    detected_features: int = 0
    detection_rate: float = 0.0
    missing_rate: float = 0.0
    mean_intensity: float = 0.0
    median_intensity: float = 0.0
    cv: float = 0.0  # Coefficient of variation
    outlier_count: int = 0
    pass_qc: bool = True
    qc_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "sample_id": self.sample_id,
            "total_features": self.total_features,
            "detected_features": self.detected_features,
            "detection_rate": self.detection_rate,
            "missing_rate": self.missing_rate,
            "mean_intensity": self.mean_intensity,
            "median_intensity": self.median_intensity,
            "cv": self.cv,
            "outlier_count": self.outlier_count,
            "pass_qc": self.pass_qc,
            "qc_flags": self.qc_flags
        }


@dataclass
class QCReport:
    """Comprehensive QC report for a dataset."""

    omics_type: str
    n_samples: int
    n_features: int
    sample_metrics: List[QCMetrics]
    overall_detection_rate: float
    overall_missing_rate: float
    samples_passed: int
    samples_failed: int
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "omics_type": self.omics_type,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "sample_metrics": [m.to_dict() for m in self.sample_metrics],
            "overall_detection_rate": self.overall_detection_rate,
            "overall_missing_rate": self.overall_missing_rate,
            "samples_passed": self.samples_passed,
            "samples_failed": self.samples_failed,
            "recommendations": self.recommendations
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def summary(self) -> str:
        """Generate human-readable summary."""
        return (
            f"=== QC Report for {self.omics_type.upper()} Data ===\n"
            f"Samples: {self.n_samples} ({self.samples_passed} passed, {self.samples_failed} failed)\n"
            f"Features: {self.n_features}\n"
            f"Overall Detection Rate: {self.overall_detection_rate:.2%}\n"
            f"Overall Missing Rate: {self.overall_missing_rate:.2%}\n"
            f"\nRecommendations:\n" + "\n".join([f"  - {r}" for r in self.recommendations])
        )


class QualityController:
    """
    Quality control for omics data.

    Performs comprehensive QC checks including:
    - Sample-level quality metrics
    - Feature coverage assessment
    - Outlier detection
    - QC pass/fail determination
    """

    def __init__(
        self,
        omics_type: str,
        min_detection_rate: float = 0.5,  # At least 50% features detected
        max_missing_rate: float = 0.5,    # At most 50% missing values
        max_cv: float = 1.0,               # Maximum coefficient of variation
        outlier_threshold: float = 3.0     # Z-score threshold for outliers
    ):
        """
        Initialize quality controller.

        Args:
            omics_type: Type of omics data
            min_detection_rate: Minimum required detection rate
            max_missing_rate: Maximum allowed missing rate
            max_cv: Maximum coefficient of variation
            outlier_threshold: Z-score threshold for outlier detection
        """
        self.omics_type = omics_type
        self.min_detection_rate = min_detection_rate
        self.max_missing_rate = max_missing_rate
        self.max_cv = max_cv
        self.outlier_threshold = outlier_threshold

    def assess_sample_quality(
        self,
        sample_data: pd.Series,
        sample_id: str
    ) -> QCMetrics:
        """
        Assess quality of a single sample.

        Args:
            sample_data: Data for one sample
            sample_id: Sample identifier

        Returns:
            QC metrics for the sample
        """
        total_features = len(sample_data)

        # Calculate detection metrics
        detected = (sample_data > 0) & (~sample_data.isna())
        detected_features = detected.sum()
        detection_rate = detected_features / total_features

        # Calculate missing metrics
        missing = sample_data.isna()
        missing_rate = missing.sum() / total_features

        # Calculate intensity metrics (on non-zero values)
        non_zero_data = sample_data[sample_data > 0]
        if len(non_zero_data) > 0:
            mean_intensity = non_zero_data.mean()
            median_intensity = non_zero_data.median()
            cv = non_zero_data.std() / (mean_intensity + 1e-8)
        else:
            mean_intensity = 0.0
            median_intensity = 0.0
            cv = 0.0

        # Count outliers
        outlier_count = self._count_outliers(sample_data)

        # Determine QC pass/fail and flags
        qc_flags = []
        pass_qc = True

        if detection_rate < self.min_detection_rate:
            qc_flags.append(f"Low detection rate: {detection_rate:.2%}")
            pass_qc = False

        if missing_rate > self.max_missing_rate:
            qc_flags.append(f"High missing rate: {missing_rate:.2%}")
            pass_qc = False

        if cv > self.max_cv:
            qc_flags.append(f"High coefficient of variation: {cv:.2f}")
            pass_qc = False

        if outlier_count > total_features * 0.1:  # More than 10% outliers
            qc_flags.append(f"Many outliers detected: {outlier_count}")
            pass_qc = False

        return QCMetrics(
            sample_id=sample_id,
            total_features=total_features,
            detected_features=detected_features,
            detection_rate=detection_rate,
            missing_rate=missing_rate,
            mean_intensity=mean_intensity,
            median_intensity=median_intensity,
            cv=cv,
            outlier_count=outlier_count,
            pass_qc=pass_qc,
            qc_flags=qc_flags
        )

    def assess_dataset_quality(
        self,
        data: pd.DataFrame
    ) -> QCReport:
        """
        Assess quality of entire dataset.

        Args:
            data: Complete dataset (samples x features)

        Returns:
            Comprehensive QC report
        """
        # Assess each sample
        sample_metrics = []
        for sample_id in data.index:
            metrics = self.assess_sample_quality(data.loc[sample_id], sample_id)
            sample_metrics.append(metrics)

        # Calculate overall metrics
        overall_detection_rate = np.mean([m.detection_rate for m in sample_metrics])
        overall_missing_rate = np.mean([m.missing_rate for m in sample_metrics])

        # Count passed/failed samples
        samples_passed = sum(1 for m in sample_metrics if m.pass_qc)
        samples_failed = len(sample_metrics) - samples_passed

        # Generate recommendations
        recommendations = self._generate_recommendations(
            sample_metrics,
            overall_detection_rate,
            overall_missing_rate
        )

        return QCReport(
            omics_type=self.omics_type,
            n_samples=len(data),
            n_features=len(data.columns),
            sample_metrics=sample_metrics,
            overall_detection_rate=overall_detection_rate,
            overall_missing_rate=overall_missing_rate,
            samples_passed=samples_passed,
            samples_failed=samples_failed,
            recommendations=recommendations
        )

    def _count_outliers(self, data: pd.Series) -> int:
        """Count outliers using z-score method."""
        # Remove zeros and NaN
        non_zero_data = data[(data > 0) & (~data.isna())]

        if len(non_zero_data) < 3:
            return 0

        # Calculate z-scores
        mean = non_zero_data.mean()
        std = non_zero_data.std()

        if std == 0:
            return 0

        z_scores = np.abs((non_zero_data - mean) / std)
        outliers = z_scores > self.outlier_threshold

        return outliers.sum()

    def _generate_recommendations(
        self,
        sample_metrics: List[QCMetrics],
        overall_detection_rate: float,
        overall_missing_rate: float
    ) -> List[str]:
        """Generate QC recommendations based on metrics."""
        recommendations = []

        # Check overall detection rate
        if overall_detection_rate < 0.5:
            recommendations.append(
                "Low overall detection rate. Consider reviewing sample preparation protocol."
            )

        # Check overall missing rate
        if overall_missing_rate > 0.3:
            recommendations.append(
                "High overall missing rate. Consider more aggressive feature filtering."
            )

        # Check failed samples
        failed_samples = [m for m in sample_metrics if not m.pass_qc]
        if len(failed_samples) > len(sample_metrics) * 0.2:
            recommendations.append(
                f"{len(failed_samples)} samples failed QC. Consider removing or re-processing."
            )

        # Check for systematic issues
        high_cv_samples = [m for m in sample_metrics if m.cv > self.max_cv]
        if len(high_cv_samples) > len(sample_metrics) * 0.3:
            recommendations.append(
                "Many samples have high coefficient of variation. Check for batch effects."
            )

        # Check feature coverage
        low_detection_samples = [
            m for m in sample_metrics
            if m.detection_rate < self.min_detection_rate
        ]
        if len(low_detection_samples) > 0:
            recommendations.append(
                f"{len(low_detection_samples)} samples have low feature detection. "
                "Consider technical replicates."
            )

        # If everything looks good
        if not recommendations:
            recommendations.append("All QC checks passed. Data quality is acceptable.")

        return recommendations

    def filter_samples(
        self,
        data: pd.DataFrame,
        qc_report: QCReport
    ) -> pd.DataFrame:
        """
        Filter out samples that failed QC.

        Args:
            data: Original dataset
            qc_report: QC report

        Returns:
            Filtered dataset with only QC-passed samples
        """
        passed_samples = [
            m.sample_id for m in qc_report.sample_metrics
            if m.pass_qc
        ]

        return data.loc[passed_samples]

    def generate_qc_plots(
        self,
        qc_report: QCReport,
        output_dir: str
    ):
        """
        Generate QC visualization plots.

        Args:
            qc_report: QC report
            output_dir: Directory to save plots

        Note: This would generate plots using matplotlib/seaborn
        """
        # TODO: Implement QC plots
        # - Detection rate distribution
        # - Missing rate distribution
        # - CV distribution
        # - Sample-level metrics heatmap
        pass
