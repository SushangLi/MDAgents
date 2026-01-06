"""
Data Annotation GUI for Oral Multi-Omics Clinical Diagnosis System.

This Streamlit application allows clinical experts to annotate multi-omics data
for training expert agents. Features include:
- Loading multi-omics data (CSV/Excel)
- Visualization (PCA, heatmaps, distributions)
- Annotation interface (diagnosis, severity, confidence, notes)
- Export to JSON format with train/validation/test split

Usage:
    streamlit run scripts/data_annotation/annotation_gui.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(
    page_title="Oral Multi-Omics Data Annotation Tool",
    page_icon="🦷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Constants
DATA_DIR = Path("data/raw")
LABELED_DIR = Path("data/labeled")
LABELED_DIR.mkdir(parents=True, exist_ok=True)

OMICS_TYPES = ["microbiome", "metabolome", "proteome"]
DIAGNOSIS_CATEGORIES = ["healthy", "periodontitis", "caries", "other"]
SEVERITY_LEVELS = ["mild", "moderate", "severe", "n/a"]
CONFIDENCE_LEVELS = ["high", "medium", "low"]

# Initialize session state
if "annotations" not in st.session_state:
    st.session_state.annotations = {}
if "current_sample_index" not in st.session_state:
    st.session_state.current_sample_index = 0
if "sample_ids" not in st.session_state:
    st.session_state.sample_ids = []
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False


def load_omics_data(omics_type: str, sample_id: str) -> Optional[pd.DataFrame]:
    """Load omics data from CSV or Excel file."""
    data_path = DATA_DIR / omics_type / f"{sample_id}.csv"

    if not data_path.exists():
        # Try Excel format
        data_path = DATA_DIR / omics_type / f"{sample_id}.xlsx"
        if not data_path.exists():
            return None

    try:
        if data_path.suffix == ".csv":
            df = pd.read_csv(data_path, index_col=0)
        else:
            df = pd.read_excel(data_path, index_col=0)
        return df
    except Exception as e:
        st.error(f"Error loading {omics_type} data: {e}")
        return None


def scan_available_samples() -> List[str]:
    """Scan data directory for available samples."""
    sample_ids = set()

    for omics_type in OMICS_TYPES:
        omics_dir = DATA_DIR / omics_type
        if omics_dir.exists():
            # Scan for CSV files
            for file_path in omics_dir.glob("*.csv"):
                sample_ids.add(file_path.stem)
            # Scan for Excel files
            for file_path in omics_dir.glob("*.xlsx"):
                sample_ids.add(file_path.stem)

    return sorted(list(sample_ids))


def visualize_pca(data_dict: Dict[str, pd.DataFrame]):
    """Visualize PCA of combined omics data."""
    st.subheader("📊 PCA Visualization")

    # Combine all omics data
    combined_features = []
    feature_names = []

    for omics_type in OMICS_TYPES:
        if omics_type in data_dict and data_dict[omics_type] is not None:
            df = data_dict[omics_type]
            if not df.empty:
                combined_features.extend(df.values.flatten())
                feature_names.extend([f"{omics_type}_{col}" for col in df.columns])

    if len(combined_features) < 2:
        st.warning("Not enough features for PCA visualization")
        return

    # Reshape for PCA (single sample)
    X = np.array(combined_features).reshape(1, -1)

    # Show feature statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Features", len(combined_features))
    with col2:
        st.metric("Mean Value", f"{np.mean(combined_features):.4f}")
    with col3:
        st.metric("Std Dev", f"{np.std(combined_features):.4f}")

    # Feature distribution
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(combined_features, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel("Feature Value")
    ax.set_ylabel("Frequency")
    ax.set_title("Feature Value Distribution")
    st.pyplot(fig)
    plt.close()


def visualize_heatmap(data: pd.DataFrame, title: str):
    """Visualize omics data as heatmap."""
    if data is None or data.empty:
        st.warning(f"No data available for {title}")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # Show top 50 features by variance
    if data.shape[1] > 50:
        feature_var = data.var(axis=0)
        top_features = feature_var.nlargest(50).index
        data_to_plot = data[top_features]
    else:
        data_to_plot = data

    sns.heatmap(
        data_to_plot.T,
        cmap='RdYlBu_r',
        center=0,
        cbar_kws={'label': 'Expression Level'},
        ax=ax
    )
    ax.set_title(title)
    ax.set_ylabel("Features")
    ax.set_xlabel("Samples")
    st.pyplot(fig)
    plt.close()


def annotation_interface(sample_id: str):
    """Display annotation interface for current sample."""
    st.subheader(f"🏷️ Annotate Sample: {sample_id}")

    # Load existing annotation if available
    existing_annotation = st.session_state.annotations.get(sample_id, {})

    # Annotation form
    with st.form(key=f"annotation_form_{sample_id}"):
        col1, col2 = st.columns(2)

        with col1:
            diagnosis = st.selectbox(
                "Diagnosis Category",
                DIAGNOSIS_CATEGORIES,
                index=DIAGNOSIS_CATEGORIES.index(existing_annotation.get("diagnosis", "healthy"))
            )

            severity = st.selectbox(
                "Severity Level",
                SEVERITY_LEVELS,
                index=SEVERITY_LEVELS.index(existing_annotation.get("severity", "n/a"))
            )

        with col2:
            confidence = st.selectbox(
                "Confidence Level",
                CONFIDENCE_LEVELS,
                index=CONFIDENCE_LEVELS.index(existing_annotation.get("confidence", "high"))
            )

            annotator = st.text_input(
                "Annotator Name",
                value=existing_annotation.get("annotator", "")
            )

        notes = st.text_area(
            "Clinical Notes",
            value=existing_annotation.get("notes", ""),
            height=100,
            placeholder="Enter any relevant clinical observations..."
        )

        submitted = st.form_submit_button("💾 Save Annotation", use_container_width=True)

        if submitted:
            # Save annotation
            st.session_state.annotations[sample_id] = {
                "diagnosis": diagnosis,
                "severity": severity,
                "confidence": confidence,
                "annotator": annotator,
                "notes": notes,
                "timestamp": datetime.now().isoformat()
            }
            st.success(f"✅ Annotation saved for {sample_id}")

    # Show annotation status
    if sample_id in st.session_state.annotations:
        st.info(f"📝 Last annotated: {st.session_state.annotations[sample_id]['timestamp']}")


def export_annotations():
    """Export annotations to JSON file."""
    if not st.session_state.annotations:
        st.warning("No annotations to export")
        return

    st.subheader("📤 Export Annotations")

    col1, col2, col3 = st.columns(3)
    with col1:
        train_ratio = st.slider("Training Set %", 0, 100, 70, 5)
    with col2:
        val_ratio = st.slider("Validation Set %", 0, 100, 15, 5)
    with col3:
        test_ratio = 100 - train_ratio - val_ratio
        st.metric("Test Set %", test_ratio)

    if train_ratio + val_ratio > 100:
        st.error("Training + Validation ratio exceeds 100%")
        return

    if st.button("💾 Export to JSON", use_container_width=True):
        # Split samples into train/val/test
        sample_ids = list(st.session_state.annotations.keys())
        n_samples = len(sample_ids)

        np.random.seed(42)
        shuffled_indices = np.random.permutation(n_samples)

        n_train = int(n_samples * train_ratio / 100)
        n_val = int(n_samples * val_ratio / 100)

        train_indices = shuffled_indices[:n_train]
        val_indices = shuffled_indices[n_train:n_train + n_val]
        test_indices = shuffled_indices[n_train + n_val:]

        # Create export data
        export_data = {
            "metadata": {
                "export_time": datetime.now().isoformat(),
                "total_samples": n_samples,
                "train_samples": len(train_indices),
                "val_samples": len(val_indices),
                "test_samples": len(test_indices),
            },
            "splits": {
                "train": [sample_ids[i] for i in train_indices],
                "validation": [sample_ids[i] for i in val_indices],
                "test": [sample_ids[i] for i in test_indices],
            },
            "annotations": st.session_state.annotations
        }

        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = LABELED_DIR / f"annotations_{timestamp}.json"

        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2)

        st.success(f"✅ Annotations exported to: {output_path}")

        # Show statistics
        st.write("### Export Statistics")
        stats_df = pd.DataFrame({
            "Split": ["Training", "Validation", "Test"],
            "Samples": [len(train_indices), len(val_indices), len(test_indices)],
            "Percentage": [train_ratio, val_ratio, test_ratio]
        })
        st.dataframe(stats_df, use_container_width=True)


def main():
    """Main application."""
    st.title("🦷 Oral Multi-Omics Data Annotation Tool")
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.header("📁 Data Management")

        if st.button("🔄 Scan Samples", use_container_width=True):
            st.session_state.sample_ids = scan_available_samples()
            st.session_state.data_loaded = True
            st.success(f"Found {len(st.session_state.sample_ids)} samples")

        if st.session_state.data_loaded and st.session_state.sample_ids:
            st.write(f"**Total Samples:** {len(st.session_state.sample_ids)}")
            st.write(f"**Annotated:** {len(st.session_state.annotations)}")
            st.write(f"**Remaining:** {len(st.session_state.sample_ids) - len(st.session_state.annotations)}")

            st.markdown("---")
            st.header("🎯 Navigation")

            # Sample navigation
            current_index = st.number_input(
                "Sample Index",
                min_value=0,
                max_value=len(st.session_state.sample_ids) - 1,
                value=st.session_state.current_sample_index,
                step=1
            )
            st.session_state.current_sample_index = current_index

            col1, col2 = st.columns(2)
            with col1:
                if st.button("⬅️ Previous", use_container_width=True):
                    if st.session_state.current_sample_index > 0:
                        st.session_state.current_sample_index -= 1
                        st.rerun()
            with col2:
                if st.button("Next ➡️", use_container_width=True):
                    if st.session_state.current_sample_index < len(st.session_state.sample_ids) - 1:
                        st.session_state.current_sample_index += 1
                        st.rerun()

    # Main content
    if not st.session_state.data_loaded:
        st.info("👈 Click 'Scan Samples' in the sidebar to load data")
        st.write("### Setup Instructions")
        st.markdown(f"""
        1. Place your omics data files in the following directories:
           - Microbiome: `{DATA_DIR / 'microbiome'}/`
           - Metabolome: `{DATA_DIR / 'metabolome'}/`
           - Proteome: `{DATA_DIR / 'proteome'}/`

        2. Files should be named consistently (e.g., `sample_001.csv`)

        3. Data format: CSV or Excel files with features as columns

        4. Click 'Scan Samples' to detect available data
        """)
        return

    if not st.session_state.sample_ids:
        st.warning("No samples found. Please check your data directory structure.")
        return

    # Display current sample
    sample_id = st.session_state.sample_ids[st.session_state.current_sample_index]

    st.header(f"Sample {st.session_state.current_sample_index + 1} of {len(st.session_state.sample_ids)}: {sample_id}")

    # Load omics data
    data_dict = {}
    tabs = st.tabs(["📊 Visualization", "🏷️ Annotation", "📤 Export"])

    with tabs[0]:
        st.subheader("Multi-Omics Data Visualization")

        # Load all omics types
        for omics_type in OMICS_TYPES:
            data_dict[omics_type] = load_omics_data(omics_type, sample_id)

        # Show data availability
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Microbiome", "✅" if data_dict["microbiome"] is not None else "❌")
        with col2:
            st.metric("Metabolome", "✅" if data_dict["metabolome"] is not None else "❌")
        with col3:
            st.metric("Proteome", "✅" if data_dict["proteome"] is not None else "❌")

        # PCA visualization
        visualize_pca(data_dict)

        # Heatmaps for each omics type
        st.markdown("---")
        for omics_type in OMICS_TYPES:
            if data_dict[omics_type] is not None:
                st.subheader(f"{omics_type.capitalize()} Heatmap")
                visualize_heatmap(data_dict[omics_type], f"{omics_type.capitalize()} Expression")

    with tabs[1]:
        annotation_interface(sample_id)

    with tabs[2]:
        export_annotations()


if __name__ == "__main__":
    main()
