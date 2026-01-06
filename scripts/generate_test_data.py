"""
Generate synthetic omics data for testing.

Creates realistic multi-omics test data for the clinical diagnosis system.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime


# Disease classes for testing
DISEASES = ["Periodontitis", "Diabetes_Associated_Dysbiosis", "Healthy", "Oral_Cancer_Risk"]

# Microbiome taxa (genus level)
MICROBIOME_TAXA = [
    "Porphyromonas_gingivalis",
    "Treponema_denticola",
    "Tannerella_forsythia",
    "Fusobacterium_nucleatum",
    "Prevotella_intermedia",
    "Aggregatibacter_actinomycetemcomitans",
    "Streptococcus_mutans",
    "Streptococcus_salivarius",
    "Lactobacillus_reuteri",
    "Neisseria_sicca",
    "Veillonella_parvula",
    "Actinomyces_naeslundii",
    "Rothia_dentocariosa",
    "Capnocytophaga_sputigena",
    "Campylobacter_concisus"
]

# Metabolites
METABOLITES = [
    "Butyrate",
    "Propionate",
    "Acetate",
    "Lactate",
    "Succinate",
    "Putrescine",
    "Spermidine",
    "Spermine",
    "Indole",
    "Skatole",
    "p-Cresol",
    "Trimethylamine",
    "Choline",
    "Betaine",
    "GABA"
]

# Proteins (salivary)
PROTEINS = [
    "MMP9",
    "IL1B",
    "IL6",
    "IL8",
    "TNF",
    "VEGFA",
    "MMP1",
    "MMP3",
    "CXCL8",
    "CCL2",
    "LDH",
    "Calprotectin",
    "Lactoferrin",
    "IgA",
    "Albumin"
]


def generate_microbiome_data(n_samples: int, disease: str) -> pd.DataFrame:
    """Generate microbiome abundance data."""
    np.random.seed(42 + hash(disease) % 1000)

    data = {}

    for taxon in MICROBIOME_TAXA:
        # Base abundance
        base = np.random.uniform(0.01, 0.1, n_samples)

        # Disease-specific patterns
        if disease == "Periodontitis":
            if taxon in ["Porphyromonas_gingivalis", "Treponema_denticola", "Tannerella_forsythia"]:
                base *= np.random.uniform(3, 6, n_samples)  # Elevated
            elif taxon in ["Streptococcus_salivarius", "Lactobacillus_reuteri"]:
                base *= np.random.uniform(0.2, 0.5, n_samples)  # Reduced

        elif disease == "Diabetes_Associated_Dysbiosis":
            if taxon in ["Prevotella_intermedia", "Fusobacterium_nucleatum", "Veillonella_parvula"]:
                base *= np.random.uniform(2, 4, n_samples)
            elif taxon in ["Streptococcus_mutans"]:
                base *= np.random.uniform(1.5, 3, n_samples)

        elif disease == "Oral_Cancer_Risk":
            if taxon in ["Fusobacterium_nucleatum", "Prevotella_intermedia"]:
                base *= np.random.uniform(2.5, 5, n_samples)

        elif disease == "Healthy":
            if taxon in ["Streptococcus_salivarius", "Lactobacillus_reuteri", "Neisseria_sicca"]:
                base *= np.random.uniform(1.5, 2.5, n_samples)

        # Add noise
        base += np.random.normal(0, 0.01, n_samples)
        base = np.clip(base, 0.001, None)

        data[taxon] = base

    # Normalize to relative abundance
    df = pd.DataFrame(data)
    df = df.div(df.sum(axis=1), axis=0)

    return df


def generate_metabolome_data(n_samples: int, disease: str) -> pd.DataFrame:
    """Generate metabolome intensity data."""
    np.random.seed(43 + hash(disease) % 1000)

    data = {}

    for metabolite in METABOLITES:
        # Base intensity (log scale)
        base = np.random.lognormal(mean=5, sigma=1, size=n_samples)

        # Disease-specific patterns
        if disease == "Periodontitis":
            if metabolite in ["Butyrate", "Propionate", "Putrescine"]:
                base *= np.random.uniform(1.5, 2.5, n_samples)

        elif disease == "Diabetes_Associated_Dysbiosis":
            if metabolite in ["Butyrate", "Propionate", "Acetate"]:
                base *= np.random.uniform(1.8, 3, n_samples)
            elif metabolite in ["Lactate"]:
                base *= np.random.uniform(2, 4, n_samples)

        elif disease == "Oral_Cancer_Risk":
            if metabolite in ["Putrescine", "Spermidine", "Spermine"]:
                base *= np.random.uniform(3, 5, n_samples)
            elif metabolite == "LDH":
                base *= np.random.uniform(3.5, 6, n_samples)

        # Add noise
        base += np.random.normal(0, base * 0.1, n_samples)
        base = np.clip(base, 100, None)

        data[metabolite] = base

    return pd.DataFrame(data)


def generate_proteome_data(n_samples: int, disease: str) -> pd.DataFrame:
    """Generate proteome expression data."""
    np.random.seed(44 + hash(disease) % 1000)

    data = {}

    for protein in PROTEINS:
        # Base expression (log scale)
        base = np.random.lognormal(mean=3, sigma=0.8, size=n_samples)

        # Disease-specific patterns
        if disease == "Periodontitis":
            if protein in ["MMP9", "IL1B", "IL6", "IL8", "TNF"]:
                base *= np.random.uniform(2, 4, n_samples)  # Inflammatory markers

        elif disease == "Diabetes_Associated_Dysbiosis":
            if protein in ["IL6", "TNF", "MMP9"]:
                base *= np.random.uniform(1.5, 3, n_samples)

        elif disease == "Oral_Cancer_Risk":
            if protein in ["MMP9", "VEGFA", "IL8", "LDH"]:
                base *= np.random.uniform(3, 6, n_samples)
            elif protein in ["MMP1", "MMP3"]:
                base *= np.random.uniform(2, 4, n_samples)

        elif disease == "Healthy":
            if protein in ["IgA", "Lactoferrin"]:
                base *= np.random.uniform(1.2, 1.8, n_samples)

        # Add noise
        base += np.random.normal(0, base * 0.15, n_samples)
        base = np.clip(base, 10, None)

        data[protein] = base

    return pd.DataFrame(data)


def generate_dataset(samples_per_disease: int = 25) -> dict:
    """Generate complete multi-omics dataset."""
    all_data = {
        "microbiome": [],
        "metabolome": [],
        "proteome": [],
        "labels": []
    }

    sample_ids = []

    for disease in DISEASES:
        print(f"Generating {samples_per_disease} samples for {disease}...")

        # Generate omics data
        microbiome = generate_microbiome_data(samples_per_disease, disease)
        metabolome = generate_metabolome_data(samples_per_disease, disease)
        proteome = generate_proteome_data(samples_per_disease, disease)

        # Add sample IDs
        disease_sample_ids = [f"{disease}_{i+1:03d}" for i in range(samples_per_disease)]
        sample_ids.extend(disease_sample_ids)

        microbiome.index = disease_sample_ids
        metabolome.index = disease_sample_ids
        proteome.index = disease_sample_ids

        # Append
        all_data["microbiome"].append(microbiome)
        all_data["metabolome"].append(metabolome)
        all_data["proteome"].append(proteome)
        all_data["labels"].extend([disease] * samples_per_disease)

    # Concatenate
    microbiome_df = pd.concat(all_data["microbiome"])
    metabolome_df = pd.concat(all_data["metabolome"])
    proteome_df = pd.concat(all_data["proteome"])
    labels_df = pd.DataFrame({
        "sample_id": sample_ids,
        "diagnosis": all_data["labels"]
    })

    return {
        "microbiome": microbiome_df,
        "metabolome": metabolome_df,
        "proteome": proteome_df,
        "labels": labels_df
    }


def save_dataset(dataset: dict, output_dir: str = "data/test"):
    """Save dataset to files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save raw data
    dataset["microbiome"].to_csv(output_dir / "microbiome_raw.csv")
    dataset["metabolome"].to_csv(output_dir / "metabolome_raw.csv")
    dataset["proteome"].to_csv(output_dir / "proteome_raw.csv")
    dataset["labels"].to_csv(output_dir / "labels.csv", index=False)

    print(f"\n✓ Dataset saved to {output_dir}/")
    print(f"  Samples: {len(dataset['labels'])}")
    print(f"  Microbiome features: {len(dataset['microbiome'].columns)}")
    print(f"  Metabolome features: {len(dataset['metabolome'].columns)}")
    print(f"  Proteome features: {len(dataset['proteome'].columns)}")
    print(f"  Classes: {dataset['labels']['diagnosis'].unique()}")


def generate_annotations(labels_df: pd.DataFrame, output_dir: str = "data/labeled"):
    """Generate annotation file for training."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    annotations = []

    for idx, row in labels_df.iterrows():
        annotations.append({
            "sample_id": row["sample_id"],
            "diagnosis": row["diagnosis"],
            "annotator": "synthetic_data",
            "confidence": 1.0,
            "notes": f"Synthetic sample for {row['diagnosis']}",
            "timestamp": datetime.now().isoformat()
        })

    with open(output_dir / "annotations.json", "w") as f:
        json.dump(annotations, f, indent=2)

    print(f"\n✓ Annotations saved to {output_dir}/annotations.json")


def main():
    """Generate all test data."""
    print("="*60)
    print("Generating Synthetic Multi-Omics Test Data")
    print("="*60)

    # Generate dataset
    dataset = generate_dataset(samples_per_disease=25)

    # Save to files
    save_dataset(dataset)

    # Generate annotations
    generate_annotations(dataset["labels"])

    # Generate train/val/test split
    print("\nCreating train/val/test split...")
    labels_df = dataset["labels"]

    # 70/15/15 split
    train_samples = []
    val_samples = []
    test_samples = []

    for disease in DISEASES:
        disease_samples = labels_df[labels_df["diagnosis"] == disease]["sample_id"].tolist()
        n = len(disease_samples)

        train_n = int(n * 0.7)
        val_n = int(n * 0.15)

        train_samples.extend(disease_samples[:train_n])
        val_samples.extend(disease_samples[train_n:train_n+val_n])
        test_samples.extend(disease_samples[train_n+val_n:])

    splits = {
        "train": train_samples,
        "validation": val_samples,
        "test": test_samples
    }

    with open("data/labeled/splits.json", "w") as f:
        json.dump(splits, f, indent=2)

    print(f"  Train: {len(train_samples)} samples")
    print(f"  Validation: {len(val_samples)} samples")
    print(f"  Test: {len(test_samples)} samples")

    print("\n" + "="*60)
    print("Test Data Generation Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
