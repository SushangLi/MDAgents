"""
Generate synthetic omics data with OBVIOUS FEATURES for testing debate system.

Each disease has very distinctive biomarker patterns to avoid misclassification.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime


# Disease classes with VERY DISTINCT features
DISEASES = ["Periodontitis", "Diabetes", "Healthy"]

# Microbiome taxa
MICROBIOME_TAXA = [
    "Porphyromonas_gingivalis",    # HIGH in Periodontitis
    "Treponema_denticola",          # HIGH in Periodontitis
    "Prevotella_intermedia",        # HIGH in Diabetes
    "Fusobacterium_nucleatum",      # HIGH in Diabetes
    "Streptococcus_salivarius",     # HIGH in Healthy
    "Lactobacillus_reuteri",        # HIGH in Healthy
    "Veillonella_parvula",
    "Actinomyces_naeslundii"
]

# Metabolites
METABOLITES = [
    "Butyrate",         # HIGH in Periodontitis
    "Propionate",       # HIGH in Periodontitis
    "Lactate",          # HIGH in Diabetes
    "Glucose",          # HIGH in Diabetes
    "GABA",             # HIGH in Healthy
    "Acetate",
    "Succinate"
]

# Proteins
PROTEINS = [
    "MMP9",      # HIGH in Periodontitis
    "IL6",       # HIGH in Periodontitis
    "TNF",       # HIGH in Diabetes
    "CRP",       # HIGH in Diabetes
    "IgA",       # HIGH in Healthy
    "Lactoferrin",  # HIGH in Healthy
    "IL8"
]


def generate_microbiome_data(n_samples: int, disease: str) -> pd.DataFrame:
    """Generate microbiome with OBVIOUS disease-specific patterns."""
    np.random.seed(42 + hash(disease) % 1000)

    data = {}

    for taxon in MICROBIOME_TAXA:
        # Base abundance (very low)
        base = np.random.uniform(0.001, 0.01, n_samples)

        if disease == "Periodontitis":
            if taxon in ["Porphyromonas_gingivalis", "Treponema_denticola"]:
                # EXTREMELY HIGH (10-20x)
                base *= np.random.uniform(15, 25, n_samples)
            elif taxon in ["Streptococcus_salivarius", "Lactobacillus_reuteri"]:
                # Very low
                base *= np.random.uniform(0.05, 0.1, n_samples)

        elif disease == "Diabetes":
            if taxon in ["Prevotella_intermedia", "Fusobacterium_nucleatum"]:
                # EXTREMELY HIGH (10-20x)
                base *= np.random.uniform(15, 25, n_samples)
            elif taxon in ["Streptococcus_salivarius", "Lactobacillus_reuteri"]:
                # Very low
                base *= np.random.uniform(0.05, 0.1, n_samples)

        elif disease == "Healthy":
            if taxon in ["Streptococcus_salivarius", "Lactobacillus_reuteri"]:
                # EXTREMELY HIGH (10-20x)
                base *= np.random.uniform(15, 25, n_samples)
            elif taxon in ["Porphyromonas_gingivalis", "Treponema_denticola",
                          "Prevotella_intermedia", "Fusobacterium_nucleatum"]:
                # Very low (almost zero)
                base *= np.random.uniform(0.01, 0.05, n_samples)

        # Add minimal noise
        base += np.random.normal(0, 0.001, n_samples)
        base = np.clip(base, 0.0001, None)

        data[taxon] = base

    # Normalize to relative abundance
    df = pd.DataFrame(data)
    df = df.div(df.sum(axis=1), axis=0)

    return df


def generate_metabolome_data(n_samples: int, disease: str) -> pd.DataFrame:
    """Generate metabolome with OBVIOUS disease-specific patterns."""
    np.random.seed(43 + hash(disease) % 1000)

    data = {}

    for metabolite in METABOLITES:
        # Base intensity (low)
        base = np.random.lognormal(mean=4, sigma=0.3, size=n_samples)

        if disease == "Periodontitis":
            if metabolite in ["Butyrate", "Propionate"]:
                # EXTREMELY HIGH (20-40x)
                base *= np.random.uniform(25, 45, n_samples)
            elif metabolite in ["GABA"]:
                # Very low
                base *= np.random.uniform(0.05, 0.1, n_samples)

        elif disease == "Diabetes":
            if metabolite in ["Lactate", "Glucose"]:
                # EXTREMELY HIGH (20-40x)
                base *= np.random.uniform(25, 45, n_samples)
            elif metabolite in ["GABA"]:
                # Very low
                base *= np.random.uniform(0.05, 0.1, n_samples)

        elif disease == "Healthy":
            if metabolite == "GABA":
                # EXTREMELY HIGH (20-40x)
                base *= np.random.uniform(25, 45, n_samples)
            elif metabolite in ["Butyrate", "Propionate", "Lactate", "Glucose"]:
                # Very low
                base *= np.random.uniform(0.05, 0.1, n_samples)

        # Minimal noise
        base += np.random.normal(0, base * 0.02, n_samples)
        base = np.clip(base, 10, None)

        data[metabolite] = base

    return pd.DataFrame(data)


def generate_proteome_data(n_samples: int, disease: str) -> pd.DataFrame:
    """Generate proteome with OBVIOUS disease-specific patterns."""
    np.random.seed(44 + hash(disease) % 1000)

    data = {}

    for protein in PROTEINS:
        # Base expression (low)
        base = np.random.lognormal(mean=2, sigma=0.3, size=n_samples)

        if disease == "Periodontitis":
            if protein in ["MMP9", "IL6"]:
                # EXTREMELY HIGH (30-50x)
                base *= np.random.uniform(35, 55, n_samples)
            elif protein in ["IgA", "Lactoferrin"]:
                # Very low
                base *= np.random.uniform(0.05, 0.1, n_samples)

        elif disease == "Diabetes":
            if protein in ["TNF", "CRP"]:
                # EXTREMELY HIGH (30-50x)
                base *= np.random.uniform(35, 55, n_samples)
            elif protein in ["IgA", "Lactoferrin"]:
                # Very low
                base *= np.random.uniform(0.05, 0.1, n_samples)

        elif disease == "Healthy":
            if protein in ["IgA", "Lactoferrin"]:
                # EXTREMELY HIGH (30-50x)
                base *= np.random.uniform(35, 55, n_samples)
            elif protein in ["MMP9", "IL6", "TNF", "CRP"]:
                # Very low (almost zero)
                base *= np.random.uniform(0.02, 0.05, n_samples)

        # Minimal noise
        base += np.random.normal(0, base * 0.02, n_samples)
        base = np.clip(base, 5, None)

        data[protein] = base

    return pd.DataFrame(data)


def generate_dataset(samples_per_disease: int = 30) -> dict:
    """Generate complete multi-omics dataset with OBVIOUS features."""
    all_data = {
        "microbiome": [],
        "metabolome": [],
        "proteome": [],
        "labels": []
    }

    sample_ids = []

    for disease in DISEASES:
        print(f"Generating {samples_per_disease} samples for {disease}...")
        print(f"  Distinctive features:")

        if disease == "Periodontitis":
            print(f"    - HIGH: P.gingivalis, T.denticola, Butyrate, Propionate, MMP9, IL6")
            print(f"    - LOW: Beneficial bacteria, GABA, IgA")
        elif disease == "Diabetes":
            print(f"    - HIGH: Prevotella, Fusobacterium, Lactate, Glucose, TNF, CRP")
            print(f"    - LOW: Beneficial bacteria, GABA, IgA")
        else:  # Healthy
            print(f"    - HIGH: Streptococcus, Lactobacillus, GABA, IgA, Lactoferrin")
            print(f"    - LOW: All pathogens, inflammatory markers")

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


def save_dataset(dataset: dict, output_dir: str = "data/training"):
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


def generate_annotations(labels_df: pd.DataFrame, output_dir: str = "data/training"):
    """Generate annotation file for training."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    annotations = []

    for idx, row in labels_df.iterrows():
        annotations.append({
            "sample_id": row["sample_id"],
            "diagnosis": row["diagnosis"],
            "annotator": "synthetic_obvious_features",
            "confidence": 1.0,
            "notes": f"Synthetic sample with obvious {row['diagnosis']} features",
            "timestamp": datetime.now().isoformat()
        })

    with open(output_dir / "annotations.json", "w") as f:
        json.dump(annotations, f, indent=2)

    print(f"\n✓ Annotations saved to {output_dir}/annotations.json")


def create_splits(labels_df: pd.DataFrame, output_dir: str = "data/training"):
    """Create train/test splits (80/20)."""
    output_dir = Path(output_dir)

    train_samples = []
    test_samples = []

    for disease in DISEASES:
        disease_samples = labels_df[labels_df["diagnosis"] == disease]["sample_id"].tolist()
        n = len(disease_samples)

        train_n = int(n * 0.8)

        train_samples.extend(disease_samples[:train_n])
        test_samples.extend(disease_samples[train_n:])

    splits = {
        "train": train_samples,
        "test": test_samples
    }

    with open(output_dir / "splits.json", "w") as f:
        json.dump(splits, f, indent=2)

    print(f"\n✓ Train/Test split created:")
    print(f"  Train: {len(train_samples)} samples")
    print(f"  Test: {len(test_samples)} samples")


def main():
    """Generate all training data with obvious features."""
    print("\n" + "="*60)
    print("Generating Training Data with OBVIOUS Disease Features")
    print("="*60)
    print("\nFeature Design:")
    print("- Periodontitis: HIGH periodontal pathogens + inflammatory markers")
    print("- Diabetes: HIGH diabetes-related bacteria + metabolites")
    print("- Healthy: HIGH beneficial bacteria + protective proteins")
    print("="*60)

    # Generate dataset (30 samples per disease = 90 total)
    dataset = generate_dataset(samples_per_disease=30)

    # Save to files
    save_dataset(dataset)

    # Generate annotations
    generate_annotations(dataset["labels"])

    # Create train/test split
    create_splits(dataset["labels"])

    print("\n" + "="*60)
    print("Training Data Generation Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Train models: python main_clinical.py train")
    print("2. Test with mismatched data to trigger debate")


if __name__ == "__main__":
    main()
