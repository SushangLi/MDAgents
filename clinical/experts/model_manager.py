"""
Model Manager for Expert Agents.

Centralized management of expert models including loading, saving,
and version control.
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
import json
from datetime import datetime

from clinical.experts.microbiome_expert import MicrobiomeExpert
from clinical.experts.metabolome_expert import MetabolomeExpert
from clinical.experts.proteome_expert import ProteomeExpert
from clinical.experts.base_expert import BaseExpert


class ModelManager:
    """
    Manages loading and saving of expert models.

    Provides centralized access to all expert models with version control.
    """

    def __init__(self, models_dir: str = "data/models"):
        """
        Initialize model manager.

        Args:
            models_dir: Directory to store/load models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Registry of loaded models
        self.loaded_models: Dict[str, BaseExpert] = {}

        # Model metadata
        self.metadata_file = self.models_dir / "models_metadata.json"
        self.metadata = self._load_metadata()

    def save_expert(
        self,
        expert: BaseExpert,
        version: Optional[str] = None,
        notes: str = ""
    ):
        """
        Save an expert model.

        Args:
            expert: Expert model to save
            version: Version string (default: auto-generate timestamp)
            notes: Optional notes about this version
        """
        if not expert.is_fitted_:
            raise ValueError("Cannot save unfitted model")

        # Generate version if not provided
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create filename
        filename = f"{expert.expert_name}_v{version}.pkl"
        filepath = self.models_dir / filename

        # Save model
        expert.save_model(str(filepath))

        # Update metadata
        model_info = {
            "expert_name": expert.expert_name,
            "omics_type": expert.omics_type,
            "version": version,
            "filepath": str(filepath),
            "saved_at": datetime.now().isoformat(),
            "notes": notes,
            "model_metadata": expert.get_model_metadata()
        }

        if expert.expert_name not in self.metadata:
            self.metadata[expert.expert_name] = []

        self.metadata[expert.expert_name].append(model_info)
        self._save_metadata()

        print(f"✓ Saved {expert.expert_name} version {version} to {filepath}")

    def load_expert(
        self,
        expert_name: str,
        version: Optional[str] = None
    ) -> BaseExpert:
        """
        Load an expert model.

        Args:
            expert_name: Name of the expert ("microbiome_expert", "metabolome_expert", "proteome_expert")
            version: Version to load (default: latest)

        Returns:
            Loaded expert model
        """
        # Check if already loaded
        cache_key = f"{expert_name}_{version if version else 'latest'}"
        if cache_key in self.loaded_models:
            return self.loaded_models[cache_key]

        # Get model info
        if expert_name not in self.metadata:
            raise ValueError(f"No saved models found for {expert_name}")

        versions = self.metadata[expert_name]

        if version is None:
            # Load latest version
            model_info = versions[-1]
        else:
            # Load specific version
            matching = [v for v in versions if v["version"] == version]
            if not matching:
                raise ValueError(f"Version {version} not found for {expert_name}")
            model_info = matching[0]

        # Create expert instance
        expert = self._create_expert_instance(expert_name)

        # Load model
        expert.load_model(model_info["filepath"])

        # Cache it
        self.loaded_models[cache_key] = expert

        print(f"✓ Loaded {expert_name} version {model_info['version']}")
        return expert

    def _create_expert_instance(self, expert_name: str) -> BaseExpert:
        """Create a new expert instance based on name."""
        if expert_name == "microbiome_expert":
            return MicrobiomeExpert()
        elif expert_name == "metabolome_expert":
            return MetabolomeExpert()
        elif expert_name == "proteome_expert":
            return ProteomeExpert()
        else:
            raise ValueError(f"Unknown expert: {expert_name}")

    def load_all_experts(
        self,
        version: Optional[str] = None
    ) -> Dict[str, BaseExpert]:
        """
        Load all three expert models.

        Args:
            version: Version to load for all experts (default: latest)

        Returns:
            Dictionary mapping expert names to loaded models
        """
        expert_names = ["microbiome_expert", "metabolome_expert", "proteome_expert"]
        experts = {}

        for name in expert_names:
            try:
                experts[name] = self.load_expert(name, version)
            except Exception as e:
                print(f"Warning: Failed to load {name}: {e}")

        return experts

    def list_versions(self, expert_name: Optional[str] = None) -> Dict[str, List[Dict]]:
        """
        List all saved model versions.

        Args:
            expert_name: Name of specific expert (default: all experts)

        Returns:
            Dictionary of expert names to version info
        """
        if expert_name:
            if expert_name not in self.metadata:
                return {}
            return {expert_name: self.metadata[expert_name]}
        else:
            return self.metadata

    def delete_version(self, expert_name: str, version: str):
        """
        Delete a specific model version.

        Args:
            expert_name: Name of the expert
            version: Version to delete
        """
        if expert_name not in self.metadata:
            raise ValueError(f"No models found for {expert_name}")

        versions = self.metadata[expert_name]
        matching = [v for v in versions if v["version"] == version]

        if not matching:
            raise ValueError(f"Version {version} not found")

        model_info = matching[0]

        # Delete file
        filepath = Path(model_info["filepath"])
        if filepath.exists():
            filepath.unlink()

        # Remove from metadata
        self.metadata[expert_name] = [v for v in versions if v["version"] != version]

        # Clean up empty expert entries
        if not self.metadata[expert_name]:
            del self.metadata[expert_name]

        self._save_metadata()

        print(f"✓ Deleted {expert_name} version {version}")

    def get_model_info(self, expert_name: str, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Get detailed information about a model.

        Args:
            expert_name: Name of the expert
            version: Version (default: latest)

        Returns:
            Model information dictionary
        """
        if expert_name not in self.metadata:
            raise ValueError(f"No models found for {expert_name}")

        versions = self.metadata[expert_name]

        if version is None:
            return versions[-1]
        else:
            matching = [v for v in versions if v["version"] == version]
            if not matching:
                raise ValueError(f"Version {version} not found")
            return matching[0]

    def _load_metadata(self) -> Dict[str, List[Dict]]:
        """Load metadata from file."""
        if self.metadata_file.exists():
            with open(self.metadata_file, "r") as f:
                return json.load(f)
        else:
            return {}

    def _save_metadata(self):
        """Save metadata to file."""
        with open(self.metadata_file, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def summary(self) -> str:
        """
        Generate summary of all models.

        Returns:
            Human-readable summary string
        """
        lines = ["=== Model Manager Summary ==="]
        lines.append(f"Models Directory: {self.models_dir}")
        lines.append("")

        if not self.metadata:
            lines.append("No models saved yet.")
            return "\n".join(lines)

        for expert_name, versions in self.metadata.items():
            lines.append(f"{expert_name}:")
            lines.append(f"  Total Versions: {len(versions)}")

            if versions:
                latest = versions[-1]
                lines.append(f"  Latest Version: {latest['version']}")
                lines.append(f"  Saved At: {latest['saved_at']}")

                if latest.get('notes'):
                    lines.append(f"  Notes: {latest['notes']}")

            lines.append("")

        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation."""
        n_experts = len(self.metadata)
        total_versions = sum(len(v) for v in self.metadata.values())
        return f"ModelManager(experts={n_experts}, total_versions={total_versions})"
