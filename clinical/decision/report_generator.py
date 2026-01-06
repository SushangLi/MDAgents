"""
Clinical Diagnostic Report Generator.

Generates comprehensive markdown-formatted clinical diagnostic reports
from diagnosis results.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from clinical.models.diagnosis_result import DiagnosisResult
from clinical.models.expert_opinion import ExpertOpinion


class ReportGenerator:
    """
    Generates clinical diagnostic reports in markdown format.

    Produces comprehensive reports with executive summary,
    multi-omics analysis, diagnostic rationale, and recommendations.
    """

    def __init__(
        self,
        include_metadata: bool = True,
        include_expert_details: bool = True,
        include_biomarkers: bool = True
    ):
        """
        Initialize report generator.

        Args:
            include_metadata: Include metadata section
            include_expert_details: Include detailed expert opinions
            include_biomarkers: Include biomarker details
        """
        self.include_metadata = include_metadata
        self.include_expert_details = include_expert_details
        self.include_biomarkers = include_biomarkers

        print("✓ Report Generator initialized")

    def generate_report(
        self,
        diagnosis_result: DiagnosisResult,
        patient_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate complete diagnostic report.

        Args:
            diagnosis_result: Diagnosis result
            patient_metadata: Optional patient metadata

        Returns:
            Markdown-formatted report
        """
        sections = []

        # Header
        sections.append(self._generate_header(diagnosis_result, patient_metadata))

        # Executive Summary
        sections.append(self._generate_executive_summary(diagnosis_result))

        # Multi-Omics Analysis
        sections.append(self._generate_multiumics_analysis(diagnosis_result))

        # Diagnostic Rationale
        sections.append(self._generate_diagnostic_rationale(diagnosis_result))

        # Key Biomarkers
        if self.include_biomarkers and diagnosis_result.key_biomarkers:
            sections.append(self._generate_biomarkers_section(diagnosis_result))

        # Expert Opinions
        if self.include_expert_details:
            sections.append(self._generate_expert_opinions_section(diagnosis_result))

        # Conflict Resolution (if applicable)
        if diagnosis_result.conflict_resolution:
            sections.append(self._generate_conflict_resolution_section(diagnosis_result))

        # Differential Diagnoses
        if diagnosis_result.differential_diagnoses:
            sections.append(self._generate_differential_diagnoses_section(diagnosis_result))

        # Clinical Recommendations
        sections.append(self._generate_recommendations_section(diagnosis_result))

        # References and Evidence
        if diagnosis_result.rag_citations or diagnosis_result.cag_similar_cases:
            sections.append(self._generate_references_section(diagnosis_result))

        # Limitations and Follow-up
        sections.append(self._generate_limitations_section(diagnosis_result))

        # Metadata
        if self.include_metadata:
            sections.append(self._generate_metadata_section(diagnosis_result))

        # Footer
        sections.append(self._generate_footer())

        return "\n\n".join(sections)

    def _generate_header(
        self,
        diagnosis_result: DiagnosisResult,
        patient_metadata: Optional[Dict[str, Any]]
    ) -> str:
        """Generate report header."""
        lines = []

        lines.append("# Multi-Omics Clinical Diagnostic Report")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Patient info
        if patient_metadata:
            lines.append("## Patient Information")
            lines.append("")
            for key, value in patient_metadata.items():
                lines.append(f"- **{key.replace('_', ' ').title()}**: {value}")
        else:
            lines.append(f"- **Patient ID**: {diagnosis_result.patient_id or 'N/A'}")

        lines.append(f"- **Report Date**: {diagnosis_result.timestamp}")
        lines.append(f"- **Report ID**: {id(diagnosis_result)}")  # Simple ID
        lines.append("")
        lines.append("---")

        return "\n".join(lines)

    def _generate_executive_summary(self, diagnosis_result: DiagnosisResult) -> str:
        """Generate executive summary."""
        lines = []

        lines.append("## Executive Summary")
        lines.append("")

        # Diagnosis box
        lines.append("### Final Diagnosis")
        lines.append("")
        lines.append(f"**{diagnosis_result.diagnosis}**")
        lines.append("")

        # Confidence indicator
        conf = diagnosis_result.confidence
        if conf >= 0.85:
            conf_label = "High"
            conf_emoji = "✅"
        elif conf >= 0.70:
            conf_label = "Moderate"
            conf_emoji = "⚠️"
        else:
            conf_label = "Low"
            conf_emoji = "❌"

        lines.append(f"**Confidence Level**: {conf_label} {conf_emoji} ({conf:.1%})")
        lines.append("")

        # Quick summary
        lines.append("### Key Findings")
        lines.append("")
        lines.append(f"- **Expert Consensus**: {len(diagnosis_result.expert_opinions)} expert opinions analyzed")

        if diagnosis_result.conflict_resolution:
            lines.append(f"- **Conflict Resolution**: {diagnosis_result.conflict_resolution.resolution_method}")
            if diagnosis_result.conflict_resolution.debate_rounds > 0:
                lines.append(f"- **Debate Rounds**: {diagnosis_result.conflict_resolution.debate_rounds}")
        else:
            lines.append("- **Conflict Resolution**: Not required (consensus achieved)")

        if diagnosis_result.key_biomarkers:
            top_biomarkers = [b["name"] for b in diagnosis_result.key_biomarkers[:3]]
            lines.append(f"- **Key Biomarkers**: {', '.join(top_biomarkers)}")

        return "\n".join(lines)

    def _generate_multiumics_analysis(self, diagnosis_result: DiagnosisResult) -> str:
        """Generate multi-omics analysis section."""
        lines = []

        lines.append("## Multi-Omics Analysis")
        lines.append("")

        # Group expert opinions by omics type
        omics_summary = {}
        for opinion in diagnosis_result.expert_opinions:
            omics_summary[opinion.omics_type] = {
                "diagnosis": opinion.diagnosis,
                "confidence": opinion.confidence,
                "probability": opinion.probability,
                "top_features": opinion.top_features[:3]
            }

        # Display each omics type
        for omics_type, data in omics_summary.items():
            lines.append(f"### {omics_type.title()} Analysis")
            lines.append("")
            lines.append(f"- **Predicted Diagnosis**: {data['diagnosis']}")
            lines.append(f"- **Confidence**: {data['confidence']:.1%}")
            lines.append(f"- **Probability**: {data['probability']:.1%}")
            lines.append("")

            if data['top_features']:
                lines.append("**Top Biomarkers**:")
                lines.append("")
                for feature in data['top_features']:
                    lines.append(
                        f"- {feature.feature_name}: {feature.direction}regulated "
                        f"(importance: {feature.importance:.3f})"
                    )
                lines.append("")

        return "\n".join(lines)

    def _generate_diagnostic_rationale(self, diagnosis_result: DiagnosisResult) -> str:
        """Generate diagnostic rationale section."""
        lines = []

        lines.append("## Diagnostic Rationale")
        lines.append("")

        if diagnosis_result.reasoning_chain:
            lines.append("### Reasoning Chain")
            lines.append("")
            for i, step in enumerate(diagnosis_result.reasoning_chain, 1):
                lines.append(f"{i}. {step}")
        else:
            lines.append("Diagnosis based on multi-expert consensus and biomarker analysis.")

        return "\n".join(lines)

    def _generate_biomarkers_section(self, diagnosis_result: DiagnosisResult) -> str:
        """Generate key biomarkers section."""
        lines = []

        lines.append("## Key Biomarkers")
        lines.append("")

        # Create table
        lines.append("| Biomarker | Omics Type | Direction | Importance | Description |")
        lines.append("|-----------|------------|-----------|------------|-------------|")

        for biomarker in diagnosis_result.key_biomarkers[:10]:
            lines.append(
                f"| {biomarker['name']} | "
                f"{biomarker['omics_type']} | "
                f"{biomarker.get('direction', 'N/A')} | "
                f"{biomarker.get('importance', 0):.3f} | "
                f"{biomarker.get('description', 'N/A')} |"
            )

        return "\n".join(lines)

    def _generate_expert_opinions_section(self, diagnosis_result: DiagnosisResult) -> str:
        """Generate detailed expert opinions section."""
        lines = []

        lines.append("## Expert Opinions")
        lines.append("")

        for i, opinion in enumerate(diagnosis_result.expert_opinions, 1):
            lines.append(f"### Expert {i}: {opinion.expert_name}")
            lines.append("")
            lines.append(f"**Omics Type**: {opinion.omics_type}")
            lines.append("")
            lines.append(f"**Diagnosis**: {opinion.diagnosis}")
            lines.append(f"**Probability**: {opinion.probability:.1%}")
            lines.append(f"**Confidence**: {opinion.confidence:.1%}")
            lines.append("")

            # Biological explanation
            lines.append("**Biological Explanation**:")
            lines.append("")
            lines.append(opinion.biological_explanation)
            lines.append("")

            # Evidence chain
            if opinion.evidence_chain:
                lines.append("**Evidence Chain**:")
                lines.append("")
                for evidence in opinion.evidence_chain:
                    lines.append(f"- {evidence}")
                lines.append("")

        return "\n".join(lines)

    def _generate_conflict_resolution_section(self, diagnosis_result: DiagnosisResult) -> str:
        """Generate conflict resolution details."""
        lines = []

        lines.append("## Conflict Resolution")
        lines.append("")

        resolution = diagnosis_result.conflict_resolution

        lines.append(f"**Conflict Types**: {', '.join(resolution.conflict_types)}")
        lines.append(f"**Resolution Method**: {resolution.resolution_method}")
        lines.append(f"**Debate Rounds**: {resolution.debate_rounds}")
        lines.append(f"**RAG Used**: {'Yes' if resolution.rag_used else 'No'}")
        lines.append(f"**CAG Used**: {'Yes' if resolution.cag_used else 'No'}")
        lines.append("")

        if resolution.final_reasoning:
            lines.append("**Final Reasoning**:")
            lines.append("")
            for i, reason in enumerate(resolution.final_reasoning, 1):
                lines.append(f"{i}. {reason}")

        return "\n".join(lines)

    def _generate_differential_diagnoses_section(self, diagnosis_result: DiagnosisResult) -> str:
        """Generate differential diagnoses section."""
        lines = []

        lines.append("## Differential Diagnoses")
        lines.append("")

        for diff_dx in diagnosis_result.differential_diagnoses:
            lines.append(f"### {diff_dx.get('diagnosis', 'Unknown')}")
            lines.append("")
            lines.append(f"**Probability**: {diff_dx.get('probability', 0):.1%}")
            lines.append("")
            lines.append(f"**Rationale**: {diff_dx.get('rationale', 'N/A')}")
            lines.append("")

        return "\n".join(lines)

    def _generate_recommendations_section(self, diagnosis_result: DiagnosisResult) -> str:
        """Generate clinical recommendations section."""
        lines = []

        lines.append("## Clinical Recommendations")
        lines.append("")

        if diagnosis_result.recommendations:
            for i, rec in enumerate(diagnosis_result.recommendations, 1):
                lines.append(f"{i}. {rec}")
        else:
            lines.append("1. Follow standard treatment protocols")
            lines.append("2. Monitor patient response")
            lines.append("3. Consider follow-up diagnostics")

        return "\n".join(lines)

    def _generate_references_section(self, diagnosis_result: DiagnosisResult) -> str:
        """Generate references and evidence section."""
        lines = []

        lines.append("## References and Evidence")
        lines.append("")

        # RAG citations
        if diagnosis_result.rag_citations:
            lines.append("### Supporting Medical Literature")
            lines.append("")
            for i, citation in enumerate(diagnosis_result.rag_citations, 1):
                lines.append(f"{i}. {citation}")
            lines.append("")

        # CAG similar cases
        if diagnosis_result.cag_similar_cases:
            lines.append("### Similar Historical Cases")
            lines.append("")
            for i, case in enumerate(diagnosis_result.cag_similar_cases, 1):
                lines.append(f"{i}. {case}")

        return "\n".join(lines)

    def _generate_limitations_section(self, diagnosis_result: DiagnosisResult) -> str:
        """Generate limitations and follow-up section."""
        lines = []

        lines.append("## Limitations and Follow-up")
        lines.append("")

        lines.append("### Limitations")
        lines.append("")

        # Confidence-based limitations
        if diagnosis_result.confidence < 0.7:
            lines.append("- **Low Confidence**: This diagnosis has below-threshold confidence")
            lines.append("  and should be validated with additional testing")

        if diagnosis_result.conflict_resolution:
            lines.append("- **Expert Disagreement**: Experts had conflicting opinions")
            lines.append("  which were resolved through debate and evidence review")

        lines.append("- **Multi-omics Integration**: Results are based on computational")
        lines.append("  integration of multi-omics data and should be validated clinically")
        lines.append("")

        lines.append("### Recommended Follow-up")
        lines.append("")
        lines.append("1. Clinical validation by specialist")
        lines.append("2. Additional diagnostic tests if confidence < 80%")
        lines.append("3. Longitudinal monitoring of key biomarkers")
        lines.append("4. Re-evaluation if symptoms change")

        return "\n".join(lines)

    def _generate_metadata_section(self, diagnosis_result: DiagnosisResult) -> str:
        """Generate technical metadata section."""
        lines = []

        lines.append("## Technical Metadata")
        lines.append("")

        metadata = diagnosis_result.metadata

        lines.append(f"- **Decision Type**: {metadata.get('decision_type', 'N/A')}")
        lines.append(f"- **Conflict Detected**: {metadata.get('conflict_detected', False)}")
        lines.append(f"- **Number of Experts**: {metadata.get('n_experts', len(diagnosis_result.expert_opinions))}")

        if metadata.get('llm_provider'):
            lines.append(f"- **LLM Provider**: {metadata['llm_provider']}")

        if metadata.get('llm_model'):
            lines.append(f"- **LLM Model**: {metadata['llm_model']}")

        return "\n".join(lines)

    def _generate_footer(self) -> str:
        """Generate report footer."""
        lines = []

        lines.append("---")
        lines.append("")
        lines.append("*This report was generated by the Multi-Omics Clinical Diagnosis System.*")
        lines.append("")
        lines.append("*For questions or concerns, please consult with a qualified healthcare professional.*")

        return "\n".join(lines)

    def save_report(
        self,
        report: str,
        output_path: str
    ):
        """
        Save report to file.

        Args:
            report: Report content
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            f.write(report)

        print(f"✓ Report saved to {output_path}")

    def __repr__(self) -> str:
        """String representation."""
        return f"ReportGenerator(metadata={self.include_metadata})"
