"""
Bilingual Clinical Diagnostic Report Generator.

Generates comprehensive markdown-formatted clinical diagnostic reports
in bilingual format (Chinese | English).
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from clinical.decision.report_generator import ReportGenerator
from clinical.models.diagnosis_result import DiagnosisResult
from clinical.models.expert_opinion import ExpertOpinion


class BilingualReportGenerator(ReportGenerator):
    """
    Generates clinical diagnostic reports in bilingual format (Chinese | English).

    Inherits from ReportGenerator and overrides methods to produce
    bilingual output with Chinese and English side by side.
    """

    def __init__(
        self,
        include_metadata: bool = True,
        include_expert_details: bool = True,
        include_biomarkers: bool = True
    ):
        """
        Initialize bilingual report generator.

        Args:
            include_metadata: Include metadata section
            include_expert_details: Include detailed expert opinions
            include_biomarkers: Include biomarker details
        """
        super().__init__(include_metadata, include_expert_details, include_biomarkers)
        print("✓ Bilingual Report Generator initialized")

    def generate_report(
        self,
        diagnosis_result: DiagnosisResult,
        patient_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate bilingual diagnostic report.

        Args:
            diagnosis_result: Diagnosis result
            patient_metadata: Optional patient metadata

        Returns:
            Bilingual markdown-formatted report
        """
        sections = []

        # Bilingual header
        sections.append(self._generate_bilingual_header(diagnosis_result, patient_metadata))

        # Bilingual executive summary
        sections.append(self._generate_bilingual_executive_summary(diagnosis_result))

        # Multi-Omics Analysis
        sections.append(self._generate_bilingual_multiumics_analysis(diagnosis_result))

        # Diagnostic Rationale
        sections.append(self._generate_bilingual_diagnostic_rationale(diagnosis_result))

        # Key Biomarkers
        if self.include_biomarkers and diagnosis_result.key_biomarkers:
            sections.append(self._generate_bilingual_biomarkers_section(diagnosis_result))

        # Expert Opinions
        if self.include_expert_details:
            sections.append(self._generate_bilingual_expert_opinions_section(diagnosis_result))

        # Conflict Resolution
        if diagnosis_result.conflict_resolution:
            sections.append(self._generate_bilingual_conflict_resolution_section(diagnosis_result))

        # Recommendations
        sections.append(self._generate_bilingual_recommendations_section(diagnosis_result))

        # References
        if diagnosis_result.references or diagnosis_result.conflict_resolution:
            sections.append(self._generate_bilingual_references_section(diagnosis_result))

        # Limitations
        sections.append(self._generate_bilingual_limitations_section(diagnosis_result))

        # Metadata
        if self.include_metadata:
            sections.append(self._generate_bilingual_metadata_section(diagnosis_result))

        # Footer
        sections.append(self._generate_bilingual_footer())

        return "\n\n".join(sections)

    def _generate_bilingual_header(
        self,
        diagnosis_result: DiagnosisResult,
        patient_metadata: Optional[Dict[str, Any]]
    ) -> str:
        """Generate bilingual report header."""
        lines = []

        lines.append("# 多组学临床诊断报告 | Multi-Omics Clinical Diagnostic Report")
        lines.append("")
        lines.append("---")
        lines.append("")

        lines.append("## 患者信息 | Patient Information")
        lines.append("")

        if patient_metadata:
            for key, value in patient_metadata.items():
                cn_key, en_key = self._translate_key(key)
                lines.append(f"- **{cn_key} | {en_key}**: {value}")

        lines.append(f"- **报告日期 | Report Date**: {diagnosis_result.timestamp}")
        lines.append(f"- **患者编号 | Patient ID**: {diagnosis_result.patient_id or 'N/A'}")
        lines.append("")
        lines.append("---")

        return "\n".join(lines)

    def _generate_bilingual_executive_summary(self, diagnosis_result: DiagnosisResult) -> str:
        """Generate bilingual executive summary."""
        lines = []

        lines.append("## 执行摘要 | Executive Summary")
        lines.append("")

        lines.append("### 最终诊断 | Final Diagnosis")
        lines.append("")

        # Split or translate diagnosis
        diagnosis_cn, diagnosis_en = self._split_or_translate(diagnosis_result.diagnosis)
        lines.append(f"**{diagnosis_cn} | {diagnosis_en}**")
        lines.append("")

        # Confidence level
        conf = diagnosis_result.confidence
        if conf >= 0.85:
            label_cn, label_en = "高", "High"
            emoji = "✅"
        elif conf >= 0.70:
            label_cn, label_en = "中等", "Moderate"
            emoji = "⚠️"
        else:
            label_cn, label_en = "低", "Low"
            emoji = "❌"

        lines.append(f"**置信度 | Confidence**: {label_cn} {emoji} | {label_en} {emoji} ({conf:.1%})")
        lines.append("")

        # Key findings
        lines.append("### 关键发现 | Key Findings")
        lines.append("")

        num_experts = len(diagnosis_result.expert_opinions)
        lines.append(f"- **专家共识 | Expert Consensus**: {num_experts} 位专家意见 | {num_experts} expert opinions analyzed")

        if diagnosis_result.conflict_resolution:
            method = diagnosis_result.conflict_resolution.get('method', 'N/A')
            method_cn, method_en = self._translate_method(method)
            lines.append(f"- **冲突解决 | Conflict Resolution**: {method_cn} | {method_en}")
        else:
            lines.append("- **冲突解决 | Conflict Resolution**: 未需要（达成共识） | Not required (consensus achieved)")

        if diagnosis_result.key_biomarkers:
            top_3 = [b.get("name", "N/A") for b in diagnosis_result.key_biomarkers[:3]]
            lines.append(f"- **关键生物标志物 | Key Biomarkers**: {', '.join(top_3)}")

        return "\n".join(lines)

    def _generate_bilingual_multiumics_analysis(self, diagnosis_result: DiagnosisResult) -> str:
        """Generate bilingual multi-omics analysis."""
        lines = []

        lines.append("## 多组学分析 | Multi-Omics Analysis")
        lines.append("")

        # Group opinions by omics type
        omics_groups = {}
        for opinion in diagnosis_result.expert_opinions:
            omics_type = opinion.omics_type
            if omics_type not in omics_groups:
                omics_groups[omics_type] = []
            omics_groups[omics_type].append(opinion)

        for omics_type, opinions in omics_groups.items():
            omics_cn, omics_en = self._translate_omics(omics_type)
            lines.append(f"### {omics_cn} | {omics_en}")
            lines.append("")

            for opinion in opinions:
                diag_cn, diag_en = self._split_or_translate(opinion.diagnosis)
                lines.append(f"- **诊断 | Diagnosis**: {diag_cn} | {diag_en}")
                lines.append(f"- **概率 | Probability**: {opinion.probability:.1%}")
                lines.append(f"- **置信度 | Confidence**: {opinion.confidence:.1%}")

                if opinion.biological_explanation:
                    lines.append(f"- **生物学解释 | Biological Explanation**: {opinion.biological_explanation}")

                lines.append("")

        return "\n".join(lines)

    def _generate_bilingual_diagnostic_rationale(self, diagnosis_result: DiagnosisResult) -> str:
        """Generate bilingual diagnostic rationale with CoT reasoning."""
        lines = []

        lines.append("## 诊断理由 | Diagnostic Rationale")
        lines.append("")

        # If conflict resolution exists, show debate evolution and CMO reasoning
        if diagnosis_result.conflict_resolution:
            cr = diagnosis_result.conflict_resolution

            # Show debate evolution (round-by-round changes)
            if cr.get("threshold_history"):
                lines.append("### 辩论演化 | Debate Evolution")
                lines.append("")

                threshold_hist = cr["threshold_history"]

                # Deduplicate by round number (keep only unique rounds)
                unique_rounds = {}
                for round_data in threshold_hist:
                    round_num = round_data.get("round", 0)
                    if round_num not in unique_rounds:
                        unique_rounds[round_num] = round_data

                # Sort by round number and display
                for round_num in sorted(unique_rounds.keys()):
                    round_data = unique_rounds[round_num]
                    expert_ops = round_data.get("expert_opinions", [])

                    # Group diagnoses by count
                    diag_counts = {}
                    for op in expert_ops:
                        diag = op.get("diagnosis", "Unknown")
                        diag_counts[diag] = diag_counts.get(diag, 0) + 1

                    lines.append(f"**第 {round_num} 轮 | Round {round_num}**:")
                    for diag, count in sorted(diag_counts.items(), key=lambda x: x[1], reverse=True):
                        lines.append(f"- {diag}: {count} 位专家 | {count} expert(s)")

                    lines.append("")

                # Show consensus status
                is_resolved = cr.get("debate_resolved", False)
                if is_resolved:
                    lines.append("**共识状态 | Consensus Status**: ✅ 专家已达成共识 | Experts reached consensus")
                else:
                    lines.append("**共识状态 | Consensus Status**: ⚠ 存在分歧，由CMO做出最终裁定 | Divergent opinions, CMO made final adjudication")
                lines.append("")

            # Show CMO Chain-of-Thought reasoning
            cot_response = cr.get("cmo_cot_response")
            if cot_response:
                lines.append("### CMO推理思维链 | CMO Chain-of-Thought Reasoning")
                lines.append("")

                cot_steps = [
                    ("step1_consensus_analysis", "步骤1: 专家共识分析 | Step 1: Expert Consensus Analysis"),
                    ("step2_biomarker_evaluation", "步骤2: 生物标志物评估 | Step 2: Biomarker Evaluation"),
                    ("step3_external_evidence", "步骤3: 外部证据整合 | Step 3: External Evidence Integration"),
                    ("step4_alternatives", "步骤4: 备择诊断考虑 | Step 4: Alternative Diagnoses Considered"),
                    ("step5_evidence_weighting", "步骤5: 证据权重分配 | Step 5: Evidence Weighting"),
                    ("step6_final_diagnosis", "步骤6: 最终结论 | Step 6: Final Conclusion")
                ]

                for step_key, step_title in cot_steps:
                    step_content = cot_response.get(step_key)
                    if step_content:
                        lines.append(f"**{step_title}**")
                        lines.append("")
                        lines.append(step_content)
                        lines.append("")

        # Show general explanation
        if diagnosis_result.explanation:
            lines.append("### 综合诊断说明 | Comprehensive Diagnostic Explanation")
            lines.append("")
            lines.append(diagnosis_result.explanation)
        else:
            lines.append("基于专家共识和生物标志物证据 | Based on expert consensus and biomarker evidence")

        return "\n".join(lines)

    def _generate_bilingual_biomarkers_section(self, diagnosis_result: DiagnosisResult) -> str:
        """Generate bilingual biomarkers table."""
        lines = []

        lines.append("## 关键生物标志物 | Key Biomarkers")
        lines.append("")

        # Bilingual table
        lines.append("| 标志物<br>Biomarker | 组学类型<br>Omics | 方向<br>Direction | 重要性<br>Importance | 描述<br>Description |")
        lines.append("|---------------------|-------------------|-------------------|----------------------|---------------------|")

        for bm in diagnosis_result.key_biomarkers[:10]:
            omics_cn, omics_en = self._translate_omics(bm.get('omics_type', 'N/A'))
            dir_cn, dir_en = self._translate_direction(bm.get('direction', 'N/A'))

            name = bm.get('name', 'N/A')
            importance = bm.get('importance', 0)
            description = bm.get('description', 'N/A')

            lines.append(
                f"| {name} | "
                f"{omics_cn}<br>{omics_en} | "
                f"{dir_cn}<br>{dir_en} | "
                f"{importance:.3f} | "
                f"{description} |"
            )

        return "\n".join(lines)

    def _generate_bilingual_expert_opinions_section(self, diagnosis_result: DiagnosisResult) -> str:
        """Generate bilingual expert opinions."""
        lines = []

        lines.append("## 专家意见详情 | Detailed Expert Opinions")
        lines.append("")

        for i, opinion in enumerate(diagnosis_result.expert_opinions, 1):
            omics_cn, omics_en = self._translate_omics(opinion.omics_type)
            lines.append(f"### 专家 {i} | Expert {i}: {omics_cn} | {omics_en}")
            lines.append("")

            diag_cn, diag_en = self._split_or_translate(opinion.diagnosis)
            lines.append(f"- **诊断 | Diagnosis**: {diag_cn} | {diag_en}")
            lines.append(f"- **概率 | Probability**: {opinion.probability:.1%}")
            lines.append(f"- **置信度 | Confidence**: {opinion.confidence:.1%}")
            lines.append("")

            if opinion.top_features:
                lines.append("**关键特征 | Top Features**:")
                for feature in opinion.top_features[:5]:
                    dir_cn, dir_en = self._translate_direction(feature.direction)
                    lines.append(
                        f"- {feature.feature_name}: {dir_cn} | {dir_en} "
                        f"(importance: {feature.importance_score:.3f})"
                    )
                lines.append("")

        return "\n".join(lines)

    def _generate_bilingual_conflict_resolution_section(self, diagnosis_result: DiagnosisResult) -> str:
        """Generate bilingual conflict resolution section."""
        lines = []

        lines.append("## 冲突解决过程 | Conflict Resolution Process")
        lines.append("")

        if diagnosis_result.conflict_resolution:
            resolution = diagnosis_result.conflict_resolution

            method = resolution.get('method', 'N/A')
            method_cn, method_en = self._translate_method(method)
            lines.append(f"- **解决方法 | Resolution Method**: {method_cn} | {method_en}")

            if 'debate_rounds' in resolution:
                rounds = resolution['debate_rounds']
                lines.append(f"- **辩论轮次 | Debate Rounds**: {rounds}")

            if 'rag_used' in resolution:
                used = "是 | Yes" if resolution['rag_used'] else "否 | No"
                lines.append(f"- **使用文献检索 | RAG Used**: {used}")

            if 'cag_used' in resolution:
                used = "是 | Yes" if resolution['cag_used'] else "否 | No"
                lines.append(f"- **使用案例检索 | CAG Used**: {used}")

        return "\n".join(lines)

    def _generate_bilingual_recommendations_section(self, diagnosis_result: DiagnosisResult) -> str:
        """Generate bilingual clinical recommendations."""
        lines = []

        lines.append("## 临床建议 | Clinical Recommendations")
        lines.append("")

        if diagnosis_result.clinical_recommendations:
            for i, rec in enumerate(diagnosis_result.clinical_recommendations, 1):
                # Recommendations should already be bilingual from CMO
                lines.append(f"{i}. {rec}")
        else:
            lines.append("1. 遵循标准治疗方案 | Follow standard treatment protocols")
            lines.append("2. 监测患者进展 | Monitor patient progress")
            lines.append("3. 考虑后续诊断 | Consider follow-up diagnostics")

        return "\n".join(lines)

    def _generate_bilingual_references_section(self, diagnosis_result: DiagnosisResult) -> str:
        """Generate bilingual references."""
        lines = []

        lines.append("## 参考文献和证据 | References and Evidence")
        lines.append("")

        if diagnosis_result.references:
            for i, ref in enumerate(diagnosis_result.references, 1):
                lines.append(f"{i}. {ref}")

        return "\n".join(lines)

    def _generate_bilingual_limitations_section(self, diagnosis_result: DiagnosisResult) -> str:
        """Generate bilingual limitations section."""
        lines = []

        lines.append("## 限制和注意事项 | Limitations and Considerations")
        lines.append("")

        lines.append("本报告基于多组学数据分析，仅供临床参考。最终诊断应结合临床检查和医生专业判断。")
        lines.append("")
        lines.append("This report is based on multi-omics data analysis and is for clinical reference only. "
                    "Final diagnosis should be made in conjunction with clinical examination and professional medical judgment.")

        return "\n".join(lines)

    def _generate_bilingual_metadata_section(self, diagnosis_result: DiagnosisResult) -> str:
        """Generate bilingual metadata."""
        lines = []

        lines.append("## 技术信息 | Technical Information")
        lines.append("")

        if diagnosis_result.metadata:
            for key, value in diagnosis_result.metadata.items():
                cn_key, en_key = self._translate_key(key)
                lines.append(f"- **{cn_key} | {en_key}**: {value}")

        return "\n".join(lines)

    def _generate_bilingual_footer(self) -> str:
        """Generate bilingual footer."""
        lines = []

        lines.append("---")
        lines.append("")
        lines.append("*报告生成时间 | Report Generated*: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        lines.append("")
        lines.append("*本报告由AI辅助诊断系统生成 | This report was generated by an AI-assisted diagnostic system*")

        return "\n".join(lines)

    # Translation helper methods

    def _split_or_translate(self, text: str) -> Tuple[str, str]:
        """Split bilingual text or translate if monolingual."""
        if " | " in text:
            parts = text.split(" | ", 1)
            return parts[0], parts[1]

        # Translation mapping for common diagnoses
        translations = {
            "Periodontitis": ("牙周炎", "Periodontitis"),
            "Gingivitis": ("牙龈炎", "Gingivitis"),
            "Healthy": ("健康", "Healthy"),
            "Diabetes": ("糖尿病", "Diabetes"),
            "Normal": ("正常", "Normal")
        }
        return translations.get(text, (text, text))

    def _translate_key(self, key: str) -> Tuple[str, str]:
        """Translate metadata keys to Chinese and English."""
        translations = {
            "patient_id": ("患者编号", "Patient ID"),
            "age": ("年龄", "Age"),
            "sex": ("性别", "Sex"),
            "gender": ("性别", "Gender"),
            "diagnosis": ("诊断", "Diagnosis"),
            "confidence": ("置信度", "Confidence"),
            "timestamp": ("时间戳", "Timestamp"),
            "model_version": ("模型版本", "Model Version"),
            "processing_time": ("处理时间", "Processing Time")
        }

        key_lower = key.lower().replace("_", " ")
        for eng_key, (cn, en) in translations.items():
            if eng_key in key_lower:
                return (cn, en)

        # Default: capitalize English, use as-is for Chinese
        return (key.replace("_", " ").title(), key.replace("_", " ").title())

    def _translate_omics(self, omics_type: str) -> Tuple[str, str]:
        """Translate omics type."""
        translations = {
            "microbiome": ("微生物组", "Microbiome"),
            "metabolome": ("代谢组", "Metabolome"),
            "proteome": ("蛋白质组", "Proteome"),
            "genomics": ("基因组", "Genomics"),
            "transcriptomics": ("转录组", "Transcriptomics")
        }
        return translations.get(omics_type.lower(), (omics_type, omics_type))

    def _translate_direction(self, direction: str) -> Tuple[str, str]:
        """Translate regulation direction."""
        translations = {
            "up": ("上调", "Upregulated"),
            "down": ("下调", "Downregulated"),
            "upregulated": ("上调", "Upregulated"),
            "downregulated": ("下调", "Downregulated"),
            "N/A": ("N/A", "N/A"),
            "none": ("无变化", "No change")
        }
        return translations.get(direction.lower(), (direction, direction))

    def _translate_method(self, method: str) -> Tuple[str, str]:
        """Translate resolution method."""
        translations = {
            "cmo_llm_reasoning": ("CMO-LLM推理", "CMO LLM Reasoning"),
            "voting": ("投票", "Voting"),
            "fallback_voting": ("回退投票", "Fallback Voting"),
            "consensus": ("共识", "Consensus"),
            "debate": ("辩论", "Debate"),
            "rag_assisted": ("文献辅助", "RAG-Assisted"),
            "cag_assisted": ("案例辅助", "CAG-Assisted")
        }
        return translations.get(method.lower(), (method, method))
