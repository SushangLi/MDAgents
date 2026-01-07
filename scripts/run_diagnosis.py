#!/usr/bin/env python3
"""
口腔多组学诊断系统 - 完整版

包含完整的RAG（文献检索）和CAG（案例检索）系统。

使用方法:
    python scripts/run_diagnosis.py                    # 交互式模式
    python scripts/run_diagnosis.py --patient-id P001  # 指定患者
    python scripts/run_diagnosis.py --help             # 帮助
"""

import sys
import argparse
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

# Load environment variables
env_local = project_root / ".env.local"
if env_local.exists():
    load_dotenv(env_local)

# Import system components
from clinical.preprocessing.microbiome_preprocessor import MicrobiomePreprocessor
from clinical.preprocessing.metabolome_preprocessor import MetabolomePreprocessor
from clinical.preprocessing.proteome_preprocessor import ProteomePreprocessor
from clinical.experts.microbiome_expert import MicrobiomeExpert
from clinical.experts.metabolome_expert import MetabolomeExpert
from clinical.experts.proteome_expert import ProteomeExpert
from clinical.decision.conflict_resolver import ConflictResolver
from clinical.decision.debate_system import DebateSystem, DebateConfig
from clinical.decision.cmo_coordinator import CMOCoordinator
from clinical.decision.report_generator import ReportGenerator
from clinical.decision.llm_wrapper import create_llm_wrapper
from clinical.collaboration.rag_system import RAGSystem
from clinical.collaboration.cag_system import CAGSystem


class OralMultiomicsDiagnosisSystem:
    """
    完整的口腔多组学诊断系统

    集成了:
    - 3个专家模型（微生物组、代谢组、蛋白质组）
    - 冲突检测和辩论系统
    - RAG文献检索系统
    - CAG案例检索系统
    - CMO决策协调器（支持LLM推理）
    - 报告生成系统
    """

    def __init__(
        self,
        use_llm: bool = True,
        enable_rag: bool = True,
        enable_cag: bool = True,
        use_mock_llm: bool = False
    ):
        """
        初始化诊断系统

        Args:
            use_llm: 是否使用LLM进行CMO决策
            enable_rag: 是否启用RAG文献检索
            enable_cag: 是否启用CAG案例检索
            use_mock_llm: 是否使用Mock LLM（测试用）
        """
        print("\n" + "="*70)
        print("初始化口腔多组学诊断系统")
        print("="*70)

        # 1. 初始化预处理器
        print("\n[1/8] 初始化预处理器...")
        self.preprocessors = {
            'microbiome': MicrobiomePreprocessor(),
            'metabolome': MetabolomePreprocessor(),
            'proteome': ProteomePreprocessor()
        }

        # 2. 初始化专家模型
        print("\n[2/8] 加载专家模型...")

        # 找到最新的模型文件
        models_dir = project_root / "data" / "models"

        def find_latest_model(prefix: str) -> str:
            """找到最新的模型文件"""
            model_files = list(models_dir.glob(f"{prefix}_expert_v*.pkl"))
            if not model_files:
                raise FileNotFoundError(f"No model files found for {prefix}_expert")
            # 按修改时间排序，返回最新的
            latest = max(model_files, key=lambda p: p.stat().st_mtime)
            return str(latest)

        self.experts = {
            'microbiome': MicrobiomeExpert(),
            'metabolome': MetabolomeExpert(),
            'proteome': ProteomeExpert()
        }

        # 加载模型
        for name, expert in self.experts.items():
            model_path = find_latest_model(name)
            expert.load_model(model_path)
            print(f"  ✓ {name}: {Path(model_path).name}")

        # 3. 初始化RAG系统
        print("\n[3/8] 初始化RAG文献检索系统...")
        if enable_rag:
            try:
                self.rag_system = RAGSystem()
                # 添加示例文献（如果知识库为空）
                if self.rag_system.vector_store.count() == 0:
                    self._initialize_sample_literature()
            except Exception as e:
                print(f"  ⚠ RAG初始化失败: {e}")
                print(f"  使用空RAG系统")
                self.rag_system = None
        else:
            print("  RAG已禁用")
            self.rag_system = None

        # 4. 初始化CAG系统
        print("\n[4/8] 初始化CAG案例检索系统...")
        if enable_cag:
            try:
                self.cag_system = CAGSystem()
                # 添加示例案例（如果数据库为空）
                if len(self.cag_system.cases) == 0:
                    self._initialize_sample_cases()
            except Exception as e:
                print(f"  ⚠ CAG初始化失败: {e}")
                print(f"  使用空CAG系统")
                self.cag_system = None
        else:
            print("  CAG已禁用")
            self.cag_system = None

        # 5. 初始化冲突检测器
        print("\n[5/8] 初始化冲突检测器...")
        self.conflict_resolver = ConflictResolver()

        # 6. 初始化辩论系统
        print("\n[6/8] 初始化辩论系统...")
        debate_config = DebateConfig(
            max_rounds=3,
            threshold_adjustment=0.1,
            confidence_threshold=0.7,
            enable_rag=enable_rag and self.rag_system is not None,
            enable_cag=enable_cag and self.cag_system is not None
        )
        self.debate_system = DebateSystem(
            conflict_resolver=self.conflict_resolver,
            rag_system=self.rag_system,
            cag_system=self.cag_system,
            config=debate_config
        )

        # 7. 初始化CMO协调器
        print("\n[7/8] 初始化CMO决策协调器...")
        if use_llm:
            llm_wrapper = create_llm_wrapper(use_mock=use_mock_llm)
            self.cmo = CMOCoordinator(
                llm_call_func=llm_wrapper.call,
                temperature=0.3
            )
        else:
            print("  LLM已禁用，使用fallback voting")
            self.cmo = CMOCoordinator()

        # 8. 初始化报告生成器
        print("\n[8/8] 初始化报告生成器...")
        self.report_generator = ReportGenerator(
            include_metadata=True,
            include_expert_details=True,
            include_biomarkers=True
        )

        print("\n" + "="*70)
        print("✓ 系统初始化完成")
        print("="*70)
        print(f"\n配置:")
        print(f"  - LLM决策: {'启用' if use_llm else '禁用'}")
        print(f"  - RAG文献检索: {'启用 ({} docs)'.format(self.rag_system.vector_store.count()) if self.rag_system else '禁用'}")
        print(f"  - CAG案例检索: {'启用 ({} cases)'.format(len(self.cag_system.cases)) if self.cag_system else '禁用'}")
        print()

    async def diagnose(
        self,
        patient_id: str,
        microbiome_data: pd.DataFrame,
        metabolome_data: pd.DataFrame,
        proteome_data: pd.DataFrame,
        patient_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        执行完整诊断流程

        Args:
            patient_id: 患者ID
            microbiome_data: 微生物组数据（DataFrame，包含该患者的行）
            metabolome_data: 代谢组数据
            proteome_data: 蛋白质组数据
            patient_metadata: 患者元数据（年龄、性别等）

        Returns:
            包含诊断结果和报告的字典
        """
        print("\n" + "="*70)
        print(f"开始诊断患者: {patient_id}")
        print("="*70)

        # 步骤1: 数据预处理
        print("\n[步骤 1/6] 预处理组学数据...")
        preprocessed = {}
        for omics_type, preprocessor in self.preprocessors.items():
            if omics_type == 'microbiome':
                result = preprocessor.transform(microbiome_data)
            elif omics_type == 'metabolome':
                result = preprocessor.transform(metabolome_data)
            elif omics_type == 'proteome':
                result = preprocessor.transform(proteome_data)

            preprocessed[omics_type] = result
            print(f"  ✓ {omics_type}: {result.data.shape}")

        # 步骤2: 专家预测
        print("\n[步骤 2/6] 专家模型预测...")
        expert_opinions = []
        for omics_type, expert in self.experts.items():
            # Extract DataFrame from PreprocessingResult
            data = preprocessed[omics_type].data
            opinion = expert.predict(data)[0]
            expert_opinions.append(opinion)
            print(f"  ✓ {omics_type}: {opinion.diagnosis} (置信度: {opinion.confidence:.1%})")

        # 步骤3: 冲突检测
        print("\n[步骤 3/6] 检测专家意见冲突...")
        conflict_analysis = self.conflict_resolver.detect_conflict(expert_opinions)

        if conflict_analysis.has_conflict:
            print(f"  ⚠ 检测到冲突:")
            print(f"    - 冲突类型: {[ct.value for ct in conflict_analysis.conflict_types]}")
            print(f"    - 诊断分布: {conflict_analysis.diagnosis_distribution}")
            print(f"    - 需要辩论: {conflict_analysis.requires_debate}")
        else:
            print(f"  ✓ 无冲突，专家意见一致")

        # 步骤4: 辩论系统（如有冲突且需要辩论）
        debate_result = None
        if conflict_analysis.requires_debate:
            print("\n[步骤 4/6] 启动辩论系统...")

            # 准备样本数据（原始数据，用于CAG）
            sample_data = {
                'microbiome': microbiome_data.iloc[0],
                'metabolome': metabolome_data.iloc[0],
                'proteome': proteome_data.iloc[0]
            }

            debate_result = self.debate_system.run_debate(
                expert_opinions=expert_opinions,
                sample_data=sample_data
            )

            print(f"  ✓ 辩论完成: {debate_result['current_round']} 轮")
        else:
            print("\n[步骤 4/6] 跳过辩论（无需辩论）")

        # 步骤5: CMO决策
        print("\n[步骤 5/6] CMO协调器做出最终决策...")

        if conflict_analysis.has_conflict:
            # 有冲突，使用冲突解决
            diagnosis_result = await self.cmo.make_conflict_resolution(
                expert_opinions=expert_opinions,
                conflict_analysis=conflict_analysis,
                rag_context=debate_result.get('rag_context') if debate_result else None,
                cag_context=debate_result.get('cag_context') if debate_result else None,
                debate_history=debate_result.get('debate_history') if debate_result else None,
                patient_metadata=patient_metadata
            )
        else:
            # 无冲突，快速决策
            diagnosis_result = await self.cmo.make_quick_decision(
                expert_opinions=expert_opinions,
                conflict_analysis=conflict_analysis
            )

        # 设置患者ID
        diagnosis_result.patient_id = patient_id

        print(f"  ✓ 最终诊断: {diagnosis_result.diagnosis}")
        print(f"  ✓ 置信度: {diagnosis_result.confidence:.1%}")

        # 步骤6: 生成报告
        print("\n[步骤 6/6] 生成诊断报告...")
        report = self.report_generator.generate_report(
            diagnosis_result=diagnosis_result,
            patient_metadata=patient_metadata or {"patient_id": patient_id}
        )

        print(f"  ✓ 报告生成完成 ({len(report)} 字符)")

        # 保存报告
        output_dir = Path("data/diagnosis_reports")
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / f"{patient_id}_report.md"
        self.report_generator.save_report(report, str(report_path))

        print("\n" + "="*70)
        print("诊断完成")
        print("="*70)
        print(f"\n结果:")
        print(f"  - 诊断: {diagnosis_result.diagnosis}")
        print(f"  - 置信度: {diagnosis_result.confidence:.1%}")
        print(f"  - 报告路径: {report_path}")
        print()

        return {
            "patient_id": patient_id,
            "diagnosis": diagnosis_result.diagnosis,
            "confidence": diagnosis_result.confidence,
            "diagnosis_result": diagnosis_result,
            "report": report,
            "report_path": str(report_path),
            "debate_result": debate_result
        }

    def _initialize_sample_literature(self):
        """初始化示例医学文献"""
        print("  添加示例医学文献...")

        sample_docs = [
            {
                "text": "Periodontitis is characterized by elevated levels of red complex bacteria including Porphyromonas gingivalis, Treponema denticola, and Tannerella forsythia. These pathogens trigger host immune responses leading to tissue destruction and alveolar bone loss. Matrix metalloproteinases (MMPs), particularly MMP-8 and MMP-9, serve as key biomarkers of periodontal tissue degradation.",
                "metadata": {
                    "title": "Red Complex Bacteria in Periodontitis Pathogenesis",
                    "year": "2023",
                    "doi": "10.1177/periodontology.2023.001",
                    "type": "research_article"
                }
            },
            {
                "text": "Metabolomic profiling reveals elevated short-chain fatty acids (SCFAs) including butyrate and propionate in periodontitis patients. These metabolites are produced by periodontal pathogens and contribute to local inflammation through activation of inflammatory pathways. IL-6 and TNF-α levels correlate with disease severity.",
                "metadata": {
                    "title": "Metabolomic Signatures of Periodontal Disease",
                    "year": "2023",
                    "doi": "10.1016/metabolomics.2023.045",
                    "type": "research_article"
                }
            },
            {
                "text": "Differential diagnosis between gingivitis and periodontitis requires multi-omics assessment. Gingivitis shows reversible gingival inflammation without attachment loss, while periodontitis demonstrates irreversible periodontal tissue destruction. Key differentiators include alveolar bone resorption, clinical attachment loss >3mm, and elevated MMP-8 levels >25 ng/mL in gingival crevicular fluid.",
                "metadata": {
                    "title": "Clinical Guidelines for Periodontal Disease Diagnosis",
                    "year": "2024",
                    "type": "clinical_guideline"
                }
            },
            {
                "text": "Diabetes mellitus and periodontitis exhibit bidirectional relationship. Hyperglycemia impairs host immune response and promotes dysbiotic microbiota. Conversely, periodontal inflammation exacerbates insulin resistance through systemic inflammation. Co-occurring conditions require integrated management approach with glycemic control and periodontal therapy.",
                "metadata": {
                    "title": "Diabetes-Periodontitis Interaction: A Systematic Review",
                    "year": "2024",
                    "doi": "10.2337/diabetes.2024.012",
                    "type": "systematic_review"
                }
            },
            {
                "text": "Proteomic analysis identifies lactoferrin, lysozyme, and IgA as protective salivary proteins in healthy individuals. Decreased levels correlate with susceptibility to periodontal disease. Conversely, elevated calprotectin and S100A8/A9 proteins indicate active inflammation and neutrophil infiltration in diseased tissues.",
                "metadata": {
                    "title": "Salivary Proteomics in Oral Health and Disease",
                    "year": "2023",
                    "doi": "10.1021/proteomics.2023.089",
                    "type": "research_article"
                }
            }
        ]

        documents = [doc["text"] for doc in sample_docs]
        metadatas = [doc["metadata"] for doc in sample_docs]

        self.rag_system.add_literature(documents, metadatas)
        print(f"  ✓ 添加了 {len(documents)} 篇文献")

    def _initialize_sample_cases(self):
        """初始化示例临床案例"""
        print("  添加示例临床案例...")

        sample_cases = [
            {
                "patient_id": "P_HIST_001",
                "diagnosis": "Periodontitis",
                "microbiome_features": {
                    "Porphyromonas_gingivalis": 0.32,
                    "Treponema_denticola": 0.28,
                    "Tannerella_forsythia": 0.22,
                    "Fusobacterium_nucleatum": 0.18
                },
                "metabolome_features": {
                    "Butyrate": 0.25,
                    "Propionate": 0.20,
                    "IL6": 0.22
                },
                "proteome_features": {
                    "MMP8": 0.30,
                    "MMP9": 0.26,
                    "Cathepsin": 0.18
                },
                "clinical_notes": "45-year-old male patient with severe periodontitis. High levels of red complex bacteria detected. Significant alveolar bone loss observed. MMP-8 levels elevated at 45 ng/mL. Patient presents with deep periodontal pockets (6-8mm) and bleeding on probing.",
                "severity": "Severe",
                "treatment_outcome": "Successful response to scaling, root planing, and adjunct antibiotic therapy. Pocket depths reduced to 3-4mm after 3 months."
            },
            {
                "patient_id": "P_HIST_002",
                "diagnosis": "Periodontitis",
                "microbiome_features": {
                    "Porphyromonas_gingivalis": 0.28,
                    "Aggregatibacter_actinomycetemcomitans": 0.24,
                    "Prevotella_intermedia": 0.20
                },
                "metabolome_features": {
                    "Butyrate": 0.22,
                    "IL6": 0.25,
                    "CRP": 0.20
                },
                "proteome_features": {
                    "MMP9": 0.28,
                    "Calprotectin": 0.24,
                    "S100A9": 0.20
                },
                "clinical_notes": "38-year-old female with moderate chronic periodontitis. Family history of periodontal disease. Clinical attachment loss 4-6mm. Radiographic evidence of horizontal bone loss. Elevated inflammatory markers.",
                "severity": "Moderate",
                "treatment_outcome": "Good response to non-surgical periodontal therapy. Maintained with 3-month recall intervals."
            },
            {
                "patient_id": "P_HIST_003",
                "diagnosis": "Gingivitis",
                "microbiome_features": {
                    "Streptococcus_salivarius": 0.15,
                    "Fusobacterium_nucleatum": 0.12,
                    "Prevotella_melaninogenica": 0.10
                },
                "metabolome_features": {
                    "IL6": 0.15,
                    "PGE2": 0.12
                },
                "proteome_features": {
                    "IgA": 0.18,
                    "Lactoferrin": 0.15
                },
                "clinical_notes": "28-year-old patient with plaque-induced gingivitis. Gingival inflammation present but no attachment loss. No radiographic bone loss. Reversible condition with improved oral hygiene.",
                "severity": "Mild",
                "treatment_outcome": "Complete resolution with professional prophylaxis and improved home care. No periodontal therapy required."
            }
        ]

        for case in sample_cases:
            self.cag_system.add_case(
                patient_id=case["patient_id"],
                diagnosis=case["diagnosis"],
                microbiome_features=case["microbiome_features"],
                metabolome_features=case.get("metabolome_features"),
                proteome_features=case.get("proteome_features"),
                clinical_notes=case["clinical_notes"],
                severity=case.get("severity"),
                treatment_outcome=case.get("treatment_outcome")
            )

        print(f"  ✓ 添加了 {len(sample_cases)} 个案例")


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="口腔多组学诊断系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s                              # 交互式模式
  %(prog)s --patient-id P001            # 诊断指定患者
  %(prog)s --no-llm                     # 禁用LLM推理
  %(prog)s --no-rag                     # 禁用RAG文献检索
  %(prog)s --mock-llm                   # 使用Mock LLM（测试）
        """
    )

    parser.add_argument(
        "--patient-id",
        type=str,
        help="患者ID（如果不指定，使用训练数据中的样本）"
    )

    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="禁用LLM推理，使用fallback voting"
    )

    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="禁用RAG文献检索系统"
    )

    parser.add_argument(
        "--no-cag",
        action="store_true",
        help="禁用CAG案例检索系统"
    )

    parser.add_argument(
        "--mock-llm",
        action="store_true",
        help="使用Mock LLM（测试用，不消耗API费用）"
    )

    args = parser.parse_args()

    # 初始化系统
    system = OralMultiomicsDiagnosisSystem(
        use_llm=not args.no_llm,
        enable_rag=not args.no_rag,
        enable_cag=not args.no_cag,
        use_mock_llm=args.mock_llm
    )

    # 加载数据
    print("\n加载患者数据...")

    # 使用训练数据作为示例 (raw data)
    microbiome_raw = pd.read_csv("data/training/microbiome_raw.csv", index_col=0)
    metabolome_raw = pd.read_csv("data/training/metabolome_raw.csv", index_col=0)
    proteome_raw = pd.read_csv("data/training/proteome_raw.csv", index_col=0)
    labels_df = pd.read_csv("data/training/labels.csv")

    # Preprocess data
    print("  预处理数据...")
    microbiome_result = system.preprocessors['microbiome'].fit_transform(microbiome_raw)
    metabolome_result = system.preprocessors['metabolome'].fit_transform(metabolome_raw)
    proteome_result = system.preprocessors['proteome'].fit_transform(proteome_raw)

    # Extract DataFrames from PreprocessingResult
    microbiome_df = microbiome_result.data
    metabolome_df = metabolome_result.data
    proteome_df = proteome_result.data

    # 选择患者
    if args.patient_id:
        patient_idx = labels_df[labels_df['sample_id'] == args.patient_id].index[0]
    else:
        # 使用第一个Periodontitis样本
        patient_idx = labels_df[labels_df['diagnosis'] == 'Periodontitis'].index[0]

    patient_id = labels_df.iloc[patient_idx]['sample_id']
    true_diagnosis = labels_df.iloc[patient_idx]['diagnosis']

    print(f"  ✓ 选择患者: {patient_id}")
    print(f"  ✓ 真实诊断: {true_diagnosis}")

    # 提取患者数据
    patient_microbiome = microbiome_df.iloc[[patient_idx]]
    patient_metabolome = metabolome_df.iloc[[patient_idx]]
    patient_proteome = proteome_df.iloc[[patient_idx]]

    patient_metadata = {
        "patient_id": patient_id,
        "age": 45,
        "sex": "M",
        "true_diagnosis": true_diagnosis
    }

    # 执行诊断
    result = await system.diagnose(
        patient_id=patient_id,
        microbiome_data=patient_microbiome,
        metabolome_data=patient_metabolome,
        proteome_data=patient_proteome,
        patient_metadata=patient_metadata
    )

    # 验证
    print("\n" + "="*70)
    print("诊断验证")
    print("="*70)
    print(f"  真实诊断: {true_diagnosis}")
    print(f"  预测诊断: {result['diagnosis']}")
    print(f"  匹配: {'✓ 正确' if result['diagnosis'] == true_diagnosis else '✗ 不匹配'}")
    print()

    return result


if __name__ == "__main__":
    asyncio.run(main())
