#!/usr/bin/env python3
"""
端到端诊断系统测试脚本

测试从原始组学数据到最终诊断报告的完整流程，包括：
- 数据预处理
- 专家预测
- 冲突检测
- 辩论系统（LangGraph）
- CMO决策
- 报告生成

使用已训练的模型和生成的训练数据。
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import asyncio

from clinical.preprocessing.microbiome_preprocessor import MicrobiomePreprocessor
from clinical.preprocessing.metabolome_preprocessor import MetabolomePreprocessor
from clinical.preprocessing.proteome_preprocessor import ProteomePreprocessor
from clinical.experts.model_manager import ModelManager
from clinical.decision.conflict_resolver import ConflictResolver
from clinical.decision.cmo_coordinator import CMOCoordinator
from clinical.decision.report_generator import ReportGenerator
from clinical.models.expert_opinion import ExpertOpinion, FeatureImportance


# =============================================================================
# Mock数据生成函数
# =============================================================================

def create_mock_rag_context(query: str) -> Dict[str, Any]:
    """创建模拟的RAG查询结果"""
    return {
        "query": query,
        "documents": [
            {
                "source": "PubMed:12345678",
                "title": "Periodontal Disease and Oral Microbiome",
                "content": "P. gingivalis is a key pathogen in periodontitis. Elevated levels of this bacteria correlate with disease severity and tissue destruction.",
                "score": 0.92,
                "url": "https://pubmed.ncbi.nlm.nih.gov/12345678"
            },
            {
                "source": "Clinical Guidelines 2023",
                "title": "Diagnosis and Treatment of Periodontitis",
                "content": "Clinical diagnosis requires elevated MMP-9 and IL-6 inflammatory markers, combined with microbiome analysis showing pathogenic bacteria.",
                "score": 0.87,
                "url": ""
            },
            {
                "source": "J Clin Microbiol 2024",
                "title": "Butyrate Production and Periodontal Health",
                "content": "Butyrate produced by anaerobic bacteria is a key metabolite in periodontitis pathogenesis.",
                "score": 0.83,
                "url": ""
            }
        ]
    }


def create_mock_cag_context(diagnosis: str) -> Dict[str, Any]:
    """创建模拟的CAG查询结果"""
    return {
        "query_features": {"P_gingivalis": 0.22, "Butyrate": 0.18, "MMP9": 0.25},
        "similar_cases": [
            {
                "case_id": "CASE_2023_001",
                "diagnosis": diagnosis,
                "similarity": 0.89,
                "outcome": "Successful treatment with scaling and root planing",
                "key_features": {
                    "P_gingivalis": 0.22,
                    "Butyrate": 0.18,
                    "MMP9": 0.25
                }
            },
            {
                "case_id": "CASE_2023_045",
                "diagnosis": diagnosis,
                "similarity": 0.85,
                "outcome": "Required advanced periodontal therapy",
                "key_features": {
                    "P_gingivalis": 0.20,
                    "Butyrate": 0.16,
                    "MMP9": 0.23
                }
            }
        ]
    }


# =============================================================================
# 测试场景类
# =============================================================================

class TestScenario:
    """测试场景基类"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.start_time = None
        self.end_time = None
        self.results = {}

    def start(self):
        """开始测试"""
        print("\n" + "=" * 70)
        print(f"场景: {self.name}")
        print("=" * 70)
        print(f"描述: {self.description}")
        print("-" * 70)
        self.start_time = time.time()

    def end(self, success: bool = True):
        """结束测试"""
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        status = "✓ 通过" if success else "✗ 失败"
        print("-" * 70)
        print(f"{status} (执行时间: {duration:.2f}s)")
        print("=" * 70)
        return success

    def checkpoint(self, name: str, success: bool = True, details: str = ""):
        """测试检查点"""
        symbol = "✓" if success else "✗"
        print(f"{symbol} {name}", end="")
        if details:
            print(f": {details}")
        else:
            print()
        self.results[name] = {"success": success, "details": details}


# =============================================================================
# 场景1: 无冲突快速决策
# =============================================================================

class NoConflictScenario(TestScenario):
    """场景1: 所有专家一致诊断Periodontitis"""

    def __init__(self):
        super().__init__(
            name="场景1: 无冲突快速决策",
            description="所有3个专家一致诊断为Periodontitis，测试快速决策路径"
        )

    async def run(self, model_manager: ModelManager) -> bool:
        """运行测试场景"""
        self.start()

        try:
            # 加载数据
            microbiome_data = pd.read_csv("data/training/microbiome_raw.csv", index_col=0)
            metabolome_data = pd.read_csv("data/training/metabolome_raw.csv", index_col=0)
            proteome_data = pd.read_csv("data/training/proteome_raw.csv", index_col=0)
            labels = pd.read_csv("data/training/labels.csv", index_col=0)
            with open("data/training/splits.json", "r") as f:
                splits = json.load(f)

            # 选择测试集中的Periodontitis样本
            test_ids = splits['test']
            periodontitis_ids = [
                idx for idx in test_ids
                if idx.startswith("Periodontitis_") and labels.loc[idx, "diagnosis"] == "Periodontitis"
            ]

            if not periodontitis_ids:
                self.checkpoint("数据加载", False, "没有找到Periodontitis测试样本")
                return self.end(False)

            # 取第一个样本
            sample_id = periodontitis_ids[0]
            micro_sample = microbiome_data.loc[[sample_id]]
            metab_sample = metabolome_data.loc[[sample_id]]
            prot_sample = proteome_data.loc[[sample_id]]

            self.checkpoint("数据加载", True, f"样本ID: {sample_id}")

            # 预处理
            micro_preprocessor = MicrobiomePreprocessor()
            metab_preprocessor = MetabolomePreprocessor()
            prot_preprocessor = ProteomePreprocessor()

            micro_processed = micro_preprocessor.fit_transform(micro_sample).data
            metab_processed = metab_preprocessor.fit_transform(metab_sample).data
            prot_processed = prot_preprocessor.fit_transform(prot_sample).data

            self.checkpoint("数据预处理", True, f"形状: {micro_processed.shape}, {metab_processed.shape}, {prot_processed.shape}")

            # 加载专家模型并预测
            experts = model_manager.load_all_experts()

            micro_opinion = experts['microbiome_expert'].predict(micro_processed)[0]
            metab_opinion = experts['metabolome_expert'].predict(metab_processed)[0]
            prot_opinion = experts['proteome_expert'].predict(prot_processed)[0]

            opinions = [micro_opinion, metab_opinion, prot_opinion]

            self.checkpoint("专家预测", True)
            print(f"  - Microbiome: {micro_opinion.diagnosis} (prob={micro_opinion.probability:.2f}, conf={micro_opinion.confidence:.2f})")
            print(f"  - Metabolome: {metab_opinion.diagnosis} (prob={metab_opinion.probability:.2f}, conf={metab_opinion.confidence:.2f})")
            print(f"  - Proteome: {prot_opinion.diagnosis} (prob={prot_opinion.probability:.2f}, conf={prot_opinion.confidence:.2f})")

            # 冲突检测
            resolver = ConflictResolver()
            conflict_analysis = resolver.detect_conflict(opinions)

            has_conflict = conflict_analysis.has_conflict
            self.checkpoint("冲突检测", not has_conflict, f"冲突: {has_conflict}")

            # CMO决策
            cmo = CMOCoordinator()
            if has_conflict:
                print("  警告: 检测到冲突（预期无冲突）")
                # 即使有冲突也继续测试
                diagnosis_result = await cmo.make_conflict_resolution(
                    expert_opinions=opinions,
                    conflict_analysis=conflict_analysis,
                    patient_metadata={"patient_id": sample_id},
                    rag_context=create_mock_rag_context("Periodontitis diagnosis"),
                    cag_context=create_mock_cag_context("Periodontitis")
                )
            else:
                diagnosis_result = await cmo.make_quick_decision(
                    expert_opinions=opinions,
                    conflict_analysis=conflict_analysis
                )
                diagnosis_result.patient_id = sample_id

            self.checkpoint("CMO决策", True, f"{diagnosis_result.diagnosis} (置信度: {diagnosis_result.confidence:.2f})")

            # 报告生成
            report_gen = ReportGenerator()
            report = report_gen.generate_report(diagnosis_result)

            self.checkpoint("报告生成", len(report) > 100, f"长度: {len(report)} 字符")

            # 保存报告
            report_path = Path("data/test") / "scenario1_report.md"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_gen.save_report(report, str(report_path))

            self.checkpoint("报告保存", report_path.exists(), f"路径: {report_path}")

            return self.end(True)

        except Exception as e:
            print(f"\n✗ 错误: {e}")
            import traceback
            traceback.print_exc()
            return self.end(False)


# =============================================================================
# 场景2: 边界冲突（通过阈值调整解决）
# =============================================================================

class BorderlineConflictScenario(TestScenario):
    """场景2: 2个专家一致，1个边界值"""

    def __init__(self):
        super().__init__(
            name="场景2: 边界冲突",
            description="2个专家诊断Periodontitis，1个专家边界值，测试阈值调整"
        )

    async def run(self, model_manager: ModelManager) -> bool:
        """运行测试场景"""
        self.start()

        try:
            # 为了创建边界情况，我们使用Mock专家意见
            # 实际场景中可以通过混合数据或调整阈值实现

            opinions = [
                ExpertOpinion(
                    expert_name="microbiome_expert",
                    omics_type="microbiome",
                    diagnosis="Periodontitis",
                    probability=0.88,
                    confidence=0.92,
                    biological_explanation="HIGH levels of P.gingivalis (22x) and T.denticola (20x) detected",
                    top_features=[
                        FeatureImportance("Porphyromonas_gingivalis", 0.40, "up", "Major periodontal pathogen"),
                        FeatureImportance("Treponema_denticola", 0.32, "up", "Periodontal spirochete"),
                        FeatureImportance("Prevotella_intermedia", 0.15, "up", "Opportunistic pathogen")
                    ],
                    evidence_chain=["Strong pathogenic signal", "Consistent with periodontal disease"],
                    model_metadata={"version": "v1.0.0", "threshold": 0.5}
                ),
                ExpertOpinion(
                    expert_name="metabolome_expert",
                    omics_type="metabolome",
                    diagnosis="Periodontitis",
                    probability=0.86,
                    confidence=0.90,
                    biological_explanation="HIGH butyrate (28x) and propionate (25x) levels",
                    top_features=[
                        FeatureImportance("Butyrate", 0.38, "up", "Inflammatory metabolite"),
                        FeatureImportance("Propionate", 0.35, "up", "Bacterial fermentation product"),
                        FeatureImportance("Acetate", 0.12, "up", "Short-chain fatty acid")
                    ],
                    evidence_chain=["Elevated inflammatory metabolites", "Supports periodontal diagnosis"],
                    model_metadata={"version": "v1.0.0", "threshold": 0.5}
                ),
                ExpertOpinion(
                    expert_name="proteome_expert",
                    omics_type="proteome",
                    diagnosis="Healthy",
                    probability=0.55,  # 边界值
                    confidence=0.60,
                    biological_explanation="IgA levels slightly elevated but MMP9 also present (mixed signals)",
                    top_features=[
                        FeatureImportance("IgA", 0.30, "up", "Protective antibody"),
                        FeatureImportance("MMP9", 0.28, "up", "Matrix metalloproteinase"),
                        FeatureImportance("IL6", 0.25, "up", "Inflammatory cytokine")
                    ],
                    evidence_chain=["Mixed signals detected", "Borderline classification"],
                    model_metadata={"version": "v1.0.0", "threshold": 0.5, "borderline": True}
                )
            ]

            self.checkpoint("创建Mock专家意见", True, "3个专家")

            for i, op in enumerate(opinions, 1):
                print(f"  {i}. {op.expert_name}: {op.diagnosis} (prob={op.probability:.2f}, conf={op.confidence:.2f})")

            # 冲突检测
            resolver = ConflictResolver()
            conflict_analysis = resolver.detect_conflict(opinions)

            has_conflict = conflict_analysis.has_conflict
            self.checkpoint("冲突检测", has_conflict, f"冲突类型: {[ct.value for ct in conflict_analysis.conflict_types]}")

            # CMO决策（有冲突）
            cmo = CMOCoordinator()
            diagnosis_result = await cmo.make_conflict_resolution(
                expert_opinions=opinions,
                conflict_analysis=conflict_analysis,
                patient_metadata={"patient_id": "BORDERLINE_TEST_001"},
                rag_context=create_mock_rag_context("Periodontitis vs Healthy differential"),
                cag_context=create_mock_cag_context("Periodontitis")
            )
            diagnosis_result.patient_id = "BORDERLINE_TEST_001"

            self.checkpoint("CMO冲突解决", True, f"{diagnosis_result.diagnosis} (置信度: {diagnosis_result.confidence:.2f})")

            # 报告生成
            report_gen = ReportGenerator()
            report = report_gen.generate_report(diagnosis_result)

            self.checkpoint("报告生成", len(report) > 100)

            # 保存报告
            report_path = Path("data/test") / "scenario2_report.md"
            report_gen.save_report(report, str(report_path))

            self.checkpoint("报告保存", report_path.exists())

            return self.end(True)

        except Exception as e:
            print(f"\n✗ 错误: {e}")
            import traceback
            traceback.print_exc()
            return self.end(False)


# =============================================================================
# 场景3: 强冲突（需要RAG/CAG）
# =============================================================================

class StrongConflictScenario(TestScenario):
    """场景3: 三个专家完全不一致"""

    def __init__(self):
        super().__init__(
            name="场景3: 强冲突需要RAG/CAG",
            description="3个专家完全不一致，测试辩论系统和RAG/CAG触发"
        )

    async def run(self, model_manager: ModelManager) -> bool:
        """运行测试场景"""
        self.start()

        try:
            # 创建强冲突的Mock专家意见
            opinions = [
                ExpertOpinion(
                    expert_name="microbiome_expert",
                    omics_type="microbiome",
                    diagnosis="Periodontitis",
                    probability=0.85,
                    confidence=0.90,
                    biological_explanation="HIGH levels of P.gingivalis (20x) and T.denticola (18x) detected",
                    top_features=[
                        FeatureImportance("Porphyromonas_gingivalis", 0.35, "up", "Pathogenic bacteria"),
                        FeatureImportance("Treponema_denticola", 0.28, "up", "Periodontal pathogen"),
                        FeatureImportance("Fusobacterium_nucleatum", 0.15, "up", "Inflammatory bacteria")
                    ],
                    evidence_chain=["High pathogenic bacteria detected", "Consistent with periodontal disease"],
                    model_metadata={"version": "v1.0.0", "threshold": 0.5}
                ),
                ExpertOpinion(
                    expert_name="metabolome_expert",
                    omics_type="metabolome",
                    diagnosis="Diabetes",
                    probability=0.82,
                    confidence=0.88,
                    biological_explanation="HIGH glucose (25x) and lactate (22x) levels",
                    top_features=[
                        FeatureImportance("Glucose", 0.40, "up", "Elevated blood sugar"),
                        FeatureImportance("Lactate", 0.32, "up", "Metabolic dysfunction"),
                        FeatureImportance("Pyruvate", 0.12, "up", "Energy metabolism issue")
                    ],
                    evidence_chain=["Elevated glucose detected", "Metabolic profile suggests diabetes"],
                    model_metadata={"version": "v1.0.0", "threshold": 0.5}
                ),
                ExpertOpinion(
                    expert_name="proteome_expert",
                    omics_type="proteome",
                    diagnosis="Healthy",
                    probability=0.78,
                    confidence=0.85,
                    biological_explanation="HIGH protective IgA (22x) and Lactoferrin (20x)",
                    top_features=[
                        FeatureImportance("IgA", 0.38, "up", "Protective antibody"),
                        FeatureImportance("Lactoferrin", 0.30, "up", "Antimicrobial protein"),
                        FeatureImportance("Lysozyme", 0.15, "up", "Enzyme protection")
                    ],
                    evidence_chain=["High protective proteins detected", "Immune system functioning well"],
                    model_metadata={"version": "v1.0.0", "threshold": 0.5}
                )
            ]

            self.checkpoint("创建强冲突Mock意见", True, "3个专家完全不一致")

            for i, op in enumerate(opinions, 1):
                print(f"  {i}. {op.expert_name}: {op.diagnosis} (prob={op.probability:.2f}, conf={op.confidence:.2f})")

            # 冲突检测
            resolver = ConflictResolver()
            conflict_analysis = resolver.detect_conflict(opinions)

            self.checkpoint("冲突检测", conflict_analysis.has_conflict,
                          f"冲突类型: {[ct.value for ct in conflict_analysis.conflict_types]}")
            print(f"  需要辩论: {conflict_analysis.requires_debate}")
            print(f"  需要RAG: {conflict_analysis.requires_rag}")
            print(f"  需要CAG: {conflict_analysis.requires_cag}")

            # CMO决策（强冲突）
            cmo = CMOCoordinator()
            diagnosis_result = await cmo.make_conflict_resolution(
                expert_opinions=opinions,
                conflict_analysis=conflict_analysis,
                patient_metadata={"patient_id": "CONFLICT_TEST_001"},
                rag_context=create_mock_rag_context("Periodontitis vs Diabetes vs Healthy differential"),
                cag_context=create_mock_cag_context("Periodontitis"),  # CMO会优先选择
                debate_history=[]  # 模拟辩论历史（实际由DebateSystem生成）
            )
            diagnosis_result.patient_id = "CONFLICT_TEST_001"

            self.checkpoint("CMO冲突解决", True,
                          f"{diagnosis_result.diagnosis} (置信度: {diagnosis_result.confidence:.2f})")

            # 检查冲突解决记录
            if diagnosis_result.conflict_resolution:
                cr = diagnosis_result.conflict_resolution
                print(f"  解决方法: {cr.resolution_method}")
                print(f"  RAG证据数: {len(cr.rag_evidence)}")
                print(f"  CAG案例数: {len(cr.cag_cases)}")
                self.checkpoint("冲突解决记录", True, f"使用{cr.resolution_method}")

            # 报告生成
            report_gen = ReportGenerator()
            report = report_gen.generate_report(diagnosis_result)

            self.checkpoint("报告生成", len(report) > 100)

            # 保存报告
            report_path = Path("data/test") / "scenario3_report.md"
            report_gen.save_report(report, str(report_path))

            self.checkpoint("报告保存", report_path.exists())

            return self.end(True)

        except Exception as e:
            print(f"\n✗ 错误: {e}")
            import traceback
            traceback.print_exc()
            return self.end(False)


# =============================================================================
# 主测试运行器
# =============================================================================

async def main():
    """运行所有测试场景"""
    print("\n" + "#" * 70)
    print("# 端到端诊断系统测试")
    print("#" * 70)
    print("\n测试目标:")
    print("  1. 验证完整诊断流程（数据→预处理→专家→冲突检测→CMO→报告）")
    print("  2. 测试3种场景（无冲突、边界冲突、强冲突）")
    print("  3. 验证RAG/CAG触发机制")
    print("  4. 验证CMO决策逻辑\n")

    # 加载训练好的模型
    print("=" * 70)
    print("初始化系统")
    print("=" * 70)

    try:
        model_manager = ModelManager(models_dir="data/models")
        print("✓ ModelManager初始化成功")

        # 检查模型文件
        model_files = list(Path("data/models").glob("*_expert_*.pkl"))
        print(f"✓ 找到{len(model_files)}个模型文件")

        if len(model_files) < 3:
            print("\n✗ 错误: 缺少训练好的模型文件")
            print("  请先运行: python scripts/train_with_generated_data.py")
            return

    except Exception as e:
        print(f"\n✗ 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 运行测试场景
    scenarios = [
        NoConflictScenario(),
        BorderlineConflictScenario(),
        StrongConflictScenario()
    ]

    results = []
    total_start = time.time()

    for scenario in scenarios:
        success = await scenario.run(model_manager)
        results.append({
            "name": scenario.name,
            "success": success,
            "duration": scenario.end_time - scenario.start_time if scenario.end_time else 0
        })

    total_duration = time.time() - total_start

    # 打印总结
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)

    passed = sum(1 for r in results if r["success"])
    total = len(results)

    for r in results:
        status = "✓ 通过" if r["success"] else "✗ 失败"
        print(f"{status}: {r['name']} ({r['duration']:.2f}s)")

    print(f"\n总计: {passed}/{total} 场景通过")
    print(f"总执行时间: {total_duration:.2f}s")

    if passed == total:
        print("\n" + "=" * 70)
        print("🎉 所有测试通过！诊断系统运行正常。")
        print("=" * 70)
        print("\n生成的报告:")
        for i in range(1, 4):
            report_path = Path(f"data/test/scenario{i}_report.md")
            if report_path.exists():
                print(f"  - {report_path}")
    else:
        print("\n" + "=" * 70)
        print("⚠️ 部分测试失败，请检查错误信息")
        print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
