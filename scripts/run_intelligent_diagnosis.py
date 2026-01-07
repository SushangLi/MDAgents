#!/usr/bin/env python3
"""
CMO智能诊断系统 - 交互式入口

基于文件级缓存的智能诊断系统：
- 自然语言请求解析
- 按需预处理（检查文件→不存在则预处理并保存→加载）
- 智能筛选病人/行范围
- 生成双语诊断报告
"""

import sys
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
    print(f"✓ Loaded environment from {env_local}")
else:
    print(f"⚠ No .env.local found, using system environment")

# Import system components
from clinical.preprocessing.cache_manager import PreprocessingCacheManager
from clinical.experts.microbiome_expert import MicrobiomeExpert
from clinical.experts.metabolome_expert import MetabolomeExpert
from clinical.experts.proteome_expert import ProteomeExpert
from clinical.decision.intelligent_debate_system import IntelligentDebateSystem
from clinical.decision.request_parser import RequestParser
from clinical.decision.bilingual_report_generator import BilingualReportGenerator
from clinical.decision.debate_system import DebateConfig
from clinical.decision.llm_wrapper import create_llm_wrapper
from clinical.collaboration.rag_system import RAGSystem
from clinical.collaboration.cag_system import CAGSystem


class IntelligentDiagnosisInterface:
    """智能诊断交互式接口（基于文件级缓存）"""

    def __init__(self):
        """初始化系统"""
        print("\n" + "="*70)
        print("CMO智能诊断系统初始化中...")
        print("="*70)

        # 1. 初始化缓存管理器（不预加载）
        print("\n[1/8] 初始化文件级缓存管理器...")
        self.cache_manager = PreprocessingCacheManager()
        print("  ✓ 缓存管理器就绪（按需加载模式）")

        # 2. 初始化并加载专家模型
        print("\n[2/8] 初始化并加载专家模型...")
        self.experts = {}

        # 查找并加载最新的模型文件
        models_dir = project_root / "data" / "models"

        for omics_type, ExpertClass in [
            ("microbiome", MicrobiomeExpert),
            ("metabolome", MetabolomeExpert),
            ("proteome", ProteomeExpert)
        ]:
            expert = ExpertClass()

            # 查找该组学类型的最新模型文件
            if models_dir.exists():
                model_files = list(models_dir.glob(f"{omics_type}_expert_v*.pkl"))
                if model_files:
                    # 按文件名排序，获取最新的
                    latest_model = sorted(model_files)[-1]
                    try:
                        expert.load_model(str(latest_model))
                        print(f"  ✓ {omics_type}: 已加载模型 {latest_model.name}")
                    except Exception as e:
                        print(f"  ⚠ {omics_type}: 模型加载失败 {e}")
                else:
                    print(f"  ⚠ {omics_type}: 未找到训练好的模型")
            else:
                print(f"  ⚠ 模型目录不存在: {models_dir}")

            self.experts[omics_type] = expert

        print("  ✓ 专家模型加载完成")

        # 3. 初始化RAG系统
        print("\n[3/8] 初始化RAG文献检索系统...")
        try:
            self.rag_system = RAGSystem()
            doc_count = self.rag_system.vector_store.count()
            print(f"  ✓ RAG系统就绪 ({doc_count} 篇文献)")
        except Exception as e:
            print(f"  ⚠ RAG初始化失败: {e}")
            self.rag_system = None

        # 4. 初始化CAG系统
        print("\n[4/8] 初始化CAG案例检索系统...")
        try:
            self.cag_system = CAGSystem()
            case_count = len(self.cag_system.cases)
            print(f"  ✓ CAG系统就绪 ({case_count} 个案例)")
        except Exception as e:
            print(f"  ⚠ CAG初始化失败: {e}")
            self.cag_system = None

        # 5. 初始化LLM
        print("\n[5/8] 初始化LLM...")
        self.llm_wrapper = create_llm_wrapper(use_mock=False)
        print("  ✓ LLM就绪（真实API调用）")

        # 6. 初始化请求解析器和报告生成器
        print("\n[6/8] 初始化智能组件...")
        self.request_parser = RequestParser(llm_call_func=self.llm_wrapper.call)
        self.bilingual_generator = BilingualReportGenerator(
            include_metadata=True,
            include_expert_details=True,
            include_biomarkers=True
        )
        print("  ✓ 请求解析器就绪")
        print("  ✓ 双语报告生成器就绪")

        # 7. 初始化智能辩论系统
        print("\n[7/8] 初始化智能辩论系统...")
        self.intelligent_system = IntelligentDebateSystem(
            request_parser=self.request_parser,
            preprocessors={},  # 不使用内部预处理，从缓存加载
            experts=self.experts,
            bilingual_generator=self.bilingual_generator,
            rag_system=self.rag_system,
            cag_system=self.cag_system,
            config=DebateConfig(max_rounds=3, threshold_adjustment=0.1),
            llm_wrapper=self.llm_wrapper  # Pass LLM wrapper for CMO CoT reasoning
        )
        print("  ✓ 智能辩论系统就绪 (CMO CoT推理已启用)")

        # 8. 预处理数据目录信息
        print("\n[8/8] 检查数据目录...")
        self.data_dir = project_root / "data" / "training"
        if self.data_dir.exists():
            print(f"  ✓ 数据目录: {self.data_dir}")
        else:
            print(f"  ⚠ 数据目录不存在: {self.data_dir}")

        print("\n" + "="*70)
        print("✅ 系统初始化完成！")
        print("="*70)

    def check_and_preprocess(self, omics_type: str, data_source: str = "training"):
        """
        检查预处理文件并按需预处理

        Args:
            omics_type: 组学类型
            data_source: 数据源标识
        """
        # 检查文件是否存在
        if not self.cache_manager.check_if_preprocessed_exists(omics_type, data_source):
            # 不存在，运行预处理
            raw_path = self.data_dir / f"{omics_type}_raw.csv"

            if not raw_path.exists():
                raise FileNotFoundError(f"原始数据不存在: {raw_path}")

            print(f"  → 首次使用，开始预处理 {omics_type}...")
            self.cache_manager.preprocess_and_save(omics_type, raw_path, data_source)
        else:
            print(f"  → 使用缓存文件")

    def load_and_filter_data(
        self,
        omics_types: list,
        patient_ids: Optional[list] = None,
        row_range: Optional[tuple] = None,
        data_source: str = "training"
    ) -> Dict[str, pd.DataFrame]:
        """
        加载并筛选数据

        Args:
            omics_types: 组学类型列表
            patient_ids: 病人ID列表
            row_range: 行范围 (start, end)
            data_source: 数据源

        Returns:
            筛选后的数据字典 {omics_type: DataFrame}
        """
        print("\n[数据加载] 加载组学数据...")

        available_data = {}

        for omics_type in omics_types:
            print(f"\n  {omics_type.upper()}:")

            # 步骤1: 检查并按需预处理
            self.check_and_preprocess(omics_type, data_source)

            # 步骤2: 从文件加载
            print(f"  → 从Parquet文件加载...")
            full_data = self.cache_manager.load_preprocessed(omics_type, data_source)
            print(f"     加载数据形状: {full_data.shape}")

            # 步骤3: 筛选数据
            if patient_ids:
                print(f"  → 筛选病人: {patient_ids}")
                # 从labels.csv获取索引
                labels_path = self.data_dir / "labels.csv"
                if labels_path.exists():
                    labels_df = pd.read_csv(labels_path)

                    # Try multiple matching strategies
                    indices = []

                    # Strategy 1: Exact match
                    exact_matches = labels_df[labels_df['sample_id'].isin(patient_ids)].index.tolist()
                    if exact_matches:
                        indices = exact_matches
                        print(f"     ✓ 精确匹配: {len(indices)} 个病人")
                    else:
                        # Strategy 2: Fuzzy match (e.g., "P002" matches "Periodontitis_002")
                        print(f"     未找到精确匹配，尝试模糊匹配...")
                        for pid in patient_ids:
                            # Extract number from pattern like "P002" or "P2"
                            import re
                            match = re.search(r'[Pp]?(\d+)', pid)
                            if match:
                                num = match.group(1).zfill(3)  # Pad to 3 digits
                                # Try different diagnosis types
                                for diagnosis_type in ['Periodontitis', 'Healthy', 'Diabetes_Associated_Dysbiosis', 'Oral_Cancer_Risk']:
                                    candidate = f"{diagnosis_type}_{num}"
                                    candidate_idx = labels_df[labels_df['sample_id'] == candidate].index.tolist()
                                    if candidate_idx:
                                        indices.extend(candidate_idx)
                                        print(f"     ✓ 模糊匹配: {pid} → {candidate}")
                                        break

                    if len(indices) == 0:
                        print(f"     ⚠ 未找到匹配的病人ID")
                        print(f"     提示: 实际病人ID格式如 'Periodontitis_001', 'Healthy_001' 等")
                        print(f"     您可以尝试: '分析第{patient_ids[0][-3:]}个病人' 或 '分析第2行数据'")
                        filtered_data = full_data.iloc[:0]  # 空DataFrame
                    else:
                        filtered_data = full_data.iloc[indices]
                        print(f"     筛选后形状: {filtered_data.shape}")
                else:
                    print(f"     ⚠ labels.csv不存在，使用全部数据")
                    filtered_data = full_data

            elif row_range:
                start, end = row_range
                print(f"  → 筛选行范围: [{start}:{end}]")
                filtered_data = full_data.iloc[start:end]
                print(f"     筛选后形状: {filtered_data.shape}")

            else:
                print(f"  → 使用全部数据")
                filtered_data = full_data

            available_data[omics_type] = filtered_data

        print(f"\n  ✓ 数据加载完成，共 {len(available_data)} 种组学类型")
        return available_data

    async def run_diagnosis(
        self,
        natural_request: str,
        patient_metadata: Optional[Dict] = None
    ):
        """
        运行智能诊断

        Args:
            natural_request: 自然语言请求
            patient_metadata: 病人元数据（可选）
        """
        print("\n" + "="*70)
        print("开始智能诊断")
        print("="*70)
        print(f"\n用户请求: {natural_request}")

        try:
            # 步骤1: 解析自然语言请求
            print("\n[步骤1] 解析自然语言请求...")
            config = await self.request_parser.parse_request(natural_request)

            print(f"  ✓ 解析完成:")
            print(f"    组学类型: {config.omics_types}")
            print(f"    病人ID: {config.patient_ids}")
            print(f"    行范围: {config.row_range}")
            print(f"    启用RAG: {config.enable_rag}")
            print(f"    启用CAG: {config.enable_cag}")
            print(f"    强制RAG: {config.force_rag_even_no_conflict}")
            print(f"    辩论轮数: {config.max_debate_rounds}")
            print(f"    报告详细度: {config.detail_level}")

            # 步骤2: 加载和筛选数据
            available_data = self.load_and_filter_data(
                omics_types=config.omics_types,
                patient_ids=config.patient_ids,
                row_range=config.row_range
            )

            # 步骤3: 运行智能诊断系统
            print("\n[步骤3] 运行智能诊断工作流...")
            final_state = await self.intelligent_system.run_intelligent_diagnosis(
                user_request=natural_request,
                available_data=available_data,
                patient_metadata=patient_metadata or {},
                parsed_config=config
            )

            # 步骤4: 获取并显示报告
            bilingual_report = final_state.get("bilingual_report")

            if bilingual_report:
                print("\n" + "="*70)
                print("诊断报告生成完成")
                print("="*70)
                print("\n" + bilingual_report)

                # 步骤5: 保存报告
                self._save_report(bilingual_report)

                return final_state
            else:
                print("\n✗ 报告生成失败")
                return None

        except Exception as e:
            print(f"\n✗ 诊断过程出错: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _save_report(self, report: str):
        """保存报告到文件"""
        try:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            report_dir = project_root / "data" / "diagnosis_reports"
            report_dir.mkdir(parents=True, exist_ok=True)
            report_path = report_dir / f"diagnosis_{timestamp}.md"

            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report)

            print(f"\n✓ 报告已保存到: {report_path}")

        except Exception as e:
            print(f"\n⚠ 报告保存失败: {e}")

    async def interactive_mode(self):
        """交互式模式主循环"""
        print("\n" + "="*70)
        print("CMO智能诊断系统 - 交互式模式")
        print("="*70)

        print("\n欢迎使用CMO智能诊断系统！")
        print("\n您可以使用自然语言描述诊断需求，例如：")
        print("  - '分析所有数据的微生物组和代谢组'")
        print("  - '分析前10个病人，使用文献支持'")
        print("  - '只分析微生物组，简要报告'")
        print("  - '全面分析，3轮辩论，生成详细双语报告'")
        print("\n输入 'quit' 或 'exit' 退出")

        # 主循环
        while True:
            print("\n" + "="*70)
            natural_request = input("\n请描述您的诊断需求（或输入 quit 退出）: ").strip()

            if natural_request.lower() in ['quit', 'exit', 'q']:
                print("\n感谢使用CMO智能诊断系统！再见！")
                break

            if not natural_request:
                print("✗ 请输入有效的诊断需求")
                continue

            # 询问病人元数据（可选）
            print("\n[可选] 病人元数据（直接回车跳过）:")
            age = input("  年龄: ").strip()
            sex = input("  性别 (M/F): ").strip()

            patient_metadata = {}
            if age:
                try:
                    patient_metadata["age"] = int(age)
                except:
                    pass
            if sex:
                patient_metadata["sex"] = sex

            # 运行诊断
            await self.run_diagnosis(natural_request, patient_metadata)

            # 询问是否继续
            print("\n" + "-"*70)
            continue_choice = input("\n是否继续诊断？(y/n): ").strip().lower()
            if continue_choice not in ['y', 'yes', '是']:
                print("\n感谢使用CMO智能诊断系统！再见！")
                break


async def main():
    """主函数"""
    try:
        # 初始化系统
        interface = IntelligentDiagnosisInterface()

        # 运行交互式模式
        await interface.interactive_mode()

    except KeyboardInterrupt:
        print("\n\n用户中断，程序退出")
    except Exception as e:
        print(f"\n✗ 系统错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
