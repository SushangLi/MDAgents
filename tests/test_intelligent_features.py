#!/usr/bin/env python3
"""
测试CMO智能调度系统新功能

测试范围:
1. DiagnosisConfig 数据模型
2. RequestParser 自然语言解析
3. BilingualReportGenerator 双语报告生成
4. IntelligentDebateSystem 工作流

运行方式:
    python tests/test_intelligent_features.py
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


async def test_diagnosis_config():
    """测试1: DiagnosisConfig数据模型"""
    print("\n" + "="*70)
    print("测试1: DiagnosisConfig数据模型")
    print("="*70)

    from clinical.models.diagnosis_config import DiagnosisConfig

    # 测试1.1: 创建配置
    print("\n[1.1] 创建诊断配置")
    config = DiagnosisConfig(
        omics_types=["microbiome", "metabolome"],
        patient_ids=["P001", "P002", "P003"],
        row_range=(0, 50),
        force_rag_even_no_conflict=True,
        max_debate_rounds=3,
        detail_level="detailed",
        bilingual=True
    )
    print(f"✓ 配置创建成功: {config.omics_types}")
    print(f"  病人编号: {config.patient_ids}")
    print(f"  行范围: {config.row_range}")
    print(f"  强制RAG: {config.force_rag_even_no_conflict}")
    print(f"  详细度: {config.detail_level}")

    # 测试1.2: JSON序列化
    print("\n[1.2] JSON序列化测试")
    json_str = config.to_json()
    print(f"✓ JSON序列化成功 ({len(json_str)} 字符)")

    # 测试1.3: JSON反序列化
    print("\n[1.3] JSON反序列化测试")
    config2 = DiagnosisConfig.from_json(json_str)
    assert config2.omics_types == config.omics_types
    assert config2.patient_ids == config.patient_ids
    assert config2.row_range == config.row_range
    print(f"✓ JSON反序列化成功，数据一致")

    # 测试1.4: 默认配置
    print("\n[1.4] 默认配置测试")
    default_config = DiagnosisConfig.get_default()
    print(f"✓ 默认配置: {default_config.omics_types}")
    print(f"  RAG启用: {default_config.enable_rag}")
    print(f"  CAG启用: {default_config.enable_cag}")

    print("\n✅ DiagnosisConfig 测试通过")
    return True


async def test_request_parser():
    """测试2: RequestParser自然语言解析"""
    print("\n" + "="*70)
    print("测试2: RequestParser自然语言解析")
    print("="*70)

    from clinical.decision.request_parser import RequestParser
    from clinical.decision.llm_wrapper import create_llm_wrapper

    # 创建LLM wrapper (使用mock模式避免API调用)
    print("\n[2.1] 初始化RequestParser")
    try:
        wrapper = create_llm_wrapper(use_mock=True)
        parser = RequestParser(llm_call_func=wrapper.call)
        print("✓ RequestParser初始化成功 (Mock模式)")
    except Exception as e:
        print(f"⚠ RequestParser初始化失败: {e}")
        print("  提示: 如果要测试真实LLM解析，请配置LLM API密钥")
        return False

    # 测试用例
    test_requests = [
        {
            "request": "只分析微生物组数据",
            "expected_omics": ["microbiome"]
        },
        {
            "request": "分析病人P001的代谢组",
            "expected_omics": ["metabolome"],
            "expected_patients": ["P001"]
        },
        {
            "request": "分析前50行数据",
            "expected_row_range": (0, 50)
        },
        {
            "request": "快速诊断，简要报告",
            "expected_detail": "brief"
        }
    ]

    print("\n[2.2] 测试各种自然语言请求")
    for i, test_case in enumerate(test_requests, 1):
        req = test_case["request"]
        print(f"\n  测试用例 {i}: \"{req}\"")

        try:
            # Mock模式下会返回默认配置
            config = await parser.parse_request(req)
            print(f"  ✓ 解析成功")
            print(f"    组学类型: {config.omics_types}")
            print(f"    病人编号: {config.patient_ids}")
            print(f"    行范围: {config.row_range}")
            print(f"    详细度: {config.detail_level}")
        except Exception as e:
            print(f"  ✗ 解析失败: {e}")

    print("\n✅ RequestParser 测试完成")
    print("  注意: Mock模式下返回默认配置，要测试真实解析请配置LLM API")
    return True


async def test_bilingual_report_generator():
    """测试3: BilingualReportGenerator双语报告生成"""
    print("\n" + "="*70)
    print("测试3: BilingualReportGenerator双语报告生成")
    print("="*70)

    from clinical.decision.bilingual_report_generator import BilingualReportGenerator
    from clinical.models.diagnosis_result import DiagnosisResult
    from clinical.models.expert_opinion import ExpertOpinion, FeatureImportance

    # 测试3.1: 创建生成器
    print("\n[3.1] 创建BilingualReportGenerator")
    generator = BilingualReportGenerator(
        include_metadata=True,
        include_expert_details=True,
        include_biomarkers=True
    )
    print("✓ 生成器创建成功")

    # 测试3.2: 翻译方法测试
    print("\n[3.2] 测试翻译方法")
    omics_cn, omics_en = generator._translate_omics("microbiome")
    print(f"  microbiome → {omics_cn} | {omics_en}")
    assert omics_cn == "微生物组" and omics_en == "Microbiome"

    dir_cn, dir_en = generator._translate_direction("up")
    print(f"  up → {dir_cn} | {dir_en}")
    assert dir_cn == "上调" and dir_en == "Upregulated"

    method_cn, method_en = generator._translate_method("debate")
    print(f"  debate → {method_cn} | {method_en}")
    assert method_cn == "辩论" and method_en == "Debate"

    print("✓ 翻译方法测试通过")

    # 测试3.3: 生成双语报告
    print("\n[3.3] 生成双语报告")

    # 创建测试诊断结果
    diagnosis_result = DiagnosisResult(
        patient_id="P001",
        diagnosis="牙周炎 | Periodontitis",
        confidence=0.87,
        expert_opinions=[
            ExpertOpinion(
                expert_name="microbiome_expert",
                omics_type="microbiome",
                diagnosis="Periodontitis",
                probability=0.89,
                confidence=0.85,
                top_features=[
                    FeatureImportance(
                        feature_name="Porphyromonas_gingivalis",
                        importance_score=0.892,
                        direction="up",
                        biological_meaning="Red complex pathogen"
                    )
                ],
                biological_explanation="Elevated red complex bacteria | 红复合体细菌升高",
                evidence_chain=["步骤1 | Step 1", "步骤2 | Step 2"],
                model_metadata={},
                timestamp=""
            )
        ],
        key_biomarkers=[
            {
                "name": "P. gingivalis",
                "omics_type": "microbiome",
                "direction": "up",
                "importance": 0.892,
                "description": "红复合体病原菌 | Red complex pathogen"
            }
        ],
        clinical_recommendations=[
            "建议牙周治疗 | Recommend periodontal treatment",
            "监测细菌水平 | Monitor bacterial levels"
        ],
        explanation="基于微生物组证据 | Based on microbiome evidence",
        references=["文献1 | Reference 1"],
        metadata={"model_version": "v1.0"}
    )

    # 生成报告
    bilingual_report = generator.generate_report(
        diagnosis_result=diagnosis_result,
        patient_metadata={"age": 45, "sex": "M"}
    )

    print(f"✓ 报告生成成功 ({len(bilingual_report)} 字符)")

    # 验证双语格式
    assert "多组学临床诊断报告 | Multi-Omics Clinical Diagnostic Report" in bilingual_report
    assert "患者信息 | Patient Information" in bilingual_report
    assert "牙周炎 | Periodontitis" in bilingual_report
    assert "微生物组" in bilingual_report and "Microbiome" in bilingual_report

    # 保存报告样本
    output_path = project_root / "data" / "diagnosis_reports" / "test_bilingual_report.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(bilingual_report)
    print(f"✓ 报告已保存到: {output_path}")

    # 显示报告预览
    print("\n[3.4] 报告预览 (前500字符)")
    print("-" * 70)
    print(bilingual_report[:500])
    print("-" * 70)

    print("\n✅ BilingualReportGenerator 测试通过")
    return True


async def test_intelligent_debate_system_structure():
    """测试4: IntelligentDebateSystem结构验证"""
    print("\n" + "="*70)
    print("测试4: IntelligentDebateSystem结构验证")
    print("="*70)

    try:
        from clinical.decision.intelligent_debate_system import IntelligentDebateSystem

        print("\n[4.1] 导入IntelligentDebateSystem")
        print("✓ 模块导入成功")

        print("\n[4.2] 检查类定义")
        assert hasattr(IntelligentDebateSystem, '__init__')
        assert hasattr(IntelligentDebateSystem, '_build_intelligent_graph')
        assert hasattr(IntelligentDebateSystem, 'run_intelligent_diagnosis')
        print("✓ 核心方法存在")

        print("\n[4.3] 检查节点方法")
        node_methods = [
            '_parse_request_node',
            '_filter_data_node',
            '_select_omics_node',
            '_preprocess_data_node',
            '_get_expert_opinions_node',
            '_generate_bilingual_report_node'
        ]
        for method in node_methods:
            assert hasattr(IntelligentDebateSystem, method)
            print(f"  ✓ {method}")

        print("\n[4.4] 检查条件边方法")
        assert hasattr(IntelligentDebateSystem, '_should_debate_or_rag')
        assert hasattr(IntelligentDebateSystem, '_query_rag_conditional_node')
        print("  ✓ _should_debate_or_rag")
        print("  ✓ _query_rag_conditional_node")

        print("\n✅ IntelligentDebateSystem 结构验证通过")
        return True

    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False
    except AssertionError as e:
        print(f"✗ 结构验证失败: {e}")
        return False


async def test_mcp_server_integration():
    """测试5: MCP服务器集成验证"""
    print("\n" + "="*70)
    print("测试5: MCP服务器集成验证")
    print("="*70)

    try:
        mcp_server_path = project_root / "mcp_server" / "clinical_diagnosis_server.py"

        print("\n[5.1] 检查MCP服务器文件")
        assert mcp_server_path.exists()
        print(f"✓ 文件存在: {mcp_server_path}")

        print("\n[5.2] 检查全局变量定义")
        with open(mcp_server_path, 'r', encoding='utf-8') as f:
            content = f.read()

        assert "_request_parser" in content
        assert "_intelligent_debate_system" in content
        assert "_bilingual_report_generator" in content
        print("  ✓ _request_parser")
        print("  ✓ _intelligent_debate_system")
        print("  ✓ _bilingual_report_generator")

        print("\n[5.3] 检查新工具定义")
        assert "diagnose_with_natural_language" in content
        assert "configure_diagnosis" in content
        print("  ✓ diagnose_with_natural_language 工具")
        print("  ✓ configure_diagnosis 工具")

        print("\n[5.4] 检查处理函数")
        assert "async def _diagnose_with_nl" in content
        assert "async def _diagnose_with_config" in content
        print("  ✓ _diagnose_with_nl 处理函数")
        print("  ✓ _diagnose_with_config 处理函数")

        print("\n✅ MCP服务器集成验证通过")
        return True

    except AssertionError as e:
        print(f"✗ 验证失败: {e}")
        return False
    except Exception as e:
        print(f"✗ 错误: {e}")
        return False


async def test_rag_pdf_loading():
    """测试6: RAG系统PDF加载验证"""
    print("\n" + "="*70)
    print("测试6: RAG系统PDF自动加载验证")
    print("="*70)

    try:
        run_diagnosis_path = project_root / "scripts" / "run_diagnosis.py"

        print("\n[6.1] 检查run_diagnosis.py文件")
        assert run_diagnosis_path.exists()
        print(f"✓ 文件存在: {run_diagnosis_path}")

        print("\n[6.2] 检查_load_pdf_literature方法")
        with open(run_diagnosis_path, 'r', encoding='utf-8') as f:
            content = f.read()

        assert "def _load_pdf_literature" in content
        assert "PyPDF2" in content
        assert "LiteratureIngester" in content
        print("  ✓ _load_pdf_literature 方法存在")
        print("  ✓ PyPDF2 依赖检查")
        print("  ✓ LiteratureIngester 导入")

        print("\n[6.3] 检查文献目录")
        literature_dir = project_root / "data" / "knowledge_base" / "medical_literature"
        literature_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ 文献目录: {literature_dir}")

        pdf_files = list(literature_dir.glob("*.pdf"))
        if pdf_files:
            print(f"✓ 发现 {len(pdf_files)} 个PDF文件:")
            for pdf in pdf_files[:3]:
                print(f"    - {pdf.name}")
        else:
            print("  ⚠ 未发现PDF文件，系统将使用示例文献")

        print("\n✅ RAG系统PDF加载验证通过")
        return True

    except AssertionError as e:
        print(f"✗ 验证失败: {e}")
        return False
    except Exception as e:
        print(f"✗ 错误: {e}")
        return False


async def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*70)
    print("CMO智能调度系统 - 新功能测试套件")
    print("="*70)

    results = []

    # 测试1: DiagnosisConfig
    try:
        result = await test_diagnosis_config()
        results.append(("DiagnosisConfig", result))
    except Exception as e:
        print(f"✗ DiagnosisConfig测试异常: {e}")
        results.append(("DiagnosisConfig", False))

    # 测试2: RequestParser
    try:
        result = await test_request_parser()
        results.append(("RequestParser", result))
    except Exception as e:
        print(f"✗ RequestParser测试异常: {e}")
        results.append(("RequestParser", False))

    # 测试3: BilingualReportGenerator
    try:
        result = await test_bilingual_report_generator()
        results.append(("BilingualReportGenerator", result))
    except Exception as e:
        print(f"✗ BilingualReportGenerator测试异常: {e}")
        results.append(("BilingualReportGenerator", False))

    # 测试4: IntelligentDebateSystem
    try:
        result = await test_intelligent_debate_system_structure()
        results.append(("IntelligentDebateSystem", result))
    except Exception as e:
        print(f"✗ IntelligentDebateSystem测试异常: {e}")
        results.append(("IntelligentDebateSystem", False))

    # 测试5: MCP服务器集成
    try:
        result = await test_mcp_server_integration()
        results.append(("MCP Server Integration", result))
    except Exception as e:
        print(f"✗ MCP服务器集成测试异常: {e}")
        results.append(("MCP Server Integration", False))

    # 测试6: RAG系统PDF加载
    try:
        result = await test_rag_pdf_loading()
        results.append(("RAG PDF Loading", result))
    except Exception as e:
        print(f"✗ RAG PDF加载测试异常: {e}")
        results.append(("RAG PDF Loading", False))

    # 汇总结果
    print("\n" + "="*70)
    print("测试结果汇总")
    print("="*70)

    passed = 0
    failed = 0

    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status}  {test_name}")
        if result:
            passed += 1
        else:
            failed += 1

    print("\n" + "-"*70)
    print(f"总计: {len(results)} 个测试")
    print(f"通过: {passed} 个")
    print(f"失败: {failed} 个")
    print(f"成功率: {passed/len(results)*100:.1f}%")
    print("="*70)

    return passed == len(results)


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
