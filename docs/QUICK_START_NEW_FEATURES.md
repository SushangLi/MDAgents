# CMO智能调度系统 - 快速使用指南

## 简介

本指南展示如何使用新实现的CMO智能调度系统核心功能。

## 前提条件

```bash
# 确保安装所有依赖
pip install PyPDF2  # PDF解析
pip install langgraph langchain langchain-core  # 已有
```

## 功能1: 自然语言请求解析

### 基本用法

```python
import asyncio
from clinical.decision.request_parser import RequestParser
from clinical.decision.llm_wrapper import create_llm_wrapper

async def test_request_parser():
    # 初始化LLM和解析器
    wrapper = create_llm_wrapper(use_mock=False)  # 使用真实LLM
    parser = RequestParser(llm_call_func=wrapper.call)

    # 测试各种请求
    requests = [
        "只分析微生物组数据，使用文献支持",
        "分析病人P001的代谢组",
        "分析前50行数据的微生物组和蛋白质组",
        "全面分析病人P002、P003、P005，3轮辩论，详细报告",
        "快速诊断第100-200行数据，简要报告"
    ]

    for req in requests:
        print(f"\n请求: {req}")
        config = await parser.parse_request(req)
        print(f"配置: {config}")
        print(f"组学类型: {config.omics_types}")
        print(f"病人编号: {config.patient_ids}")
        print(f"行范围: {config.row_range}")
        print(f"强制RAG: {config.force_rag_even_no_conflict}")
        print(f"辩论轮次: {config.max_debate_rounds}")
        print(f"详细度: {config.detail_level}")

# 运行
asyncio.run(test_request_parser())
```

### 输出示例

```
请求: 只分析微生物组数据，使用文献支持
[RequestParser] Parsing request: "只分析微生物组数据，使用文献支持"
[RequestParser] ✓ Parsed successfully: DiagnosisConfig(omics=microbiome, RAG=on, force_RAG=on, ...)
配置: DiagnosisConfig(omics=microbiome, RAG=on, force_RAG=on, rounds=3, detail=standard, bilingual=yes)
组学类型: ['microbiome']
病人编号: None
行范围: None
强制RAG: True
辩论轮次: 3
详细度: standard
```

## 功能2: 配置序列化

### 保存和加载配置

```python
from clinical.models.diagnosis_config import DiagnosisConfig

# 创建配置
config = DiagnosisConfig(
    omics_types=["microbiome", "metabolome"],
    patient_ids=["P001", "P002", "P003"],
    row_range=(0, 100),
    force_rag_even_no_conflict=True,
    max_debate_rounds=3,
    detail_level="detailed",
    bilingual=True
)

# 保存到JSON
json_str = config.to_json()
with open("diagnosis_config.json", "w") as f:
    f.write(json_str)

# 从JSON加载
with open("diagnosis_config.json", "r") as f:
    loaded_config = DiagnosisConfig.from_json(f.read())

print(f"原配置: {config}")
print(f"加载配置: {loaded_config}")
```

## 功能3: 使用修复后的RAG系统

### 自动加载PDF文献

```python
from scripts.run_diagnosis import OralMultiomicsDiagnosisSystem
from pathlib import Path

# 1. 将PDF文献放入目录
literature_dir = Path("data/knowledge_base/medical_literature")
literature_dir.mkdir(parents=True, exist_ok=True)

# 假设您已将PDF文件放入该目录
# 例如: nutrients-15-05030.pdf

# 2. 初始化系统（自动加载PDF）
system = OralMultiomicsDiagnosisSystem(
    use_llm=True,
    enable_rag=True,  # 启用RAG
    enable_cag=True,
    use_mock_llm=False
)

# 系统会自动：
# [3/8] 初始化RAG文献检索系统...
#   Vector store empty. Scanning for PDFs...
#   Found PDFs in data/knowledge_base/medical_literature
#   ✓ PyPDF2 available
#   ✓ LiteratureIngester imported
#   Ingesting PDFs from data/knowledge_base/medical_literature...
#   ✓ Successfully loaded 1 PDFs (45 chunks)
#   ✓ Vector store now contains 45 documents

# 3. 验证RAG系统
if system.rag_system:
    doc_count = system.rag_system.vector_store.count()
    print(f"✓ RAG系统已加载 {doc_count} 个文档")

    # 测试检索
    results = system.rag_system.search("periodontitis bacteria", top_k=3)
    print(f"✓ 检索到 {len(results.documents)} 个相关文档")
    for i, doc in enumerate(results.documents[:3], 1):
        print(f"  {i}. 相关度: {results.relevance_scores[i-1]:.2f}")
```

## 功能4: 生成双语报告

### 完整示例

```python
from clinical.decision.bilingual_report_generator import BilingualReportGenerator
from clinical.models.diagnosis_result import DiagnosisResult
from clinical.models.expert_opinion import ExpertOpinion, FeatureImportance

# 创建诊断结果（示例）
diagnosis_result = DiagnosisResult(
    patient_id="P001",
    diagnosis="牙周炎 | Periodontitis",  # 已经是双语格式
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
                    biological_meaning="Red complex pathogen elevation"
                )
            ],
            biological_explanation="Elevated red complex bacteria | 红复合体细菌升高",
            evidence_chain=["步骤1 | Step 1", "步骤2 | Step 2"]
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
    explanation="基于微生物组证据的诊断 | Diagnosis based on microbiome evidence",
    references=["文献1 | Reference 1"],
    metadata={"model_version": "v1.0"}
)

# 生成双语报告
generator = BilingualReportGenerator(
    include_metadata=True,
    include_expert_details=True,
    include_biomarkers=True
)

bilingual_report = generator.generate_report(
    diagnosis_result=diagnosis_result,
    patient_metadata={"age": 45, "sex": "M"}
)

# 保存报告
output_path = "data/diagnosis_reports/P001_bilingual_report.md"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(bilingual_report)

print(f"✓ 双语报告已生成: {output_path}")
print("\n预览:")
print(bilingual_report[:500])
```

### 报告输出示例

```markdown
# 多组学临床诊断报告 | Multi-Omics Clinical Diagnostic Report

---

## 患者信息 | Patient Information

- **年龄 | Age**: 45
- **性别 | Sex**: M
- **报告日期 | Report Date**: 2026-01-07 10:30:00
- **患者编号 | Patient ID**: P001

---

## 执行摘要 | Executive Summary

### 最终诊断 | Final Diagnosis

**牙周炎 | Periodontitis**

**置信度 | Confidence**: 高 ✅ | High ✅ (87.0%)

### 关键发现 | Key Findings

- **专家共识 | Expert Consensus**: 1 位专家意见 | 1 expert opinions analyzed
- **冲突解决 | Conflict Resolution**: 未需要（达成共识） | Not required (consensus achieved)
- **关键生物标志物 | Key Biomarkers**: P. gingivalis

## 关键生物标志物 | Key Biomarkers

| 标志物<br>Biomarker | 组学类型<br>Omics | 方向<br>Direction | 重要性<br>Importance | 描述<br>Description |
|---------------------|-------------------|-------------------|----------------------|---------------------|
| P. gingivalis | 微生物组<br>Microbiome | 上调<br>Upregulated | 0.892 | 红复合体病原菌 \| Red complex pathogen |
```

## 功能5: 组合使用

### 端到端工作流

```python
import asyncio
from pathlib import Path
from clinical.decision.request_parser import RequestParser
from clinical.decision.llm_wrapper import create_llm_wrapper
from clinical.decision.bilingual_report_generator import BilingualReportGenerator
from scripts.run_diagnosis import OralMultiomicsDiagnosisSystem

async def intelligent_diagnosis_workflow():
    """
    智能诊断工作流示例
    """
    # 步骤1: 解析用户请求
    print("步骤1: 解析用户自然语言请求")
    wrapper = create_llm_wrapper()
    parser = RequestParser(llm_call_func=wrapper.call)

    user_request = "分析病人P001的微生物组和代谢组，使用文献支持，生成详细双语报告"
    config = await parser.parse_request(user_request)
    print(f"✓ 解析完成: {config}")

    # 步骤2: 初始化诊断系统（自动加载PDF）
    print("\n步骤2: 初始化诊断系统")
    system = OralMultiomicsDiagnosisSystem(
        use_llm=True,
        enable_rag=config.enable_rag,
        enable_cag=config.enable_cag,
        use_mock_llm=False
    )
    print("✓ 系统初始化完成")

    # 步骤3: 准备数据（根据配置筛选）
    print("\n步骤3: 准备诊断数据")
    # 这里应该根据config.patient_ids和config.omics_types加载数据
    # 简化示例中省略实际数据加载

    # 步骤4: 执行诊断（使用现有系统）
    print("\n步骤4: 执行诊断")
    # result = await system.diagnose(...)
    # 简化示例中使用模拟结果

    # 步骤5: 生成双语报告
    print("\n步骤5: 生成双语报告")
    generator = BilingualReportGenerator(
        include_metadata=(config.detail_level != "brief"),
        include_expert_details=(config.detail_level == "detailed"),
        include_biomarkers=True
    )

    # bilingual_report = generator.generate_report(diagnosis_result)
    print("✓ 双语报告生成完成")

    return config

# 运行工作流
asyncio.run(intelligent_diagnosis_workflow())
```

## 测试脚本

### 快速测试所有功能

创建 `test_new_features.py`:

```python
#!/usr/bin/env python3
"""
测试CMO智能调度系统新功能
"""

import asyncio
from pathlib import Path
from clinical.models.diagnosis_config import DiagnosisConfig
from clinical.decision.request_parser import RequestParser
from clinical.decision.llm_wrapper import create_llm_wrapper

async def test_all_features():
    """测试所有新功能"""

    print("="*70)
    print("CMO智能调度系统 - 功能测试")
    print("="*70)

    # 测试1: DiagnosisConfig
    print("\n[测试1] DiagnosisConfig数据模型")
    config = DiagnosisConfig(
        omics_types=["microbiome"],
        patient_ids=["P001", "P002"],
        row_range=(0, 50),
        force_rag_even_no_conflict=True,
        detail_level="detailed"
    )
    print(f"✓ 创建配置: {config}")

    # 序列化
    json_str = config.to_json()
    print(f"✓ JSON序列化: {len(json_str)} 字符")

    # 反序列化
    config2 = DiagnosisConfig.from_json(json_str)
    print(f"✓ JSON反序列化: {config2.omics_types}")

    # 测试2: RequestParser
    print("\n[测试2] RequestParser自然语言解析")
    wrapper = create_llm_wrapper(use_mock=True)  # 使用Mock避免API调用
    parser = RequestParser(llm_call_func=wrapper.call)

    test_requests = [
        "只分析微生物组",
        "分析病人P001-P003",
        "快速诊断，简要报告"
    ]

    for req in test_requests:
        try:
            parsed_config = await parser.parse_request(req)
            print(f"✓ \"{req}\" → {parsed_config.omics_types}")
        except Exception as e:
            print(f"✗ \"{req}\" → 错误: {e}")

    # 测试3: RAG系统
    print("\n[测试3] RAG系统PDF加载")
    literature_dir = Path("data/knowledge_base/medical_literature")
    if literature_dir.exists():
        pdf_files = list(literature_dir.glob("*.pdf"))
        print(f"✓ 发现 {len(pdf_files)} 个PDF文件")
        for pdf in pdf_files[:3]:
            print(f"  - {pdf.name}")
    else:
        print(f"⚠ 文献目录不存在: {literature_dir}")

    # 测试4: BilingualReportGenerator
    print("\n[测试4] BilingualReportGenerator")
    from clinical.decision.bilingual_report_generator import BilingualReportGenerator

    generator = BilingualReportGenerator()
    print(f"✓ 双语报告生成器已创建")

    # 测试翻译方法
    omics_cn, omics_en = generator._translate_omics("microbiome")
    print(f"✓ 翻译测试: microbiome → {omics_cn} | {omics_en}")

    print("\n"+"="*70)
    print("所有功能测试完成！")
    print("="*70)

if __name__ == "__main__":
    asyncio.run(test_all_features())
```

运行测试:

```bash
python test_new_features.py
```

## 故障排查

### 问题1: PyPDF2未安装

```bash
# 症状
⚠ PyPDF2 not installed. Install with: pip install PyPDF2

# 解决
pip install PyPDF2
```

### 问题2: LLM API密钥未配置

```bash
# 症状
⚠ No LLM API keys found - using MOCK mode

# 解决
# 在 .env.local 文件中配置至少一个API密钥:
DEEPSEEK_API_KEY=your_key_here
# 或
GEMINI_API_KEY=your_key_here
# 或
OPENAI_API_KEY=your_key_here
```

### 问题3: PDF文献目录不存在

```bash
# 症状
⚠ No PDFs found in data/knowledge_base/medical_literature

# 解决
mkdir -p data/knowledge_base/medical_literature
# 将PDF文件复制到该目录
```

## 下一步

1. **测试新功能**: 运行上述示例代码
2. **集成到现有流程**: 将新功能整合到您的诊断工作流
3. **准备真实PDF**: 将医学文献PDF放入指定目录
4. **配置LLM API**: 确保有可用的LLM服务
5. **反馈和优化**: 根据使用体验提出改进建议

---

生成时间: 2026-01-07
