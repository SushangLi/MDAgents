#!/usr/bin/env python3
"""
预处理缓存管理器

管理预处理数据的文件级缓存，避免重复预处理：
- 检查预处理文件是否存在
- 运行一次预处理并保存到文件
- 从文件加载预处理数据
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple
import json
from datetime import datetime

from clinical.preprocessing.microbiome_preprocessor import MicrobiomePreprocessor
from clinical.preprocessing.metabolome_preprocessor import MetabolomePreprocessor
from clinical.preprocessing.proteome_preprocessor import ProteomePreprocessor


class PreprocessingCacheManager:
    """文件级预处理缓存管理器"""

    def __init__(self, cache_dir: Path = None):
        """
        初始化缓存管理器

        Args:
            cache_dir: 缓存目录，默认为 data/preprocessed
        """
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent.parent / "data" / "preprocessed"

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 初始化预处理器
        self.preprocessors = {
            "microbiome": MicrobiomePreprocessor(),
            "metabolome": MetabolomePreprocessor(),
            "proteome": ProteomePreprocessor()
        }

    def _get_cache_paths(self, omics_type: str, data_source: str = "training") -> Tuple[Path, Path, Path]:
        """
        获取缓存文件路径

        Args:
            omics_type: 组学类型 (microbiome, metabolome, proteome)
            data_source: 数据源 (training, test)

        Returns:
            (data_path, params_path, metadata_path)
        """
        base_name = f"{data_source}_{omics_type}"
        data_path = self.cache_dir / f"{base_name}.parquet"
        params_path = self.cache_dir / f"{base_name}_params.json"
        metadata_path = self.cache_dir / f"{base_name}_metadata.json"

        return data_path, params_path, metadata_path

    def check_if_preprocessed_exists(self, omics_type: str, data_source: str = "training") -> bool:
        """
        检查预处理文件是否存在

        Args:
            omics_type: 组学类型
            data_source: 数据源

        Returns:
            True if all required files exist
        """
        data_path, params_path, metadata_path = self._get_cache_paths(omics_type, data_source)

        exists = data_path.exists() and params_path.exists()

        if exists:
            print(f"  ✓ 发现预处理缓存: {data_path.name}")
        else:
            print(f"  ⚠ 未发现预处理缓存: {data_path.name}")

        return exists

    def preprocess_and_save(
        self,
        omics_type: str,
        raw_data_path: Path,
        data_source: str = "training"
    ):
        """
        运行预处理并保存到文件

        Args:
            omics_type: 组学类型
            raw_data_path: 原始数据路径
            data_source: 数据源
        """
        print(f"\n[预处理] {omics_type.upper()} 数据...")

        # 检查原始数据是否存在
        raw_data_path = Path(raw_data_path)
        if not raw_data_path.exists():
            raise FileNotFoundError(f"原始数据不存在: {raw_data_path}")

        # 读取原始数据
        print(f"  1. 读取原始数据: {raw_data_path.name}")
        raw_data = pd.read_csv(raw_data_path, index_col=0)
        print(f"     形状: {raw_data.shape}")

        # 运行预处理
        print(f"  2. 运行预处理...")
        preprocessor = self.preprocessors[omics_type]
        result = preprocessor.fit_transform(raw_data)

        # 获取缓存路径
        data_path, params_path, metadata_path = self._get_cache_paths(omics_type, data_source)

        # 保存预处理数据（Parquet格式）
        print(f"  3. 保存预处理数据: {data_path.name}")
        result.data.to_parquet(data_path, index=True)

        # 保存预处理器参数
        print(f"  4. 保存预处理器参数: {params_path.name}")
        preprocessor.save_params(str(params_path))

        # 保存元数据
        metadata = {
            "omics_type": omics_type,
            "data_source": data_source,
            "raw_data_path": str(raw_data_path),
            "original_shape": raw_data.shape,
            "preprocessed_shape": result.data.shape,
            "preprocessing_timestamp": datetime.now().isoformat(),
            "feature_count": len(result.feature_names),
            "sample_count": len(result.data)
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"  ✓ 预处理完成并保存")
        print(f"     数据形状: {result.data.shape}")
        print(f"     特征数: {len(result.feature_names)}")

    def load_preprocessed(
        self,
        omics_type: str,
        data_source: str = "training"
    ) -> pd.DataFrame:
        """
        从文件加载预处理数据

        Args:
            omics_type: 组学类型
            data_source: 数据源

        Returns:
            预处理后的DataFrame
        """
        data_path, params_path, metadata_path = self._get_cache_paths(omics_type, data_source)

        if not data_path.exists():
            raise FileNotFoundError(
                f"预处理数据不存在: {data_path}\n"
                f"请先调用 preprocess_and_save() 进行预处理"
            )

        # 加载数据
        data = pd.read_parquet(data_path)

        # 加载预处理器参数（用于后续transform）
        if params_path.exists():
            self.preprocessors[omics_type].load_params(str(params_path))

        return data

    def preprocess_all(
        self,
        data_dir: Path = None,
        data_source: str = "training",
        force: bool = False
    ):
        """
        预处理所有组学数据

        Args:
            data_dir: 原始数据目录，默认为 data/training
            data_source: 数据源标识
            force: 是否强制重新预处理（即使缓存存在）
        """
        if data_dir is None:
            data_dir = Path(__file__).parent.parent.parent / "data" / data_source

        data_dir = Path(data_dir)

        print("\n" + "="*70)
        print(f"批量预处理 - {data_source.upper()} 数据集")
        print("="*70)

        for omics_type in ["microbiome", "metabolome", "proteome"]:
            raw_data_path = data_dir / f"{omics_type}_raw.csv"

            if not raw_data_path.exists():
                print(f"\n✗ {omics_type.upper()}: 原始数据不存在，跳过")
                continue

            # 检查缓存
            if not force and self.check_if_preprocessed_exists(omics_type, data_source):
                print(f"  → 使用现有缓存，跳过预处理")
                continue

            # 预处理并保存
            try:
                self.preprocess_and_save(omics_type, raw_data_path, data_source)
            except Exception as e:
                print(f"\n✗ {omics_type.upper()} 预处理失败: {e}")
                import traceback
                traceback.print_exc()

        print("\n" + "="*70)
        print("✅ 批量预处理完成")
        print("="*70)

    def get_metadata(self, omics_type: str, data_source: str = "training") -> Optional[Dict]:
        """
        获取预处理元数据

        Args:
            omics_type: 组学类型
            data_source: 数据源

        Returns:
            元数据字典，如果不存在返回None
        """
        _, _, metadata_path = self._get_cache_paths(omics_type, data_source)

        if not metadata_path.exists():
            return None

        with open(metadata_path, "r") as f:
            return json.load(f)

    def list_cached_data(self):
        """列出所有缓存的预处理数据"""
        print("\n" + "="*70)
        print("已缓存的预处理数据")
        print("="*70)

        parquet_files = list(self.cache_dir.glob("*.parquet"))

        if not parquet_files:
            print("\n暂无缓存数据")
            return

        for data_file in sorted(parquet_files):
            base_name = data_file.stem
            metadata_file = self.cache_dir / f"{base_name}_metadata.json"

            if metadata_file.exists():
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)

                print(f"\n✓ {base_name}")
                print(f"  类型: {metadata.get('omics_type', 'N/A')}")
                print(f"  数据源: {metadata.get('data_source', 'N/A')}")
                print(f"  形状: {metadata.get('preprocessed_shape', 'N/A')}")
                print(f"  时间: {metadata.get('preprocessing_timestamp', 'N/A')}")
            else:
                print(f"\n✓ {base_name} (无元数据)")

        print("\n" + "="*70)


if __name__ == "__main__":
    """测试缓存管理器"""
    import sys

    manager = PreprocessingCacheManager()

    if len(sys.argv) > 1 and sys.argv[1] == "preprocess":
        # 批量预处理
        manager.preprocess_all(force=False)
    else:
        # 列出缓存
        manager.list_cached_data()
