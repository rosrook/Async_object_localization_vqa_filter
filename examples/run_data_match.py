#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据匹配示例脚本
演示如何使用Router的数据匹配功能
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.router import Router
import config


def main():
    """
    运行数据匹配的主函数
    """
    print("[INFO] 开始数据匹配...")
    
    # 方式1: 使用默认配置（从config.py和.env文件读取）
    # output_dir = Router.match_benchmark_data()
    
    # 方式2: 自定义配置
    output_dir = Router.match_benchmark_data(
        recat_root="/home/zhuxuzhou/recat",
        benchmark_file="/mnt/tidal-alsh01/dataset/perceptionVLMData/processed_v1.5/bench/MMBench_DEV_EN_V11/MMBench_DEV_EN_V11.parquet",
        output_dir="/home/zhuxuzhou/test_localization/object_localization/data/match_results",
        target_categories=[("object_localization", "finegrained_perception (instance-level)")],
        test_mode=False,
        test_samples=5,
        test_max_categories=2
    )
    
    print(f"[INFO] 数据匹配完成，输出目录: {output_dir}")


if __name__ == "__main__":
    main()

