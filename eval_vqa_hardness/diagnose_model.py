#!/usr/bin/env python3
"""
诊断模型目录，检查自定义代码模块问题
"""
import json
import os
from pathlib import Path

def diagnose_model(model_path):
    """诊断模型目录"""
    model_path = Path(model_path)
    
    print("=" * 70)
    print(f"诊断模型目录: {model_path}")
    print("=" * 70)
    
    # 检查 config.json
    config_path = model_path / "config.json"
    if config_path.exists():
        print("\n✓ 找到 config.json")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 检查 auto_map
        if 'auto_map' in config:
            print(f"\nauto_map 配置:")
            auto_map = config['auto_map']
            if isinstance(auto_map, dict):
                for key, value in auto_map.items():
                    print(f"  {key}: {value}")
                    # 检查是否有问题的模块名
                    if isinstance(value, str):
                        if 'LLaVA-OneVision' in value or '-' in value.split('.')[-1]:
                            print(f"    ⚠️  警告: 模块名包含连字符，可能导致导入失败")
        
        # 检查 model_type
        if 'model_type' in config:
            print(f"\nmodel_type: {config['model_type']}")
        
        # 检查 architectures
        if 'architectures' in config:
            print(f"\narchitectures: {config['architectures']}")
    else:
        print("\n✗ 未找到 config.json")
    
    # 检查自定义代码文件
    print("\n检查自定义代码文件:")
    custom_files = []
    for pattern in ['modeling_*.py', 'configuration_*.py', 'tokenization_*.py', 'processing_*.py']:
        files = list(model_path.glob(pattern))
        if files:
            custom_files.extend(files)
            for f in files:
                print(f"  ✓ {f.name}")
    
    if not custom_files:
        print("  ✗ 未找到自定义代码文件")
    
    # 检查是否有 transformers_modules 目录
    transformers_modules_dir = model_path / "transformers_modules"
    if transformers_modules_dir.exists():
        print(f"\n✓ 找到 transformers_modules 目录")
        subdirs = [d for d in transformers_modules_dir.iterdir() if d.is_dir()]
        for subdir in subdirs:
            print(f"  - {subdir.name}")
            if '-' in subdir.name:
                print(f"    ⚠️  警告: 目录名包含连字符: {subdir.name}")
    
    print("\n" + "=" * 70)
    print("诊断完成")
    print("=" * 70)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        diagnose_model(sys.argv[1])
    else:
        print("用法: python diagnose_model.py <model_path>")
        print("示例: python diagnose_model.py /path/to/LLaVA-OneVision-1.5-4B-Instruct")

