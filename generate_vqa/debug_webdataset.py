#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试 WebDataset 数据，查看样本的实际结构
"""
import sys
from pathlib import Path
import webdataset as wds
import json

def debug_webdataset_sample(tar_path, sample_key=None):
    """
    调试单个样本，查看其结构
    """
    tar_path = Path(tar_path)
    if tar_path.is_dir():
        tar_files = list(tar_path.glob("*.tar"))
        if not tar_files:
            print(f"[ERROR] 目录中没有找到 .tar 文件")
            return
        tar_file = tar_files[0]
    elif tar_path.is_file():
        tar_file = tar_path
    else:
        print(f"[ERROR] 无效的路径: {tar_path}")
        return
    
    print(f"打开 WebDataset: {tar_file}\n")
    
    dataset = wds.WebDataset(str(tar_file))
    
    # 找到指定的样本或第一个样本
    target_key = None
    if sample_key:
        # 如果指定了 key，找到匹配的样本
        for sample in dataset:
            key = sample.get("__key__", "unknown")
            if sample_key in key:
                target_key = key
                print_sample(sample, target_key)
                break
    else:
        # 否则显示前几个样本
        count = 0
        for sample in dataset:
            count += 1
            key = sample.get("__key__", "unknown")
            print(f"\n{'='*80}")
            print(f"样本 {count}: {key}")
            print_sample(sample, key)
            
            if count >= 3:
                break

def print_sample(sample, key):
    """打印样本的详细信息"""
    print(f"\n[样本 Key]: {key}")
    print(f"\n[所有字段名]: {list(sample.keys())}")
    
    # 检查 JSON
    if "json" in sample:
        try:
            if isinstance(sample["json"], bytes):
                json_data = json.loads(sample["json"].decode('utf-8'))
            else:
                json_data = sample["json"]
            
            print(f"\n[JSON 元数据]:")
            print(f"  媒体类型: {json_data.get('media')}")
            print(f"  图片名称 (name): {json_data.get('name')}")
            print(f"  对话轮数: {len(json_data.get('texts', []))}")
            
            # 检查 name 字段指定的字段名是否存在于 sample 中
            expected_names = json_data.get('name', [])
            if expected_names:
                print(f"\n[字段名匹配检查]:")
                for name in expected_names:
                    exists = name in sample
                    value = sample.get(name)
                    value_type = type(value).__name__
                    value_size = len(value) if isinstance(value, bytes) else "N/A"
                    
                    print(f"  '{name}':")
                    print(f"    存在: {exists}")
                    print(f"    类型: {value_type}")
                    if isinstance(value, bytes):
                        print(f"    大小: {value_size} bytes ({value_size/1024:.2f} KB)")
                        # 检查文件头
                        if value.startswith(b'\xff\xd8\xff'):
                            print(f"    格式: JPEG (从文件头判断)")
                        elif value.startswith(b'\x89PNG'):
                            print(f"    格式: PNG (从文件头判断)")
                    elif value is None:
                        print(f"    ⚠️  值: None - 这就是问题所在！")
        except Exception as e:
            print(f"[ERROR] 解析 JSON 失败: {e}")
    
    # 列出所有可能的图片字段
    print(f"\n[所有可能的图片字段]:")
    image_fields = []
    for k, v in sample.items():
        if k in ['jpg', 'jpeg', 'png', 'image'] or '_image' in k:
            image_fields.append(k)
            if isinstance(v, bytes):
                print(f"  {k}: {len(v)} bytes ({type(v).__name__})")
            else:
                print(f"  {k}: {v} ({type(v).__name__})")
    
    if not image_fields:
        print("  ⚠️  未找到任何图片字段！")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python debug_webdataset.py <tar文件路径> [sample_key]")
        print("示例: python debug_webdataset.py /path/to/subtaskdata-0.tar")
        print("示例: python debug_webdataset.py /path/to/subtaskdata-0.tar image_2729")
        sys.exit(1)
    
    tar_path = sys.argv[1]
    sample_key = sys.argv[2] if len(sys.argv) > 2 else None
    
    debug_webdataset_sample(tar_path, sample_key)

