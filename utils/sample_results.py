#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从输出结果文件中随机采样n个样本
"""
import json
import argparse
import random
from pathlib import Path
from typing import List, Dict, Any


def sample_results(
    input_file: Path,
    output_file: Path,
    n: int,
    seed: int = None,
    preserve_order: bool = False,
    exclude_errors: bool = False,
    only_passed: bool = False,
    only_failed: bool = False
) -> None:
    """
    从输入文件中随机采样n个样本
    
    Args:
        input_file: 输入JSON文件路径
        output_file: 输出JSON文件路径
        n: 采样数量
        seed: 随机种子（用于可重复性）
        preserve_order: 是否保持原始顺序（True则按原始顺序输出，False则随机顺序）
    """
    # 读取输入文件
    print(f"[INFO] 读取输入文件: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError(f"输入文件应该包含一个数组，但得到: {type(data)}")
    
    total = len(data)
    print(f"[INFO] 总记录数: {total}")
    
    # 应用过滤条件
    if exclude_errors:
        data = [item for item in data if "error" not in item]
        print(f"[INFO] 过滤错误记录后: {len(data)} 条")
    
    if only_passed:
        data = [item for item in data if item.get("passed") == True]
        print(f"[INFO] 仅保留通过记录: {len(data)} 条")
    
    if only_failed:
        data = [item for item in data if item.get("passed") == False and "error" not in item]
        print(f"[INFO] 仅保留未通过记录: {len(data)} 条")
    
    if len(data) == 0:
        print(f"[ERROR] 过滤后没有可用记录")
        return
    
    # 检查采样数量
    if n > len(data):
        print(f"[WARNING] 采样数量 ({n}) 大于可用记录数 ({len(data)})，将采样所有记录")
        n = len(data)
    
    # 设置随机种子
    if seed is not None:
        random.seed(seed)
        print(f"[INFO] 使用随机种子: {seed}")
    
    # 随机采样
    sampled = random.sample(data, n)
    
    # 是否保持原始顺序
    if preserve_order:
        # 获取原始索引并排序
        indices = [data.index(item) for item in sampled]
        indices.sort()
        sampled = [data[i] for i in indices]
        print(f"[INFO] 保持原始顺序，采样索引: {indices[:10]}{'...' if len(indices) > 10 else ''}")
    else:
        print(f"[INFO] 随机顺序输出")
    
    # 保存输出文件
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sampled, f, ensure_ascii=False, indent=2)
    
    print(f"[INFO] 已采样 {n} 条记录，保存到: {output_file}")
    
    # 打印统计信息
    if sampled:
        passed_count = sum(1 for item in sampled if item.get("passed") == True)
        failed_count = sum(1 for item in sampled if item.get("passed") == False and "error" not in item)
        error_count = sum(1 for item in sampled if "error" in item)
        
        scores = [item.get("total_score", 0) for item in sampled if "total_score" in item]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        print(f"\n[统计] 采样结果:")
        print(f"  总记录数: {len(sampled)}")
        print(f"  通过: {passed_count}")
        print(f"  未通过: {failed_count}")
        print(f"  错误: {error_count}")
        if scores:
            print(f"  平均分数: {avg_score:.3f}")
            print(f"  最高分数: {max(scores):.3f}")
            print(f"  最低分数: {min(scores):.3f}")
        
        # 显示错误记录的详细信息
        if error_count > 0:
            print(f"\n[错误详情] 错误记录类型统计:")
            error_types = {}
            for item in sampled:
                if "error" in item:
                    error_msg = item.get("error", "未知错误")
                    # 提取错误类型（取前50个字符）
                    error_key = error_msg[:50] if len(error_msg) > 50 else error_msg
                    error_types[error_key] = error_types.get(error_key, 0) + 1
            
            for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
                print(f"  [{count}次] {error_type}")
            
            # 显示前3个错误记录的完整信息
            print(f"\n[错误详情] 前3个错误记录示例:")
            error_items = [item for item in sampled if "error" in item][:3]
            for i, item in enumerate(error_items, 1):
                error_msg = item.get("error", "未知错误")
                item_id = item.get("id", item.get("sample_index", "unknown"))
                print(f"  示例 {i} (ID: {item_id}):")
                print(f"    错误: {error_msg}")
                if "pipeline_type" in item:
                    print(f"    Pipeline: {item.get('pipeline_type', 'unknown')}")
                print()


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description='从输出结果文件中随机采样n个样本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 从results.json中随机采样100条记录
  python utils/sample_results.py results.json sample_100.json -n 100
  
  # 使用随机种子确保可重复性
  python utils/sample_results.py results.json sample_100.json -n 100 --seed 42
  
  # 保持原始顺序（按原始索引排序）
  python utils/sample_results.py results.json sample_100.json -n 100 --preserve-order
  
  # 排除错误记录，只采样有筛选结果的记录
  python utils/sample_results.py results.json sample_100.json -n 100 --exclude-errors
  
  # 只采样通过的记录
  python utils/sample_results.py results.json passed_samples.json -n 100 --only-passed
  
  # 只采样未通过的记录（排除错误）
  python utils/sample_results.py results.json failed_samples.json -n 100 --only-failed
        """
    )
    
    parser.add_argument(
        'input_file',
        type=str,
        help='输入JSON文件路径'
    )
    parser.add_argument(
        'output_file',
        type=str,
        help='输出JSON文件路径'
    )
    parser.add_argument(
        '-n', '--num-samples',
        type=int,
        required=True,
        help='采样数量'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='随机种子（用于可重复性）'
    )
    parser.add_argument(
        '--preserve-order',
        action='store_true',
        help='保持原始顺序（按原始索引排序）'
    )
    parser.add_argument(
        '--exclude-errors',
        action='store_true',
        help='排除错误记录（只采样有筛选结果的记录）'
    )
    parser.add_argument(
        '--only-passed',
        action='store_true',
        help='只采样通过的记录（passed=true）'
    )
    parser.add_argument(
        '--only-failed',
        action='store_true',
        help='只采样未通过的记录（passed=false，排除错误记录）'
    )
    
    args = parser.parse_args()
    
    input_file = Path(args.input_file)
    output_file = Path(args.output_file)
    
    if not input_file.exists():
        print(f"[ERROR] 输入文件不存在: {input_file}")
        return
    
    try:
        sample_results(
            input_file=input_file,
            output_file=output_file,
            n=args.num_samples,
            seed=args.seed,
            preserve_order=args.preserve_order,
            exclude_errors=args.exclude_errors,
            only_passed=args.only_passed,
            only_failed=args.only_failed
        )
    except Exception as e:
        print(f"[ERROR] 处理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()



# # 从results.json中随机采样100条记录
# python utils/sample_results.py /home/zhuxuzhou/test_localization/object_localization/final_output.json sample_10.json -n 10

# # 使用随机种子确保可重复性
# python utils/sample_results.py /home/zhuxuzhou/test_localization/object_localization/output.json sample_100.json -n 100 --seed 42

# # 保持原始顺序
# python utils/sample_results.py /home/zhuxuzhou/test_localization/object_localization/output.json sample_100.json -n 100 --preserve-order