#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据分数将结果文件分流到两个文件（按类别智能划分）
根据各类别的数量和分数分布，按保留比例动态计算每个类别的划分阈值
同时保留最低分数线，低于最低分数的记录必须剔除
"""
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import statistics


def get_category_key(item: Dict[str, Any], category_field: str) -> str:
    """
    获取记录的类别键
    
    Args:
        item: 数据记录
        category_field: 类别字段名（如 "pipeline_name", "question_type", 或 "pipeline_name,question_type"）
    
    Returns:
        类别键字符串
    """
    if ',' in category_field:
        # 多字段组合
        fields = [f.strip() for f in category_field.split(',')]
        values = []
        for field in fields:
            value = item.get(field, "unknown")
            if value is None:
                value = "unknown"
            values.append(str(value))
        return "|".join(values)
    else:
        value = item.get(category_field, "unknown")
        if value is None:
            value = "unknown"
        return str(value)


def calculate_dynamic_threshold(
    scores: List[float],
    keep_ratio: float,
    min_threshold: float = 0.6,
    adaptive_high_score: bool = True
) -> float:
    """
    根据分数分布和保留比例动态计算阈值
    
    Args:
        scores: 分数列表（已排序，从高到低）
        keep_ratio: 保留比例（0-1之间，如0.8表示保留80%）
        min_threshold: 最低分数线（默认0.6）
        adaptive_high_score: 如果分数普遍高，是否提高筛选要求（默认True）
    
    Returns:
        计算出的阈值
    """
    if not scores:
        return min_threshold
    
    # 过滤掉无效分数
    valid_scores = [s for s in scores if s is not None and 0 <= s <= 1]
    if not valid_scores:
        return min_threshold
    
    # 排序（从高到低）
    valid_scores.sort(reverse=True)
    
    # 计算统计信息
    median_score = statistics.median(valid_scores)
    mean_score = statistics.mean(valid_scores)
    max_score = max(valid_scores)
    min_score = min(valid_scores)
    
    # 如果分数普遍高（中位数>0.8），提高筛选要求
    if adaptive_high_score and median_score > 0.8:
        # 使用更高的百分位数（如90%而不是80%）
        adjusted_ratio = min(keep_ratio + 0.1, 0.95)  # 最多提高到95%
        print(f"    [自适应] 分数普遍较高（中位数={median_score:.3f}），提高筛选要求：{keep_ratio:.1%} -> {adjusted_ratio:.1%}")
        keep_ratio = adjusted_ratio
    
    # 计算需要保留的数量
    keep_count = max(1, int(len(valid_scores) * keep_ratio))
    
    # 如果保留数量超过总数，保留所有
    if keep_count >= len(valid_scores):
        threshold = min(valid_scores)
    else:
        # 找到保留keep_count个记录时的最低分数
        threshold = valid_scores[keep_count - 1]
    
    # 确保阈值不低于最低分数线
    threshold = max(threshold, min_threshold)
    
    return threshold


def split_by_score_category_aware(
    input_file: Path,
    high_score_file: Path,
    low_score_file: Path,
    keep_ratio: float = 0.8,
    min_threshold: float = 0.6,
        category_field: str = "pipeline_type",
    score_field: str = "total_score",
    adaptive_high_score: bool = True,
    include_no_score_in_high: bool = False
) -> Tuple[int, int, Dict[str, Dict[str, Any]]]:
    """
    根据类别和分数分布智能分流
    
    Args:
        input_file: 输入JSON文件路径
        high_score_file: 高分输出文件路径
        low_score_file: 低分输出文件路径
        keep_ratio: 每个类别的保留比例（0-1之间，默认0.8即保留80%）
        min_threshold: 最低分数线（默认0.6，低于此分数的必须剔除）
        category_field: 类别字段名（如 "pipeline_name", "question_type", 或 "pipeline_name,question_type"）
        score_field: 分数字段名（默认 "total_score"，也支持 "validation_score"）
        adaptive_high_score: 如果分数普遍高，是否提高筛选要求（默认True）
        include_no_score_in_high: 是否将无分数记录保存到高分文件（默认False）
        
    Returns:
        (高分记录数, 低分记录数, 类别统计信息)
    """
    # 读取输入文件
    print(f"[INFO] 读取输入文件: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError(f"输入文件应该包含一个数组，但得到: {type(data)}")
    
    total = len(data)
    print(f"[INFO] 总记录数: {total}")
    print(f"[INFO] 保留比例: {keep_ratio:.1%}")
    print(f"[INFO] 最低分数线: {min_threshold}")
    print(f"[INFO] 类别字段: {category_field}")
    print(f"[INFO] 分数字段: {score_field}")
    
    # 按类别分组
    category_data = defaultdict(list)
    no_score_data = []  # 没有分数的记录
    
    for item in data:
        score = item.get(score_field) or item.get("validation_score")  # 兼容两种字段名
        
        if score is None:
            no_score_data.append(item)
        else:
            category_key = get_category_key(item, category_field)
            category_data[category_key].append(item)
    
    print(f"\n[INFO] 发现 {len(category_data)} 个类别")
    print(f"[INFO] 无分数记录: {len(no_score_data)} 条")
    
    # 为每个类别计算动态阈值
    category_thresholds = {}
    category_stats = {}
    
    print("\n[INFO] 计算各类别的动态阈值:")
    print("-" * 80)
    
    for category_key, items in sorted(category_data.items()):
        # 提取分数
        scores = []
        for item in items:
            score = item.get(score_field) or item.get("validation_score")
            if score is not None:
                scores.append(float(score))
        
        if not scores:
            print(f"  {category_key}: 无有效分数，跳过")
            continue
        
        # 计算动态阈值
        threshold = calculate_dynamic_threshold(
            scores=scores,
            keep_ratio=keep_ratio,
            min_threshold=min_threshold,
            adaptive_high_score=adaptive_high_score
        )
        
        category_thresholds[category_key] = threshold
        
        # 统计信息
        scores_sorted = sorted(scores, reverse=True)
        median_score = statistics.median(scores)
        mean_score = statistics.mean(scores)
        keep_count = max(1, int(len(scores) * keep_ratio))
        actual_keep_count = sum(1 for s in scores if s >= threshold)
        
        category_stats[category_key] = {
            "count": len(items),
            "threshold": threshold,
            "median_score": median_score,
            "mean_score": mean_score,
            "min_score": min(scores),
            "max_score": max(scores),
            "expected_keep": keep_count,
            "actual_keep": actual_keep_count,
            "keep_ratio_actual": actual_keep_count / len(items) if items else 0
        }
        
        print(f"  {category_key}:")
        print(f"    记录数: {len(items)}")
        print(f"    分数范围: {min(scores):.3f} - {max(scores):.3f}")
        print(f"    中位数: {median_score:.3f}, 平均值: {mean_score:.3f}")
        print(f"    动态阈值: {threshold:.3f} (期望保留 {keep_count} 条，实际保留 {actual_keep_count} 条)")
    
    print("-" * 80)
    
    # 根据阈值分流
    high_score_data = []
    low_score_data = []
    
    for category_key, items in category_data.items():
        threshold = category_thresholds.get(category_key, min_threshold)
        
        for item in items:
            score = item.get(score_field) or item.get("validation_score")
            
            if score is None:
                # 无分数记录
                if include_no_score_in_high:
                    high_score_data.append(item)
                else:
                    low_score_data.append(item)
            elif score >= threshold:
                # 高于或等于阈值
                high_score_data.append(item)
            else:
                # 低于阈值
                low_score_data.append(item)
    
    # 处理无分数记录（如果未包含在类别数据中）
    for item in no_score_data:
        if include_no_score_in_high:
            high_score_data.append(item)
        else:
            low_score_data.append(item)
    
    # 打印统计信息
    print(f"\n[统计] 分流结果:")
    print(f"  高分组: {len(high_score_data)} 条")
    print(f"  低分组: {len(low_score_data)} 条")
    print(f"  无分数记录: {len(no_score_data)} 条")
    
    if high_score_data:
        high_scores = []
        for item in high_score_data:
            score = item.get(score_field) or item.get("validation_score")
            if score is not None:
                high_scores.append(float(score))
        if high_scores:
            print(f"  高分组分数范围: {min(high_scores):.3f} - {max(high_scores):.3f}")
            print(f"  高分组平均分数: {statistics.mean(high_scores):.3f}")
    
    if low_score_data:
        low_scores = []
        for item in low_score_data:
            score = item.get(score_field) or item.get("validation_score")
            if score is not None:
                low_scores.append(float(score))
        if low_scores:
            print(f"  低分组分数范围: {min(low_scores):.3f} - {max(low_scores):.3f}")
            print(f"  低分组平均分数: {statistics.mean(low_scores):.3f}")
    
    # 保存高分文件
    high_score_file.parent.mkdir(parents=True, exist_ok=True)
    with open(high_score_file, 'w', encoding='utf-8') as f:
        json.dump(high_score_data, f, ensure_ascii=False, indent=2)
    print(f"\n[INFO] 高分组已保存到: {high_score_file} ({len(high_score_data)} 条)")
    
    # 保存低分文件
    low_score_file.parent.mkdir(parents=True, exist_ok=True)
    with open(low_score_file, 'w', encoding='utf-8') as f:
        json.dump(low_score_data, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 低分组已保存到: {low_score_file} ({len(low_score_data)} 条)")
    
    # 保存类别统计信息
    stats_file = high_score_file.parent / f"{high_score_file.stem}_category_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": {
                "total": total,
                "high_score_count": len(high_score_data),
                "low_score_count": len(low_score_data),
                "no_score_count": len(no_score_data),
                "keep_ratio": keep_ratio,
                "min_threshold": min_threshold,
                "category_field": category_field
            },
            "categories": category_stats
        }, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 类别统计信息已保存到: {stats_file}")
    
    return len(high_score_data), len(low_score_data), category_stats


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description='根据类别和分数分布智能分流结果文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 按pipeline_type类别，保留80%，最低分数线0.6
  python utils/split_by_score.py results.json high_score.json low_score.json
  
  # 按question_type类别，保留70%
  python utils/split_by_score.py results.json high.json low.json --category-field question_type --keep-ratio 0.7
  
  # 按pipeline_type和question_type组合分类
  python utils/split_by_score.py results.json high.json low.json --category-field "pipeline_type,question_type"
  
  # 自定义最低分数线0.7
  python utils/split_by_score.py results.json high.json low.json --min-threshold 0.7
  
  # 禁用自适应高分筛选（不提高筛选要求）
  python utils/split_by_score.py results.json high.json low.json --no-adaptive
  
  # 将无分数记录也保存到高分文件
  python utils/split_by_score.py results.json high.json low.json --include-no-score
        """
    )
    
    parser.add_argument(
        'input_file',
        type=str,
        help='输入JSON文件路径'
    )
    parser.add_argument(
        'high_score_file',
        type=str,
        help='高分输出文件路径'
    )
    parser.add_argument(
        'low_score_file',
        type=str,
        help='低分输出文件路径'
    )
    parser.add_argument(
        '--keep-ratio',
        type=float,
        default=0.8,
        help='每个类别的保留比例（0-1之间，默认: 0.8即保留80%%）'
    )
    parser.add_argument(
        '--min-threshold',
        type=float,
        default=0.6,
        help='最低分数线（默认: 0.6，低于此分数的必须剔除）'
    )
    parser.add_argument(
        '--category-field',
        type=str,
        default='pipeline_type',
        help='类别字段名（默认: pipeline_type，支持 "question_type" 或 "pipeline_type,question_type"）'
    )
    parser.add_argument(
        '--score-field',
        type=str,
        default='total_score',
        help='分数字段名（默认: total_score，也支持 validation_score）'
    )
    parser.add_argument(
        '--no-adaptive',
        action='store_true',
        help='禁用自适应高分筛选（默认启用，分数普遍高时会提高筛选要求）'
    )
    parser.add_argument(
        '--include-no-score',
        action='store_true',
        help='将无分数记录（错误记录等）也保存到高分文件（默认不保存）'
    )
    
    args = parser.parse_args()
    
    input_file = Path(args.input_file)
    high_score_file = Path(args.high_score_file)
    low_score_file = Path(args.low_score_file)
    
    if not input_file.exists():
        print(f"[ERROR] 输入文件不存在: {input_file}")
        return 1
    
    if args.keep_ratio < 0 or args.keep_ratio > 1:
        print(f"[ERROR] 保留比例应该在0-1之间，当前值: {args.keep_ratio}")
        return 1
    
    if args.min_threshold < 0 or args.min_threshold > 1:
        print(f"[ERROR] 最低分数线应该在0-1之间，当前值: {args.min_threshold}")
        return 1
    
    try:
        high_count, low_count, category_stats = split_by_score_category_aware(
            input_file=input_file,
            high_score_file=high_score_file,
            low_score_file=low_score_file,
            keep_ratio=args.keep_ratio,
            min_threshold=args.min_threshold,
            category_field=args.category_field,
            score_field=args.score_field,
            adaptive_high_score=not args.no_adaptive,
            include_no_score_in_high=args.include_no_score
        )
        
        print(f"\n[完成] 分流完成！高分组: {high_count} 条，低分组: {low_count} 条")
        
        return 0
        
    except Exception as e:
        print(f"[ERROR] 处理失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
