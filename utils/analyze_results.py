#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析JSON结果文件，统计各种指标
"""
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict


def analyze_results(input_file: Path) -> Dict[str, Any]:
    """
    分析JSON结果文件
    
    Args:
        input_file: 输入JSON文件路径
        
    Returns:
        统计结果字典
    """
    print(f"[INFO] 读取输入文件: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError(f"输入文件应该包含一个数组，但得到: {type(data)}")
    
    total = len(data)
    print(f"[INFO] 总记录数: {total}")
    
    # 初始化统计
    stats = {
        "total": total,
        "by_pipeline_type": defaultdict(lambda: {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "error": 0,
            "high_score": 0,  # >= 0.6
            "medium_score": 0,  # 0.3-0.6
            "low_score": 0,  # < 0.3
            "no_score": 0,
            "scores": [],
            "confidences": []
        }),
        "overall": {
            "passed": 0,
            "failed": 0,
            "error": 0,
            "high_score": 0,  # >= 0.6
            "medium_score": 0,  # 0.3-0.6
            "low_score": 0,  # < 0.3
            "no_score": 0,
            "scores": [],
            "confidences": []
        }
    }
    
    # 统计每条记录
    for record in data:
        pipeline_type = record.get("pipeline_type", "unknown")
        pipeline_stats = stats["by_pipeline_type"][pipeline_type]
        
        # 总数
        pipeline_stats["total"] += 1
        
        # 错误记录
        if "error" in record:
            pipeline_stats["error"] += 1
            stats["overall"]["error"] += 1
            continue
        
        # 通过/未通过
        passed = record.get("passed", False)
        if passed:
            pipeline_stats["passed"] += 1
            stats["overall"]["passed"] += 1
        else:
            pipeline_stats["failed"] += 1
            stats["overall"]["failed"] += 1
        
        # 分数统计
        total_score = record.get("total_score")
        if total_score is not None:
            pipeline_stats["scores"].append(total_score)
            stats["overall"]["scores"].append(total_score)
            
            if total_score >= 0.6:
                pipeline_stats["high_score"] += 1
                stats["overall"]["high_score"] += 1
            elif total_score >= 0.3:
                pipeline_stats["medium_score"] += 1
                stats["overall"]["medium_score"] += 1
            else:
                pipeline_stats["low_score"] += 1
                stats["overall"]["low_score"] += 1
        else:
            pipeline_stats["no_score"] += 1
            stats["overall"]["no_score"] += 1
        
        # 置信度
        confidence = record.get("confidence")
        if confidence is not None:
            pipeline_stats["confidences"].append(confidence)
            stats["overall"]["confidences"].append(confidence)
    
    # 计算平均值
    for pipeline_type, pipeline_stats in stats["by_pipeline_type"].items():
        scores = pipeline_stats["scores"]
        confidences = pipeline_stats["confidences"]
        
        pipeline_stats["avg_score"] = sum(scores) / len(scores) if scores else 0.0
        pipeline_stats["min_score"] = min(scores) if scores else 0.0
        pipeline_stats["max_score"] = max(scores) if scores else 0.0
        pipeline_stats["avg_confidence"] = sum(confidences) / len(confidences) if confidences else 0.0
        
        # 计算通过率
        valid_count = pipeline_stats["total"] - pipeline_stats["error"]
        pipeline_stats["pass_rate"] = (pipeline_stats["passed"] / valid_count * 100) if valid_count > 0 else 0.0
    
    # 整体统计
    overall = stats["overall"]
    scores = overall["scores"]
    confidences = overall["confidences"]
    
    overall["avg_score"] = sum(scores) / len(scores) if scores else 0.0
    overall["min_score"] = min(scores) if scores else 0.0
    overall["max_score"] = max(scores) if scores else 0.0
    overall["avg_confidence"] = sum(confidences) / len(confidences) if confidences else 0.0
    
    valid_count = stats["total"] - overall["error"]
    overall["pass_rate"] = (overall["passed"] / valid_count * 100) if valid_count > 0 else 0.0
    
    return stats


def print_statistics(stats: Dict[str, Any], output_file: Path = None):
    """
    打印统计结果
    
    Args:
        stats: 统计结果字典
        output_file: 可选的输出文件路径（保存为文本）
    """
    lines = []
    
    # 标题
    lines.append("=" * 80)
    lines.append("结果统计分析")
    lines.append("=" * 80)
    lines.append("")
    
    # 整体统计
    overall = stats["overall"]
    total = stats["total"]
    valid_count = total - overall["error"]
    
    lines.append("## 整体统计")
    lines.append("")
    lines.append(f"**总记录数**: {total}")
    lines.append(f"**有效记录数**: {valid_count} (排除错误记录)")
    lines.append(f"**错误记录数**: {overall['error']}")
    lines.append("")
    
    if valid_count > 0:
        lines.append("### 通过/未通过统计")
        lines.append(f"- ✅ **通过**: {overall['passed']} ({overall['passed']/valid_count*100:.1f}%)")
        lines.append(f"- ❌ **未通过**: {overall['failed']} ({overall['failed']/valid_count*100:.1f}%)")
        lines.append(f"- **通过率**: {overall['pass_rate']:.2f}%")
        lines.append("")
        
        lines.append("### 分数统计")
        if overall['scores']:
            lines.append(f"- **平均分数**: {overall['avg_score']:.3f}")
            lines.append(f"- **最高分数**: {overall['max_score']:.3f}")
            lines.append(f"- **最低分数**: {overall['min_score']:.3f}")
            lines.append("")
            lines.append(f"- **高分 (≥0.6)**: {overall['high_score']} ({overall['high_score']/len(overall['scores'])*100:.1f}%)")
            lines.append(f"- **中分 (0.3-0.6)**: {overall['medium_score']} ({overall['medium_score']/len(overall['scores'])*100:.1f}%)")
            lines.append(f"- **低分 (<0.3)**: {overall['low_score']} ({overall['low_score']/len(overall['scores'])*100:.1f}%)")
            lines.append(f"- **无分数**: {overall['no_score']}")
        else:
            lines.append("- 无分数数据")
        lines.append("")
        
        if overall['confidences']:
            lines.append("### 置信度统计")
            lines.append(f"- **平均置信度**: {overall['avg_confidence']:.3f}")
            lines.append(f"- **最高置信度**: {max(overall['confidences']):.3f}")
            lines.append(f"- **最低置信度**: {min(overall['confidences']):.3f}")
            lines.append("")
    
    # 按Pipeline类型统计
    lines.append("## 按Pipeline类型统计")
    lines.append("")
    
    # 按总数排序
    sorted_pipelines = sorted(
        stats["by_pipeline_type"].items(),
        key=lambda x: x[1]["total"],
        reverse=True
    )
    
    for pipeline_type, pipeline_stats in sorted_pipelines:
        lines.append(f"### {pipeline_type}")
        lines.append("")
        lines.append(f"- **总记录数**: {pipeline_stats['total']}")
        lines.append(f"- **错误记录**: {pipeline_stats['error']}")
        
        valid = pipeline_stats['total'] - pipeline_stats['error']
        if valid > 0:
            lines.append(f"- ✅ **通过**: {pipeline_stats['passed']} ({pipeline_stats['passed']/valid*100:.1f}%)")
            lines.append(f"- ❌ **未通过**: {pipeline_stats['failed']} ({pipeline_stats['failed']/valid*100:.1f}%)")
            lines.append(f"- **通过率**: {pipeline_stats['pass_rate']:.2f}%")
            lines.append("")
            
            if pipeline_stats['scores']:
                lines.append(f"- **平均分数**: {pipeline_stats['avg_score']:.3f}")
                lines.append(f"- **分数范围**: {pipeline_stats['min_score']:.3f} - {pipeline_stats['max_score']:.3f}")
                lines.append("")
                lines.append(f"- **高分 (≥0.6)**: {pipeline_stats['high_score']} ({pipeline_stats['high_score']/len(pipeline_stats['scores'])*100:.1f}%)")
                lines.append(f"- **中分 (0.3-0.6)**: {pipeline_stats['medium_score']} ({pipeline_stats['medium_score']/len(pipeline_stats['scores'])*100:.1f}%)")
                lines.append(f"- **低分 (<0.3)**: {pipeline_stats['low_score']} ({pipeline_stats['low_score']/len(pipeline_stats['scores'])*100:.1f}%)")
            
            if pipeline_stats['confidences']:
                lines.append(f"- **平均置信度**: {pipeline_stats['avg_confidence']:.3f}")
        
        lines.append("")
        lines.append("---")
        lines.append("")
    
    # 打印到控制台
    output_text = "\n".join(lines)
    print(output_text)
    
    # 保存到文件
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(output_text)
        print(f"\n[INFO] 统计结果已保存到: {output_file}")


def export_to_csv(stats: Dict[str, Any], output_file: Path):
    """
    导出统计结果为CSV格式
    
    Args:
        stats: 统计结果字典
        output_file: 输出CSV文件路径
    """
    import csv
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        
        # 表头
        writer.writerow([
            "Pipeline Type",
            "Total",
            "Error",
            "Valid",
            "Passed",
            "Failed",
            "Pass Rate (%)",
            "High Score (≥0.6)",
            "Medium Score (0.3-0.6)",
            "Low Score (<0.3)",
            "No Score",
            "Avg Score",
            "Min Score",
            "Max Score",
            "Avg Confidence"
        ])
        
        # 按Pipeline类型写入
        sorted_pipelines = sorted(
            stats["by_pipeline_type"].items(),
            key=lambda x: x[1]["total"],
            reverse=True
        )
        
        for pipeline_type, pipeline_stats in sorted_pipelines:
            valid = pipeline_stats['total'] - pipeline_stats['error']
            writer.writerow([
                pipeline_type,
                pipeline_stats['total'],
                pipeline_stats['error'],
                valid,
                pipeline_stats['passed'],
                pipeline_stats['failed'],
                f"{pipeline_stats['pass_rate']:.2f}",
                pipeline_stats['high_score'],
                pipeline_stats['medium_score'],
                pipeline_stats['low_score'],
                pipeline_stats['no_score'],
                f"{pipeline_stats['avg_score']:.3f}",
                f"{pipeline_stats['min_score']:.3f}",
                f"{pipeline_stats['max_score']:.3f}",
                f"{pipeline_stats['avg_confidence']:.3f}"
            ])
        
        # 整体统计行
        overall = stats["overall"]
        valid = stats["total"] - overall['error']
        writer.writerow([
            "OVERALL",
            stats["total"],
            overall['error'],
            valid,
            overall['passed'],
            overall['failed'],
            f"{overall['pass_rate']:.2f}",
            overall['high_score'],
            overall['medium_score'],
            overall['low_score'],
            overall['no_score'],
            f"{overall['avg_score']:.3f}",
            f"{overall['min_score']:.3f}",
            f"{overall['max_score']:.3f}",
            f"{overall['avg_confidence']:.3f}"
        ])
    
    print(f"[INFO] CSV文件已保存到: {output_file}")


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description='分析JSON结果文件，统计各种指标',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本统计（输出到控制台）
  python utils/analyze_results.py merged_output.json
  
  # 保存统计结果到文本文件
  python utils/analyze_results.py merged_output.json -o statistics.txt
  
  # 同时导出CSV格式
  python utils/analyze_results.py merged_output.json -o statistics.txt --csv statistics.csv
        """
    )
    
    parser.add_argument(
        'input_file',
        type=str,
        help='输入JSON文件路径（合并后的结果文件）'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='输出文本文件路径（可选）'
    )
    parser.add_argument(
        '--csv',
        type=str,
        default=None,
        help='导出CSV格式文件路径（可选）'
    )
    
    args = parser.parse_args()
    
    input_file = Path(args.input_file)
    
    if not input_file.exists():
        print(f"[ERROR] 输入文件不存在: {input_file}")
        return
    
    try:
        # 分析结果
        stats = analyze_results(input_file)
        
        # 打印统计
        output_file = Path(args.output) if args.output else None
        print_statistics(stats, output_file)
        
        # 导出CSV
        if args.csv:
            export_to_csv(stats, Path(args.csv))
        
    except Exception as e:
        print(f"[ERROR] 处理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

