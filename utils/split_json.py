#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将大JSON文件分割成多个小文件，用于分段处理
"""
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any


def split_json(
    input_file: Path,
    output_dir: Path,
    chunk_size: int = 1000,
    prefix: str = "chunk"
) -> List[Path]:
    """
    将JSON文件分割成多个小文件
    
    Args:
        input_file: 输入JSON文件路径
        output_dir: 输出目录
        chunk_size: 每个文件包含的记录数
        prefix: 输出文件前缀
        
    Returns:
        输出文件路径列表
    """
    print(f"[INFO] 读取输入文件: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError(f"输入文件应该包含一个数组，但得到: {type(data)}")
    
    total = len(data)
    print(f"[INFO] 总记录数: {total}")
    print(f"[INFO] 每个文件包含: {chunk_size} 条记录")
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 计算需要多少个文件
    num_chunks = (total + chunk_size - 1) // chunk_size  # 向上取整
    print(f"[INFO] 将分割为 {num_chunks} 个文件")
    
    output_files = []
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total)
        
        chunk_data = data[start_idx:end_idx]
        
        # 生成输出文件名
        output_file = output_dir / f"{prefix}_{i+1:04d}_of_{num_chunks:04d}.json"
        
        # 保存文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunk_data, f, ensure_ascii=False, indent=2)
        
        output_files.append(output_file)
        print(f"[INFO] 已创建: {output_file.name} ({len(chunk_data)} 条记录)")
    
    print(f"\n[完成] 共创建 {len(output_files)} 个文件")
    return output_files


def merge_results(
    input_files: List[Path],
    output_file: Path
) -> None:
    """
    合并多个结果文件
    
    Args:
        input_files: 输入文件列表
        output_file: 输出文件路径
    """
    print(f"[INFO] 开始合并 {len(input_files)} 个文件")
    
    all_results = []
    
    for input_file in input_files:
        print(f"[INFO] 读取: {input_file.name}")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            all_results.extend(data)
        else:
            all_results.append(data)
    
    # 保存合并结果
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"[完成] 已合并 {len(all_results)} 条记录到: {output_file}")


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description='分割或合并JSON文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 分割大文件为每1000条记录一个文件
  python utils/split_json.py split input.json output_dir/ -s 1000
  
  # 合并多个结果文件
  python utils/split_json.py merge output_dir/*.json merged_output.json
  
  # 分割并指定前缀
  python utils/split_json.py split input.json output_dir/ -s 500 -p batch
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='命令')
    
    # 分割命令
    split_parser = subparsers.add_parser('split', help='分割JSON文件')
    split_parser.add_argument('input_file', type=str, help='输入JSON文件路径')
    split_parser.add_argument('output_dir', type=str, help='输出目录')
    split_parser.add_argument('-s', '--chunk-size', type=int, default=1000,
                             help='每个文件包含的记录数（默认: 1000）')
    split_parser.add_argument('-p', '--prefix', type=str, default='chunk',
                             help='输出文件前缀（默认: chunk）')
    
    # 合并命令
    merge_parser = subparsers.add_parser('merge', help='合并JSON文件')
    merge_parser.add_argument('input_files', type=str, nargs='+',
                             help='输入JSON文件路径（支持通配符）')
    merge_parser.add_argument('output_file', type=str, help='输出JSON文件路径')
    
    args = parser.parse_args()
    
    if args.command == 'split':
        input_file = Path(args.input_file)
        output_dir = Path(args.output_dir)
        
        if not input_file.exists():
            print(f"[ERROR] 输入文件不存在: {input_file}")
            return
        
        try:
            output_files = split_json(
                input_file=input_file,
                output_dir=output_dir,
                chunk_size=args.chunk_size,
                prefix=args.prefix
            )
            
            print(f"\n[提示] 可以使用以下命令逐个处理:")
            for i, output_file in enumerate(output_files, 1):
                print(f"  python main.py --json {output_file} --output results_{i:04d}.json")
            
        except Exception as e:
            print(f"[ERROR] 处理失败: {e}")
            import traceback
            traceback.print_exc()
    
    elif args.command == 'merge':
        import glob
        
        # 处理通配符
        input_files = []
        for pattern in args.input_files:
            input_files.extend(glob.glob(pattern))
        
        input_files = [Path(f) for f in input_files]
        input_files.sort()  # 排序确保顺序
        
        if not input_files:
            print(f"[ERROR] 未找到输入文件")
            return
        
        output_file = Path(args.output_file)
        
        try:
            merge_results(input_files, output_file)
        except Exception as e:
            print(f"[ERROR] 处理失败: {e}")
            import traceback
            traceback.print_exc()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

