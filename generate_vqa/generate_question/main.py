#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VQA问题生成系统主程序
"""
import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from generate_vqa.generate_question.vqa_generator import VQAGenerator


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='VQA问题生成系统 - 基于配置文件的声明式问题生成',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 处理单个文件，使用所有pipeline
  python generate_vqa/generate_question/main.py input.json output.json
  
  # 只使用特定的pipeline
  python generate_vqa/generate_question/main.py input.json output.json --pipelines question object_counting
  
  # 限制处理样本数（用于测试）
  python generate_vqa/generate_question/main.py input.json output.json -n 100
        """
    )
    
    parser.add_argument(
        'input_file',
        type=str,
        help='输入JSON文件路径（batch_process.sh的输出，包含source_a）'
    )
    parser.add_argument(
        'output_file',
        type=str,
        help='输出JSON文件路径（生成的VQA问题）'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='配置文件路径（默认: generate_vqa/question_config.json）'
    )
    parser.add_argument(
        '--pipelines',
        type=str,
        nargs='+',
        default=None,
        help='要使用的pipeline列表（默认: 使用所有pipeline）'
    )
    parser.add_argument(
        '-n', '--max-samples',
        type=int,
        default=None,
        help='最大处理样本数（默认: 全部）'
    )
    
    args = parser.parse_args()
    
    input_file = Path(args.input_file)
    output_file = Path(args.output_file)
    
    # 如果没有指定配置文件，使用默认路径
    if args.config:
        config_path = Path(args.config)
    else:
        # 默认配置文件路径（相对于项目根目录）
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "generate_vqa" / "question_config.json"
    
    if not input_file.exists():
        print(f"[ERROR] 输入文件不存在: {input_file}")
        return
    
    if not config_path.exists():
        print(f"[ERROR] 配置文件不存在: {config_path}")
        return
    
    try:
        # 初始化生成器
        generator = VQAGenerator(config_path=config_path)
        
        # 处理数据文件
        generator.process_data_file(
            input_file=input_file,
            output_file=output_file,
            pipeline_names=args.pipelines,
            max_samples=args.max_samples
        )
        
    except Exception as e:
        print(f"[ERROR] 处理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

