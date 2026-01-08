#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将JSON结果文件转换为Markdown格式，自动检测并可视化base64图片
"""
import json
import base64
import re
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import hashlib


def is_base64_image(data: Any) -> Tuple[bool, Optional[str]]:
    """
    检测数据是否为base64编码的图片
    
    Args:
        data: 要检测的数据
        
    Returns:
        (是否为base64图片, 图片格式)
    """
    if not isinstance(data, str):
        return False, None
    
    # 检查是否是data URI格式
    if data.startswith('data:image/'):
        # 提取格式: data:image/jpeg;base64,...
        match = re.match(r'data:image/(\w+);base64,', data)
        if match:
            return True, match.group(1)
    
    # 检查是否是纯base64字符串
    # base64字符串通常很长（至少几百字符）
    if len(data) < 100:
        return False, None
    
    # 移除空白字符
    clean_data = re.sub(r'\s', '', data)
    
    # 尝试解码
    try:
        decoded = base64.b64decode(clean_data, validate=True)
        
        # 检查是否是图片格式（通过文件头）
        if decoded.startswith(b'\xff\xd8\xff'):
            return True, 'jpeg'
        elif decoded.startswith(b'\x89PNG\r\n\x1a\n'):
            return True, 'png'
        elif decoded.startswith(b'GIF87a') or decoded.startswith(b'GIF89a'):
            return True, 'gif'
        elif decoded.startswith(b'RIFF') and b'WEBP' in decoded[:12]:
            return True, 'webp'
        elif decoded.startswith(b'BM'):
            return True, 'bmp'
        
        # 如果解码成功但不确定格式，假设是jpeg
        if len(decoded) > 100:
            return True, 'jpeg'
            
    except Exception:
        pass
    
    return False, None


def find_base64_images(obj: Any, path: str = "") -> List[Tuple[str, str, str]]:
    """
    递归查找对象中的所有base64图片
    
    Args:
        obj: 要搜索的对象
        path: 当前路径（用于标识位置）
        
    Returns:
        [(路径, base64数据, 格式), ...]
    """
    images = []
    
    if isinstance(obj, dict):
        for key, value in obj.items():
            current_path = f"{path}.{key}" if path else key
            is_img, img_format = is_base64_image(value)
            if is_img:
                images.append((current_path, value, img_format))
            else:
                # 递归搜索
                images.extend(find_base64_images(value, current_path))
    
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            current_path = f"{path}[{i}]"
            images.extend(find_base64_images(item, current_path))
    
    return images


def format_value(value: Any, indent: int = 0) -> str:
    """
    格式化值为Markdown友好的格式
    
    Args:
        value: 要格式化的值
        indent: 缩进级别
        
    Returns:
        格式化后的字符串
    """
    indent_str = "  " * indent
    
    if value is None:
        return "`None`"
    elif isinstance(value, bool):
        return f"**{value}**"
    elif isinstance(value, (int, float)):
        return f"`{value}`"
    elif isinstance(value, str):
        # 检查是否是base64图片
        is_img, img_format = is_base64_image(value)
        if is_img:
            # 如果是data URI，直接使用
            if value.startswith('data:image/'):
                return f'<img src="{value}" alt="Image" style="max-width: 500px;" />'
            else:
                # 转换为data URI
                data_uri = f"data:image/{img_format};base64,{value}"
                return f'<img src="{data_uri}" alt="Image ({img_format})" style="max-width: 500px;" />'
        
        # 长字符串截断
        if len(value) > 200:
            return f"`{value[:200]}...` (长度: {len(value)})"
        return f"`{value}`"
    elif isinstance(value, dict):
        lines = ["{"]
        for k, v in value.items():
            lines.append(f"{indent_str}  **{k}**: {format_value(v, indent + 1)}")
        lines.append(f"{indent_str}}}")
        return "\n".join(lines)
    elif isinstance(value, list):
        if len(value) == 0:
            return "`[]`"
        lines = ["["]
        for i, item in enumerate(value):
            lines.append(f"{indent_str}  {i}: {format_value(item, indent + 1)}")
        lines.append(f"{indent_str}]")
        return "\n".join(lines)
    else:
        return f"`{str(value)}`"


def json_to_markdown(
    input_file: Path,
    output_file: Path,
    max_records: Optional[int] = None,
    include_images: bool = True
) -> None:
    """
    将JSON文件转换为Markdown格式
    
    Args:
        input_file: 输入JSON文件路径
        output_file: 输出Markdown文件路径
        max_records: 最大处理记录数（None表示全部）
        include_images: 是否包含图片可视化
    """
    print(f"[INFO] 读取输入文件: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        data = [data]
    
    total = len(data)
    if max_records:
        data = data[:max_records]
        print(f"[INFO] 处理前 {max_records} 条记录（共 {total} 条）")
    else:
        print(f"[INFO] 处理 {total} 条记录")
    
    # 生成Markdown内容
    md_lines = []
    
    # 标题
    md_lines.append("# JSON结果可视化")
    md_lines.append("")
    md_lines.append(f"**总记录数**: {total}")
    md_lines.append(f"**显示记录数**: {len(data)}")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")
    
    # 处理每条记录
    for idx, record in enumerate(data, 1):
        md_lines.append(f"## 记录 {idx}")
        md_lines.append("")
        
        # 基本信息
        if "id" in record:
            md_lines.append(f"**ID**: `{record['id']}`")
        if "sample_index" in record:
            md_lines.append(f"**样本索引**: `{record['sample_index']}`")
        if "timestamp" in record:
            md_lines.append(f"**时间戳**: `{record['timestamp']}`")
        md_lines.append("")
        
        # 查找并显示base64图片
        if include_images:
            images = find_base64_images(record)
            if images:
                md_lines.append("### 📷 图片")
                md_lines.append("")
                for img_path, img_data, img_format in images:
                    md_lines.append(f"**位置**: `{img_path}`")
                    md_lines.append(f"**格式**: {img_format}")
                    md_lines.append("")
                    
                    # 转换为data URI并显示
                    if img_data.startswith('data:image/'):
                        data_uri = img_data
                    else:
                        data_uri = f"data:image/{img_format};base64,{img_data}"
                    
                    md_lines.append(f'<img src="{data_uri}" alt="Image at {img_path}" style="max-width: 600px; border: 1px solid #ddd; border-radius: 4px; padding: 5px;" />')
                    md_lines.append("")
        
        # 筛选结果
        if "pipeline_type" in record:
            md_lines.append("### Pipeline信息")
            md_lines.append("")
            md_lines.append(f"- **类型**: `{record.get('pipeline_type', 'N/A')}`")
            if "pipeline_name" in record:
                md_lines.append(f"- **名称**: {record['pipeline_name']}")
            md_lines.append("")
        
        # 筛选结果
        if "passed" in record:
            md_lines.append("### 筛选结果")
            md_lines.append("")
            status = "✅ **通过**" if record.get("passed") else "❌ **未通过**"
            md_lines.append(f"- **状态**: {status}")
            
            if "total_score" in record:
                md_lines.append(f"- **总分**: `{record['total_score']:.3f}`")
            if "basic_score" in record:
                md_lines.append(f"- **基础分**: `{record['basic_score']:.3f}`")
            if "bonus_score" in record:
                md_lines.append(f"- **奖励分**: `{record['bonus_score']:.3f}`")
            if "confidence" in record:
                md_lines.append(f"- **置信度**: `{record['confidence']:.3f}`")
            md_lines.append("")
        
        # 原因说明
        if "reason" in record:
            md_lines.append("### 原因说明")
            md_lines.append("")
            md_lines.append(record['reason'])
            md_lines.append("")
        
        # 错误信息
        if "error" in record:
            md_lines.append("### ⚠️ 错误信息")
            md_lines.append("")
            md_lines.append(f"```")
            md_lines.append(record['error'])
            md_lines.append(f"```")
            md_lines.append("")
        
        # 完整数据（折叠）
        md_lines.append("<details>")
        md_lines.append("<summary>完整数据（点击展开）</summary>")
        md_lines.append("")
        md_lines.append("```json")
        md_lines.append(json.dumps(record, ensure_ascii=False, indent=2))
        md_lines.append("```")
        md_lines.append("")
        md_lines.append("</details>")
        md_lines.append("")
        
        md_lines.append("---")
        md_lines.append("")
    
    # 保存Markdown文件
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    
    print(f"[INFO] Markdown文件已保存到: {output_file}")
    
    # 统计信息
    total_images = sum(len(find_base64_images(record)) for record in data)
    print(f"[INFO] 共找到 {total_images} 个base64图片")


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description='将JSON结果文件转换为Markdown格式，自动检测并可视化base64图片',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 转换整个文件
  python utils/json_to_markdown.py input.json output.md
  
  # 只转换前10条记录
  python utils/json_to_markdown.py input.json output.md -n 10
  
  # 不包含图片可视化（只显示文本）
  python utils/json_to_markdown.py input.json output.md --no-images
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
        help='输出Markdown文件路径'
    )
    parser.add_argument(
        '-n', '--max-records',
        type=int,
        default=None,
        help='最大处理记录数（默认: 全部）'
    )
    parser.add_argument(
        '--no-images',
        action='store_true',
        help='不包含图片可视化'
    )
    
    args = parser.parse_args()
    
    input_file = Path(args.input_file)
    output_file = Path(args.output_file)
    
    if not input_file.exists():
        print(f"[ERROR] 输入文件不存在: {input_file}")
        return
    
    try:
        json_to_markdown(
            input_file=input_file,
            output_file=output_file,
            max_records=args.max_records,
            include_images=not args.no_images
        )
    except Exception as e:
        print(f"[ERROR] 处理失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()



# # 1. 从结果中采样
# python utils/sample_results.py output.json sample_10.json -n 10

# # 2. 转换为Markdown并可视化
# python utils/json_to_markdown.py sample_10.json sample_10.md

# # 3. 在Markdown查看器中打开（如VS Code、Typora等）
# # 图片会自动显示