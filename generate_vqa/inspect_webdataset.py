#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 WebDataset 中抽取和查看样本数据，验证转换是否正确
"""
import argparse
import json
import sys
from pathlib import Path
import webdataset as wds
from PIL import Image
from io import BytesIO
import random


def inspect_webdataset(tar_path, num_samples=5, save_images=False, output_dir=None):
    """
    检查 WebDataset 文件
    
    Args:
        tar_path: WebDataset tar 文件路径或包含 tar 文件的目录
        num_samples: 要检查的样本数量
        save_images: 是否保存图片到本地
        output_dir: 保存图片的目录（如果 save_images=True）
    """
    # 查找 tar 文件
    tar_path = Path(tar_path)
    if tar_path.is_dir():
        tar_files = list(tar_path.glob("*.tar"))
        if not tar_files:
            print(f"[ERROR] 目录中没有找到 .tar 文件: {tar_path}")
            return
        tar_file = tar_files[0]
        print(f"[INFO] 找到 tar 文件: {tar_file}")
    elif tar_path.is_file() and tar_path.suffix == '.tar':
        tar_file = tar_path
    else:
        print(f"[ERROR] 无效的路径: {tar_path}")
        return
    
    # 创建输出目录（如果需要保存图片）
    if save_images:
        if output_dir is None:
            output_dir = tar_path.parent / "inspected_samples"
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] 图片将保存到: {output_dir}")
    
    # 打开 WebDataset
    print(f"\n{'='*80}")
    print(f"开始检查 WebDataset: {tar_file}")
    print(f"{'='*80}\n")
    
    try:
        dataset = wds.WebDataset(str(tar_file))
        
        # 收集所有样本（用于随机选择）
        all_samples = []
        print("[INFO] 正在读取所有样本索引...")
        for sample in dataset:
            all_samples.append(sample.get("__key__", "unknown"))
        
        total_samples = len(all_samples)
        print(f"[INFO] 总样本数: {total_samples}\n")
        
        if total_samples == 0:
            print("[ERROR] WebDataset 中没有找到样本")
            return
        
        # 随机选择要检查的样本
        if num_samples > total_samples:
            num_samples = total_samples
            print(f"[WARNING] 请求的样本数超过总数，将检查所有 {total_samples} 个样本")
        
        selected_keys = random.sample(all_samples, num_samples) if num_samples < total_samples else all_samples
        selected_keys_set = set(selected_keys)
        
        # 重新打开数据集并查找选中的样本
        dataset = wds.WebDataset(str(tar_file))
        found_samples = []
        
        print(f"[INFO] 正在提取 {len(selected_keys)} 个样本...\n")
        
        for sample in dataset:
            key = sample.get("__key__", "unknown")
            if key in selected_keys_set:
                found_samples.append((key, sample))
                if len(found_samples) >= len(selected_keys):
                    break
        
        # 显示每个样本的详细信息
        for idx, (key, sample) in enumerate(found_samples, 1):
            print(f"{'='*80}")
            print(f"样本 {idx}/{len(found_samples)}: {key}")
            print(f"{'='*80}")
            
            # 解析 JSON 元数据
            json_data = None
            if "json" in sample:
                try:
                    if isinstance(sample["json"], bytes):
                        json_data = json.loads(sample["json"].decode('utf-8'))
                    else:
                        json_data = sample["json"]
                except Exception as e:
                    print(f"[ERROR] 解析 JSON 失败: {e}")
                    continue
            
            if json_data:
                print(f"\n[元数据]")
                print(f"  媒体类型: {json_data.get('media', 'unknown')}")
                print(f"  图片名称: {json_data.get('name', [])}")
                
                # 验证元数据格式
                media_type = json_data.get('media', '')
                if media_type not in ['image', 'video', 'mix']:
                    print(f"  ⚠ WARNING: 媒体类型 '{media_type}' 不在预期的 ['image', 'video', 'mix'] 中")
                
                # 显示对话内容
                texts = json_data.get('texts', [])
                print(f"\n[对话内容] (共 {len(texts)} 轮)")
                
                # 验证对话格式
                if len(texts) < 2:
                    print(f"  ⚠ WARNING: 对话轮次少于 2 轮，预期至少包含 user 和 assistant")
                elif texts[0].get('role') != 'user':
                    print(f"  ⚠ WARNING: 第一轮对话的角色不是 'user'")
                elif texts[1].get('role') != 'assistant':
                    print(f"  ⚠ WARNING: 第二轮对话的角色不是 'assistant'")
                
                has_image_tag = False
                for turn_idx, turn in enumerate(texts, 1):
                    role = turn.get('role', 'unknown')
                    content = turn.get('content', '')
                    
                    # 验证角色
                    if role not in ['user', 'assistant', 'system']:
                        print(f"  ⚠ WARNING: 轮次 {turn_idx} 的角色 '{role}' 不在预期值中")
                    
                    # 验证内容不为空
                    if not content or len(content.strip()) == 0:
                        print(f"  ⚠ WARNING: 轮次 {turn_idx} 的内容为空")
                    
                    # 截断过长的内容
                    content_preview = content[:300] + "..." if len(content) > 300 else content
                    
                    print(f"\n  轮次 {turn_idx} [{role}]:")
                    print(f"    {content_preview}")
                    
                    # 检查是否有图片标记
                    if role == 'user' and '<image>' in content:
                        print(f"    ✓ 包含图片标记 <image>")
                        has_image_tag = True
                
                if not has_image_tag:
                    print(f"  ⚠ WARNING: user 消息中未找到 <image> 标记")
            
            # 查找并显示图片信息（支持新的格式：jpg/png 或旧的格式：0_image）
            image_keys = [k for k in sample.keys() if '_image' in k or k == 'jpg' or k == 'png' or k == 'jpeg']
            if image_keys:
                print(f"\n[图片信息]")
                
                # 验证 JSON 中的 name 字段是否与实际的图片键名匹配
                if json_data:
                    expected_names = json_data.get('name', [])
                    if expected_names:
                        print(f"  期望的图片字段名: {expected_names}")
                        for exp_name in expected_names:
                            if exp_name not in image_keys:
                                print(f"    ⚠ WARNING: JSON 中指定的字段名 '{exp_name}' 在样本中不存在")
                
                for img_key in image_keys:
                    img_data = sample[img_key]
                    if isinstance(img_data, bytes):
                        img_size = len(img_data)
                        print(f"  {img_key}: {img_size:,} bytes ({img_size/1024:.2f} KB)")
                        
                        # 验证键名与文件扩展名是否匹配
                        if img_key in ['jpg', 'jpeg']:
                            # 验证是否为 JPEG
                            if not img_data.startswith(b'\xff\xd8\xff'):
                                print(f"    ⚠ WARNING: 字段名是 '{img_key}' 但文件头不是 JPEG 格式")
                        elif img_key == 'png':
                            # 验证是否为 PNG
                            if not img_data.startswith(b'\x89PNG'):
                                print(f"    ⚠ WARNING: 字段名是 'png' 但文件头不是 PNG 格式")
                        
                        # 尝试打开图片获取尺寸信息
                        try:
                            img = Image.open(BytesIO(img_data))
                            print(f"    ✓ 尺寸: {img.size[0]}x{img.size[1]}, 格式: {img.format}, 模式: {img.mode}")
                            
                            # 验证格式一致性
                            format_map = {'JPEG': 'jpg', 'PNG': 'png', 'JPG': 'jpg'}
                            expected_ext = format_map.get(img.format, 'jpg')
                            if img_key not in [expected_ext, f"0_image"]:  # 允许旧的 0_image 格式
                                print(f"    ⚠ WARNING: 实际图片格式 {img.format} 与字段名 '{img_key}' 不一致")
                        except Exception as e:
                            print(f"    ✗ 无法解析图片: {e}")
                        
                        # 保存图片（如果需要）
                        if save_images:
                            try:
                                img = Image.open(BytesIO(img_data))
                                img_ext = img.format.lower() if img.format else img_key
                                if img_ext == 'jpeg':
                                    img_ext = 'jpg'
                            except:
                                img_ext = img_key if img_key in ['jpg', 'png'] else 'jpg'
                            
                            img_path = output_dir / f"{key}.{img_ext}"
                            with open(img_path, 'wb') as f:
                                f.write(img_data)
                            print(f"    ✓ 已保存: {img_path}")
            else:
                print(f"\n[WARNING] 未找到图片数据")
                if json_data and json_data.get('name'):
                    print(f"  但 JSON 中指定了图片字段名: {json_data.get('name')}")
            
            # 显示其他字段
            other_keys = [k for k in sample.keys() 
                         if k not in ['json'] + image_keys + ['__key__', '__url__']]
            if other_keys:
                print(f"\n[其他字段]")
                for k in other_keys:
                    val = sample[k]
                    if isinstance(val, bytes):
                        print(f"  {k}: {len(val):,} bytes")
                    else:
                        val_str = str(val)[:100]
                        print(f"  {k}: {val_str}")
            
            print()
        
        # 统计信息
        print(f"\n{'='*80}")
        print(f"检查完成 - 统计信息")
        print(f"{'='*80}")
        print(f"总样本数: {total_samples}")
        print(f"已检查: {len(found_samples)}")
        
        # 统计图片格式
        if found_samples:
            format_count = {}
            for key, sample in found_samples:
                image_keys = [k for k in sample.keys() if '_image' in k or k == 'jpg' or k == 'png' or k == 'jpeg']
                for img_key in image_keys:
                    format_count[img_key] = format_count.get(img_key, 0) + 1
            if format_count:
                print(f"\n图片格式统计:")
                for fmt, count in sorted(format_count.items()):
                    print(f"  {fmt}: {count} 个样本")
        
        if save_images:
            print(f"\n图片保存目录: {output_dir}")
        print()
        
    except Exception as e:
        print(f"[ERROR] 读取 WebDataset 失败: {e}")
        import traceback
        traceback.print_exc()


def list_webdataset_files(data_dir):
    """
    列出目录中的所有 WebDataset 文件
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        print(f"[ERROR] 目录不存在: {data_dir}")
        return
    
    tar_files = list(data_dir.glob("*.tar"))
    if not tar_files:
        print(f"[INFO] 目录中没有找到 .tar 文件: {data_dir}")
        return
    
    print(f"\n找到 {len(tar_files)} 个 WebDataset 文件:\n")
    for tar_file in sorted(tar_files):
        file_size = tar_file.stat().st_size / (1024 * 1024)  # MB
        print(f"  {tar_file.name} ({file_size:.2f} MB)")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='检查 WebDataset 文件，验证转换是否正确',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 检查指定 tar 文件中的 5 个随机样本
  python generate_vqa/inspect_webdataset.py /path/to/subtaskdata-00000.tar -n 5
  
  # 检查目录中的第一个 tar 文件
  python generate_vqa/inspect_webdataset.py /path/to/output_dir -n 10
  
  # 保存图片到本地
  python generate_vqa/inspect_webdataset.py /path/to/subtaskdata-00000.tar -n 3 --save-images
  
  # 列出目录中的所有 tar 文件
  python generate_vqa/inspect_webdataset.py /path/to/output_dir --list-only
        """
    )
    
    parser.add_argument(
        'input_path',
        type=str,
        help='WebDataset tar 文件路径或包含 tar 文件的目录'
    )
    
    parser.add_argument(
        '-n', '--num-samples',
        type=int,
        default=5,
        help='要检查的样本数量（默认: 5）'
    )
    
    parser.add_argument(
        '--save-images',
        action='store_true',
        help='是否保存图片到本地'
    )
    
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default=None,
        help='保存图片的目录（默认: input_path/inspected_samples）'
    )
    
    parser.add_argument(
        '--list-only',
        action='store_true',
        help='仅列出目录中的 tar 文件，不检查内容'
    )
    
    args = parser.parse_args()
    
    if args.list_only:
        list_webdataset_files(args.input_path)
    else:
        inspect_webdataset(
            tar_path=args.input_path,
            num_samples=args.num_samples,
            save_images=args.save_images,
            output_dir=args.output_dir
        )


if __name__ == '__main__':
    main()


