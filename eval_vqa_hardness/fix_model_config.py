#!/usr/bin/env python3
"""
修复模型目录中的模块名问题（将连字符替换为下划线）
"""
import json
import os
import shutil
from pathlib import Path

def fix_model_config(model_path, dry_run=True):
    """修复模型目录中的模块名问题"""
    model_path = Path(model_path)
    
    print("=" * 70)
    print(f"修复模型目录: {model_path}")
    print(f"模式: {'预览（不实际修改）' if dry_run else '实际修改'}")
    print("=" * 70)
    
    changes_made = False
    
    # 1. 修复 config.json 中的 auto_map
    config_path = model_path / "config.json"
    if config_path.exists():
        print("\n检查 config.json...")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        if 'auto_map' in config:
            auto_map = config['auto_map']
            if isinstance(auto_map, dict):
                fixed_auto_map = {}
                for key, value in auto_map.items():
                    if isinstance(value, str):
                        # 检查并修复包含连字符的模块名
                        if 'LLaVA-OneVision' in value or ('-' in value and 'transformers_modules' in value):
                            # 修复模块名：将连字符替换为下划线
                            # 注意：需要先处理更长的匹配（如 1.5），再处理短的（如 1）
                            fixed_value = value
                            # 先处理 1.5 版本
                            fixed_value = fixed_value.replace('LLaVA-OneVision-1.5', 'LLaVA_OneVision_1_5')
                            fixed_value = fixed_value.replace('LLaVA-OneVision-1', 'LLaVA_OneVision_1')
                            # 再处理其他版本（如 8B）
                            fixed_value = fixed_value.replace('LLaVA-OneVision-8', 'LLaVA_OneVision_8')
                            # 最后处理通用情况
                            fixed_value = fixed_value.replace('LLaVA-OneVision', 'LLaVA_OneVision')
                            fixed_auto_map[key] = fixed_value
                            if fixed_value != value:
                                print(f"  修复 auto_map[{key}]:")
                                print(f"    原值: {value}")
                                print(f"    新值: {fixed_value}")
                                changes_made = True
                        else:
                            fixed_auto_map[key] = value
                    else:
                        fixed_auto_map[key] = value
                
                if changes_made and not dry_run:
                    # 备份原文件
                    backup_path = config_path.with_suffix('.json.backup')
                    shutil.copy2(config_path, backup_path)
                    print(f"  已备份原文件到: {backup_path}")
                    
                    # 写入修复后的配置
                    config['auto_map'] = fixed_auto_map
                    with open(config_path, 'w', encoding='utf-8') as f:
                        json.dump(config, f, indent=2, ensure_ascii=False)
                    print(f"  ✓ 已修复 config.json")
    
    # 2. 修复 transformers_modules 目录名
    transformers_modules_dir = model_path / "transformers_modules"
    if transformers_modules_dir.exists():
        print("\n检查 transformers_modules 目录...")
        subdirs = [d for d in transformers_modules_dir.iterdir() if d.is_dir()]
        for subdir in subdirs:
            if '-' in subdir.name:
                new_name = subdir.name.replace('-', '_')
                new_path = subdir.parent / new_name
                print(f"  发现包含连字符的目录: {subdir.name}")
                print(f"  建议重命名为: {new_name}")
                if not dry_run:
                    if new_path.exists():
                        print(f"  ⚠️  目标目录已存在，跳过: {new_name}")
                    else:
                        subdir.rename(new_path)
                        print(f"  ✓ 已重命名: {subdir.name} -> {new_name}")
                        changes_made = True
    
    print("\n" + "=" * 70)
    if dry_run:
        print("预览完成（未实际修改）")
        if changes_made:
            print("\n发现需要修复的问题。要实际修复，请运行：")
            print(f"  python fix_model_config.py {model_path} --apply")
    else:
        if changes_made:
            print("✓ 修复完成！")
        else:
            print("未发现需要修复的问题")
    print("=" * 70)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法:")
        print("  预览（不实际修改）: python fix_model_config.py <model_path>")
        print("  实际修复: python fix_model_config.py <model_path> --apply")
        print("\n示例:")
        print("  python fix_model_config.py /path/to/LLaVA-OneVision-1.5-4B-Instruct")
        sys.exit(1)
    
    model_path = sys.argv[1]
    dry_run = '--apply' not in sys.argv
    
    if not dry_run:
        response = input(f"\n确定要修改模型目录 {model_path} 吗？(yes/no): ")
        if response.lower() != 'yes':
            print("已取消")
            sys.exit(0)
    
    fix_model_config(model_path, dry_run=dry_run)

