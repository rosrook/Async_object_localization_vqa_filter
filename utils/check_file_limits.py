#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查系统文件描述符限制
"""
import resource
import os


def check_file_limits():
    """检查当前文件描述符限制"""
    # 获取软限制和硬限制
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    
    print("=" * 60)
    print("系统文件描述符限制检查")
    print("=" * 60)
    print(f"当前软限制: {soft}")
    print(f"当前硬限制: {hard}")
    print(f"当前打开的文件数: {len(os.listdir('/proc/self/fd')) if os.path.exists('/proc/self/fd') else 'N/A'}")
    print()
    
    # 建议
    if soft < 1024:
        print("⚠️  警告: 软限制过低，可能导致 'Too many open files' 错误")
        print(f"   建议: 将软限制提高到至少 4096")
    elif soft < 4096:
        print("⚠️  警告: 软限制较低，高并发处理时可能出现问题")
        print(f"   建议: 将软限制提高到 8192 或更高")
    else:
        print("✓ 软限制正常")
    
    print()
    print("如何提高限制:")
    print("1. 临时提高（当前会话）:")
    print(f"   ulimit -n 8192")
    print()
    print("2. 永久提高（需要root权限）:")
    print("   编辑 /etc/security/limits.conf，添加:")
    print("   * soft nofile 8192")
    print("   * hard nofile 16384")
    print()
    print("3. 在代码中提高（需要权限）:")
    print("   import resource")
    print(f"   resource.setrlimit(resource.RLIMIT_NOFILE, (8192, hard))")
    print("=" * 60)


if __name__ == "__main__":
    check_file_limits()

