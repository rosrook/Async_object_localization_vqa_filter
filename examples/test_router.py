#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试Router的路由功能
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.router import Router, PipelineType


def test_question_matching():
    """测试question匹配功能"""
    router = Router()
    
    print("=" * 60)
    print("测试Question匹配")
    print("=" * 60)
    
    # 测试QUESTION pipeline的匹配
    test_cases = [
        "Which term matches the picture?",
        "Which term matches the picture",
        "Some text Which term matches the picture? more text",
    ]
    
    for question in test_cases:
        result = router._match_pipeline_by_question(question)
        expected = [PipelineType.QUESTION]
        status = "✓" if result == expected else "✗"
        print(f"{status} Question: '{question}'")
        print(f"   匹配结果: {[pt.value for pt in result]}")
        print(f"   期望结果: {[pt.value for pt in expected]}")
        print()
    
    print("=" * 60)
    print("测试Caption匹配")
    print("=" * 60)
    
    # 测试CAPTION pipeline的匹配
    test_cases = [
        "Which one is the correct caption of this image?（共2个）",
        "Which one is the correct caption of this image?",
        "Which one is the correct caption",
    ]
    
    for question in test_cases:
        result = router._match_pipeline_by_question(question)
        expected = [PipelineType.CAPTION]
        status = "✓" if result == expected else "✗"
        print(f"{status} Question: '{question}'")
        print(f"   匹配结果: {[pt.value for pt in result]}")
        print(f"   期望结果: {[pt.value for pt in expected]}")
        print()
    
    print("=" * 60)
    print("测试route方法")
    print("=" * 60)
    
    # 测试route方法
    test_image_path = Path("/tmp/test.jpg")
    
    # 测试1: 从metadata直接获取question
    metadata1 = {"question": "Which term matches the picture?"}
    result1 = router.route(test_image_path, metadata1)
    print(f"Metadata with question: {result1}")
    assert PipelineType.QUESTION in result1, "应该匹配到QUESTION pipeline"
    
    # 测试2: 从source_b获取question
    metadata2 = {
        "source_b": {
            "question": "Which one is the correct caption of this image?（共2个）"
        }
    }
    result2 = router.route(test_image_path, metadata2)
    print(f"Metadata with source_b.question: {result2}")
    assert PipelineType.CAPTION in result2, "应该匹配到CAPTION pipeline"
    
    print("\n✅ 所有测试通过！")


if __name__ == "__main__":
    test_question_matching()



