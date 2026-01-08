"""
基础Pipeline抽象类
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from utils.gemini_client import GeminiClient

import config


class BasePipeline(ABC):
    """Pipeline基类，所有筛选pipeline都应继承此类"""
    
    def __init__(self, gemini_client: Optional["GeminiClient"] = None, pipeline_type: str=None):
        """
        初始化Pipeline
        
        Args:
            pipeline_type: Pipeline类型（如'question', 'caption'）
            gemini_client: Gemini客户端实例，如果为None则创建新实例
        """
        self.pipeline_type = pipeline_type
        self.config = config.PIPELINE_CONFIG.get(pipeline_type, {})
        if gemini_client is None:
            from utils.gemini_client import GeminiClient
            self.gemini_client = GeminiClient()
        else:
            self.gemini_client = gemini_client
    
    @abstractmethod
    def filter(self, image_path: Path) -> Dict[str, Any]:
        """
        筛选图片
        
        Args:
            image_path: 图片路径
            
        Returns:
            筛选结果字典，包含：
            - passed: bool - 是否通过筛选
            - reason: str - 筛选原因
            - confidence: float - 置信度
            - pipeline_type: str - Pipeline类型
        """
        pass
    
    def get_criteria_description(self) -> str:
        """
        获取筛选标准描述
        
        Returns:
            格式化的筛选标准描述文本
        """
        # 添加调试信息
        if not self.config:
            print(f"[WARNING] BasePipeline.get_criteria_description: config is empty!")
            print(f"  - pipeline_type: {self.pipeline_type}")
            print(f"  - config type: {type(self.config)}")
            print(f"  - config value: {self.config}")
            return ""
        
        criteria = self.config.get("criteria", [])
        description = self.config.get("description", "")
        
        # 检查criteria是否为空
        if not criteria:
            print(f"[WARNING] BasePipeline.get_criteria_description: criteria is empty!")
            print(f"  - pipeline_type: {self.pipeline_type}")
            print(f"  - config keys: {list(self.config.keys())}")
            print(f"  - criteria value: {criteria}")
        
        criteria_text = "\n".join([f"- {criterion}" for criterion in criteria]) if criteria else "(No criteria specified)"
        
        result = f"""{description}

筛选标准：
{criteria_text}"""
        
        # 如果结果为空或只有空白，输出警告
        if not result.strip() or result.strip() == "筛选标准：\n(No criteria specified)":
            print(f"[WARNING] BasePipeline.get_criteria_description: result is empty or invalid!")
            print(f"  - pipeline_type: {self.pipeline_type}")
            print(f"  - description: {repr(description)}")
            print(f"  - criteria: {criteria}")
            print(f"  - result: {repr(result)}")
        
        return result
    
    def get_question(self) -> str:
        """获取问题描述"""
        question = self.config.get("question", "")
        
        # 检查question是否为空
        if not question or question.strip() == "":
            print(f"[WARNING] BasePipeline.get_question: question is empty!")
            print(f"  - pipeline_type: {self.pipeline_type}")
            print(f"  - config keys: {list(self.config.keys())}")
            print(f"  - question value: {repr(question)}")
        
        return question

