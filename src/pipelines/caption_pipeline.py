"""
Caption Pipeline - 场景描述/caption
"""
from pathlib import Path
from typing import Dict, Any
from .base_pipeline import BasePipeline


class CaptionPipeline(BasePipeline):
    """Caption类型Pipeline：场景描述筛选"""
    
    def __init__(self, gemini_client=None):
        super().__init__(gemini_client, "caption")
    
    def filter(self, image_input) -> Dict[str, Any]:
        """
        筛选图片：判断是否符合"场景描述/caption"的要求
        
        Args:
            image_path: 图片路径
            
        Returns:
            筛选结果字典
        """
        criteria_description = self.get_criteria_description()
        question = self.get_question()
        
        result = self.gemini_client.filter_image(
            image_input=image_input,
            criteria_description=criteria_description,
            question=question,
            temperature=0.3  # 使用较低温度以获得更稳定的筛选结果
        )
        
        result["pipeline_type"] = "caption"
        result["pipeline_name"] = self.config.get("name", "Caption Pipeline")
        
        return result

