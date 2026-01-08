from pathlib import Path
from typing import Dict, Any
from .base_pipeline import BasePipeline


class ObjectAbsencePipeline(BasePipeline):
    """
    物体缺失 Pipeline
    用于：识别图片中没有某类物体的区域
    示例：Which corner doesn't have any fruits?
    """
    
    def __init__(self, gemini_client=None):
        super().__init__(gemini_client, "object_absence")

    def filter(self, image_input) -> Dict[str, Any]:
        """
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
        
        result["pipeline_type"] = "object_absence"
        result["pipeline_name"] = self.config.get("name", "ObjectAbsence Pipeline")
        
        return result