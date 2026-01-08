from pathlib import Path
from typing import Dict, Any
from .base_pipeline import BasePipeline


class ObjectPositionPipeline(BasePipeline):
    """
    物体位置 Pipeline
    用于：判断物体在图片中的位置
    示例：Where is the woman located in the picture?
    """
    
    def __init__(self, gemini_client=None):
        super().__init__(gemini_client, "object_position")
    
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
        
        result["pipeline_type"] = "object_position"
        result["pipeline_name"] = self.config.get("name", "ObjectPosition Pipeline")
        
        return result

