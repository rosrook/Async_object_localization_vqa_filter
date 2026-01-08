from pathlib import Path
from typing import Dict, Any
from .base_pipeline import BasePipeline


class ObjectCountingPipeline(BasePipeline):
    """
    物体计数 Pipeline
    用于：统计图片中特定物体的数量
    示例：How many motorcycles are in the picture?
    """
    
    def __init__(self, gemini_client=None):
        super().__init__(gemini_client, "object_counting")
    
    def filter(self, image_input: Path) -> Dict[str, Any]:
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
        
        result["pipeline_type"] = "object_counting"
        result["pipeline_name"] = self.config.get("name", "ObjectCounting Pipeline")
        
        return result

