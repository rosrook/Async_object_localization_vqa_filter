from pathlib import Path
from typing import Dict, Any
from .base_pipeline import BasePipeline


class TextAssociationPipeline(BasePipeline):
    """
    文本关联 Pipeline
    用于：判断图片与文本的相关性
    示例：Which can be the associated text with this image posted on twitter?
    """
    
    def __init__(self, gemini_client=None):
        super().__init__(gemini_client, "text_association")
    
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
        
        result["pipeline_type"] = "text_association"
        result["pipeline_name"] = self.config.get("name", "TextAssociation Pipeline")
        
        return result

