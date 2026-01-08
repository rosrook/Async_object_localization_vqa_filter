from pathlib import Path
from typing import Dict, Any
from .base_pipeline import BasePipeline


class QuestionPipeline(BasePipeline):
    """Question类型Pipeline：识别物体+概念匹配"""
    
    def __init__(self, gemini_client=None):
        super().__init__(gemini_client, "question")
    
    def filter(self, image_input) -> Dict[str, Any]:
        """
        筛选图片：判断是否符合"识别物体+概念匹配"的要求
        
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
        
        result["pipeline_type"] = "question"
        result["pipeline_name"] = self.config.get("name", "Question Pipeline")
        
        return result

