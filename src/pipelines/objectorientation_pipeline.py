from pathlib import Path
from typing import Dict, Any
from .base_pipeline import BasePipeline


class ObjectOrientationPipeline(BasePipeline):
    """
    物体朝向 Pipeline
    用于：判断物体的朝向/方向
    示例：In the picture, which direction is this man facing?
    """
    
    def __init__(self, gemini_client=None):
        super().__init__(gemini_client, "object_orientation")
    
    def filter(self, image_input) -> Dict[str, Any]:
        """
        Args:
            image_path: 图片路径
            
        Returns:
            筛选结果字典
        """
        # 添加调试信息
        print(f"[DEBUG] ObjectOrientationPipeline.filter called")
        print(f"  - pipeline_type: {self.pipeline_type}")
        print(f"  - config keys: {list(self.config.keys())}")
        print(f"  - config has question: {'question' in self.config}")
        print(f"  - config has criteria: {'criteria' in self.config}")
        
        criteria_description = self.get_criteria_description()
        question = self.get_question()
        
        print(f"[DEBUG] ObjectOrientationPipeline.filter: extracted values")
        print(f"  - question length: {len(question) if question else 0}")
        print(f"  - criteria_description length: {len(criteria_description) if criteria_description else 0}")
        print(f"  - question preview: {question[:50] if question else 'EMPTY'}...")
        print(f"  - criteria_description preview: {criteria_description[:100] if criteria_description else 'EMPTY'}...")
        
        result = self.gemini_client.filter_image(
            image_input=image_input,
            criteria_description=criteria_description,
            question=question,
            temperature=0.3  # 使用较低温度以获得更稳定的筛选结果
        )
        
        result["pipeline_type"] = "object_orientation"
        result["pipeline_name"] = self.config.get("name", "ObjectOrientation Pipeline")
        
        return result

