"""
对象选择模块
根据配置选择图像中的目标对象
"""
import json
from typing import Dict, Any, Optional, List
from utils.gemini_client import GeminiClient


class ObjectSelector:
    """对象选择器"""
    
    def __init__(self, gemini_client: Optional[GeminiClient] = None):
        """
        初始化对象选择器
        
        Args:
            gemini_client: Gemini客户端实例
        """
        self.gemini_client = gemini_client or GeminiClient()
    
    def select_object(
        self,
        image_input: Any,
        pipeline_config: Dict[str, Any],
        global_policy: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        根据配置选择图像中的目标对象
        
        Args:
            image_input: 图片输入（路径、base64、bytes等）
            pipeline_config: Pipeline配置
            global_policy: 全局对象选择策略
            
        Returns:
            选中的对象信息字典，如果无法选择则返回None
        """
        # 检查是否需要对象选择
        object_grounding = pipeline_config.get("object_grounding")
        if not object_grounding:
            return None
        
        if not object_grounding.get("selection_required", False):
            return None
        
        # 获取选择策略和约束
        selection_strategy = object_grounding.get("selection_strategy", "best_fit")
        constraints = object_grounding.get("constraints", [])
        general_criteria = global_policy.get("general_criteria", [])
        
        # 使用LLM进行对象选择
        selected_object = self._select_with_llm(
            image_input=image_input,
            pipeline_intent=pipeline_config.get("intent", ""),
            selection_strategy=selection_strategy,
            constraints=constraints,
            general_criteria=general_criteria
        )
        
        return selected_object
    
    def _select_with_llm(
        self,
        image_input: Any,
        pipeline_intent: str,
        selection_strategy: str,
        constraints: List[str],
        general_criteria: List[str]
    ) -> Optional[Dict[str, Any]]:
        """
        使用LLM选择对象
        
        Args:
            image_input: 图片输入
            pipeline_intent: Pipeline意图
            selection_strategy: 选择策略
            constraints: Pipeline特定约束
            general_criteria: 通用标准
            
        Returns:
            选中的对象信息，如果无法选择返回None
        """
        # 构建prompt
        prompt = f"""You are an object selection expert. Your task is to select the most suitable target object from the image according to the given criteria.

Pipeline Intent: {pipeline_intent}
Selection Strategy: {selection_strategy}

General Criteria:
{chr(10).join(f"- {criterion}" for criterion in general_criteria)}

Pipeline-Specific Constraints:
{chr(10).join(f"- {constraint}" for constraint in constraints)}

Analyze the image and select the most suitable object. If no suitable object can be found according to the criteria, return null.

Return ONLY a JSON object in this format:
{{
    "selected": true/false,
    "object_name": "name of the selected object (e.g., 'person', 'car', 'tree')",
    "object_category": "category of the object",
    "reason": "brief explanation of why this object was selected",
    "confidence": 0.0-1.0
}}

If no suitable object can be selected, return:
{{
    "selected": false,
    "reason": "explanation of why no object can be selected",
    "confidence": 0.0
}}

Return only JSON, no other text."""

        try:
            response = self.gemini_client.analyze_image(
                image_input=image_input,
                prompt=prompt,
                temperature=0.3,
                context="object_selection"
            )
            
            # 解析JSON响应
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                
                if result.get("selected", False):
                    return {
                        "name": result.get("object_name", ""),
                        "category": result.get("object_category", ""),
                        "reason": result.get("reason", ""),
                        "confidence": result.get("confidence", 0.0)
                    }
            
            return None
            
        except Exception as e:
            print(f"[WARNING] 对象选择失败: {e}")
            return None

