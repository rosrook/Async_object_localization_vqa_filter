"""
问题验证模块
验证生成的问题是否符合约束条件
"""
import json
import re
from typing import Dict, Any, Optional, Tuple
from utils.gemini_client import GeminiClient


class QuestionValidator:
    """问题验证器"""
    
    def __init__(self, gemini_client: Optional[GeminiClient] = None):
        """
        初始化验证器
        
        Args:
            gemini_client: Gemini客户端实例
        """
        self.gemini_client = gemini_client or GeminiClient()
    
    def validate(
        self,
        question: str,
        image_input: Any,
        pipeline_config: Dict[str, Any],
        global_constraints: Dict[str, Any]
    ) -> tuple[bool, str]:
        """
        验证问题是否符合约束
        
        Args:
            question: 生成的问题文本
            image_input: 图片输入
            pipeline_config: Pipeline配置
            global_constraints: 全局约束
            
        Returns:
            (是否通过验证, 验证原因)
        """
        # 基本检查
        if not question or len(question.strip()) == 0:
            return False, "问题为空"
        
        # 检查全局约束
        if not self._check_global_constraints(question, global_constraints):
            return False, "违反全局约束"
        
        # 检查Pipeline特定约束
        if not self._check_pipeline_constraints(question, pipeline_config):
            return False, "违反Pipeline约束"
        
        # 使用LLM进行深度验证
        is_valid, reason = self._validate_with_llm(
            question=question,
            image_input=image_input,
            pipeline_config=pipeline_config,
            global_constraints=global_constraints
        )
        
        return is_valid, reason
    
    def _check_global_constraints(
        self,
        question: str,
        global_constraints: Dict[str, Any]
    ) -> bool:
        """检查全局约束"""
        # 检查禁止的问题类型
        forbidden_types = global_constraints.get("forbidden_question_types", [])
        question_lower = question.lower()
        
        forbidden_keywords = {
            "generic_scene_description": ["describe", "what do you see", "what's in"],
            "opinion_based": ["do you like", "do you think", "prefer", "favorite"],
            "hypothetical": ["what if", "suppose", "imagine", "if"],
            "commonsense_only": ["why", "how come", "what causes"],
            "unanswerable_from_image": ["what happened before", "what will happen"]
        }
        
        for forbidden_type in forbidden_types:
            keywords = forbidden_keywords.get(forbidden_type, [])
            if any(keyword in question_lower for keyword in keywords):
                return False
        
        return True
    
    def _check_pipeline_constraints(
        self,
        question: str,
        pipeline_config: Dict[str, Any]
    ) -> bool:
        """检查Pipeline特定约束"""
        constraints = pipeline_config.get("question_constraints", [])
        
        # 这里可以添加更详细的约束检查
        # 目前简化处理，主要依赖LLM验证
        
        return True
    
    def _validate_with_llm(
        self,
        question: str,
        image_input: Any,
        pipeline_config: Dict[str, Any],
        global_constraints: Dict[str, Any]
    ) -> tuple[bool, str]:
        """使用LLM进行深度验证"""
        validation_rules = global_constraints.get("validation_rules", [])
        question_constraints = pipeline_config.get("question_constraints", [])
        
        prompt = f"""You are a VQA question validation expert. Validate whether the given question meets all requirements.

Question: "{question}"

Pipeline Intent: {pipeline_config.get("intent", "")}

Global Validation Rules:
{chr(10).join(f"- {rule}" for rule in validation_rules)}

Pipeline-Specific Constraints:
{chr(10).join(f"- {constraint}" for constraint in question_constraints)}

Check if the question:
1. Explicitly references at least one visual entity or region in the image
2. Is answerable solely based on the image
3. Would have a different answer if the image content changes
4. Does NOT rely on external knowledge or commonsense only
5. Follows the pipeline intent and constraints

Return ONLY a JSON object in this format:
{{
    "valid": true/false,
    "reason": "detailed explanation of why the question is valid or invalid"
}}

Return only JSON, no other text."""

        try:
            response = self.gemini_client.analyze_image(
                image_input=image_input,
                prompt=prompt,
                temperature=0.3,
                context="question_validation"
            )
            
            # 解析JSON响应
            import json
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                is_valid = result.get("valid", False)
                reason = result.get("reason", "验证失败")
                return is_valid, reason
            
            return False, "无法解析验证结果"
            
        except Exception as e:
            print(f"[WARNING] 问题验证失败: {e}")
            return False, f"验证过程出错: {str(e)}"

