# """
# 对象选择模块
# 根据配置选择图像中的目标对象
# """
# import json
# import re
# from typing import Dict, Any, Optional, List
# from utils.gemini_client import GeminiClient
# from utils.async_client import AsyncGeminiClient


# class ObjectSelector:
#     """对象选择器"""
    
#     def __init__(self, gemini_client: Optional[GeminiClient] = None):
#         """
#         初始化对象选择器
        
#         Args:
#             gemini_client: Gemini客户端实例
#         """
#         self.gemini_client = gemini_client or GeminiClient()
    
#     def select_object(
#         self,
#         image_input: Any,
#         pipeline_config: Dict[str, Any],
#         global_policy: Dict[str, Any]
#     ) -> Optional[Dict[str, Any]]:
#         """
#         根据配置选择图像中的目标对象
        
#         Args:
#             image_input: 图片输入（路径、base64、bytes等）
#             pipeline_config: Pipeline配置
#             global_policy: 全局对象选择策略
            
#         Returns:
#             选中的对象信息字典，如果无法选择则返回None
#         """
#         # 检查是否需要对象选择
#         object_grounding = pipeline_config.get("object_grounding")
#         if not object_grounding:
#             return None
        
#         if not object_grounding.get("selection_required", False):
#             return None
        
#         # 获取选择策略和约束
#         selection_strategy = object_grounding.get("selection_strategy", "best_fit")
#         constraints = object_grounding.get("constraints", [])
#         general_criteria = global_policy.get("general_criteria", [])
        
#         # 使用LLM进行对象选择
#         selected_object = self._select_with_llm(
#             image_input=image_input,
#             pipeline_intent=pipeline_config.get("intent", ""),
#             selection_strategy=selection_strategy,
#             constraints=constraints,
#             general_criteria=general_criteria
#         )
        
#         return selected_object
    
#     def _select_with_llm(
#         self,
#         image_input: Any,
#         pipeline_intent: str,
#         selection_strategy: str,
#         constraints: List[str],
#         general_criteria: List[str]
#     ) -> Optional[Dict[str, Any]]:
#         """
#         使用LLM选择对象
        
#         Args:
#             image_input: 图片输入
#             pipeline_intent: Pipeline意图
#             selection_strategy: 选择策略
#             constraints: Pipeline特定约束
#             general_criteria: 通用标准
            
#         Returns:
#             选中的对象信息，如果无法选择返回None
#         """
#         # 构建prompt
#         prompt = f"""You are an object selection expert. Your task is to select the most suitable target object from the image according to the given criteria.

# Pipeline Intent: {pipeline_intent}
# Selection Strategy: {selection_strategy}

# General Criteria:
# {chr(10).join(f"- {criterion}" for criterion in general_criteria)}

# Pipeline-Specific Constraints:
# {chr(10).join(f"- {constraint}" for constraint in constraints)}

# Analyze the image and select the most suitable object. If no suitable object can be found according to the criteria, return null.

# Return ONLY a JSON object in this format:
# {{
#     "selected": true/false,
#     "object_name": "name of the selected object (e.g., 'person', 'car', 'tree')",
#     "object_category": "category of the object",
#     "reason": "brief explanation of why this object was selected",
#     "confidence": 0.0-1.0
# }}

# If no suitable object can be selected, return:
# {{
#     "selected": false,
#     "reason": "explanation of why no object can be selected",
#     "confidence": 0.0
# }}

# Return only JSON, no other text."""

#         try:
#             response = self.gemini_client.analyze_image(
#                 image_input=image_input,
#                 prompt=prompt,
#                 temperature=0.3,
#                 context="object_selection"
#             )
            
#             # 解析JSON响应
#             import re
#             json_match = re.search(r'\{.*\}', response, re.DOTALL)
#             if json_match:
#                 result = json.loads(json_match.group())
                
#                 if result.get("selected", False):
#                     return {
#                         "name": result.get("object_name", ""),
#                         "category": result.get("object_category", ""),
#                         "reason": result.get("reason", ""),
#                         "confidence": result.get("confidence", 0.0)
#                     }
            
#             return None
            
#         except Exception as e:
#             print(f"[WARNING] 对象选择失败: {e}")
#             return None
    
#     async def select_object_async(
#         self,
#         image_base64: str,
#         pipeline_config: Dict[str, Any],
#         global_policy: Dict[str, Any],
#         async_client: Optional[AsyncGeminiClient] = None,
#         model: Optional[str] = None
#     ) -> Optional[Dict[str, Any]]:
#         """
#         异步根据配置选择图像中的目标对象
        
#         Args:
#             image_base64: 图片的base64编码
#             pipeline_config: Pipeline配置
#             global_policy: 全局对象选择策略
#             async_client: 异步客户端实例（可选）
#             model: 模型名称（可选）
            
#         Returns:
#             选中的对象信息字典，如果无法选择则返回None
#         """
#         # 检查是否需要对象选择
#         object_grounding = pipeline_config.get("object_grounding")
#         if not object_grounding:
#             return None
        
#         if not object_grounding.get("selection_required", False):
#             return None
        
#         # 获取选择策略和约束
#         selection_strategy = object_grounding.get("selection_strategy", "best_fit")
#         constraints = object_grounding.get("constraints", [])
#         general_criteria = global_policy.get("general_criteria", [])
        
#         # 使用LLM进行对象选择
#         selected_object = await self._select_with_llm_async(
#             image_base64=image_base64,
#             pipeline_intent=pipeline_config.get("intent", ""),
#             selection_strategy=selection_strategy,
#             constraints=constraints,
#             general_criteria=general_criteria,
#             async_client=async_client,
#             model=model
#         )
        
#         return selected_object
    
#     async def _select_with_llm_async(
#         self,
#         image_base64: str,
#         pipeline_intent: str,
#         selection_strategy: str,
#         constraints: List[str],
#         general_criteria: List[str],
#         async_client: Optional[AsyncGeminiClient] = None,
#         model: Optional[str] = None
#     ) -> Optional[Dict[str, Any]]:
#         """
#         异步使用LLM选择对象
        
#         Args:
#             image_base64: 图片的base64编码
#             pipeline_intent: Pipeline意图
#             selection_strategy: 选择策略
#             constraints: Pipeline特定约束
#             general_criteria: 通用标准
#             async_client: 异步客户端实例（可选）
#             model: 模型名称（可选）
            
#         Returns:
#             选中的对象信息，如果无法选择返回None
#         """
#         # 构建prompt
#         prompt = f"""You are an object selection expert. Your task is to select the most suitable target object from the image according to the given criteria.

# Pipeline Intent: {pipeline_intent}
# Selection Strategy: {selection_strategy}

# General Criteria:
# {chr(10).join(f"- {criterion}" for criterion in general_criteria)}

# Pipeline-Specific Constraints:
# {chr(10).join(f"- {constraint}" for constraint in constraints)}

# Analyze the image and select the most suitable object. If no suitable object can be found according to the criteria, return null.

# You MUST return a valid JSON object with the following structure:
# {{
#     "selected": true/false,
#     "object_name": "name of the selected object (e.g., 'person', 'car', 'tree')",
#     "object_category": "category of the object",
#     "reason": "brief explanation of why this object was selected",
#     "confidence": 0.0-1.0
# }}

# If no suitable object can be selected, return:
# {{
#     "selected": false,
#     "reason": "explanation of why no object can be selected",
#     "confidence": 0.0
# }}

# Return ONLY valid JSON, no other text."""

#         try:
#             # 构建图像内容（OpenAI兼容格式）
#             image_content = {
#                 "type": "image_url",
#                 "image_url": {
#                     "url": f"data:image/jpeg;base64,{image_base64}"
#                 }
#             }
            
#             text_content = {
#                 "type": "text",
#                 "text": prompt
#             }
            
#             # 确定使用的模型名称
#             if model is None:
#                 if async_client is not None:
#                     model = async_client.model_name
#                 else:
#                     import config
#                     model = config.MODEL_NAME
            
#             # 使用异步客户端
#             if async_client is None:
#                 async with AsyncGeminiClient() as client:
#                     response = await client.chat.completions.create(
#                         model=model,
#                         messages=[
#                             {
#                                 "role": "user",
#                                 "content": [text_content, image_content]
#                             }
#                         ],
#                         max_tokens=1000,
#                         temperature=0.3,
#                         response_format={"type": "json_object"}
#                     )
#             else:
#                 response = await async_client.chat.completions.create(
#                     model=model,
#                     messages=[
#                         {
#                             "role": "user",
#                             "content": [text_content, image_content]
#                         }
#                     ],
#                     max_tokens=1000,
#                     temperature=0.3,
#                     response_format={"type": "json_object"}
#                 )
            
#             # 提取响应内容
#             response_text = response.choices[0].message.content
            
#             # 解析JSON响应
#             try:
#                 result = json.loads(response_text)
#             except json.JSONDecodeError:
#                 # 如果JSON解析失败，尝试从文本中提取JSON
#                 json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
#                 if json_match:
#                     result = json.loads(json_match.group())
#                 else:
#                     return None
            
#             if result.get("selected", False):
#                 return {
#                     "name": result.get("object_name", ""),
#                     "category": result.get("object_category", ""),
#                     "reason": result.get("reason", ""),
#                     "confidence": result.get("confidence", 0.0)
#                 }
            
#             return None
            
#         except Exception as e:
#             print(f"[WARNING] 异步对象选择失败: {e}")
#             return None







"""
对象选择模块
根据配置选择图像中的目标对象
"""
import json
import re
from typing import Dict, Any, Optional, List
from utils.gemini_client import GeminiClient
from utils.async_client import AsyncGeminiClient


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
        # 构建prompt（改进：明确要求返回JSON，不使用null）
        prompt = f"""You are an object selection expert. Your task is to select the most suitable target object from the image according to the given criteria.

Pipeline Intent: {pipeline_intent}
Selection Strategy: {selection_strategy}

General Criteria:
{chr(10).join(f"- {criterion}" for criterion in general_criteria)}

Pipeline-Specific Constraints:
{chr(10).join(f"- {constraint}" for constraint in constraints)}

Analyze the image and select the most suitable object.

You MUST return a valid JSON object. Do NOT return null or any non-JSON text.

If a suitable object can be selected, return:
{{
    "selected": true,
    "object_name": "name of the selected object (e.g., 'person', 'car', 'tree')",
    "object_category": "category of the object",
    "reason": "brief explanation of why this object was selected",
    "confidence": 0.0-1.0
}}

If NO suitable object can be selected, return:
{{
    "selected": false,
    "reason": "detailed explanation of why no object meets the criteria",
    "confidence": 0.0
}}

IMPORTANT: Always return valid JSON format. Never return null or text without JSON structure."""

        try:
            response = self.gemini_client.analyze_image(
                image_input=image_input,
                prompt=prompt,
                temperature=0.3,
                context="object_selection"
            )
            
            # 记录原始响应（用于调试）
            print(f"[DEBUG] 对象选择原始响应: {response[:500]}..." if len(response) > 500 else f"[DEBUG] 对象选择原始响应: {response}")
            
            # 解析JSON响应（改进：多种解析方式）
            def _parse_json_response(response_text):
                """尝试多种方式解析JSON响应"""
                # 方法1: 直接解析
                try:
                    return json.loads(response_text)
                except json.JSONDecodeError:
                    pass
                
                # 方法2: 提取JSON块（移除可能的markdown代码块标记）
                cleaned = re.sub(r'```(?:json)?\s*', '', response_text)
                cleaned = re.sub(r'\s*```', '', cleaned)
                try:
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    pass
                
                # 方法3: 使用正则表达式提取JSON对象
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
                if json_match:
                    try:
                        return json.loads(json_match.group())
                    except json.JSONDecodeError:
                        pass
                
                return None
            
            result = _parse_json_response(response)
            
            if result is None:
                print(f"[ERROR] 无法从响应中提取有效JSON")
                print(f"[ERROR] 原始响应: {response[:300]}...")
                return None
            
            # 检查selected字段（改进：记录未选择的原因）
            if result.get("selected", False):
                confidence = result.get("confidence", 0.0)
                # 可选：如果置信度过低，也可以拒绝
                # if confidence < 0.3:
                #     print(f"[INFO] 对象选择置信度过低: {confidence}，拒绝选择")
                #     return None
                
                return {
                    "name": result.get("object_name", ""),
                    "category": result.get("object_category", ""),
                    "reason": result.get("reason", ""),
                    "confidence": confidence
                }
            else:
                # 记录未选择的原因（重要！用于诊断）
                reason = result.get("reason", "未知原因")
                confidence = result.get("confidence", 0.0)
                print(f"[INFO] 对象未选择 - 原因: {reason} (置信度: {confidence})")
                return None
            
        except Exception as e:
            print(f"[ERROR] 对象选择异常: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def select_object_async(
        self,
        image_base64: str,
        pipeline_config: Dict[str, Any],
        global_policy: Dict[str, Any],
        async_client: Optional[AsyncGeminiClient] = None,
        model: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        异步根据配置选择图像中的目标对象
        
        Args:
            image_base64: 图片的base64编码
            pipeline_config: Pipeline配置
            global_policy: 全局对象选择策略
            async_client: 异步客户端实例（可选）
            model: 模型名称（可选）
            
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
        selected_object = await self._select_with_llm_async(
            image_base64=image_base64,
            pipeline_intent=pipeline_config.get("intent", ""),
            selection_strategy=selection_strategy,
            constraints=constraints,
            general_criteria=general_criteria,
            async_client=async_client,
            model=model
        )
        
        return selected_object
    
    async def _select_with_llm_async(
        self,
        image_base64: str,
        pipeline_intent: str,
        selection_strategy: str,
        constraints: List[str],
        general_criteria: List[str],
        async_client: Optional[AsyncGeminiClient] = None,
        model: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        异步使用LLM选择对象
        
        Args:
            image_base64: 图片的base64编码
            pipeline_intent: Pipeline意图
            selection_strategy: 选择策略
            constraints: Pipeline特定约束
            general_criteria: 通用标准
            async_client: 异步客户端实例（可选）
            model: 模型名称（可选）
            
        Returns:
            选中的对象信息，如果无法选择返回None
        """
        # 构建prompt（改进：明确要求返回JSON，不使用null）
        prompt = f"""You are an object selection expert. Your task is to select the most suitable target object from the image according to the given criteria.

Pipeline Intent: {pipeline_intent}
Selection Strategy: {selection_strategy}

General Criteria:
{chr(10).join(f"- {criterion}" for criterion in general_criteria)}

Pipeline-Specific Constraints:
{chr(10).join(f"- {constraint}" for constraint in constraints)}

Analyze the image and select the most suitable object.

You MUST return a valid JSON object. Do NOT return null or any non-JSON text.

If a suitable object can be selected, return:
{{
    "selected": true,
    "object_name": "name of the selected object (e.g., 'person', 'car', 'tree')",
    "object_category": "category of the object",
    "reason": "brief explanation of why this object was selected",
    "confidence": 0.0-1.0
}}

If NO suitable object can be selected, return:
{{
    "selected": false,
    "reason": "detailed explanation of why no object meets the criteria",
    "confidence": 0.0
}}

IMPORTANT: Always return valid JSON format. Never return null or text without JSON structure."""

        try:
            # 构建图像内容（OpenAI兼容格式）
            image_content = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}"
                }
            }
            
            text_content = {
                "type": "text",
                "text": prompt
            }
            
            # 确定使用的模型名称
            if model is None:
                if async_client is not None:
                    model = async_client.model_name
                else:
                    import config
                    model = config.MODEL_NAME
            
            # 使用异步客户端
            if async_client is None:
                async with AsyncGeminiClient() as client:
                    response = await client.chat.completions.create(
                        model=model,
                        messages=[
                            {
                                "role": "user",
                                "content": [text_content, image_content]
                            }
                        ],
                        max_tokens=1000,
                        temperature=0.3,
                        response_format={"type": "json_object"}
                    )
            else:
                response = await async_client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": [text_content, image_content]
                        }
                    ],
                    max_tokens=1000,
                    temperature=0.3,
                    response_format={"type": "json_object"}
                )
            
            # 提取响应内容
            response_text = response.choices[0].message.content
            
            # 记录原始响应（用于调试）
            print(f"[DEBUG] 异步对象选择原始响应: {response_text[:500]}..." if len(response_text) > 500 else f"[DEBUG] 异步对象选择原始响应: {response_text}")
            
            # 解析JSON响应（改进：多种解析方式）
            def _parse_json_response(response_text):
                """尝试多种方式解析JSON响应"""
                # 方法1: 直接解析
                try:
                    return json.loads(response_text)
                except json.JSONDecodeError:
                    pass
                
                # 方法2: 提取JSON块（移除可能的markdown代码块标记）
                cleaned = re.sub(r'```(?:json)?\s*', '', response_text)
                cleaned = re.sub(r'\s*```', '', cleaned)
                try:
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    pass
                
                # 方法3: 使用正则表达式提取JSON对象
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
                if json_match:
                    try:
                        return json.loads(json_match.group())
                    except json.JSONDecodeError:
                        pass
                
                return None
            
            result = _parse_json_response(response_text)
            
            if result is None:
                print(f"[ERROR] 无法从异步响应中提取有效JSON")
                print(f"[ERROR] 原始响应: {response_text[:300]}...")
                return None
            
            # 检查selected字段（改进：记录未选择的原因）
            if result.get("selected", False):
                confidence = result.get("confidence", 0.0)
                # 可选：如果置信度过低，也可以拒绝
                # if confidence < 0.3:
                #     print(f"[INFO] 对象选择置信度过低: {confidence}，拒绝选择")
                #     return None
                
                return {
                    "name": result.get("object_name", ""),
                    "category": result.get("object_category", ""),
                    "reason": result.get("reason", ""),
                    "confidence": confidence
                }
            else:
                # 记录未选择的原因（重要！用于诊断）
                reason = result.get("reason", "未知原因")
                confidence = result.get("confidence", 0.0)
                print(f"[INFO] 异步对象未选择 - 原因: {reason} (置信度: {confidence})")
                return None
            
        except Exception as e:
            print(f"[ERROR] 异步对象选择异常: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return None

