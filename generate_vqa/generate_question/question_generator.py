"""
问题生成模块
根据配置和填充的槽位生成VQA问题
"""
from typing import Dict, Any, Optional
from utils.gemini_client import GeminiClient
import re


class QuestionGenerator:
    """问题生成器"""
    
    def __init__(self, gemini_client: Optional[GeminiClient] = None):
        """
        初始化问题生成器
        
        Args:
            gemini_client: Gemini客户端实例
        """
        self.gemini_client = gemini_client or GeminiClient()
    
    def generate_question(
        self,
        image_input: Any,
        pipeline_config: Dict[str, Any],
        slots: Dict[str, str],
        selected_object: Optional[Dict[str, Any]] = None,
        question_type: Optional[str] = None
    ) -> Optional[str]:
        """
        生成VQA问题
        
        Args:
            image_input: 图片输入
            pipeline_config: Pipeline配置
            slots: 填充的槽位字典
            selected_object: 选中的对象信息（如果有）
            question_type: 题型，可选值："multiple_choice"（选择题）或"fill_in_blank"（填空题）
            
        Returns:
            生成的问题文本，如果生成失败返回None
        """
        # 获取配置
        intent = pipeline_config.get("intent", "")
        example_template = pipeline_config.get("example_template", "")
        question_constraints = pipeline_config.get("question_constraints", [])
        description = pipeline_config.get("description", "")
        
        # 构建prompt
        prompt = self._build_generation_prompt(
            intent=intent,
            description=description,
            example_template=example_template,
            question_constraints=question_constraints,
            slots=slots,
            selected_object=selected_object,
            question_type=question_type
        )
        
        try:
            # 使用LLM生成问题
            response = self.gemini_client.analyze_image(
                image_input=image_input,
                prompt=prompt,
                temperature=0.7,  # 使用较高温度以增加多样性
                context="question_generation"
            )
            
            # 提取问题（可能包含引号或其他格式）
            question = self._extract_question(response)
            
            return question
            
        except Exception as e:
            print(f"[WARNING] 问题生成失败: {e}")
            return None
    
    def _build_generation_prompt(
        self,
        intent: str,
        description: str,
        example_template: str,
        question_constraints: list,
        slots: Dict[str, str],
        selected_object: Optional[Dict[str, Any]] = None,
        question_type: Optional[str] = None
    ) -> str:
        """构建问题生成prompt"""
        
        # 槽位信息
        slot_info = ""
        if slots:
            slot_info = "\nFilled Slots:\n"
            for key, value in slots.items():
                slot_info += f"- {key}: {value}\n"
        
        # 对象信息
        object_info = ""
        if selected_object:
            object_info = f"\nSelected Object:\n- Name: {selected_object.get('name', '')}\n- Category: {selected_object.get('category', '')}\n"
        
        # 题型要求
        question_type_instruction = ""
        if question_type == "multiple_choice":
            question_type_instruction = "\nQuestion Type: MULTIPLE CHOICE\n- Generate ONLY the question stem, NOT the answer options\n- The question should be phrased to expect a selection from multiple options (e.g., 'Which one...?', 'What is the...?', 'Choose the...?')\n- Do NOT include any answer choices or options in your response\n- Only generate the question text that would appear before the options\n"
        elif question_type == "fill_in_blank":
            question_type_instruction = "\nQuestion Type: FILL IN THE BLANK\n- Generate a question that requires a direct answer (e.g., 'What is...?', 'How many...?', 'Where is...?')\n- The question should be phrased to expect a direct textual or numerical answer, not a selection from options\n"
        
        prompt = f"""You are a VQA question generation expert. Generate a natural language question based on the given specifications.

Pipeline Intent: {intent}
Description: {description}
{object_info}
{slot_info}
Example Template: "{example_template}"
{question_type_instruction}
Question Constraints:
{chr(10).join(f"- {constraint}" for constraint in question_constraints)}

Requirements:
1. The question must be grounded in the image (explicitly reference visual entities)
2. The question must be answerable using the image alone
3. You may paraphrase and vary surface forms, but must NOT change the intent
4. You must NOT introduce new objects not in the image
5. You must NOT use external or commonsense-only knowledge
{("6. The question MUST be formatted as a " + ("multiple choice question stem (题干 only, NO options)" if question_type == "multiple_choice" else "fill-in-the-blank question") + " as specified above") if question_type else ""}

Generate a natural, fluent question that follows the template and constraints. Return ONLY the question text (question stem for multiple choice, NO options), no explanation or additional text."""

        return prompt
    
    def _extract_question(self, response: str) -> str:
        """从LLM响应中提取问题文本"""
        # 移除可能的引号
        question = response.strip()
        
        # 移除首尾引号
        if question.startswith('"') and question.endswith('"'):
            question = question[1:-1]
        elif question.startswith("'") and question.endswith("'"):
            question = question[1:-1]
        
        # 移除可能的"Question:"等前缀
        question = re.sub(r'^(Question|Q|问题)[:：]\s*', '', question, flags=re.IGNORECASE)
        
        # 只取第一行（问题通常是一行）
        question = question.split('\n')[0].strip()
        
        return question

