"""
答案生成模块
根据问题和图片生成答案
"""
import json
import random
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from utils.gemini_client import GeminiClient
from utils.async_client import AsyncGeminiClient


class AnswerGenerator:
    """答案生成器"""
    
    def __init__(self, config_path: Optional[Path] = None, gemini_client: Optional[GeminiClient] = None):
        """
        初始化答案生成器
        
        Args:
            config_path: 配置文件路径（可选）
            gemini_client: Gemini客户端实例（可选）
        """
        self.gemini_client = gemini_client or GeminiClient()
        
        # 加载配置
        if config_path and config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        else:
            # 默认配置
            self.config = {
                "multiple_choice": {
                    "wrong_options": {
                        "min_count": 2,
                        "max_count": 4
                    }
                },
                "generation_settings": {
                    "temperature": 0.7,
                    "max_tokens": 512
                }
            }
        
        self.mc_config = self.config.get("multiple_choice", {})
        self.gen_settings = self.config.get("generation_settings", {})
    
    def generate_answer(
        self,
        question: str,
        image_base64: str,
        question_type: str,
        pipeline_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        生成答案
        
        Args:
            question: 问题文本
            image_base64: 图片的base64编码
            question_type: 题型，"multiple_choice" 或 "fill_in_blank"
            pipeline_info: 可选的pipeline信息（用于上下文）
            
        Returns:
            包含answer、explanation等字段的字典
        """
        if question_type == "multiple_choice":
            return self._generate_multiple_choice_answer(
                question=question,
                image_base64=image_base64,
                pipeline_info=pipeline_info
            )
        elif question_type == "fill_in_blank":
            return self._generate_fill_in_blank_answer(
                question=question,
                image_base64=image_base64,
                pipeline_info=pipeline_info
            )
        else:
            raise ValueError(f"不支持的题型: {question_type}")
    
    async def generate_answer_async(
        self,
        question: str,
        image_base64: str,
        question_type: str,
        pipeline_info: Optional[Dict[str, Any]] = None,
        async_client: Optional[AsyncGeminiClient] = None
    ) -> Dict[str, Any]:
        """
        异步生成答案（使用 AsyncGeminiClient）
        """
        if question_type == "multiple_choice":
            return await self._generate_multiple_choice_answer_async(
                question=question,
                image_base64=image_base64,
                pipeline_info=pipeline_info,
                async_client=async_client
            )
        elif question_type == "fill_in_blank":
            return await self._generate_fill_in_blank_answer_async(
                question=question,
                image_base64=image_base64,
                pipeline_info=pipeline_info,
                async_client=async_client
            )
        else:
            raise ValueError(f"不支持的题型: {question_type}")
    
    def _generate_multiple_choice_answer(
        self,
        question: str,
        image_base64: str,
        pipeline_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        生成选择题答案
        
        步骤：
        1. 生成正确答案
        2. 生成错误选项
        3. 打乱顺序，组合成完整选择题
        4. 确定正确答案的选项字母
        """
        # Step 1: 生成正确答案
        correct_answer = self._generate_correct_answer(
            question=question,
            image_base64=image_base64,
            pipeline_info=pipeline_info
        )
        
        if not correct_answer or not correct_answer.get("answer"):
            return {
                "answer": None,
                "explanation": "无法生成正确答案",
                "full_question": question,
                "options": None
            }
        
        correct_answer_text = correct_answer["answer"]
        explanation = correct_answer.get("explanation", "")
        
        # Step 2: 生成错误选项
        wrong_options = self._generate_wrong_options(
            question=question,
            image_base64=image_base64,
            correct_answer=correct_answer_text,
            pipeline_info=pipeline_info
        )
        
        if not wrong_options:
            return {
                "answer": None,
                "explanation": explanation,
                "full_question": question,
                "options": None
            }
        
        # Step 3: 组合所有选项并打乱顺序
        all_options = [correct_answer_text] + wrong_options
        random.shuffle(all_options)
        
        # Step 4: 确定正确答案的选项字母
        correct_index = all_options.index(correct_answer_text)
        correct_letter = chr(65 + correct_index)  # A, B, C, D...
        
        # Step 5: 构建完整选择题
        options_text = "\n".join([f"{chr(65 + i)}: {option}" for i, option in enumerate(all_options)])
        full_question = f"{question}\n{options_text}"
        
        return {
            "answer": correct_letter,  # 如 "A", "B", "C"
            "explanation": explanation,
            "full_question": full_question,
            "options": {chr(65 + i): option for i, option in enumerate(all_options)},
            "correct_option": correct_letter
        }
    
    async def _generate_multiple_choice_answer_async(
        self,
        question: str,
        image_base64: str,
        pipeline_info: Optional[Dict[str, Any]] = None,
        async_client: Optional[AsyncGeminiClient] = None
    ) -> Dict[str, Any]:
        """
        异步生成选择题答案
        """
        correct_answer = await self._generate_correct_answer_async(
            question=question,
            image_base64=image_base64,
            pipeline_info=pipeline_info,
            async_client=async_client
        )
        
        if not correct_answer or not correct_answer.get("answer"):
            return {
                "answer": None,
                "explanation": "无法生成正确答案",
                "full_question": question,
                "options": None
            }
        
        correct_answer_text = correct_answer["answer"]
        explanation = correct_answer.get("explanation", "")
        
        wrong_options = await self._generate_wrong_options_async(
            question=question,
            image_base64=image_base64,
            correct_answer=correct_answer_text,
            pipeline_info=pipeline_info,
            async_client=async_client
        )
        
        if not wrong_options:
            return {
                "answer": None,
                "explanation": explanation,
                "full_question": question,
                "options": None
            }
        
        import random as _random
        all_options = [correct_answer_text] + wrong_options
        _random.shuffle(all_options)
        
        correct_index = all_options.index(correct_answer_text)
        correct_letter = chr(65 + correct_index)
        
        options_text = "\n".join([f"{chr(65 + i)}: {option}" for i, option in enumerate(all_options)])
        full_question = f"{question}\n{options_text}"
        
        return {
            "answer": correct_letter,
            "explanation": explanation,
            "full_question": full_question,
            "options": {chr(65 + i): option for i, option in enumerate(all_options)},
            "correct_option": correct_letter
        }
    
    def _generate_fill_in_blank_answer(
        self,
        question: str,
        image_base64: str,
        pipeline_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        生成填空题答案
        """
        result = self._generate_correct_answer(
            question=question,
            image_base64=image_base64,
            pipeline_info=pipeline_info
        )
        
        if not result or not result.get("answer"):
            return {
                "answer": None,
                "explanation": "无法生成答案",
                "full_question": question
            }
        
        return {
            "answer": result["answer"],
            "explanation": result.get("explanation", ""),
            "full_question": question
        }
    
    async def _generate_fill_in_blank_answer_async(
        self,
        question: str,
        image_base64: str,
        pipeline_info: Optional[Dict[str, Any]] = None,
        async_client: Optional[AsyncGeminiClient] = None
    ) -> Dict[str, Any]:
        """
        异步生成填空题答案
        """
        result = await self._generate_correct_answer_async(
            question=question,
            image_base64=image_base64,
            pipeline_info=pipeline_info,
            async_client=async_client
        )
        
        if not result or not result.get("answer"):
            return {
                "answer": None,
                "explanation": "无法生成答案",
                "full_question": question
            }
        
        return {
            "answer": result["answer"],
            "explanation": result.get("explanation", ""),
            "full_question": question
        }
    
    def _generate_correct_answer(
        self,
        question: str,
        image_base64: str,
        pipeline_info: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, str]]:
        """
        生成正确答案
        
        Returns:
            {"answer": "答案文本", "explanation": "解释"} 或 None
        """
        prompt = f"""Based on the image and the question, provide a concise and accurate answer.

Question: {question}

Requirements:
1. Provide ONLY the answer text, keep it concise and direct (typically 1-5 words)
2. Do NOT include any analysis, reasoning, or explanation in the answer field
3. The answer should be factual and based solely on the image content
4. If the question asks for a specific format (e.g., number, location, object name), provide the answer in that format
5. Keep the answer brief - avoid unnecessary words

Provide your response in the following format:
Answer: [your concise answer here, 1-5 words typically]
Explanation: [optional brief explanation, only if needed for clarity]

Important: The Answer field should contain ONLY the answer itself, nothing else. Keep it very brief."""

        try:
            response = self.gemini_client.analyze_image(
                image_input=image_base64,
                prompt=prompt,
                temperature=self.gen_settings.get("temperature", 0.7),
                context="generate_correct_answer",
                max_tokens=self.gen_settings.get("max_tokens", 512)
            )
            
            # 解析响应
            answer, explanation = self._parse_answer_response(response)
            
            return {
                "answer": answer,
                "explanation": explanation
            }
            
        except Exception as e:
            print(f"[ERROR] 生成正确答案失败: {e}")
            return None
    
    async def _generate_correct_answer_async(
        self,
        question: str,
        image_base64: str,
        pipeline_info: Optional[Dict[str, Any]] = None,
        async_client: Optional[AsyncGeminiClient] = None
    ) -> Optional[Dict[str, str]]:
        """
        异步生成正确答案（使用OpenAI兼容接口，参考vlmtool/generate_vqa的成功实践）
        """
        prompt = f"""Based on the image and the question, provide a concise and accurate answer.

Question: {question}

Requirements:
1. Provide ONLY the answer text, keep it concise and direct (typically 1-5 words)
2. Do NOT include any analysis, reasoning, or explanation in the answer field
3. The answer should be factual and based solely on the image content
4. If the question asks for a specific format (e.g., number, location, object name), provide the answer in that format
5. Keep the answer brief - avoid unnecessary words

Provide your response in the following format:
Answer: [your concise answer here, 1-5 words typically]
Explanation: [optional brief explanation, only if needed for clarity]

Important: The Answer field should contain ONLY the answer itself, nothing else. Keep it very brief."""
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
            
            # 使用OpenAI兼容接口（参考vlmtool/generate_vqa的成功实践）
            if async_client is None:
                async with AsyncGeminiClient() as client:
                    response = await client.chat.completions.create(
                        model=client.model_name,
                        messages=[
                            {
                                "role": "user",
                                "content": [text_content, image_content]
                            }
                        ],
                        max_tokens=512,
                        temperature=self.gen_settings.get("temperature", 0.7),
                        response_format={"type": "json_object"}
                    )
            else:
                response = await async_client.chat.completions.create(
                    model=async_client.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [text_content, image_content]
                        }
                    ],
                    max_tokens=512,
                    temperature=self.gen_settings.get("temperature", 0.7),
                    response_format={"type": "json_object"}
                )
            
            # 提取响应内容
            response_text = response.choices[0].message.content
            answer, explanation = self._parse_answer_response(response_text)
            return {"answer": answer, "explanation": explanation}
        except Exception as e:
            error_msg = str(e)
            # 如果是 400 错误，添加更详细的诊断信息
            if "400" in error_msg or "Bad Request" in error_msg:
                print(f"[ERROR] 异步生成正确答案失败 (400 Bad Request): {error_msg}")
                print(f"  问题: {question[:100]}...")
                print(f"  图片base64长度: {len(image_base64) if image_base64 else 0}")
                print("  可能的原因:")
                print("    1. 图片 base64 编码格式不正确")
                print("    2. 请求体大小超过 API 限制")
                print("    3. API 参数格式不匹配")
            else:
                print(f"[ERROR] 异步生成正确答案失败: {error_msg}")
            return None
    
    def _generate_wrong_options(
        self,
        question: str,
        image_base64: str,
        correct_answer: str,
        pipeline_info: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        生成错误选项（迷惑选项）
        
        Args:
            question: 问题文本
            image_base64: 图片base64
            correct_answer: 正确答案
            pipeline_info: pipeline信息
            
        Returns:
            错误选项列表
        """
        # 随机选择错误选项数量
        min_count = self.mc_config.get("wrong_options", {}).get("min_count", 2)
        max_count = self.mc_config.get("wrong_options", {}).get("max_count", 4)
        wrong_count = random.randint(min_count, max_count)
        
        prompt = f"""Based on the image and question, generate {wrong_count} plausible but incorrect answer options.

Question: {question}
Correct Answer: {correct_answer}

Requirements:
1. Generate exactly {wrong_count} wrong options
2. Each option should be similar in format and length to the correct answer ("{correct_answer}")
3. Options should be plausible but clearly incorrect when compared to the image
4. Options should be diverse and not repetitive
5. Each option should be concise (similar length to the correct answer, typically 1-5 words)
6. Do NOT include any analysis or explanation, only the options
7. Make sure each option is a single, concise phrase similar to the correct answer format

Provide your response in the following format (one option per line):
Option 1: [option text]
Option 2: [option text]
...
Option {wrong_count}: [option text]

Important: Each option should be brief and similar in style to "{correct_answer}"."""

        try:
            response = self.gemini_client.analyze_image(
                image_input=image_base64,
                prompt=prompt,
                temperature=self.gen_settings.get("temperature", 0.9),  # 更高温度以增加多样性
                context="generate_wrong_options",
                max_tokens=self.gen_settings.get("max_tokens", 512)
            )
            
            # 解析错误选项
            wrong_options = self._parse_wrong_options_response(response, wrong_count)
            
            return wrong_options
            
        except Exception as e:
            print(f"[ERROR] 生成错误选项失败: {e}")
            return []
    
    def _parse_answer_response(self, response: str) -> Tuple[str, str]:
        """
        解析答案响应
        
        Returns:
            (answer, explanation)
        """
        answer = ""
        explanation = ""
        
        # 提取Answer行
        answer_match = re.search(r'Answer:\s*(.+?)(?:\n|$)', response, re.IGNORECASE | re.MULTILINE)
        if answer_match:
            answer = answer_match.group(1).strip()
        
        # 提取Explanation行
        explanation_match = re.search(r'Explanation:\s*(.+?)(?:\n|$)', response, re.IGNORECASE | re.MULTILINE | re.DOTALL)
        if explanation_match:
            explanation = explanation_match.group(1).strip()
        
        # 如果没有找到Answer格式，尝试提取第一行或整个响应
        if not answer:
            lines = response.strip().split('\n')
            if lines:
                # 移除可能的"Answer:"前缀
                answer = re.sub(r'^Answer:\s*', '', lines[0], flags=re.IGNORECASE).strip()
                if not answer:
                    answer = lines[0].strip()
        
        # 清理答案（移除引号等）
        answer = answer.strip('"\'')
        
        return answer, explanation
    
    def _parse_wrong_options_response(self, response: str, expected_count: int) -> List[str]:
        """
        解析错误选项响应
        
        Returns:
            错误选项列表
        """
        options = []
        
        # 尝试按格式解析：Option 1: ... Option 2: ...
        pattern = r'Option\s+\d+:\s*(.+?)(?:\n|Option|$)'
        matches = re.findall(pattern, response, re.IGNORECASE | re.MULTILINE)
        
        if matches:
            options = [match.strip().strip('"\'') for match in matches]
        else:
            # 如果没有找到格式化的选项，尝试按行分割
            lines = [line.strip() for line in response.strip().split('\n') if line.strip()]
            # 过滤掉明显的标题行
            lines = [line for line in lines if not re.match(r'^(Option|选项|错误选项)', line, re.IGNORECASE)]
            options = lines[:expected_count]
        
        # 清理选项
        options = [opt.strip('"\'') for opt in options if opt.strip()]
        
        # 如果选项数量不足，返回已有的
        return options[:expected_count] if options else []
    
    async def _generate_wrong_options_async(
        self,
        question: str,
        image_base64: str,
        correct_answer: str,
        pipeline_info: Optional[Dict[str, Any]] = None,
        async_client: Optional[AsyncGeminiClient] = None
    ) -> List[str]:
        """
        异步生成错误选项（迷惑选项，使用OpenAI兼容接口）
        """
        min_count = self.mc_config.get("wrong_options", {}).get("min_count", 2)
        max_count = self.mc_config.get("wrong_options", {}).get("max_count", 4)
        wrong_count = random.randint(min_count, max_count)
        
        prompt = f"""Based on the image and question, generate {wrong_count} plausible but incorrect answer options.

Question: {question}
Correct Answer: {correct_answer}

Requirements:
1. Generate exactly {wrong_count} wrong options
2. Each option should be similar in format and length to the correct answer ("{correct_answer}")
3. Options should be plausible but clearly incorrect when compared to the image
4. Options should be diverse and not repetitive
5. Each option should be concise (similar length to the correct answer, typically 1-5 words)
6. Do NOT include any analysis or explanation, only the options
7. Make sure each option is a single, concise phrase similar to the correct answer format

Provide your response in the following format (one option per line):
Option 1: [option text]
Option 2: [option text]
...
Option {wrong_count}: [option text]

Important: Each option should be brief and similar in style to "{correct_answer}"."""
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
            
            # 使用OpenAI兼容接口
            if async_client is None:
                async with AsyncGeminiClient() as client:
                    response = await client.chat.completions.create(
                        model=client.model_name,
                        messages=[
                            {
                                "role": "user",
                                "content": [text_content, image_content]
                            }
                        ],
                        max_tokens=512,
                        temperature=self.gen_settings.get("temperature", 0.9),
                        response_format={"type": "json_object"}
                    )
            else:
                response = await async_client.chat.completions.create(
                    model=async_client.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": [text_content, image_content]
                        }
                    ],
                    max_tokens=512,
                    temperature=self.gen_settings.get("temperature", 0.9),
                    response_format={"type": "json_object"}
                )
            
            # 提取响应内容
            response_text = response.choices[0].message.content
            wrong_options = self._parse_wrong_options_response(response_text, wrong_count)
            return wrong_options
        except Exception as e:
            print(f"[ERROR] 异步生成错误选项失败: {e}")
            return []

