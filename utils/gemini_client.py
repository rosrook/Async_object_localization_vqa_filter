# import io
# import base64
# import re
# from pathlib import Path
# from typing import Union, Optional
# from PIL import Image
# from openai import OpenAI
# import config


# class GeminiClient:
#     """Gemini API客户端封装（使用OpenAI兼容格式）"""
    
#     def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None, base_url: Optional[str] = None, 
#                  save_debug_images: bool = True, debug_image_dir: str = "/home/zhuxuzhou/test_localization/object_localization/data/view_import_img/"):
#         """
#         初始化Gemini客户端
        
#         Args:
#             api_key: API密钥，如果为None则从config读取
#             model_name: 模型名称，如果为None则从config读取
#             base_url: API基础URL，如果为None则从config读取
#             save_debug_images: 是否保存调试图片（保存传入MLLM的图片）
#             debug_image_dir: 调试图片保存目录
#         """
#         self.api_key = api_key or config.API_KEY
#         self.model_name = model_name or config.MODEL_NAME
#         self.base_url = base_url or config.BASE_URL
#         self.save_debug_images = save_debug_images
#         self.debug_image_dir = Path(debug_image_dir)
#         self.debug_image_counter = 0
        
#         # 如果开启调试模式，创建保存目录
#         if self.save_debug_images:
#             self.debug_image_dir.mkdir(parents=True, exist_ok=True)
#             print(f"调试模式已开启，图片将保存到: {self.debug_image_dir}")
        
#         if not self.api_key:
#             raise ValueError("API Key未设置，请在.env文件中设置API_KEY")
        
#         if not self.base_url:
#             raise ValueError("Base URL未设置，请在config.py中设置BASE_URL")
        
#         self.client = OpenAI(
#             api_key=self.api_key,
#             base_url=self.base_url
#         )
    
#     def _detect_image_type(self, image_input: Union[str, Path, bytes, Image.Image]) -> str:
#         """
#         检测图片输入类型
        
#         Args:
#             image_input: 图片输入
            
#         Returns:
#             输入类型: 'path', 'url', 'base64', 'bytes', 'pil'
#         """
#         if isinstance(image_input, Image.Image):
#             return 'pil'
        
#         if isinstance(image_input, bytes):
#             return 'bytes'
        
#         if isinstance(image_input, (str, Path)):
#             image_str = str(image_input)
            
#             # 检测是否为URL
#             if image_str.startswith(('http://', 'https://')):
#                 return 'url'
            
#             # 检测是否为base64数据URI
#             if image_str.startswith('data:image'):
#                 return 'base64'
            
#             # 检测是否为纯base64字符串（启发式判断）
#             # 注意：必须在检查文件路径之前，且避免对长字符串使用 Path().exists()
#             if len(image_str) > 100:
#                 # 尝试base64解码
#                 try:
#                     # 移除可能的空白字符
#                     clean_str = re.sub(r'\s', '', image_str)
#                     base64.b64decode(clean_str)
#                     return 'base64'
#                 except Exception:
#                     pass
            
#             # 检测是否为文件路径（只对短字符串检查，避免"文件名过长"错误）
#             if len(image_str) < 500:  # 限制长度，避免 base64 被误判
#                 try:
#                     if Path(image_str).exists():
#                         return 'path'
#                 except OSError:
#                     # 如果路径过长或包含非法字符，跳过
#                     pass
                
#                 # 如果路径不存在但看起来像路径，仍然返回path
#                 if '/' in image_str or '\\' in image_str or '.' in image_str:
#                     return 'path'
        
#         raise ValueError(f"无法识别的图片输入类型: {type(image_input)}")
    
#     def _load_image(self, image_input: Union[str, Path, bytes, Image.Image]) -> Image.Image:
#         """
#         从各种输入类型加载图片为PIL Image对象
        
#         Args:
#             image_input: 图片输入（路径、URL、base64、bytes或PIL Image）
            
#         Returns:
#             PIL Image对象
#         """
#         input_type = self._detect_image_type(image_input)
        
#         if input_type == 'pil':
#             return image_input
        
#         elif input_type == 'bytes':
#             return Image.open(io.BytesIO(image_input))
        
#         elif input_type == 'path':
#             return Image.open(image_input)
        
#         elif input_type == 'url':
#             import requests
#             response = requests.get(str(image_input))
#             response.raise_for_status()
#             return Image.open(io.BytesIO(response.content))
        
#         elif input_type == 'base64':
#             image_str = str(image_input)
            
#             # 如果是data URI，提取base64部分
#             if image_str.startswith('data:image'):
#                 # 格式: data:image/jpeg;base64,/9j/4AAQ...
#                 base64_data = image_str.split(',', 1)[1]
#             else:
#                 base64_data = image_str
            
#             # 移除可能的空白字符
#             base64_data = re.sub(r'\s', '', base64_data)
            
#             # 解码base64
#             image_bytes = base64.b64decode(base64_data)
#             return Image.open(io.BytesIO(image_bytes))
        
#         else:
#             raise ValueError(f"不支持的图片类型: {input_type}")
    
#     def _save_debug_image(self, image: Image.Image, context: str = "") -> str:
#         """
#         保存调试图片
        
#         Args:
#             image: PIL Image对象
#             context: 上下文信息（用于文件命名）
            
#         Returns:
#             保存的文件路径
#         """
#         if not self.save_debug_images:
#             return ""
        
#         # 生成唯一的文件名
#         self.debug_image_counter += 1
#         timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
        
#         # 清理上下文信息，移除非法字符
#         clean_context = re.sub(r'[^\w\s-]', '', context)[:50]  # 限制长度
#         clean_context = re.sub(r'\s+', '_', clean_context)
        
#         if clean_context:
#             filename = f"{self.debug_image_counter:04d}_{timestamp}_{clean_context}.jpg"
#         else:
#             filename = f"{self.debug_image_counter:04d}_{timestamp}.jpg"
        
#         filepath = self.debug_image_dir / filename
        
#         # 保存图片
#         try:
#             # 确保是RGB模式
#             if image.mode == 'RGBA':
#                 rgb_image = Image.new('RGB', image.size, (255, 255, 255))
#                 rgb_image.paste(image, mask=image.split()[3])
#                 image = rgb_image
#             elif image.mode not in ('RGB', 'L'):
#                 image = image.convert('RGB')
            
#             image.save(filepath, format='JPEG', quality=95)
#             print(f"[DEBUG] 图片已保存: {filepath}")
#             return str(filepath)
#         except Exception as e:
#             print(f"[DEBUG] 保存图片失败: {e}")
#             return ""
    
#     def _encode_image(self, image_input: Union[str, Path, bytes, Image.Image], context: str = "") -> str:
#         """
#         将图片编码为base64字符串（自动检测输入类型）
        
#         Args:
#             image_input: 图片输入（支持路径、URL、base64、bytes或PIL Image）
            
#         Returns:
#             base64编码的图片字符串
#         """
#         # 加载图片
#         image = self._load_image(image_input)
        
#         # 如果图片是RGBA模式，转换为RGB
#         if image.mode == 'RGBA':
#             # 创建白色背景
#             rgb_image = Image.new('RGB', image.size, (255, 255, 255))
#             rgb_image.paste(image, mask=image.split()[3])  # 使用alpha通道作为mask
#             image = rgb_image
#         elif image.mode not in ('RGB', 'L'):
#             image = image.convert('RGB')
        
#         # 将图片转换为字节流
#         buffer = io.BytesIO()
#         image.save(buffer, format='JPEG')
#         buffer.seek(0)
        
#         # 编码为base64
#         image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
#         return image_base64
    
#     def analyze_image(
#         self, 
#         image_input: Union[str, Path, bytes, Image.Image], 
#         prompt: str,
#         temperature: float = 0.7,
#         context: str = ""
#     ) -> str:
#         """
#         使用模型分析图片
        
#         Args:
#             image_input: 图片输入（支持文件路径、URL、base64字符串、bytes或PIL Image对象）
#             prompt: 提示词
#             temperature: 温度参数，控制输出的随机性
#             context: 上下文信息（用于调试保存时的命名）
            
#         Returns:
#             模型的响应文本
#         """
#         # 编码图片（自动检测输入类型）
#         image_base64 = self._encode_image(image_input, context=context or "analyze")
        
#         # 调用API
#         completion = self.client.chat.completions.create(
#             model=self.model_name,
#             messages=[
#                 {
#                     "role": "user",
#                     "content": [
#                         {"type": "text", "text": prompt},
#                         {
#                             "type": "image_url",
#                             "image_url": {
#                                 "url": f"data:image/jpeg;base64,{image_base64}"
#                             }
#                         }
#                     ]
#                 }
#             ],
#             stream=False,
#             max_tokens=4096,
#             temperature=temperature
#         )
        
#         return completion.choices[0].message.content
    
#     def filter_image(
#         self,
#         image_input: Union[str, Path, bytes, Image.Image],
#         criteria_description: str,
#         question: str,
#         temperature: float = 0.3
#     ) -> dict:
#         """
#         使用模型筛选图片
        
#         Args:
#             image_input: 图片输入（支持文件路径、URL、base64字符串、bytes或PIL Image对象）
#             criteria_description: 筛选标准描述
#             question: 问题描述
#             temperature: 温度参数，筛选时使用较低温度以获得更稳定的结果
            
#         Returns:
#             包含筛选结果的字典，格式为：
#             {
#                 "passed": bool,  # 是否通过筛选
#                 "reason": str,    # 筛选原因
#                 "confidence": float  # 置信度（0-1）
#             }
#         """
# #         prompt = f"""You are a professional image filtering expert. Please filter the image based on the following criteria.

# # Question Type: {question}

# # Filtering Criteria:
# # {criteria_description}

# # Please carefully analyze the image and determine whether it meets all the above criteria.

# # Return the result in JSON format as follows:
# # {{
# #     "passed": true/false,
# #     "reason": "Detailed filtering reason, explaining whether the image meets the criteria and why",
# #     "confidence": A float between 0.0-1.0 representing your confidence in the judgment
# # }}

# # Return only JSON, no other content."""





#         # 添加调试信息：检查criteria和question是否为空
#         if not criteria_description or criteria_description.strip() == "":
#             print(f"[WARNING] filter_image: criteria_description is empty!")
#             print(f"  - question: {question}")
#             print(f"  - criteria_description type: {type(criteria_description)}")
#             print(f"  - criteria_description value: {repr(criteria_description)}")
        
#         if not question or question.strip() == "":
#             print(f"[WARNING] filter_image: question is empty!")
#             print(f"  - question type: {type(question)}")
#             print(f"  - question value: {repr(question)}")
        
#         # 格式化prompt，替换占位符
#         prompt = """
# You are a professional image filtering and quality evaluation expert.

# Your task is to evaluate whether an image meets the required criteria, and to assign a quality score based on how well it satisfies both required and optional standards.

# Question Type:
# {question}

# Evaluation Criteria:
# {criteria_description}

# Please follow the rules below strictly.


# 1. Required (non-optional) Criteria:
# - These are mandatory criteria.
# - ALL required criteria must be satisfied for the image to pass.
# - If ANY required criterion is not satisfied:
#   - "passed" must be false
#   - "score" must be 0.0
#   - Optional criteria must NOT be considered
# - If ALL required criteria are satisfied:
#   - "passed" is true
#   - The score starts from a base score of 0.1
#   - The quality score for required criteria ranges from 0.1 to 0.6
#   - The closer the image matches the required criteria (clarity, correctness, completeness, alignment),
#     the closer the score should be to 0.6

# 2. Optional Criteria:
# - Optional criteria are marked explicitly as "optional" in the criteria.
# - Optional criteria are considered ONLY IF all required criteria are satisfied.
# - Optional criteria represent higher difficulty and higher value.

# Scoring with Optional Criteria:
# - If there are NO optional criteria:
#   - Automatically add 0.4 to the score (i.e., total score = required score + 0.4)
# - If there ARE optional criteria:
#   - A bonus score up to 0.4 is available
#   - The more optional criteria are satisfied, and the better they are satisfied,
#     the higher the bonus score (from 0.0 to 0.4)

# 3. Final Score:
# - Total score = required score (max 0.6) + optional bonus (max 0.4)
# - Final score must be in the range [0.0, 1.0]
# - Images satisfying optional criteria well should score higher than those that only satisfy required criteria.

# 4. Confidence:
# - "confidence" represents how confident you are in your judgment
# - It must be a float between 0.0 and 1.0
# - Confidence should be lower if the image is ambiguous or borderline


# Return the result in JSON format ONLY, with no extra text:

# {{
#   "passed": true/false,
#   "basic_score": float,
#   "bonus_score": float (0.0-0.4),
#   "total_score": float (0.0–1.0),
#   "reason": "Detailed explanation of which required and optional criteria are satisfied or violated, and how the score is determined",
#   "confidence": float (0.0–1.0)
# }}""".format(question=question, criteria_description=criteria_description)
        
#         # 调试：打印prompt的前500个字符以验证格式化是否正确
#         if not criteria_description or not question:
#             print(f"[DEBUG] Generated prompt (first 500 chars):\n{prompt[:500]}")
        
#         try:
#             # 使用question的前30个字符作为context
#             context = f"filter_{question[:30] if question else 'empty'}"
#             response_text = self.analyze_image(image_input, prompt, temperature=temperature, context=context)
#             if not response_text:
#                 raise ValueError("analyze_image 返回为空")
            
#             if isinstance(response_text, str):
#                 raise ValueError("is str!!!") # 需要直接当作str处理
            
#             # 尝试解析JSON响应
#             import json
            
#             # 提取JSON部分（处理可能的markdown代码块）
#             # 先尝试提取 ```json ... ``` 代码块
#             json_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
#             if json_block_match:
#                 response_text = json_block_match.group(1)
#             else:
#                 # 如果没有代码块，尝试提取第一个完整的JSON对象
#                 # 使用更智能的方法：找到第一个 { 和最后一个 }，中间包含 "passed"
#                 start_idx = response_text.find('{')
#                 if start_idx != -1:
#                     brace_count = 0
#                     for i in range(start_idx, len(response_text)):
#                         if response_text[i] == '{':
#                             brace_count += 1
#                         elif response_text[i] == '}':
#                             brace_count -= 1
#                             if brace_count == 0 and '"passed"' in response_text[start_idx:i+1]:
#                                 response_text = response_text[start_idx:i+1]
#                                 break
            
#             result = json.loads(response_text)
            
#             # 验证结果格式
#             if "passed" not in result:
#                 result["passed"] = False
#             if "reason" not in result:
#                 result["reason"] = "无法解析筛选结果"
#             if "confidence" not in result:
#                 result["confidence"] = 0.5
            
#             return result
            
#         except Exception as e:
#             # 如果解析失败，返回默认结果
#             return {
#                 "passed": False,
#                 "reason": f"筛选过程出错: {str(e)}",
#                 "confidence": 0.0
#             }


# # 使用示例
# if __name__ == "__main__":
#     # 示例1: 关闭调试模式（默认）
#     client = GeminiClient()
#     result = client.analyze_image("path/to/image.jpg", "描述这张图片")
    
#     # 示例2: 开启调试模式
#     client_debug = GeminiClient(save_debug_images=True)
    
#     # 使用不同的输入类型
#     result1 = client_debug.analyze_image("path/to/image.jpg", "描述这张图片", context="test1")
    
#     with open("image.jpg", "rb") as f:
#         image_bytes = f.read()
#         image_base64 = base64.b64encode(image_bytes).decode()
    
#     result2 = client_debug.analyze_image(image_base64, "描述这张图片", context="test2_base64")
#     result3 = client_debug.analyze_image(image_bytes, "描述这张图片", context="test3_bytes")
    
#     pil_image = Image.open("image.jpg")
#     result4 = client_debug.analyze_image(pil_image, "描述这张图片", context="test4_pil")
    
#     # 示例3: 自定义调试目录
#     client_custom = GeminiClient(
#         save_debug_images=True,
#         debug_image_dir="/custom/path/debug_images/"
#     )
    
#     # 查看保存的图片
#     # 文件命名格式: 0001_20231215_143022_test1.jpg
#     # 序号_时间戳_上下文.jpg







import io
import base64
import re
import json
from pathlib import Path
from typing import Union, Optional, Dict, Any
from datetime import datetime
from PIL import Image
import config

# 自动安装 redeuler（如果未安装）
try:
    from redeuler.client.openai import LBOpenAIClient
except ImportError:
    print("redeuler 未安装，正在自动安装...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "redeuler"])
    print("安装完成！")
    from redeuler.client.openai import LBOpenAIClient


class GeminiClient:
    """Gemini API客户端封装（使用OpenAI兼容格式）"""
    
    def __init__(
        self, 
        service_name: Optional[str] = None,
        env: Optional[str] = None,
        api_key: Optional[str] = None, 
        model_name: Optional[str] = None, 
        save_debug_images: bool = False, 
        debug_image_dir: str = "/home/zhuxuzhou/test_localization/object_localization/data/view_import_img/"
    ):
        """
        初始化Gemini客户端（使用LBOpenAIClient）
        
        Args:
            service_name: 服务名称，如果为None则从config读取
            env: 环境（如"prod"），如果为None则从config读取
            api_key: API密钥，如果为None则从config读取（LBOpenAIClient需要，但可能不使用）
            model_name: 模型名称，如果为None则从config读取
            save_debug_images: 是否保存调试图片
            debug_image_dir: 调试图片保存目录
        """
        self.service_name = service_name or getattr(config, 'SERVICE_NAME', None)
        self.env = env or getattr(config, 'ENV', 'prod')
        self.api_key = api_key or config.API_KEY or "1"  # LBOpenAIClient需要api_key，但可能不使用
        self.model_name = model_name or config.MODEL_NAME
        self.save_debug_images = save_debug_images
        self.debug_image_dir = Path(debug_image_dir)
        self.debug_image_counter = 0
        
        # 验证必需参数
        if not self.service_name:
            raise ValueError("Service Name未设置，请在config.py中设置SERVICE_NAME或在初始化时传入")
        if not self.model_name:
            raise ValueError("Model Name未设置，请在config.py中设置MODEL_NAME")
        
        # 创建调试目录
        if self.save_debug_images:
            self.debug_image_dir.mkdir(parents=True, exist_ok=True)
            print(f"调试模式已开启，图片将保存到: {self.debug_image_dir}")
        
        # 初始化LBOpenAIClient
        self.client = LBOpenAIClient(
            service_name=self.service_name,
            env=self.env,
            api_key=self.api_key
        )
        self._closed = False
    
    def _detect_image_type(self, image_input: Union[str, Path, bytes, Image.Image]) -> str:
        """检测图片输入类型"""
        if isinstance(image_input, Image.Image):
            return 'pil'
        
        if isinstance(image_input, bytes):
            return 'bytes'
        
        if isinstance(image_input, (str, Path)):
            image_str = str(image_input)
            
            # 检测URL
            if image_str.startswith(('http://', 'https://')):
                return 'url'
            
            # 检测base64 data URI
            if image_str.startswith('data:image'):
                return 'base64'
            
            # 检测纯base64字符串（对长字符串进行启发式判断）
            if len(image_str) > 100:
                try:
                    clean_str = re.sub(r'\s', '', image_str)
                    base64.b64decode(clean_str)
                    return 'base64'
                except Exception:
                    pass
            
            # 检测文件路径（限制长度避免错误）
            if len(image_str) < 500:
                try:
                    if Path(image_str).exists():
                        return 'path'
                except OSError:
                    pass
                
                # 如果包含路径分隔符，可能是路径
                if '/' in image_str or '\\' in image_str or '.' in image_str:
                    return 'path'
        
        raise ValueError(f"无法识别的图片输入类型: {type(image_input)}")
    
    def _load_image(self, image_input: Union[str, Path, bytes, Image.Image]) -> Image.Image:
        """从各种输入类型加载图片为PIL Image对象"""
        input_type = self._detect_image_type(image_input)
        
        if input_type == 'pil':
            return image_input
        
        elif input_type == 'bytes':
            # BytesIO不需要关闭，会在垃圾回收时自动释放
            return Image.open(io.BytesIO(image_input))
        
        elif input_type == 'path':
            # 对于文件路径，需要确保文件句柄被正确关闭
            # 使用with语句确保文件在使用后立即关闭
            with open(image_input, 'rb') as f:
                image_bytes = f.read()
            return Image.open(io.BytesIO(image_bytes))
        
        elif input_type == 'url':
            import requests
            response = requests.get(str(image_input))
            response.raise_for_status()
            # 立即关闭响应，释放连接
            try:
                return Image.open(io.BytesIO(response.content))
            finally:
                response.close()
        
        elif input_type == 'base64':
            image_str = str(image_input)
            
            # 提取base64数据
            if image_str.startswith('data:image'):
                base64_data = image_str.split(',', 1)[1]
            else:
                base64_data = image_str
            
            # 移除空白字符并解码
            base64_data = re.sub(r'\s', '', base64_data)
            image_bytes = base64.b64decode(base64_data)
            return Image.open(io.BytesIO(image_bytes))
        
        else:
            raise ValueError(f"不支持的图片类型: {input_type}")
    
    def close(self):
        """
        关闭客户端，释放资源（包括HTTP连接）
        
        注意：如果LBOpenAIClient有close方法，会尝试调用
        """
        if self._closed:
            return
        
        try:
            # 尝试关闭LBOpenAIClient（如果支持）
            if hasattr(self.client, 'close'):
                self.client.close()
            # 尝试关闭内部的HTTP客户端（如果存在）
            elif hasattr(self.client, '_client') and hasattr(self.client._client, 'close'):
                self.client._client.close()
            elif hasattr(self.client, 'client') and hasattr(self.client.client, 'close'):
                self.client.client.close()
            # 尝试关闭session（如果使用requests）
            elif hasattr(self.client, '_session') and hasattr(self.client._session, 'close'):
                self.client._session.close()
            elif hasattr(self.client, 'session') and hasattr(self.client.session, 'close'):
                self.client.session.close()
        except Exception as e:
            # 静默处理，避免影响正常流程
            print(f"[DEBUG] 关闭客户端时出错（可忽略）: {e}")
        finally:
            self._closed = True
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口，确保资源释放"""
        self.close()
    
    def __del__(self):
        """析构函数，确保资源释放"""
        try:
            self.close()
        except:
            pass
    
    def _convert_to_rgb(self, image: Image.Image) -> Image.Image:
        """将图片转换为RGB模式"""
        if image.mode == 'RGBA':
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            rgb_image.paste(image, mask=image.split()[3])
            return rgb_image
        elif image.mode not in ('RGB', 'L'):
            return image.convert('RGB')
        return image
    
    def _save_debug_image(self, image: Image.Image, context: str = "") -> str:
        """保存调试图片"""
        if not self.save_debug_images:
            return ""
        
        try:
            self.debug_image_counter += 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 清理上下文信息
            clean_context = re.sub(r'[^\w\s-]', '', context)[:50]
            clean_context = re.sub(r'\s+', '_', clean_context)
            
            filename = f"{self.debug_image_counter:04d}_{timestamp}"
            if clean_context:
                filename += f"_{clean_context}"
            filename += ".jpg"
            
            filepath = self.debug_image_dir / filename
            
            # 转换为RGB并保存
            rgb_image = self._convert_to_rgb(image)
            rgb_image.save(filepath, format='JPEG', quality=95)
            print(f"[DEBUG] 图片已保存: {filepath}")
            return str(filepath)
        except Exception as e:
            print(f"[DEBUG] 保存图片失败: {e}")
            return ""
    
    def _encode_image(self, image_input: Union[str, Path, bytes, Image.Image], context: str = "") -> str:
        """将图片编码为base64字符串"""
        # 加载图片
        image = self._load_image(image_input)
        
        try:
            # 保存调试图片
            if self.save_debug_images:
                self._save_debug_image(image, context)
            
            # 转换为RGB
            image = self._convert_to_rgb(image)
            
            # 编码为base64
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG')
            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode('utf-8')
        finally:
            # 确保图片对象被关闭，释放文件句柄
            # 注意：PIL Image对象在垃圾回收时会自动关闭，但显式关闭更安全
            if hasattr(image, 'close') and image.fp is not None:
                try:
                    image.close()
                except:
                    pass
    
    def analyze_image(
        self, 
        image_input: Union[str, Path, bytes, Image.Image], 
        prompt: str,
        temperature: float = 0.7,
        context: str = "",
        max_tokens: int = 4096
    ) -> str:
        """
        使用模型分析图片
        
        Args:
            image_input: 图片输入
            prompt: 提示词
            temperature: 温度参数
            context: 上下文信息（用于调试命名）
            max_tokens: 最大token数
            
        Returns:
            模型的响应文本
        """
        try:
            # 编码图片
            image_base64 = self._encode_image(image_input, context=context or "analyze")
            
            # 完全按照工作示例的格式构建消息
            # 重要：只有一条消息，content 是列表，先文本后图片
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            
            # 添加图片（完全复制示例代码的逻辑）
            img_cont = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}"
                }
            }
            messages[0]['content'].append(img_cont)
            
            print(f"[DEBUG] 调用API - model: {self.model_name}")
            print(f"[DEBUG] prompt长度: {len(prompt)}, image_base64长度: {len(image_base64)}")
            print(f"[DEBUG] messages 数量: {len(messages)}")
            print(f"[DEBUG] messages[0]['content'] 数量: {len(messages[0]['content'])}")
            
            # 打印实际发送的 messages 结构（用于调试）
            print(f"[DEBUG] 完整 messages 结构（前500字符）:")
            messages_str = json.dumps(messages, ensure_ascii=False)[:500]
            print(messages_str)
            
            # 调用API（使用LBOpenAIClient，完全按照debug_qwen3vl.py的格式）
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            print(f"[DEBUG] API调用成功")
            
            # 检查是否有错误响应（自定义错误格式）
            if hasattr(completion, 'success') and completion.success is False:
                error_message = getattr(completion, 'message', '未知错误')
                print(f"[ERROR] API 返回错误: {error_message}")
                raise ValueError(f"API 调用失败: {error_message}")
            
            # ========== 详细调试信息（仅在需要时启用）==========
            # print(f"[DEBUG] completion 对象类型: {type(completion)}")
            # print(f"[DEBUG] completion 是否为 None: {completion is None}")
            
            # 尝试打印 completion 对象
            try:
                completion_dump = completion.model_dump() if hasattr(completion, 'model_dump') else str(completion)
                print(f"[DEBUG] completion 对象: {completion_dump}")
            except Exception as e:
                print(f"[DEBUG] 无法打印 completion 对象: {e}")
            # ========== 调试信息结束 ==========
            
            # 验证响应
            if completion is None:
                raise ValueError("API 返回了 None")
            
            # 检查是否有 choices 属性
            if not hasattr(completion, 'choices'):
                print(f"[ERROR] completion 没有 choices 属性")
                raise ValueError("API 响应中没有 choices 字段")
            
            if completion.choices is None:
                print(f"[ERROR] completion.choices 为 None")
                raise ValueError("choices 为 None")
            
            if not completion.choices:
                print(f"[ERROR] completion.choices 为空（空列表或False）")
                raise ValueError("choices 为空")
            
            if len(completion.choices) == 0:
                raise ValueError("choices 列表长度为 0")
            
            print(f"[DEBUG] choices 长度: {len(completion.choices)}")
            print(f"[DEBUG] choices[0]: {completion.choices[0]}")
            
            if not hasattr(completion.choices[0], 'message'):
                print(f"[ERROR] choices[0] 没有 message 属性")
                print(f"[DEBUG] choices[0] 的所有属性: {dir(completion.choices[0])}")
                raise ValueError("API 响应中没有 message 字段")
            
            print(f"[DEBUG] message: {completion.choices[0].message}")
            
            content = completion.choices[0].message.content
            
            if content is None:
                raise ValueError("API 返回的 content 为 None")
            
            if content == "":
                print(f"[WARNING] API 返回的内容为空字符串")
            
            print(f"[DEBUG] 响应内容长度: {len(content)}")
            
            return content
        
        except AttributeError as e:
            error_msg = f"API 响应格式错误: {e}"
            print(f"[ERROR] {error_msg}")
            try:
                print(f"[DEBUG] completion 对象类型: {type(completion)}")
                print(f"[DEBUG] completion 对象: {completion}")
            except:
                print(f"[DEBUG] 无法打印 completion 对象")
            raise ValueError(error_msg)
        
        except Exception as e:
            print(f"[ERROR] analyze_image 失败: {type(e).__name__}: {e}")
            print(f"[DEBUG] model: {self.model_name}, service_name: {self.service_name}, env: {self.env}")
            import traceback
            print(f"[DEBUG] 完整错误堆栈:\n{traceback.format_exc()}")
            raise
    
    def _extract_json_from_response(self, response_text: str) -> Dict[str, Any]:
        """从响应文本中提取JSON"""
        if not response_text:
            raise ValueError("响应文本为空")
        
        # 尝试提取 ```json ... ``` 代码块
        json_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_block_match:
            response_text = json_block_match.group(1)
        else:
            # 提取第一个完整的JSON对象
            start_idx = response_text.find('{')
            if start_idx != -1:
                brace_count = 0
                for i in range(start_idx, len(response_text)):
                    if response_text[i] == '{':
                        brace_count += 1
                    elif response_text[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            response_text = response_text[start_idx:i+1]
                            break
        
        return json.loads(response_text)
    
    def filter_image(
        self,
        image_input: Union[str, Path, bytes, Image.Image],
        criteria_description: str,
        question: str,
        temperature: float = 0.3,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """
        使用模型筛选图片
        
        Args:
            image_input: 图片输入
            criteria_description: 筛选标准描述
            question: 问题描述
            temperature: 温度参数
            max_tokens: 最大token数
            
        Returns:
            筛选结果字典
        """
        # 验证输入
        if not criteria_description or not criteria_description.strip():
            print(f"[WARNING] criteria_description 为空!")
            criteria_description = "无特定标准"
        
        if not question or not question.strip():
            print(f"[WARNING] question 为空!")
            question = "通用筛选"
        
        # 构建prompt
        prompt = f"""You are a professional image filtering and quality evaluation expert.

Your task is to evaluate whether an image meets the required criteria, and to assign a quality score based on how well it satisfies both required and optional standards.

Question Type:
{question}

Evaluation Criteria:
{criteria_description}

Please follow the rules below strictly:

1. Required (non-optional) Criteria:
- These are mandatory criteria.
- ALL required criteria must be satisfied for the image to pass.
- If ANY required criterion is not satisfied:
  - "passed" must be false
  - "total_score" must be 0.0
- If ALL required criteria are satisfied:
  - "passed" is true
  - The basic_score starts from 0.1 to 0.6 based on quality

2. Optional Criteria:
- Optional criteria are marked explicitly as "optional".
- Optional criteria are considered ONLY IF all required criteria are satisfied.
- Bonus score ranges from 0.0 to 0.4 based on optional criteria satisfaction.

3. Final Score:
- total_score = basic_score (0.0-0.6) + bonus_score (0.0-0.4)
- Final score must be in range [0.0, 1.0]

Return the result in JSON format ONLY, with no extra text:

{{
  "passed": true/false,
  "basic_score": float (0.0-0.6),
  "bonus_score": float (0.0-0.4),
  "total_score": float (0.0-1.0),
  "reason": "Detailed explanation",
  "confidence": float (0.0-1.0)
}}"""
        
        try:
            # 使用question前30字符作为context
            context = f"filter_{question[:30]}"
            
            print(f"[INFO] 开始筛选图片，问题类型: {question[:50]}")
            
            response_text = self.analyze_image(
                image_input, 
                prompt, 
                temperature=temperature,  # 保留 temperature 参数
                context=context,
                max_tokens=max_tokens
            )
            
            print(f"[DEBUG] API 返回内容长度: {len(response_text) if response_text else 0}")
            if response_text:
                print(f"[DEBUG] API 返回内容前200字符: {response_text[:200]}")
            else:
                print(f"[DEBUG] API 返回内容为 None 或空")
            
            # 解析JSON响应
            result = self._extract_json_from_response(response_text)
            
            # 验证并补充缺失字段
            result.setdefault("passed", False)
            result.setdefault("reason", "无法解析筛选结果")
            result.setdefault("confidence", 0.5)
            result.setdefault("basic_score", 0.0)
            result.setdefault("bonus_score", 0.0)
            result.setdefault("total_score", 0.0)
            
            print(f"[INFO] 筛选完成，passed: {result['passed']}, score: {result['total_score']}")
            
            return result
            
        except ValueError as e:
            # API 调用或响应解析错误
            error_msg = f"筛选过程出错: {str(e)}"
            print(f"[ERROR] {error_msg}")
            return {
                "passed": False,
                "basic_score": 0.0,
                "bonus_score": 0.0,
                "total_score": 0.0,
                "reason": error_msg,
                "confidence": 0.0
            }
        
        except Exception as e:
            # 其他未预期的错误
            error_msg = f"未预期错误: {type(e).__name__}: {str(e)}"
            print(f"[ERROR] {error_msg}")
            import traceback
            print(f"[DEBUG] 完整错误堆栈:\n{traceback.format_exc()}")
            return {
                "passed": False,
                "basic_score": 0.0,
                "bonus_score": 0.0,
                "total_score": 0.0,
                "reason": error_msg,
                "confidence": 0.0
            }


# 使用示例
if __name__ == "__main__":
    # 示例1: 基本使用（使用config中的默认配置）
    client = GeminiClient(save_debug_images=False)
    result = client.analyze_image("path/to/image.jpg", "描述这张图片")
    print(result)
    
    # 示例2: 指定服务名称和环境
    client = GeminiClient(
        service_name="mediak8s-editprompt-qwen235b",
        env="prod",
        model_name="/workspace/Qwen3-VL-235B-A22B-Instruct",
        save_debug_images=False
    )
    
    # 示例3: 开启调试模式
    client_debug = GeminiClient(save_debug_images=True)
    
    # 使用不同输入类型
    result1 = client_debug.analyze_image("path/to/image.jpg", "描述这张图片", context="test1")
    
    # 示例4: 图片筛选
    filter_result = client_debug.filter_image(
        "path/to/image.jpg",
        criteria_description="图片必须清晰，包含人物",
        question="人物识别"
    )
    print(f"筛选结果: {filter_result}")