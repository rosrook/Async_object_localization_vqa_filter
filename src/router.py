# """
# 数据分流器
# 根据图片特征将图片分流到不同的pipeline
# """
# from pathlib import Path
# from typing import Dict, List, Optional, Any
# from enum import Enum
# import sys
# import os

# # 添加项目根目录到路径以导入utils模块
# _project_root = Path(__file__).parent.parent.parent
# if str(_project_root) not in sys.path:
#     sys.path.insert(0, str(_project_root))

# from utils.data_matcher import match_data
# import config


# class PipelineType(Enum):
#     """Pipeline类型枚举"""
#     QUESTION = "question"
#     CAPTION = "caption"
#     # 可以在这里添加更多pipeline类型


# class Router:
#     """数据分流器"""
    
#     def __init__(self):
#         """初始化分流器"""
#         # 从config读取pipeline配置中的问题文本，用于匹配
#         self.question_patterns = {
#             PipelineType.QUESTION: config.PIPELINE_CONFIG["question"]["question"],
#             PipelineType.CAPTION: config.PIPELINE_CONFIG["caption"]["question"]
#         }
    
#     def _match_pipeline_by_question(self, question: Optional[str]) -> List[PipelineType]:
#         """
#         根据question字段匹配到对应的pipeline
        
#         匹配规则：
#         - 包含 "Which term matches the picture" -> PipelineType.QUESTION
#         - 包含 "Which one is the correct caption" -> PipelineType.CAPTION
        
#         Args:
#             question: 问题文本
            
#         Returns:
#             匹配到的pipeline类型列表
#         """
#         if not question:
#             # 如果没有question，返回空列表
#             return []
        
#         question = question.strip()
#         matched_pipelines = []
        
#         # 匹配QUESTION pipeline: "Which term matches the picture?"
#         question_pattern = self.question_patterns[PipelineType.QUESTION]
#         if question == question_pattern or "Which term matches the picture" in question:
#             matched_pipelines.append(PipelineType.QUESTION)
        
#         # 匹配CAPTION pipeline: "Which one is the correct caption of this image?"
#         caption_pattern = self.question_patterns[PipelineType.CAPTION]
#         caption_key = "Which one is the correct caption"
#         if question == caption_pattern or caption_key in question:
#             matched_pipelines.append(PipelineType.CAPTION)
        
#         return matched_pipelines
    
#     def route(self, image_input: Path, metadata: Optional[Dict] = None) -> List[PipelineType]:
#         """
#         将图片分流到对应的pipeline
        
#         Args:
#             image_input: 图片路径
#             metadata: 可选的图片元数据，可以包含：
#                 - question: 问题文本，用于匹配pipeline
#                 - source_b: 包含question字段的字典
            
#         Returns:
#             应该使用的pipeline类型列表（一个图片可能进入多个pipeline）
#         """
#         # 优先从metadata中获取question
#         question = None
#         if metadata:
#             # 方式: 从source_b中获取question字段
#             if "source_b" in metadata and isinstance(metadata["source_b"], dict):
#                 question = metadata["source_b"].get("question")
        
#         # 根据question匹配pipeline
#         if question:
#             matched = self._match_pipeline_by_question(question)
#             if matched:
#                 return matched
        
#         # 如果没有匹配到或没有question，返回所有pipeline类型（向后兼容）
#         print("No matching pipeline!")
#         return [PipelineType.QUESTION, PipelineType.CAPTION]
    
#     def route_single(self, image_input: Path, metadata: Optional[Dict] = None) -> PipelineType:
#         """
#         将图片分流到单个pipeline（如果只需要一个pipeline）
        
#         Args:
#             image_input: 图片路径
#             metadata: 可选的图片元数据
            
#         Returns:
#             应该使用的pipeline类型
#         """
#         routes = self.route(image_input, metadata)
#         return routes[0] if routes else PipelineType.QUESTION
    
#     @staticmethod
#     def route_from_json(json_file: Path) -> List[Dict[str, Any]]:
#         """
#         从JSON文件中读取数据并根据question字段路由到对应的pipeline
        
#         Args:
#             json_file: JSON文件路径，格式应该包含source_b字段，source_b中包含question字段
            
#         Returns:
#             路由结果列表，每个元素包含：
#             - sample_index: 样本索引
#             - id: 样本ID
#             - image_input: 图片路径（从source_a或source_b中获取）
#             - question: 问题文本
#             - pipeline_types: 匹配到的pipeline类型列表
#             - source_a: source_a数据
#             - source_b: source_b数据
#         """
#         import json
        
#         if not json_file.exists():
#             raise FileNotFoundError(f"JSON file does not exist: {json_file}")
        
#         with open(json_file, 'r', encoding='utf-8') as f:
#             data = json.load(f)
        
#         if not isinstance(data, list):
#             raise ValueError(f"JSON file should contain a list, got {type(data)}")
        
#         router = Router()
#         results = []
        
#         for item in data:
#             source_b = item.get("source_b")
#             if not source_b:
#                 # 如果没有source_b，跳过或使用默认路由
#                 print("Detect No source b!")
#                 continue
            
#             question = source_b.get("question")
#             if not question:
#                 continue
            
#             # 获取图片路径（优先使用source_a的image_input）
#             image_input = None
#             source_a = item.get("source_a", {})
#             source_b = item.get("source_b", {})

#             IMAGE_KEYS = [
#                 # 最常见
#                 "image_input",
#                 "image",
#                 "img",
#                 "picture",
#                 "pic",
#                 "jpg",
#                 "png",
#                 "jpeg",

#                 # base64 / 编码
#                 "image_base64",
#                 "img_base64",
#                 "base64",
#                 "image_b64",
#                 "img_b64",
#                 "b64",

#                 # 多模态 / 数据集常见
#                 "vision_input",
#                 "visual_input",
#                 "visual",
#                 "vision",
#                 "image_data",
#                 "img_data",

#                 # 其他可能出现的变体
#                 "image_input_path",
#                 "image_input_url",
#                 "image_source",
#                 "image_src",
#                 "img_src",
#             ]

#             # 先从 source_a 中找
#             for key in IMAGE_KEYS:
#                 if key in source_a and source_a[key]:
#                     print("Key name: ", key)
#                     image_input = source_a[key]
#                     break

#             # 如果 source_a 没有，再从 source_b 中找
#             if image_input is None:
#                 for key in IMAGE_KEYS:
#                     if key in source_b and source_b[key]:
#                         print("Key name: ", key)
#                         print("Warning: fallback to image in source_b")
#                         image_input = source_b[key]
#                         break
#             if image_input is None:
#                 print("Error! No image key found!")
            
#             # 路由匹配
#             pipeline_types = router._match_pipeline_by_question(question)
            
#             results.append({
#                 "sample_index": item.get("sample_index"),
#                 "id": item.get("id"),
#                 "image_input": image_input,
#                 "question": question,
#                 "pipeline_types": [pt.value for pt in pipeline_types],
#                 "source_a": source_a,
#                 "source_b": source_b
#             })
        
#         return results
    
#     @staticmethod
#     def match_benchmark_data(
#         recat_root: Optional[str] = None,
#         benchmark_file: Optional[str] = None,
#         output_dir: Optional[str] = None,
#         target_categories: Optional[list] = None,
#         test_mode: Optional[bool] = None,
#         test_samples: Optional[int] = None,
#         test_max_categories: Optional[int] = None
#     ) -> str:
#         """
#         数据匹配函数：从重分类文件夹中读取数据,与基准parquet文件匹配,并输出JSON格式
#         输出文件将作为下一步处理的依据
        
#         Args:
#             recat_root: 重分类后的根目录，如果为None则从config读取
#             benchmark_file: 基准parquet文件路径，如果为None则从config读取
#             output_dir: JSON输出目录，如果为None则从config读取
#             target_categories: 要处理的类别列表，格式: [(category, l2category), ...]，如果为None则处理所有类别
#             test_mode: 是否启用测试模式，如果为None则从config读取
#             test_samples: 测试模式下每个类别处理的样本数，如果为None则从config读取
#             test_max_categories: 测试模式下最多处理的类别数，如果为None则从config读取
            
#         Returns:
#             输出目录路径
            
#         Example:
#             # 使用默认配置
#             Router.match_benchmark_data()
            
#             # 指定配置
#             Router.match_benchmark_data(
#                 recat_root="/path/to/recat",
#                 benchmark_file="/path/to/benchmark.parquet",
#                 target_categories=[("object_localization", "finegrained_perception (instance-level)")],
#                 test_mode=True
#             )
#         """
#         # 使用提供的参数或从config读取
#         test_mode = test_mode if test_mode is not None else config.MATCH_TEST_MODE
#         test_samples = test_samples if test_samples is not None else config.MATCH_TEST_SAMPLES
#         test_max_categories = test_max_categories if test_max_categories is not None else config.MATCH_TEST_MAX_CATEGORIES
        
#         return match_data(
#             recat_root=recat_root,
#             benchmark_file=benchmark_file,
#             output_dir=output_dir,
#             target_categories=target_categories,
#             test_mode=test_mode,
#             test_samples=test_samples,
#             test_max_categories=test_max_categories
#         )







"""
数据分流器 - 基于语义相似度的路由匹配
"""
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import sys
import re

# 添加项目根目录到路径以导入utils模块
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from utils.data_matcher import match_data
from utils.gemini_client import GeminiClient
import config


class PipelineType(Enum):
    """Pipeline类型枚举"""
    QUESTION = "question"
    CAPTION = "caption"
    PLACE_RECOGNITION = "place_recognition"
    TEXT_ASSOCIATION = "text_association"
    OBJECT_PROPORTION = "object_proportion"
    OBJECT_POSITION = "object_position"
    OBJECT_ABSENCE = "object_absence"
    OBJECT_ORIENTATION = "object_orientation"
    OBJECT_COUNTING = "object_counting"


class Router:
    """数据分流器 - 使用多种匹配策略"""
    
    def __init__(self, use_llm: bool = True):
        """
        初始化分流器
        
        Args:
            use_llm: 是否使用LLM进行语义匹配（如果为False则使用关键词匹配）
        """
        self.use_llm = use_llm
        
        if use_llm:
            self.gemini_client = GeminiClient()
        
        # 为每个pipeline定义关键词和模式（作为备选方案或快速预筛选）
        self.pipeline_keywords = {
            PipelineType.QUESTION: {
                "keywords": ["term", "matches", "concept", "which term"],
                "patterns": [r"which\s+term\s+matches", r"what\s+term\s+best\s+describes"]
            },
            PipelineType.CAPTION: {
                "keywords": ["caption", "description", "describe", "correct caption"],
                "patterns": [r"which.*correct\s+caption", r"best\s+caption", r"describe.*image"]
            },
            PipelineType.PLACE_RECOGNITION: {
                "keywords": ["place", "location", "name of the place"],
                "patterns": [r"name\s+of.*place", r"what.*place.*shown", r"identify.*location"]
            },
            PipelineType.TEXT_ASSOCIATION: {
                "keywords": ["associated text", "twitter", "social media", "caption for"],
                "patterns": [r"associated\s+text", r"text.*with.*image", r"posted\s+on"]
            },
            PipelineType.OBJECT_PROPORTION: {
                "keywords": ["proportion", "percentage", "how much", "occupied by"],
                "patterns": [r"what\s+proportion", r"how\s+much.*occupied", r"percentage.*image"]
            },
            PipelineType.OBJECT_POSITION: {
                "keywords": ["where", "located", "position", "location of"],
                "patterns": [r"where\s+is.*located", r"position\s+of", r"location\s+in"]
            },
            PipelineType.OBJECT_ABSENCE: {
                "keywords": ["doesn't have", "without", "no", "absent", "which corner"],
                "patterns": [r"doesn't\s+have", r"which.*no\s+", r"corner.*without"]
            },
            PipelineType.OBJECT_ORIENTATION: {
                "keywords": ["direction", "facing", "oriented", "which way"],
                "patterns": [r"which\s+direction.*facing", r"facing\s+which", r"oriented\s+towards"]
            },
            PipelineType.OBJECT_COUNTING: {
                "keywords": ["how many", "count", "number of"],
                "patterns": [r"how\s+many", r"count.*in", r"number\s+of"]
            }
        }
    
    def _match_by_keywords(self, question: str) -> List[PipelineType]:
        """
        使用关键词和正则表达式匹配pipeline
        
        Args:
            question: 问题文本
            
        Returns:
            匹配到的pipeline类型列表
        """
        question_lower = question.lower().strip()
        matched_pipelines = []
        scores = {}
        
        for pipeline_type, config in self.pipeline_keywords.items():
            score = 0
            
            # 关键词匹配
            for keyword in config["keywords"]:
                if keyword.lower() in question_lower:
                    score += 1
            
            # 正则表达式匹配
            for pattern in config["patterns"]:
                if re.search(pattern, question_lower, re.IGNORECASE):
                    score += 2  # 正则匹配权重更高
            
            if score > 0:
                scores[pipeline_type] = score
        
        # 按分数排序，返回得分最高的
        if scores:
            max_score = max(scores.values())
            matched_pipelines = [pt for pt, score in scores.items() if score == max_score]
        
        return matched_pipelines
    
    def _match_by_llm(self, question: str) -> List[PipelineType]:
        """
        使用LLM进行语义匹配
        
        Args:
            question: 问题文本
            
        Returns:
            匹配到的pipeline类型列表
        """
        # 构建pipeline描述
        pipeline_descriptions = {}
        for pipeline_type in PipelineType:
            if pipeline_type.value in config.PIPELINE_CONFIG:
                pipeline_config = config.PIPELINE_CONFIG[pipeline_type.value]
                pipeline_descriptions[pipeline_type.value] = {
                    "name": pipeline_config.get("name"),
                    "description": pipeline_config.get("description"),
                    "example_question": pipeline_config.get("question")
                }
        
        prompt = f"""You are a question classifier. Given a question about an image, classify it into one of the following categories.

Available Categories:
{self._format_pipeline_descriptions(pipeline_descriptions)}

Question to classify: "{question}"

Analyze the question and determine which category it belongs to. Consider:
1. The main intent of the question (what is being asked)
2. The type of visual information needed to answer it
3. The similarity to example questions

Return ONLY a JSON object with this format:
{{
    "pipeline_type": "the matching category key (e.g., 'question', 'caption', etc.)",
    "confidence": a float between 0.0-1.0,
    "reasoning": "brief explanation of why this category matches"
}}

Return only the JSON, no other text."""

        try:
            response = self.gemini_client.analyze_image(
                image_input="",  # 不需要图片，只分析文本
                prompt=prompt,
                temperature=0.3
            )
            
            import json
            # 提取JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                pipeline_type_str = result.get("pipeline_type")
                confidence = result.get("confidence", 0.5)
                
                # 转换为PipelineType
                try:
                    pipeline_type = PipelineType(pipeline_type_str)
                    if confidence >= 0.6:  # 置信度阈值
                        return [pipeline_type]
                except ValueError:
                    pass
            
        except Exception as e:
            print(f"LLM matching failed: {e}")
        
        return []
    
    def _format_pipeline_descriptions(self, descriptions: Dict) -> str:
        """格式化pipeline描述用于LLM prompt"""
        lines = []
        for key, info in descriptions.items():
            lines.append(f"\n{key}:")
            lines.append(f"  Name: {info['name']}")
            lines.append(f"  Description: {info['description']}")
            lines.append(f"  Example: {info['example_question']}")
        return "\n".join(lines)
    
    def _match_pipeline_by_question(self, question: Optional[str]) -> List[PipelineType]:
        """
        根据question字段匹配到对应的pipeline
        
        Args:
            question: 问题文本
            
        Returns:
            匹配到的pipeline类型列表
        """
        if not question:
            return []
        
        question = question.strip()
        
        # 策略1: 先使用关键词快速匹配
        keyword_matches = self._match_by_keywords(question)
        
        if keyword_matches and len(keyword_matches) == 1:
            # 如果关键词匹配唯一，直接返回
            return keyword_matches
        
        # 策略2: 如果关键词匹配不明确或没有匹配，使用LLM
        if self.use_llm:
            llm_matches = self._match_by_llm(question)
            if llm_matches:
                return llm_matches
        
        # 策略3: 如果LLM也失败，返回关键词匹配结果（即使可能有多个）
        if keyword_matches:
            return keyword_matches
        
        return []
    
    def route(self, image_input: Union[str, Path], metadata: Optional[Dict] = None) -> List[PipelineType]:
        """
        将图片分流到对应的pipeline
        
        Args:
            image_input: 图片路径
            metadata: 可选的图片元数据，可以包含：
                - question: 问题文本，用于匹配pipeline
                - source_b: 包含question字段的字典
            
        Returns:
            应该使用的pipeline类型列表
        """
        # 优先从metadata中获取question
        question = None
        if metadata:
            # 方式1: 直接从metadata获取
            if "question" in metadata:
                question = metadata["question"]
            # 方式2: 从source_b中获取question字段
            elif "source_b" in metadata and isinstance(metadata["source_b"], dict):
                question = metadata["source_b"].get("question")
        
        # 根据question匹配pipeline
        if question:
            matched = self._match_pipeline_by_question(question)
            if matched:
                return matched
            else:
                print(f"Warning: No pipeline matched for question: {question}")
        
        # 如果没有匹配到，返回空列表或默认pipeline
        return []
    
    def route_single(self, image_input: Union[str, Path], metadata: Optional[Dict] = None) -> Optional[PipelineType]:
        """
        将图片分流到单个pipeline（如果只需要一个pipeline）
        
        Args:
            image_input: 图片路径
            metadata: 可选的图片元数据
            
        Returns:
            应该使用的pipeline类型，如果没有匹配返回None
        """
        routes = self.route(image_input, metadata)
        return routes[0] if routes else None
    
    @staticmethod
    def route_from_json(json_file: Path, use_llm: bool = True) -> List[Dict[str, Any]]:
        """
        从JSON文件中读取数据并根据question字段路由到对应的pipeline
        
        Args:
            json_file: JSON文件路径
            use_llm: 是否使用LLM进行语义匹配
            
        Returns:
            路由结果列表
        """
        import json
        
        if not json_file.exists():
            raise FileNotFoundError(f"JSON file does not exist: {json_file}")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError(f"JSON file should contain a list, got {type(data)}")
        
        router = Router(use_llm=use_llm)
        results = []
        
        for item in data:
            source_b = item.get("source_b")
            if not source_b:
                print("Warning: No source_b found, skipping item")
                continue
            
            question = source_b.get("question")
            if not question:
                print("Warning: No question found in source_b, skipping item")
                continue
            
            # 获取图片路径
            image_input = Router._extract_image_input(item)
            if image_input is None:
                print("Error: No image input found, skipping item")
                continue
            
            # 路由匹配
            pipeline_types = router._match_pipeline_by_question(question)
            
            results.append({
                "sample_index": item.get("sample_index"),
                "id": item.get("id"),
                "image_input": image_input,
                "question": question,
                "pipeline_types": pipeline_types,  # 直接保存PipelineType对象
                "source_a": item.get("source_a", {}),
                "source_b": source_b
            })
        
        return results
    
    @staticmethod
    def _extract_image_input(item: Dict) -> Optional[str]:
        """从数据项中提取图片路径"""
        IMAGE_KEYS = [
            "image_input", "image", "img", "picture", "pic",
            "jpg", "png", "jpeg",
            "image_base64", "img_base64", "base64",
            "image_b64", "img_b64", "b64",
            "vision_input", "visual_input", "visual", "vision",
            "image_data", "img_data",
            "image_input_path", "image_input_url",
            "image_source", "image_src", "img_src"
        ]
        
        source_a = item.get("source_a", {})
        source_b = item.get("source_b", {})
        
        # 先从 source_a 中找
        for key in IMAGE_KEYS:
            if key in source_a and source_a[key]:
                return source_a[key]
        
        # 如果 source_a 没有，再从 source_b 中找
        for key in IMAGE_KEYS:
            if key in source_b and source_b[key]:
                print("Warning: Using image from source_b")
                return source_b[key]
        
        return None
    
    @staticmethod
    def match_benchmark_data(
        recat_root: Optional[str] = None,
        benchmark_file: Optional[str] = None,
        output_dir: Optional[str] = None,
        target_categories: Optional[list] = None,
        test_mode: Optional[bool] = None,
        test_samples: Optional[int] = None,
        test_max_categories: Optional[int] = None
    ) -> str:
        """数据匹配函数"""
        test_mode = test_mode if test_mode is not None else config.MATCH_TEST_MODE
        test_samples = test_samples if test_samples is not None else config.MATCH_TEST_SAMPLES
        test_max_categories = test_max_categories if test_max_categories is not None else config.MATCH_TEST_MAX_CATEGORIES
        
        return match_data(
            recat_root=recat_root,
            benchmark_file=benchmark_file,
            output_dir=output_dir,
            target_categories=target_categories,
            test_mode=test_mode,
            test_samples=test_samples,
            test_max_categories=test_max_categories
        )