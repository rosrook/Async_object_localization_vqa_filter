"""
VQA问题生成系统主模块
实现完整的6步流程
"""
import json
import re
import random
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from .config_loader import ConfigLoader
from .object_selector import ObjectSelector
from .slot_filler import SlotFiller
from .question_generator import QuestionGenerator
from .validator import QuestionValidator
from utils.gemini_client import GeminiClient


class VQAGenerator:
    """VQA问题生成器主类"""
    
    def __init__(self, config_path: Path, gemini_client: Optional[GeminiClient] = None):
        """
        初始化VQA生成器
        
        Args:
            config_path: 配置文件路径
            gemini_client: Gemini客户端实例（可选）
        """
        self.config_loader = ConfigLoader(config_path)
        self.gemini_client = gemini_client or GeminiClient()
        
        # 初始化各个模块
        self.object_selector = ObjectSelector(self.gemini_client)
        self.slot_filler = SlotFiller(self.gemini_client)
        self.question_generator = QuestionGenerator(self.gemini_client)
        self.validator = QuestionValidator(self.gemini_client)
        
        # 获取策略配置
        self.global_constraints = self.config_loader.get_global_constraints()
        self.object_selection_policy = self.config_loader.get_object_selection_policy()
        self.generation_policy = self.config_loader.get_generation_policy()
        self.question_type_ratio = self.config_loader.get_question_type_ratio()
    
    def process_image_pipeline_pair(
        self,
        image_input: Any,
        pipeline_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        处理单个图片-pipeline对，生成VQA问题
        
        严格遵循6步流程：
        1. 加载Pipeline规范
        2. 对象选择（如果需要）
        3. 槽位填充
        4. 问题生成
        5. 验证
        6. 输出
        
        Args:
            image_input: 图片输入（路径、base64、bytes等）
            pipeline_name: Pipeline名称
            metadata: 可选的元数据
            
        Returns:
            (成功结果, 错误/丢弃信息)
            如果成功: (结果字典, None)
            如果失败: (None, 错误信息字典)
        """
        error_info = {
            "pipeline_name": pipeline_name,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
            "error_stage": None,
            "error_reason": None
        }
        
        try:
            # STEP 1: 加载Pipeline规范
            pipeline_config = self.config_loader.get_pipeline_config(pipeline_name)
            if not pipeline_config:
                error_info["error_stage"] = "config_loading"
                error_info["error_reason"] = f"Pipeline '{pipeline_name}' 不存在"
                print(f"[WARNING] {error_info['error_reason']}，跳过")
                return None, error_info
            
            # STEP 2: 对象选择（如果需要）
            selected_object = None
            if pipeline_config.get("object_grounding"):
                try:
                    selected_object = self.object_selector.select_object(
                        image_input=image_input,
                        pipeline_config=pipeline_config,
                        global_policy=self.object_selection_policy
                    )
                    
                    if selected_object is None:
                        # 根据策略，如果对象选择失败则丢弃
                        if self.object_selection_policy.get("fallback_strategy") == "discard_image":
                            error_info["error_stage"] = "object_selection"
                            error_info["error_reason"] = "无法选择对象"
                            print(f"[INFO] 无法为pipeline '{pipeline_name}' 选择对象，丢弃样本")
                            return None, error_info
                except Exception as e:
                    error_info["error_stage"] = "object_selection"
                    error_info["error_reason"] = f"对象选择过程出错: {str(e)}"
                    print(f"[ERROR] {error_info['error_reason']}")
                    return None, error_info
            
            # STEP 3: 槽位填充
            try:
                slots = self.slot_filler.fill_slots(
                    image_input=image_input,
                    pipeline_config=pipeline_config,
                    selected_object=selected_object
                )
                
                if slots is None:
                    error_info["error_stage"] = "slot_filling"
                    error_info["error_reason"] = "槽位填充失败（必需槽位无法解析）"
                    print(f"[INFO] 槽位填充失败，丢弃样本")
                    return None, error_info
            except Exception as e:
                error_info["error_stage"] = "slot_filling"
                error_info["error_reason"] = f"槽位填充过程出错: {str(e)}"
                print(f"[ERROR] {error_info['error_reason']}")
                return None, error_info
            
            # STEP 4: 问题生成
            # 按比例选择题型（在try块外初始化，确保在结果中可用）
            question_type = self._select_question_type()
            
            try:
                question = self.question_generator.generate_question(
                    image_input=image_input,
                    pipeline_config=pipeline_config,
                    slots=slots,
                    selected_object=selected_object,
                    question_type=question_type
                )
                
                if not question:
                    error_info["error_stage"] = "question_generation"
                    error_info["error_reason"] = "问题生成失败（返回空）"
                    print(f"[INFO] 问题生成失败，丢弃样本")
                    return None, error_info
            except Exception as e:
                error_info["error_stage"] = "question_generation"
                error_info["error_reason"] = f"问题生成过程出错: {str(e)}"
                print(f"[ERROR] {error_info['error_reason']}")
                return None, error_info
            
            # STEP 5: 验证
            try:
                is_valid, reason = self.validator.validate(
                    question=question,
                    image_input=image_input,
                    pipeline_config=pipeline_config,
                    global_constraints=self.global_constraints
                )
                
                if not is_valid:
                    error_info["error_stage"] = "validation"
                    error_info["error_reason"] = f"问题验证失败: {reason}"
                    print(f"[INFO] 问题验证失败: {reason}，丢弃样本")
                    return None, error_info
            except Exception as e:
                error_info["error_stage"] = "validation"
                error_info["error_reason"] = f"验证过程出错: {str(e)}"
                print(f"[ERROR] {error_info['error_reason']}")
                return None, error_info
            
            # STEP 6: 输出
            result = {
                "pipeline_name": pipeline_name,
                "pipeline_intent": pipeline_config.get("intent", ""),
                "question": question,
                "question_type": question_type,  # 添加题型字段
                "answer_type": pipeline_config.get("answer_type", ""),
                "slots": slots,
                "selected_object": selected_object,
                "validation_reason": reason,
                "timestamp": datetime.now().isoformat()
            }
            
            return result, None
            
        except Exception as e:
            error_info["error_stage"] = "unknown"
            error_info["error_reason"] = f"未知错误: {str(e)}"
            print(f"[ERROR] {error_info['error_reason']}")
            return None, error_info
    
    def process_data_file(
        self,
        input_file: Path,
        output_file: Path,
        pipeline_names: Optional[List[str]] = None,
        max_samples: Optional[int] = None
    ) -> None:
        """
        处理数据文件，为每张图片生成VQA问题
        
        Args:
            input_file: 输入JSON文件路径（batch_process.sh的输出）
            output_file: 输出JSON文件路径
            pipeline_names: 要使用的pipeline列表（None表示使用所有）
            max_samples: 最大处理样本数（None表示全部）
        """
        print(f"[INFO] 读取输入文件: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError(f"输入文件应该包含一个数组，但得到: {type(data)}")
        
        # 确定要使用的pipeline
        if pipeline_names is None:
            pipeline_names = self.config_loader.list_pipelines()
        
        print(f"[INFO] 使用pipelines: {pipeline_names}")
        print(f"[INFO] 总记录数: {len(data)}")
        
        if max_samples:
            data = data[:max_samples]
            print(f"[INFO] 限制处理前 {max_samples} 条记录")
        
        # 处理每条记录
        results = []
        errors = []  # 收集所有错误和丢弃的数据
        total_processed = 0
        total_discarded = 0
        
        for idx, record in enumerate(data, 1):
            source_a = record.get("source_a", {})
            if not source_a:
                error_info = {
                    "record_index": idx,
                    "id": record.get("id"),
                    "error_stage": "data_loading",
                    "error_reason": "记录没有source_a",
                    "timestamp": datetime.now().isoformat()
                }
                errors.append(error_info)
                print(f"[WARNING] 记录 {idx} 没有source_a，跳过")
                continue
            
            # 提取图片输入
            image_input = self._extract_image_input(source_a)
            if image_input is None:
                error_info = {
                    "record_index": idx,
                    "id": record.get("id"),
                    "source_a_id": source_a.get("id"),
                    "error_stage": "data_loading",
                    "error_reason": "无法提取图片输入",
                    "timestamp": datetime.now().isoformat()
                }
                errors.append(error_info)
                print(f"[WARNING] 记录 {idx} 无法提取图片，跳过")
                continue
            
            # 确定该记录应该使用的pipeline
            # 优先从记录中读取pipeline_type或pipeline_name
            record_pipeline = self._extract_pipeline_from_record(record)
            
            # 如果记录中指定了pipeline，使用指定的；否则使用传入的pipeline_names
            if record_pipeline:
                pipelines_to_use = [record_pipeline]
                print(f"[INFO] 记录 {idx} 使用指定的pipeline: {record_pipeline}")
            else:
                # 如果记录中没有指定pipeline，使用传入的pipeline_names（如果指定了）
                # 如果没有传入pipeline_names，则使用所有pipeline（向后兼容）
                pipelines_to_use = pipeline_names if pipeline_names else self.config_loader.list_pipelines()
                if not pipeline_names:
                    print(f"[WARNING] 记录 {idx} 未指定pipeline，且未传入pipeline_names，将使用所有pipeline: {pipelines_to_use}")
                else:
                    print(f"[INFO] 记录 {idx} 未指定pipeline，使用传入的pipeline_names: {pipelines_to_use}")
            
            # 为确定的pipeline生成问题
            for pipeline_name in pipelines_to_use:
                total_processed += 1
                
                result, error_info = self.process_image_pipeline_pair(
                    image_input=image_input,
                    pipeline_name=pipeline_name,
                    metadata={"record_index": idx, "id": record.get("id")}
                )
                
                if result:
                    # 添加原始数据信息
                    result["sample_index"] = record.get("sample_index")
                    result["id"] = record.get("id")
                    result["source_a_id"] = source_a.get("id")
                    # 添加图片的base64编码
                    image_base64 = self._extract_image_base64(source_a, image_input)
                    if image_base64:
                        result["image_base64"] = image_base64
                    results.append(result)
                else:
                    total_discarded += 1
                    # 收集错误信息
                    if error_info:
                        error_info["sample_index"] = record.get("sample_index")
                        error_info["id"] = record.get("id")
                        error_info["source_a_id"] = source_a.get("id")
                        errors.append(error_info)
                
                # 进度报告
                if total_processed % 10 == 0:
                    print(f"[进度] 已处理: {total_processed}, 成功: {len(results)}, 丢弃: {total_discarded}")
        
        # 保存成功结果
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 保存错误和丢弃的数据（带时间戳）
        if errors:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            error_file = output_file.parent / f"{output_file.stem}_errors_{timestamp}.json"
            with open(error_file, 'w', encoding='utf-8') as f:
                json.dump(errors, f, ensure_ascii=False, indent=2)
            print(f"  错误/丢弃数据已保存到: {error_file}")
        
        print(f"\n[完成] 处理完成！")
        print(f"  总处理: {total_processed}")
        print(f"  成功生成: {len(results)}")
        print(f"  丢弃/错误: {total_discarded}")
        print(f"  结果已保存到: {output_file}")
    
    def _extract_image_input(self, source_a: Dict[str, Any]) -> Optional[Any]:
        """
        从source_a中提取图片输入
        
        Args:
            source_a: source_a数据字典
            
        Returns:
            图片输入（base64字符串、路径等），如果无法提取返回None
        """
        # 可能的图片字段名
        image_keys = [
            "image_input", "image", "img", "picture", "pic",
            "image_base64", "img_base64", "base64", "image_b64",
            "vision_input", "visual_input", "image_data", "jpg"
        ]
        
        for key in image_keys:
            if key in source_a and source_a[key]:
                return source_a[key]
        
        return None
    
    def _extract_pipeline_from_record(self, record: Dict[str, Any]) -> Optional[str]:
        """
        从记录中提取pipeline信息
        
        支持以下字段：
        - pipeline_type: 如 "object_counting"
        - pipeline_name: 如 "Object Counting Pipeline"
        
        Args:
            record: 输入记录
            
        Returns:
            pipeline名称（配置文件中使用的名称），如果未找到返回None
        """
        # 优先使用pipeline_type（直接对应配置中的pipeline名称）
        pipeline_type = record.get("pipeline_type")
        if pipeline_type:
            # 验证pipeline是否存在
            available_pipelines = self.config_loader.list_pipelines()
            if pipeline_type in available_pipelines:
                return pipeline_type
            else:
                print(f"[WARNING] pipeline_type '{pipeline_type}' 不在可用pipeline列表中，可用: {available_pipelines}")
        
        # 如果没有pipeline_type，尝试从pipeline_name映射
        pipeline_name = record.get("pipeline_name")
        if pipeline_name:
            # 将pipeline_name映射到pipeline_type
            # 例如: "Object Counting Pipeline" -> "object_counting"
            pipeline_mapping = self._map_pipeline_name_to_type(pipeline_name)
            if pipeline_mapping:
                return pipeline_mapping
        
        # 也可以从source_a或source_b中查找
        source_a = record.get("source_a", {})
        if source_a:
            pipeline_type = source_a.get("pipeline_type")
            if pipeline_type:
                available_pipelines = self.config_loader.list_pipelines()
                if pipeline_type in available_pipelines:
                    return pipeline_type
        
        source_b = record.get("source_b", {})
        if source_b:
            pipeline_type = source_b.get("pipeline_type")
            if pipeline_type:
                available_pipelines = self.config_loader.list_pipelines()
                if pipeline_type in available_pipelines:
                    return pipeline_type
        
        return None
    
    def _map_pipeline_name_to_type(self, pipeline_name: str) -> Optional[str]:
        """
        将pipeline_name映射到pipeline_type
        
        Args:
            pipeline_name: Pipeline名称（如 "Object Counting Pipeline"）
            
        Returns:
            pipeline_type（如 "object_counting"），如果无法映射返回None
        """
        # 获取所有pipeline配置
        available_pipelines = self.config_loader.list_pipelines()
        
        # 尝试精确匹配
        for pipeline_type in available_pipelines:
            pipeline_config = self.config_loader.get_pipeline_config(pipeline_type)
            if pipeline_config:
                config_name = pipeline_config.get("name", "")
                if config_name == pipeline_name:
                    return pipeline_type
        
        # 尝试模糊匹配（基于关键词）
        pipeline_name_lower = pipeline_name.lower()
        name_mapping = {
            "object counting": "object_counting",
            "object recognition": "question",
            "question": "question",
            "object position": "object_position",
            "object proportion": "object_proportion",
            "object orientation": "object_orientation",
            "object absence": "object_absence",
            "place recognition": "place_recognition",
            "text association": "text_association",
            "caption": "caption"
        }
        
        for key, pipeline_type in name_mapping.items():
            if key in pipeline_name_lower and pipeline_type in available_pipelines:
                return pipeline_type
        
        return None
    
    def _select_question_type(self) -> str:
        """
        根据配置的比例选择题型
        
        Returns:
            "multiple_choice" 或 "fill_in_blank"
        """
        rand = random.random()
        if rand < self.question_type_ratio["multiple_choice"]:
            return "multiple_choice"
        else:
            return "fill_in_blank"
    
    def _extract_image_base64(self, source_a: Dict[str, Any], image_input: Any) -> Optional[str]:
        """
        从source_a中提取图片的base64编码
        
        优先顺序：
        1. source_a中的image_base64字段
        2. 如果image_input是base64字符串，直接使用
        3. 其他可能的base64字段
        
        Args:
            source_a: source_a数据字典
            image_input: 已提取的图片输入
            
        Returns:
            base64编码的字符串，如果无法提取返回None
        """
        # 优先从source_a中查找image_base64字段
        base64_keys = [
            "image_base64", "img_base64", "base64", "image_b64", "img_b64"
        ]
        
        for key in base64_keys:
            if key in source_a and source_a[key]:
                value = source_a[key]
                if isinstance(value, str) and len(value) > 50:
                    # 简单验证：base64字符串通常较长
                    # 移除可能的数据URL前缀
                    if value.startswith("data:image"):
                        # 提取base64部分: data:image/jpeg;base64,xxxxx
                        match = re.search(r'base64,(.+)', value)
                        if match:
                            return match.group(1)
                        return value
                    return value
        
        # 如果image_input是base64字符串，使用它
        if isinstance(image_input, str):
            # 检查是否是base64字符串（长度较长，只包含base64字符）
            if len(image_input) > 50:
                # 移除可能的数据URL前缀
                if image_input.startswith("data:image"):
                    match = re.search(r'base64,(.+)', image_input)
                    if match:
                        return match.group(1)
                    return image_input
                # 简单验证：检查是否可能是base64（只包含base64字符）
                base64_pattern = re.compile(r'^[A-Za-z0-9+/=]+$')
                if base64_pattern.match(image_input):
                    return image_input
        
        return None

