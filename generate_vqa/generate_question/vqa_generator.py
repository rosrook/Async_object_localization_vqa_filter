"""
VQA问题生成系统主模块
实现完整的6步流程
"""
import json
import re
import random
import asyncio
import base64
import io
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from PIL import Image

from .config_loader import ConfigLoader
from .object_selector import ObjectSelector
from .slot_filler import SlotFiller
from .question_generator import QuestionGenerator
from .validator import QuestionValidator
from utils.gemini_client import GeminiClient
from utils.async_client import AsyncGeminiClient


class VQAGenerator:
    """VQA问题生成器主类"""
    
    def __init__(self, config_path: Path, gemini_client: Optional[GeminiClient] = None,
                 failed_selection_dir: Optional[Path] = None):
        """
        初始化VQA生成器
        
        Args:
            config_path: 配置文件路径
            gemini_client: Gemini客户端实例（可选）
            failed_selection_dir: 失败案例存储目录（可选，如果为None则不保存）
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
        
        # 失败案例存储目录
        self.failed_selection_dir = failed_selection_dir
        if self.failed_selection_dir:
            self.failed_selection_dir.mkdir(parents=True, exist_ok=True)
    
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
                    
                    # 检查对象选择是否失败（包括返回None或selected=False）
                    if selected_object is not None:
                    
                    if selected_object is None or selected_object.get("selected") == False:
                        # 根据策略，如果对象选择失败则丢弃
                        if self.object_selection_policy.get("fallback_strategy") == "discard_image":
                            error_info["error_stage"] = "object_selection"
                            # 从对象选择器的响应中获取reason（如果存在）
                            if selected_object and selected_object.get("reason"):
                                error_info["error_reason"] = f"无法选择对象: {selected_object.get('reason')}"
                                error_info["model_reason"] = selected_object.get("reason")  # 保存模型的reason
                                error_info["confidence"] = selected_object.get("confidence", 0.0)
                            else:
                                error_info["error_reason"] = "无法选择对象"
                            
                            # 保存失败案例
                            if self.failed_selection_dir:
                                self._save_failed_selection_case(
                                    image_input=image_input,
                                    pipeline_config=pipeline_config,
                                    error_info=error_info,
                                    metadata=metadata,
                                    selection_result=selected_object  # 传递选择结果以获取reason
                                )
                            else:
                                print(f"[WARNING] 跳过保存失败案例，failed_selection_dir 为 None")
                            
                            return None, error_info
                    else:
                except Exception as e:
                    error_info["error_stage"] = "object_selection"
                    error_info["error_reason"] = f"对象选择过程出错: {str(e)}"
                    
                    # 保存失败案例
                    if self.failed_selection_dir:
                        self._save_failed_selection_case(
                            image_input=image_input,
                            pipeline_config=pipeline_config,
                            error_info=error_info,
                            metadata=metadata,
                            selection_result=None  # 异常情况下没有选择结果
                        )
                    else:
                    
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
                    return None, error_info
            except Exception as e:
                error_info["error_stage"] = "question_generation"
                error_info["error_reason"] = f"问题生成过程出错: {str(e)}"
                print(f"[ERROR] {error_info['error_reason']}")
                return None, error_info
            
            # STEP 5: 输出（跳过验证以提高效率）
            result = {
                "pipeline_name": pipeline_name,
                "pipeline_intent": pipeline_config.get("intent", ""),
                "question": question,
                "question_type": question_type,  # 添加题型字段
                "answer_type": pipeline_config.get("answer_type", ""),
                "slots": slots,
                "selected_object": selected_object,
                "validation_reason": "skipped",
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
        max_samples: Optional[int] = None,
        failed_selection_dir: Optional[Path] = None
    ) -> None:
        """
        处理数据文件，为每张图片生成VQA问题
        
        Args:
            input_file: 输入JSON文件路径（batch_process.sh的输出）
            output_file: 输出JSON文件路径
            pipeline_names: 要使用的pipeline列表（None表示使用所有）
            max_samples: 最大处理样本数（None表示全部）
            failed_selection_dir: 失败案例存储目录（可选，如果为None且self.failed_selection_dir也为None则不保存）
        """
        # 如果传入了failed_selection_dir，使用它；否则使用初始化时设置的
        if failed_selection_dir is not None:
            self.failed_selection_dir = failed_selection_dir
            self.failed_selection_dir.mkdir(parents=True, exist_ok=True)
        
        # 如果没有设置失败案例目录，使用输出目录的子目录
        if self.failed_selection_dir is None:
            self.failed_selection_dir = output_file.parent / "failed_selection"
            self.failed_selection_dir.mkdir(parents=True, exist_ok=True)
        
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError(f"输入文件应该包含一个数组，但得到: {type(data)}")
        
        # 确定要使用的pipeline
        if pipeline_names is None:
            pipeline_names = self.config_loader.list_pipelines()
        
        
        if max_samples:
            data = data[:max_samples]
        
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
            else:
                # 如果记录中没有指定pipeline，使用传入的pipeline_names（如果指定了）
                # 如果没有传入pipeline_names，则使用所有pipeline（向后兼容）
                pipelines_to_use = pipeline_names if pipeline_names else self.config_loader.list_pipelines()
                if not pipeline_names:
                    print(f"[WARNING] 记录 {idx} 未指定pipeline，且未传入pipeline_names，将使用所有pipeline: {pipelines_to_use}")
                else:
            
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
    
    def _save_failed_selection_case(
        self,
        image_input: Any,
        pipeline_config: Dict[str, Any],
        error_info: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        selection_result: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        保存对象选择失败的案例到子文件夹
        
        Args:
            image_input: 图片输入（base64、路径、bytes等）
            pipeline_config: Pipeline配置
            error_info: 错误信息
            metadata: 元数据
        """
        if not self.failed_selection_dir:
            return
        
        
        try:
            # 生成案例ID（基于时间戳和记录ID）
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 毫秒级时间戳
            record_id = metadata.get("id", "unknown") if metadata else "unknown"
            pipeline_name = pipeline_config.get("name", error_info.get("pipeline_name", "unknown"))
            case_id = f"{timestamp}_{record_id}"
            
            
            # 创建案例子目录
            case_dir = self.failed_selection_dir / case_id
            case_dir.mkdir(parents=True, exist_ok=True)
            
            # 1. 保存图片（JPG格式）
            image_path = case_dir / "image.jpg"
            try:
                image = self._load_image_from_input(image_input)
                if image:
                    image.save(image_path, "JPEG", quality=95)
                else:
                    print(f"[WARNING] 图片加载失败，返回None。image_input类型: {type(image_input)}")
                    # 尝试保存原始输入信息用于调试
                    debug_path = case_dir / "image_input_debug.txt"
                    try:
                        with open(debug_path, 'w', encoding='utf-8') as f:
                            f.write(f"image_input类型: {type(image_input)}\n")
                            if isinstance(image_input, str):
                                f.write(f"字符串长度: {len(image_input)}\n")
                                f.write(f"前200个字符: {image_input[:200]}\n")
                                f.write(f"是否以data:image开头: {image_input.startswith('data:image')}\n")
                                f.write(f"是否以/9j/开头（JPEG base64）: {image_input.startswith('/9j/')}\n")
                                # 检查是否只包含base64字符（不能在f-string中直接使用反斜杠）
                                base64_pattern_check = re.match(r'^[A-Za-z0-9+/=\s]+$', image_input)
                                f.write(f"是否只包含base64字符: {bool(base64_pattern_check)}\n")
                                # 尝试解码并检查
                                try:
                                    clean_base64 = image_input.strip().replace('\n', '').replace('\r', '').replace(' ', '')
                                    test_decode = base64.b64decode(clean_base64[:100] + '==')  # 测试解码前100个字符
                                    f.write(f"前100字符base64测试解码: 成功，长度={len(test_decode)} bytes\n")
                                except Exception as e:
                                    f.write(f"前100字符base64测试解码: 失败 - {type(e).__name__}: {e}\n")
                            else:
                                f.write(f"非字符串类型，值: {str(image_input)[:200]}\n")
                    except Exception as e:
                        print(f"[ERROR] 保存调试信息失败: {e}")
            except Exception as e:
                print(f"[ERROR] 保存失败案例图片时发生异常: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
            
            # 2. 保存错误信息JSON（包含模型的reason）
            error_json_path = case_dir / "error_info.json"
            error_data = {
                "case_id": case_id,
                "timestamp": datetime.now().isoformat(),
                "pipeline_name": pipeline_name,
                "pipeline_config": {
                    "intent": pipeline_config.get("intent"),
                    "description": pipeline_config.get("description"),
                    "object_grounding": pipeline_config.get("object_grounding")
                },
                "error_stage": error_info.get("error_stage"),
                "error_reason": error_info.get("error_reason"),
                "metadata": metadata or {}
            }
            
            # 如果selection_result存在，保存模型的reason和confidence
            if selection_result:
                if selection_result.get("reason"):
                    error_data["model_reason"] = selection_result.get("reason")
                    error_data["model_confidence"] = selection_result.get("confidence", 0.0)
                # 保存完整的选择结果
                error_data["selection_result"] = selection_result
            
            with open(error_json_path, 'w', encoding='utf-8') as f:
                json.dump(error_data, f, ensure_ascii=False, indent=2)
            
            # 3. 保存完整的Pipeline配置
            config_json_path = case_dir / "pipeline_config.json"
            with open(config_json_path, 'w', encoding='utf-8') as f:
                json.dump(pipeline_config, f, ensure_ascii=False, indent=2)
            
            # 4. 保存对象选择策略配置
            policy_json_path = case_dir / "selection_policy.json"
            policy_data = {
                "object_selection_policy": self.object_selection_policy,
                "global_constraints": self.global_constraints
            }
            with open(policy_json_path, 'w', encoding='utf-8') as f:
                json.dump(policy_data, f, ensure_ascii=False, indent=2)
            
            
        except Exception as e:
            print(f"[ERROR] 保存失败案例时出错: {e}")
            import traceback
            traceback.print_exc()
    
    def _load_image_from_input(self, image_input: Any) -> Optional[Image.Image]:
        """
        从各种格式的输入加载PIL图片
        
        Args:
            image_input: 图片输入（base64、路径、bytes、PIL.Image等）
            
        Returns:
            PIL Image对象，如果加载失败返回None
        """
        try:
            
            # 如果已经是PIL Image，直接返回
            if isinstance(image_input, Image.Image):
                return image_input.copy()
            
            # 如果是字符串或Path
            if isinstance(image_input, (str, Path)):
                # 先尝试作为文件路径
                if isinstance(image_input, str):
                    # 先判断是否可能是文件路径（路径通常较短，且不包含base64字符）
                    # 如果字符串很长（>500字符），或者以base64特征开头，直接作为base64处理
                    is_likely_base64 = (
                        len(image_input) > 500 or  # 文件路径很少超过500字符
                        image_input.startswith('/9j/') or  # JPEG base64开头
                        image_input.startswith('data:image') or  # data:image格式
                        image_input.startswith('iVBORw0KGgo') or  # PNG base64开头
                        (len(image_input) > 100 and all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=\n\r\t ' for c in image_input[:200]))  # 前200字符看起来像base64
                    )
                    
                    if not is_likely_base64:
                        # 只有看起来像文件路径时，才尝试作为路径处理
                        try:
                            path = Path(image_input)
                            # 检查是否是文件路径
                            if path.exists() and path.is_file():
                                return Image.open(path)
                        except (OSError, ValueError) as e:
                            # 如果创建Path对象失败（如文件名太长），跳过路径处理
                    
                    # 作为字符串处理（可能是base64）
                    if len(image_input) > 50:
                        image_data = None
                        # 检查是否是base64
                        if image_input.startswith("data:image"):
                            # 提取base64部分: data:image/jpeg;base64,xxxxx
                            match = re.search(r'base64,(.+)', image_input)
                            if match:
                                try:
                                    base64_str = match.group(1)
                                    image_data = base64.b64decode(base64_str)
                                except Exception as e:
                                    print(f"[ERROR] 解码data:image格式的base64失败: {type(e).__name__}: {e}")
                                    return None
                            else:
                                print(f"[WARNING] 无法从data:image格式中提取base64部分")
                                return None
                        else:
                            # 尝试直接解码base64（纯base64字符串）
                            image_data = None  # 初始化变量
                            try:
                                # 移除可能的空白字符、换行符等
                                clean_base64 = image_input.strip().replace('\n', '').replace('\r', '').replace(' ', '')
                                
                                # 先尝试不使用validate（更宽松）
                                try:
                                    image_data = base64.b64decode(clean_base64)
                                except Exception as e1:
                                    # 如果失败，尝试使用validate
                                    try:
                                        image_data = base64.b64decode(clean_base64, validate=True)
                                    except Exception as e2:
                                        # 如果还是失败，尝试添加padding
                                        try:
                                            padding_needed = 4 - (len(clean_base64) % 4)
                                            if padding_needed != 4:
                                                clean_base64_padded = clean_base64 + '=' * padding_needed
                                            else:
                                                clean_base64_padded = clean_base64
                                            image_data = base64.b64decode(clean_base64_padded)
                                        except Exception as e3:
                                            print(f"[ERROR] 所有base64解码尝试都失败，最后一个错误: {type(e3).__name__}: {e3}")
                                            image_data = None
                                
                                # 验证解码后的数据（如果成功）
                                if image_data:
                                    if len(image_data) == 0:
                                        print(f"[ERROR] base64解码后数据为空")
                                        image_data = None
                                    else:
                                        # JPEG文件头应该是 FF D8 FF
                                        if len(image_data) >= 3 and image_data[0:3] == b'\xff\xd8\xff':
                                        else:
                                else:
                                    print(f"[ERROR] base64解码失败，image_data 为 None")
                            except Exception as e:
                                print(f"[ERROR] 解码base64字符串时发生异常: {type(e).__name__}: {e}")
                                import traceback
                                traceback.print_exc()
                                image_data = None
                        
                        # 如果image_data成功解码，尝试创建PIL Image
                        if image_data:
                            try:
                                img = Image.open(io.BytesIO(image_data))
                                return img
                            except Exception as e:
                                print(f"[ERROR] 从bytes创建PIL Image失败: {type(e).__name__}: {e}")
                                import traceback
                                traceback.print_exc()
                                return None
                        else:
                            print(f"[ERROR] image_data 为 None，base64解码可能未执行或失败")
                            if isinstance(image_input, str):
                            return None
                    else:
                        print(f"[WARNING] 字符串太短（长度={len(image_input)}），不可能是有效的图片数据")
                        return None
            
            # 如果是bytes
            if isinstance(image_input, bytes):
                try:
                    img = Image.open(io.BytesIO(image_input))
                    return img
                except Exception as e:
                    print(f"[ERROR] 从bytes创建PIL Image失败: {type(e).__name__}: {e}")
                    return None
            
            print(f"[WARNING] 无法识别输入类型: {type(image_input)}")
            return None
            
        except Exception as e:
            print(f"[ERROR] 加载图片时发生异常: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def process_image_pipeline_pair_async(
        self,
        image_base64: str,
        pipeline_name: str,
        metadata: Optional[Dict[str, Any]] = None,
        async_client: Optional[AsyncGeminiClient] = None,
        model: Optional[str] = None
    ) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        异步处理单个图片-pipeline对，生成VQA问题
        
        严格遵循6步流程（异步版本）：
        1. 加载Pipeline规范
        2. 对象选择（如果需要）
        3. 槽位填充
        4. 问题生成
        5. 验证
        6. 输出
        
        Args:
            image_base64: 图片的base64编码
            pipeline_name: Pipeline名称
            metadata: 可选的元数据
            async_client: 异步客户端实例（可选）
            model: 模型名称（可选）
            
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
                    selected_object = await self.object_selector.select_object_async(
                        image_base64=image_base64,
                        pipeline_config=pipeline_config,
                        global_policy=self.object_selection_policy,
                        async_client=async_client,
                        model=model
                    )
                    
                    # 检查对象选择是否失败（包括返回None或selected=False）
                    if selected_object is not None:
                    
                    if selected_object is None or selected_object.get("selected") == False:
                        # 根据策略，如果对象选择失败则丢弃
                        if self.object_selection_policy.get("fallback_strategy") == "discard_image":
                            error_info["error_stage"] = "object_selection"
                            # 从对象选择器的响应中获取reason（如果存在）
                            if selected_object and selected_object.get("reason"):
                                error_info["error_reason"] = f"无法选择对象: {selected_object.get('reason')}"
                                error_info["model_reason"] = selected_object.get("reason")  # 保存模型的reason
                                error_info["confidence"] = selected_object.get("confidence", 0.0)
                            else:
                                error_info["error_reason"] = "无法选择对象"
                            
                            # 保存失败案例（异步版本，使用base64）
                            if self.failed_selection_dir:
                                self._save_failed_selection_case(
                                    image_input=image_base64,
                                    pipeline_config=pipeline_config,
                                    error_info=error_info,
                                    metadata=metadata,
                                    selection_result=selected_object  # 传递选择结果以获取reason
                                )
                            else:
                                print(f"[WARNING] [异步] 跳过保存失败案例，failed_selection_dir 为 None")
                            
                            return None, error_info
                    else:
                except Exception as e:
                    error_info["error_stage"] = "object_selection"
                    error_info["error_reason"] = f"对象选择过程出错: {str(e)}"
                    
                    # 保存失败案例（异步版本，使用base64）
                    if self.failed_selection_dir:
                        self._save_failed_selection_case(
                            image_input=image_base64,
                            pipeline_config=pipeline_config,
                            error_info=error_info,
                            metadata=metadata,
                            selection_result=None  # 异常情况下没有选择结果
                        )
                    
                    print(f"[ERROR] {error_info['error_reason']}")
                    return None, error_info
            
            # STEP 3: 槽位填充
            try:
                slots = await self.slot_filler.fill_slots_async(
                    image_base64=image_base64,
                    pipeline_config=pipeline_config,
                    selected_object=selected_object
                )
                
                if slots is None:
                    error_info["error_stage"] = "slot_filling"
                    error_info["error_reason"] = "槽位填充失败（必需槽位无法解析）"
                    return None, error_info
            except Exception as e:
                error_info["error_stage"] = "slot_filling"
                error_info["error_reason"] = f"槽位填充过程出错: {str(e)}"
                print(f"[ERROR] {error_info['error_reason']}")
                return None, error_info
            
            # STEP 4 & 5: 问题生成和验证（带重试机制，最多重试3次）
            # 按比例选择题型（在try块外初始化，确保在结果中可用）
            question_type = self._select_question_type()
            
            max_retries = 3
            retry_count = 0
            question = None
            is_valid = False
            reason = ""
            last_error = None
            
            # 最多重试3次（总共尝试4次：初始1次 + 重试3次）
            # retry_count: 0=初始尝试, 1=第1次重试, 2=第2次重试, 3=第3次重试
            while retry_count <= max_retries:
                try:
                    # 生成问题
                    question = await self.question_generator.generate_question_async(
                        image_base64=image_base64,
                        pipeline_config=pipeline_config,
                        slots=slots,
                        selected_object=selected_object,
                        question_type=question_type,
                        async_client=async_client,
                        model=model
                    )
                    
                    if not question:
                        last_error = "问题生成失败（返回空）"
                        if retry_count < max_retries:
                            retry_count += 1
                            print(f"[重试] 问题生成失败（返回空），正在重试 ({retry_count}/{max_retries})...")
                            continue
                        else:
                            error_info["error_stage"] = "question_generation"
                            error_info["error_reason"] = f"经过 {max_retries} 次重试后仍失败: {last_error}"
                            return None, error_info
                    
                    # 验证问题
                    is_valid, reason = await self.validator.validate_async(
                        question=question,
                        image_base64=image_base64,
                        pipeline_config=pipeline_config,
                        global_constraints=self.global_constraints,
                        async_client=async_client,
                        model=model
                    )
                    
                    if is_valid:
                        # 验证通过，退出重试循环
                        if retry_count > 0:
                            print(f"[成功] 问题经过 {retry_count} 次重试后验证通过")
                        break
                    else:
                        # 验证失败，需要重试
                        last_error = f"问题验证失败: {reason}"
                        if retry_count < max_retries:
                            retry_count += 1
                            print(f"[重试] 问题验证失败 ({reason})，正在重新生成 ({retry_count}/{max_retries})...")
                            continue
                        else:
                            error_info["error_stage"] = "validation"
                            error_info["error_reason"] = f"经过 {max_retries} 次重试后仍验证失败: {reason}"
                            return None, error_info
                            
                except Exception as e:
                    last_error = f"问题生成或验证过程出错: {str(e)}"
                    if retry_count < max_retries:
                        retry_count += 1
                        print(f"[重试] 问题生成或验证异常: {str(e)}，正在重试 ({retry_count}/{max_retries})...")
                        continue
                    else:
                        error_info["error_stage"] = "question_generation" if question is None else "validation"
                        error_info["error_reason"] = f"经过 {max_retries} 次重试后仍失败: {str(e)}"
                        print(f"[ERROR] {error_info['error_reason']}")
                        return None, error_info
            
            # 如果所有重试都失败，返回错误
            if not question or not is_valid:
                error_info["error_stage"] = "question_generation" if not question else "validation"
                error_info["error_reason"] = last_error or "问题生成或验证失败"
                return None, error_info
            
            # STEP 5: 输出（跳过验证以提高效率）
            result = {
                "pipeline_name": pipeline_name,
                "pipeline_intent": pipeline_config.get("intent", ""),
                "question": question,
                "question_type": question_type,  # 添加题型字段
                "answer_type": pipeline_config.get("answer_type", ""),
                "slots": slots,
                "selected_object": selected_object,
                "validation_reason": "skipped",
                "timestamp": datetime.now().isoformat()
            }
            
            return result, None
            
        except Exception as e:
            error_info["error_stage"] = "unknown"
            error_info["error_reason"] = f"未知错误: {str(e)}"
            print(f"[ERROR] {error_info['error_reason']}")
            return None, error_info
    
    async def process_data_file_async(
        self,
        input_file: Path,
        output_file: Path,
        pipeline_names: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
        num_gpus: int = 1,
        max_concurrent_per_gpu: int = 10,
        request_delay: float = 0.1,
        failed_selection_dir: Optional[Path] = None
    ) -> None:
        """
        异步处理数据文件，为每张图片生成VQA问题（并行版本）
        
        Args:
            input_file: 输入JSON文件路径（batch_process.sh的输出）
            output_file: 输出JSON文件路径
            pipeline_names: 要使用的pipeline列表（None表示使用所有）
            max_samples: 最大处理样本数（None表示全部）
            num_gpus: GPU数量（用于进程隔离，实际是API并发控制）
            max_concurrent_per_gpu: 每个GPU的最大并发数
            request_delay: 每个请求之间的延迟（秒）
            failed_selection_dir: 失败案例存储目录（可选，如果为None且self.failed_selection_dir也为None则不保存）
        """
        # 如果传入了failed_selection_dir，使用它；否则使用初始化时设置的
        if failed_selection_dir is not None:
            self.failed_selection_dir = failed_selection_dir
            self.failed_selection_dir.mkdir(parents=True, exist_ok=True)
        
        # 如果没有设置失败案例目录，使用输出目录的子目录
        if self.failed_selection_dir is None:
            self.failed_selection_dir = output_file.parent / "failed_selection"
            self.failed_selection_dir.mkdir(parents=True, exist_ok=True)
        
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError(f"输入文件应该包含一个数组，但得到: {type(data)}")
        
        # 确定要使用的pipeline
        if pipeline_names is None:
            pipeline_names = self.config_loader.list_pipelines()
        
        
        if max_samples:
            data = data[:max_samples]
        
        # 准备任务列表
        tasks = []
        for idx, record in enumerate(data, 1):
            source_a = record.get("source_a", {})
            if not source_a:
                continue
            
            # 提取图片输入
            image_base64 = self._extract_image_base64(source_a, self._extract_image_input(source_a))
            if image_base64 is None:
                continue
            
            # 确定该记录应该使用的pipeline
            record_pipeline = self._extract_pipeline_from_record(record)
            
            # 如果记录中指定了pipeline，使用指定的；否则使用传入的pipeline_names
            if record_pipeline:
                pipelines_to_use = [record_pipeline]
            else:
                pipelines_to_use = pipeline_names if pipeline_names else self.config_loader.list_pipelines()
            
            # 为每个pipeline创建任务
            for pipeline_name in pipelines_to_use:
                tasks.append({
                    "record_index": idx,
                    "record": record,
                    "image_base64": image_base64,
                    "pipeline_name": pipeline_name,
                    "source_a": source_a
                })
        
        
        # 使用多GPU异步处理
        results = await self._process_tasks_async(
            tasks=tasks,
            num_gpus=num_gpus,
            max_concurrent_per_gpu=max_concurrent_per_gpu,
            request_delay=request_delay
        )
        
        # 分离成功结果和错误
        success_results = []
        errors = []
        
        for result, error_info in results:
            if result:
                success_results.append(result)
            else:
                errors.append(error_info)
        
        # 保存成功结果
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(success_results, f, ensure_ascii=False, indent=2)
        
        # 保存错误和丢弃的数据（带时间戳）
        if errors:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            error_file = output_file.parent / f"{output_file.stem}_errors_{timestamp}.json"
            with open(error_file, 'w', encoding='utf-8') as f:
                json.dump(errors, f, ensure_ascii=False, indent=2)
            print(f"  错误/丢弃数据已保存到: {error_file}")
        
        print(f"\n[完成] 处理完成！")
        print(f"  总处理: {len(results)}")
        print(f"  成功生成: {len(success_results)}")
        print(f"  丢弃/错误: {len(errors)}")
        print(f"  结果已保存到: {output_file}")
    
    async def _process_tasks_async(
        self,
        tasks: List[Dict[str, Any]],
        num_gpus: int = 1,
        max_concurrent_per_gpu: int = 10,
        request_delay: float = 0.1
    ) -> List[tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]]:
        """
        异步并行处理任务列表
        
        Args:
            tasks: 任务列表
            num_gpus: GPU数量
            max_concurrent_per_gpu: 每个GPU的最大并发数
            request_delay: 每个请求之间的延迟（秒）
            
        Returns:
            结果列表，每个元素是 (成功结果, 错误信息) 的元组
        """
        # 将任务分配到不同的GPU组
        tasks_per_gpu = len(tasks) // num_gpus if num_gpus > 1 else len(tasks)
        gpu_tasks = []
        
        for gpu_id in range(num_gpus):
            start_idx = gpu_id * tasks_per_gpu
            if gpu_id == num_gpus - 1:
                end_idx = len(tasks)  # 最后一个GPU处理剩余所有任务
            else:
                end_idx = (gpu_id + 1) * tasks_per_gpu
            
            gpu_tasks.append((gpu_id, tasks[start_idx:end_idx]))
        
        # 为每个GPU创建处理任务
        async def process_gpu_tasks(gpu_id: int, gpu_task_list: List[Dict]):
            """处理单个GPU的任务"""
            results = []
            async with AsyncGeminiClient(
                gpu_id=gpu_id,
                max_concurrent=max_concurrent_per_gpu,
                request_delay=request_delay
            ) as async_client:
                # 创建所有异步任务
                async_task_list = []
                for task in gpu_task_list:
                    async_task = self.process_image_pipeline_pair_async(
                        image_base64=task["image_base64"],
                        pipeline_name=task["pipeline_name"],
                        metadata={
                            "record_index": task["record_index"],
                            "id": task["record"].get("id")
                        },
                        async_client=async_client,
                        model=async_client.model_name
                    )
                    async_task_list.append(async_task)
                
                # 等待所有任务完成（使用return_exceptions=True以处理异常）
                task_results = await asyncio.gather(*async_task_list, return_exceptions=True)
                
                # 处理结果和异常
                for i, task_result in enumerate(task_results):
                    if isinstance(task_result, Exception):
                        # 异常情况，创建错误信息
                        error_info = {
                            "pipeline_name": gpu_task_list[i]["pipeline_name"],
                            "metadata": {
                                "record_index": gpu_task_list[i]["record_index"],
                                "id": gpu_task_list[i]["record"].get("id")
                            },
                            "timestamp": datetime.now().isoformat(),
                            "error_stage": "unknown",
                            "error_reason": f"处理异常: {str(task_result)}",
                            "sample_index": gpu_task_list[i]["record"].get("sample_index"),
                            "id": gpu_task_list[i]["record"].get("id"),
                            "source_a_id": gpu_task_list[i]["source_a"].get("id")
                        }
                        
                        # 如果是对象选择相关的异常，尝试保存失败案例
                        if self.failed_selection_dir and "object_selection" in str(task_result).lower():
                            try:
                                # 获取pipeline配置
                                pipeline_config = self.config_loader.get_pipeline_config(gpu_task_list[i]["pipeline_name"])
                                if pipeline_config:
                                    self._save_failed_selection_case(
                                        image_input=gpu_task_list[i]["image_base64"],
                                        pipeline_config=pipeline_config,
                                        error_info=error_info,
                                        metadata=error_info["metadata"],
                                        selection_result=None
                                    )
                            except Exception as save_error:
                                print(f"[WARNING] 保存异常失败案例时出错: {save_error}")
                        
                        results.append((None, error_info))
                    else:
                        # 正常情况，提取结果
                        result, error_info = task_result
                        if result:
                            # 添加原始数据信息
                            result["sample_index"] = gpu_task_list[i]["record"].get("sample_index")
                            result["id"] = gpu_task_list[i]["record"].get("id")
                            result["source_a_id"] = gpu_task_list[i]["source_a"].get("id")
                            result["image_base64"] = gpu_task_list[i]["image_base64"]
                            results.append((result, None))
                        else:
                            # 失败情况，添加元数据到错误信息
                            if error_info:
                                error_info["sample_index"] = gpu_task_list[i]["record"].get("sample_index")
                                error_info["id"] = gpu_task_list[i]["record"].get("id")
                                error_info["source_a_id"] = gpu_task_list[i]["source_a"].get("id")
                            results.append((None, error_info))
                
                # 进度报告
                completed = len(results)
                success_count = sum(1 for r, e in results if r is not None)
                if completed > 0 and completed % 10 == 0:
                    print(f"[进度] GPU {gpu_id}: 已处理: {completed}, 成功: {success_count}")
            
            return results
        
        # 并发处理所有GPU的任务
        all_results = await asyncio.gather(*[
            process_gpu_tasks(gpu_id, task_list)
            for gpu_id, task_list in gpu_tasks
        ])
        
        # 合并结果
        final_results = []
        for gpu_results in all_results:
            final_results.extend(gpu_results)
        
        return final_results
