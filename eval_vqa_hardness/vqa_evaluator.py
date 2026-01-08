"""
VQA数据集评估框架 - 本地HF模型版
用于评估本地Hugging Face格式的VQA模型
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoTokenizer


class VQAEvaluator:
    """VQA评估器 - 支持本地HF模型"""
    
    def __init__(self, model_path: str, device: str = None):
        """
        初始化评估器
        
        Args:
            model_path: 本地模型路径
            device: 运行设备 (cuda/cpu)，None则自动选择
        """
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"正在加载模型: {model_path}")
        print(f"使用设备: {self.device}")
        
        # 加载模型和处理器
        self.load_model()
        
    def load_model(self):
        """加载模型和处理器
        
        - 对于 LLaVA-OneVision / Qwen2-VL 这类非标准本地模型，参考 vlmevalkit 的加载方式：
          使用 AutoModelForCausalLM + AutoProcessor + trust_remote_code=True
        - 对于常规 VQA 模型，使用 AutoModelForVision2Seq + AutoProcessor
        """
        model_path_lower = str(self.model_path).lower()
        
        # 1) 针对 LLaVA-OneVision / qwen2 系列模型，完全按照 vlmevalkit 的 LLaVA_OneVision_1_5 实现
        if ("llava" in model_path_lower) or ("qwen2" in model_path_lower) or ("hf_stage2" in model_path_lower):
            from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM
            print("检测到 LLaVA / Qwen2 系列模型，使用 vlmevalkit LLaVA_OneVision_1_5 的加载方式")
            
            # 检查是否安装了 qwen_vl_utils（必需依赖）
            try:
                from qwen_vl_utils import process_vision_info
                print("✓ qwen_vl_utils 已安装")
            except ImportError:
                raise ImportError(
                    "qwen_vl_utils 未安装！请运行: pip install qwen-vl-utils\n"
                    "这是 LLaVA-OneVision / Qwen2-VL 模型的必需依赖。"
                )
            
            # 步骤1: 先加载模型（完全按照 vlmevalkit 的方式）
            # 注意：模型加载会执行自定义代码，可能会注册 Qwen2Tokenizer 等类
            print("步骤1: 加载模型（AutoModelForCausalLM + trust_remote_code=True）...")
            
            import os
            import json
            import shutil
            
            # 自动修复 config.json 中的模块引用问题（如果存在）
            config_path = os.path.join(self.model_path, "config.json")
            config_fixed = False
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                    
                    if 'auto_map' in config:
                        auto_map = config.get('auto_map', {})
                        if isinstance(auto_map, dict):
                            fixed_auto_map = {}
                            needs_fix = False
                            
                            for key, value in auto_map.items():
                                if isinstance(value, str) and ('LLaVA-OneVision' in value or ('-' in value and 'transformers_modules' in value)):
                                    # 修复模块名：将连字符替换为下划线
                                    # 注意：需要先处理更长的匹配（如 1.5），再处理短的（如 1）
                                    fixed_value = value
                                    # 先处理 1.5 版本
                                    fixed_value = fixed_value.replace('LLaVA-OneVision-1.5', 'LLaVA_OneVision_1_5')
                                    fixed_value = fixed_value.replace('LLaVA-OneVision-1', 'LLaVA_OneVision_1')
                                    # 再处理其他版本（如 8B）
                                    fixed_value = fixed_value.replace('LLaVA-OneVision-8', 'LLaVA_OneVision_8')
                                    # 最后处理通用情况
                                    fixed_value = fixed_value.replace('LLaVA-OneVision', 'LLaVA_OneVision')
                                    fixed_auto_map[key] = fixed_value
                                    if fixed_value != value:
                                        print(f"  自动修复 auto_map[{key}]: {value} -> {fixed_value}")
                                        needs_fix = True
                                else:
                                    fixed_auto_map[key] = value
                            
                            if needs_fix:
                                # 备份原文件
                                backup_path = config_path + '.backup'
                                if not os.path.exists(backup_path):
                                    shutil.copy2(config_path, backup_path)
                                    print(f"  已备份 config.json 到: {backup_path}")
                                
                                # 写入修复后的配置
                                config['auto_map'] = fixed_auto_map
                                with open(config_path, 'w', encoding='utf-8') as f:
                                    json.dump(config, f, indent=2, ensure_ascii=False)
                                config_fixed = True
                                print("  ✓ 已自动修复 config.json 中的模块名")
                except Exception as e:
                    print(f"  检查/修复 config.json 时出错（可忽略）: {e}")
            
            # 修复 transformers_modules 目录名（如果存在）
            transformers_modules_dir = os.path.join(self.model_path, "transformers_modules")
            if os.path.exists(transformers_modules_dir):
                try:
                    subdirs = [d for d in os.listdir(transformers_modules_dir) 
                              if os.path.isdir(os.path.join(transformers_modules_dir, d)) and '-' in d]
                    for subdir in subdirs:
                        old_path = os.path.join(transformers_modules_dir, subdir)
                        new_name = subdir.replace('-', '_')
                        new_path = os.path.join(transformers_modules_dir, new_name)
                        if not os.path.exists(new_path):
                            os.rename(old_path, new_path)
                            print(f"  ✓ 已重命名目录: {subdir} -> {new_name}")
                            config_fixed = True
                except Exception as e:
                    print(f"  修复 transformers_modules 目录时出错（可忽略）: {e}")
            
            # 如果修复了配置，清除 transformers 的缓存以确保重新加载
            if config_fixed:
                print("  注意: 已修复配置，将重新加载...")
                # 清除可能的缓存
                import sys
                cache_keys = [k for k in sys.modules.keys() if 'transformers_modules' in k]
                for key in cache_keys:
                    del sys.modules[key]
            
            try:
                # 尝试加载模型
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype="auto",  # 使用 "auto" 让 transformers 自动选择
                    device_map="auto" if self.device == "cuda" else None,
                    trust_remote_code=True
                )
                self.model.eval()
                print("✓ 模型加载成功")
            except (ModuleNotFoundError, ImportError) as e:
                error_str = str(e)
                if "transformers_modules" in error_str or "LLaVA-OneVision" in error_str:
                    print(f"\n{'='*70}")
                    print("错误: 遇到自定义代码模块导入问题")
                    print(f"{'='*70}")
                    print(f"详细错误: {e}")
                    print("\n问题分析：")
                    print("  模型目录中的自定义代码模块名包含连字符（如 'LLaVA-OneVision-1'）")
                    print("  Python 无法导入包含连字符的模块名")
                    print("\n解决方案：")
                    print("  方案1: 运行诊断脚本检查模型目录")
                    print(f"    python diagnose_model.py {self.model_path}")
                    print("\n  方案2: 手动修复模型目录")
                    print(f"    1. 检查 {self.model_path}/config.json")
                    print("    2. 查看 'auto_map' 字段中的模块名")
                    print("    3. 如果包含连字符，需要修复为下划线")
                    print("    4. 或者重命名 transformers_modules 目录中的子目录")
                    print("\n  方案3: 尝试手动修复后重新加载")
                    # 如果之前没有修复，现在尝试修复
                    if not config_fixed:
                        print("  正在尝试自动修复配置...")
                        # 重新执行修复逻辑（简化版）
                        try:
                            if os.path.exists(config_path):
                                with open(config_path, 'r', encoding='utf-8') as f:
                                    config = json.load(f)
                                if 'auto_map' in config:
                                    auto_map = config.get('auto_map', {})
                                    if isinstance(auto_map, dict):
                                        fixed_auto_map = {}
                                        for key, value in auto_map.items():
                                            if isinstance(value, str) and ('LLaVA-OneVision' in value or ('-' in value and 'transformers_modules' in value)):
                                                # 修复模块名：将连字符替换为下划线
                                                # 注意：需要先处理更长的匹配（如 1.5），再处理短的（如 1）
                                                fixed_value = value
                                                # 先处理 1.5 版本
                                                fixed_value = fixed_value.replace('LLaVA-OneVision-1.5', 'LLaVA_OneVision_1_5')
                                                fixed_value = fixed_value.replace('LLaVA-OneVision-1', 'LLaVA_OneVision_1')
                                                # 再处理其他版本（如 8B）
                                                fixed_value = fixed_value.replace('LLaVA-OneVision-8', 'LLaVA_OneVision_8')
                                                # 最后处理通用情况
                                                fixed_value = fixed_value.replace('LLaVA-OneVision', 'LLaVA_OneVision')
                                                fixed_auto_map[key] = fixed_value
                                            else:
                                                fixed_auto_map[key] = value
                                        config['auto_map'] = fixed_auto_map
                                        backup_path = config_path + '.backup'
                                        if not os.path.exists(backup_path):
                                            shutil.copy2(config_path, backup_path)
                                        with open(config_path, 'w', encoding='utf-8') as f:
                                            json.dump(config, f, indent=2, ensure_ascii=False)
                                        print("  ✓ 已修复 config.json")
                                        # 清除缓存
                                        import sys
                                        cache_keys = [k for k in sys.modules.keys() if 'transformers_modules' in k]
                                        for key in cache_keys:
                                            del sys.modules[key]
                        except Exception as fix_e:
                            print(f"  修复失败: {fix_e}")
                    
                    # 尝试重新加载
                    try:
                        print("  尝试重新加载模型...")
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.model_path,
                            torch_dtype="auto",
                            device_map="auto" if self.device == "cuda" else None,
                            trust_remote_code=True
                        )
                        self.model.eval()
                        print("  ✓ 模型加载成功（修复后重新加载）")
                    except Exception as e3:
                        print(f"  ✗ 重新加载也失败: {e3}")
                        print("\n  方案4: 检查 transformers 版本")
                        print("    可能需要更新 transformers: pip install 'transformers>=4.40.0'")
                        print("\n  方案5: 手动修复模型目录")
                        print(f"    运行修复脚本: python fix_model_config.py {self.model_path} --apply")
                        print("\n" + "="*70)
                        raise ValueError(
                            f"无法加载模型。\n\n"
                            f"原始错误: {e}\n"
                            f"修复后错误: {e3}\n\n"
                            f"这是一个模型目录配置问题。\n"
                            f"请运行修复脚本: python fix_model_config.py {self.model_path} --apply\n"
                            f"或手动检查并修复模型目录中的配置文件。"
                        )
                else:
                    raise
            
            # 检查 device_map
            if hasattr(self.model, "hf_device_map") and self.model.hf_device_map:
                print(f"  模型使用 device_map 自动分配: {self.model.hf_device_map}")
            elif self.device == "cuda":
                self.model = self.model.to(self.device)
            
            # 步骤2: 加载 Processor（完全按照 vlmevalkit 的方式，带 max_pixels/min_pixels）
            print("步骤2: 加载 Processor（AutoProcessor + max_pixels/min_pixels）...")
            max_pixels = 3240000
            min_pixels = 200704
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                max_pixels=max_pixels,
                min_pixels=min_pixels
            )
            print("✓ Processor 加载成功")
            
            # 保存 qwen_vl_utils 的引用供后续使用
            self.process_vision_info = process_vision_info
            
            # 保存 kwargs（用于 generate 时的额外参数）
            self.model_kwargs = {}
            
            print("✓ 模型和 Processor 加载完成！（LLaVA/Qwen2 模式，完全按照 vlmevalkit 实现）")
            return
        
        # 2) 默认路径：常规 Vision2Seq 模型
        try:
            # 尝试加载processor（适用于视觉语言模型）
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"加载processor失败: {e}")
            print("尝试分别加载tokenizer...")
            self.processor = None
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
        
        # 加载模型
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        
        self.model.eval()
        print("模型加载完成！（Vision2Seq 模式）")
    
    def load_image(self, image_path: str = None, image_base64: str = None) -> Image.Image:
        """
        加载图片（支持文件路径或base64）
        
        Args:
            image_path: 图片文件路径
            image_base64: base64编码的图片字符串
        """
        if image_base64:
            # 从base64加载
            from io import BytesIO
            import base64
            
            # 移除可能的data URI前缀
            if ',' in image_base64:
                image_base64 = image_base64.split(',', 1)[1]
            
            image_data = base64.b64decode(image_base64)
            return Image.open(BytesIO(image_data)).convert('RGB')
        elif image_path:
            # 从文件加载
            return Image.open(image_path).convert('RGB')
        else:
            raise ValueError("必须提供image_path或image_base64")
    
    def query_model(self, question: str, image_path: str = None, 
                   image_base64: str = None, image: Image.Image = None,
                   max_new_tokens: int = 128, temperature: float = 0.0) -> str:
        """
        向模型查询问题答案
        
        完全按照 vlmevalkit LLaVA_OneVision_1_5.generate_inner 的实现方式
        
        Args:
            question: 问题文本
            image_path: 图片路径（可选）
            image_base64: base64编码图片（可选）
            image: PIL Image对象（可选）
            max_new_tokens: 最大生成token数
            temperature: 生成温度（LLaVA-OneVision 通常使用 0）
            
        Returns:
            模型的回答
        """
        # 检查是否是 LLaVA/Qwen2 模式（使用 qwen_vl_utils）
        if hasattr(self, 'process_vision_info'):
            # 使用 vlmevalkit 的方式：chat template + process_vision_info
            import os
            import base64
            from io import BytesIO
            from tempfile import NamedTemporaryFile
            
            # 步骤1: 加载图片并保存为临时文件（qwen_vl_utils 需要文件路径或 URL）
            if image is None:
                image = self.load_image(image_path=image_path, image_base64=image_base64)
            
            temp_image_path = None
            try:
                # 保存为临时文件
                temp_file = NamedTemporaryFile(delete=False, suffix='.jpg')
                temp_image_path = temp_file.name
                image.save(temp_image_path, 'JPEG')
                temp_file.close()
                
                # 确保路径格式正确（添加 file:// 前缀，按照 vlmevalkit 的 ensure_image_url）
                if not temp_image_path.startswith(('http://', 'https://', 'file://', 'data:image')):
                    image_url = f"file://{temp_image_path}"
                else:
                    image_url = temp_image_path
                
                # 步骤2: 构建消息（完全按照 vlmevalkit LLaVA_OneVision_1_5.generate_inner 的格式）
                content_list = [
                    {"type": "image", "image": image_url},
                    {"type": "text", "text": question}
                ]
                messages = [
                    {"role": "user", "content": content_list}
                ]
                
                # 步骤3: 使用 processor.apply_chat_template（完全按照 vlmevalkit）
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                
                # 步骤4: 使用 qwen_vl_utils.process_vision_info 处理视觉信息（完全按照 vlmevalkit）
                image_inputs, video_inputs = self.process_vision_info(messages)
                
                # 步骤5: 调用 processor（完全按照 vlmevalkit）
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to(self.device)
                
                # 步骤6: 生成答案（完全按照 vlmevalkit，使用 model_kwargs）
                generate_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    **self.model_kwargs
                }
                with torch.no_grad():
                    generated_ids = self.model.generate(**inputs, **generate_kwargs)
                
                # 步骤7: Trim input_ids（只保留新生成的部分，完全按照 vlmevalkit）
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                
                # 步骤8: 解码（完全按照 vlmevalkit）
                output_text = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                answer = output_text[0]
                
            finally:
                # 清理临时文件
                if temp_image_path and os.path.exists(temp_image_path):
                    try:
                        os.unlink(temp_image_path)
                    except:
                        pass
            
            return answer
        
        # 默认方式：常规 Vision2Seq 模型（非 LLaVA/Qwen2）
        # 加载图片（支持三种输入方式）
        if image is None:
            image = self.load_image(image_path=image_path, image_base64=image_base64)
        
        # 准备输入
        if self.processor is not None:
            # 使用processor处理
            inputs = self.processor(
                images=image,
                text=question,
                return_tensors="pt"
            ).to(self.device)
        else:
            # 自定义处理逻辑（需要根据具体模型调整）
            raise NotImplementedError("请根据你的模型实现图片和文本的处理逻辑")
        
        # 生成答案
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.processor.tokenizer.pad_token_id if self.processor else self.tokenizer.pad_token_id
            )
        
        # 解码输出
        if self.processor is not None:
            answer = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        else:
            answer = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        
        # 移除输入的question部分（如果模型输出包含输入）
        if question in answer:
            answer = answer.replace(question, "").strip()
        
        return answer
    
    def normalize_text(self, text: str) -> str:
        """标准化文本用于比较"""
        import re
        # 转小写
        text = text.lower().strip()
        # 移除标点
        text = re.sub(r'[^\w\s]', '', text)
        # 移除多余空格
        text = ' '.join(text.split())
        return text
    
    def check_answer(self, prediction: str, ground_truth: List[str], 
                    strict: bool = False) -> bool:
        """
        检查答案是否正确
        
        Args:
            prediction: 模型预测的答案
            ground_truth: 正确答案列表
            strict: 是否使用严格匹配
            
        Returns:
            是否正确
        """
        pred_norm = self.normalize_text(prediction)
        
        for gt in ground_truth:
            gt_norm = self.normalize_text(gt)
            
            if strict:
                # 严格匹配：完全相同
                if pred_norm == gt_norm:
                    return True
            else:
                # 宽松匹配：包含关系
                if gt_norm in pred_norm or pred_norm in gt_norm:
                    return True
        
        return False
    
    def evaluate_dataset(self, 
                        data_file: str = None,
                        processed_data: List[Dict] = None,
                        image_dir: str = None,
                        output_file: str = "evaluation_results.json",
                        batch_size: int = 1,
                        max_samples: Optional[int] = None,
                        strict_match: bool = False) -> Dict[str, Any]:
        """
        评估整个数据集
        
        Args:
            data_file: 数据文件路径（JSON格式，传统格式）
            processed_data: 已处理的数据列表（新格式，来自DataLoader）
            image_dir: 图片目录（仅用于传统格式）
            output_file: 输出结果文件路径
            batch_size: 批处理大小（目前仅支持1）
            max_samples: 最大评估样本数（用于快速测试）
            strict_match: 是否使用严格答案匹配
            
        Returns:
            评估结果字典
        """
        # 加载数据
        if processed_data is not None:
            # 使用预处理的数据（新格式）
            dataset = processed_data
        elif data_file is not None:
            # 从文件加载（传统格式）
            with open(data_file, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
        else:
            raise ValueError("必须提供data_file或processed_data")
        
        # 限制样本数
        if max_samples is not None:
            dataset = dataset[:max_samples]
        
        results = []
        correct = 0
        total = len(dataset)
        
        print(f"\n开始评估，共 {total} 个样本...")
        print(f"答案匹配模式: {'严格' if strict_match else '宽松'}")
        print("-" * 60)
        
        # 逐个评估
        for idx, item in enumerate(tqdm(dataset, desc="评估进度")):
            question = item['question']
            ground_truth = item['answers']
            
            # 获取图片（支持两种格式）
            if 'image_base64' in item:
                # 新格式：直接包含base64图片
                image_input = {'image_base64': item['image_base64']}
            else:
                # 传统格式：图片文件路径
                image_path = os.path.join(image_dir, item['image'])
                if not os.path.exists(image_path):
                    print(f"\n警告: 图片不存在 {image_path}")
                    results.append({
                        'id': item.get('id', idx),
                        'question': question,
                        'ground_truth': ground_truth,
                        'prediction': "ERROR: Image not found",
                        'correct': False
                    })
                    continue
                image_input = {'image_path': image_path}
            
            try:
                # 查询模型
                prediction = self.query_model(question=question, **image_input)
                
                # 检查答案
                is_correct = self.check_answer(prediction, ground_truth, strict=strict_match)
                
                if is_correct:
                    correct += 1
                
                # 记录结果
                result = {
                    'id': item.get('id', idx),
                    'question': question,
                    'ground_truth': ground_truth,
                    'prediction': prediction,
                    'correct': is_correct
                }
                
                # 保留元数据（如果有）
                if 'metadata' in item:
                    result['metadata'] = item['metadata']
                
                results.append(result)
                
            except Exception as e:
                print(f"\n处理样本 {idx} 时出错: {str(e)}")
                import traceback
                traceback.print_exc()
                
                error_result = {
                    'id': item.get('id', idx),
                    'question': question,
                    'ground_truth': ground_truth,
                    'prediction': f"ERROR: {str(e)}",
                    'correct': False
                }
                
                # 保留元数据（如果有）
                if 'metadata' in item:
                    error_result['metadata'] = item['metadata']
                
                results.append(error_result)
        
        # 计算准确率
        accuracy = correct / total if total > 0 else 0
        
        # 生成评估报告
        evaluation_report = {
            'model_path': self.model_path,
            'device': self.device,
            'strict_match': strict_match,
            'total_samples': total,
            'correct_predictions': correct,
            'accuracy': accuracy,
            'results': results
        }
        
        # 保存结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_report, f, ensure_ascii=False, indent=2)
        
        print(f"\n{'=' * 60}")
        print("评估完成！")
        print(f"{'=' * 60}")
        print(f"模型路径: {self.model_path}")
        print(f"总样本数: {total}")
        print(f"正确数: {correct}")
        print(f"准确率: {accuracy:.2%}")
        print(f"结果已保存到: {output_file}")
        print(f"{'=' * 60}\n")
        
        return evaluation_report


def main():
    """主函数示例"""
    
    # ========== 方式1: 使用新的数据加载器（推荐）==========
    from data_loader import VQADataLoader
    
    # 配置参数
    MODEL_PATH = "./your_model_path"  # 你的本地模型路径
    DATA_FILE = "vqa.json"             # 你的数据文件
    OUTPUT_FILE = "evaluation_results.json"
    
    # 加载并处理数据
    print("=" * 60)
    print("步骤1: 加载数据")
    print("=" * 60)
    loader = VQADataLoader(DATA_FILE)
    loader.print_statistics()
    
    # 处理数据（include_options=True会使用full_question）
    processed_data = loader.process_all(
        include_options=True,  # True: 包含选项, False: 仅问题
        max_samples=None       # None: 全部数据
    )
    
    # 创建评估器
    print("\n" + "=" * 60)
    print("步骤2: 加载模型")
    print("=" * 60)
    evaluator = VQAEvaluator(
        model_path=MODEL_PATH,
        device="cuda"  # 或 "cpu"
    )
    
    # 运行评估
    print("\n" + "=" * 60)
    print("步骤3: 运行评估")
    print("=" * 60)
    results = evaluator.evaluate_dataset(
        processed_data=processed_data,  # 使用处理后的数据
        output_file=OUTPUT_FILE,
        max_samples=None,      # 用于快速测试，如 max_samples=10
        strict_match=False     # False: 宽松匹配, True: 严格匹配
    )
    
    # ========== 方式2: 使用传统格式（如果你的数据是传统格式）==========
    # results = evaluator.evaluate_dataset(
    #     data_file="vqa_dataset.json",
    #     image_dir="images",
    #     output_file=OUTPUT_FILE,
    #     max_samples=None,
    #     strict_match=False
    # )
    
    # 快速查看结果
    print("\n" + "=" * 60)
    print("步骤4: 查看结果示例")
    print("=" * 60)
    print("\n前3个样本结果:")
    for i, r in enumerate(results['results'][:3]):
        print(f"\n样本 {i+1} (ID: {r['id']}):")
        print(f"  问题: {r['question'][:80]}...")
        print(f"  正确答案: {r['ground_truth']}")
        print(f"  模型预测: {r['prediction'][:80]}...")
        print(f"  是否正确: {'✓ 正确' if r['correct'] else '✗ 错误'}")
    
    print("\n" + "=" * 60)
    print("完成！运行 'python vqa_analyzer.py' 查看详细分析")
    print("=" * 60)


if __name__ == "__main__":
    main()