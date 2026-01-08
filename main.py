"""
主程序入口
"""
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator
import json
from datetime import datetime
import argparse
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import time
import gc
import os

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.router import Router, PipelineType
from src.pipelines import *
from utils.gemini_client import GeminiClient
import config


# 独立函数，用于多进程处理（避免序列化问题）
def _process_image_worker(image_input, metadata: Optional[Dict] = None, gpu_id: Optional[int] = None) -> Dict[str, Any]:
    """
    独立的工作函数，用于多进程处理单张图片
    
    Args:
        image_input: 图片输入
        metadata: 可选的图片元数据
        gpu_id: 可选的GPU ID
        
    Returns:
        处理结果字典
    """
    gemini_client = None
    try:
        # 设置GPU可见性（如果指定）
        if gpu_id is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # 在每个进程中独立创建Router和GeminiClient
        router = Router()
        gemini_client = GeminiClient()
        
        # 分流
        routes = router.route(image_input, metadata)
        
        # 进入各个pipeline进行筛选
        results = []
        for route in routes:
            # 创建对应的pipeline
            pipeline_classes = {
                PipelineType.QUESTION: QuestionPipeline,
                PipelineType.CAPTION: CaptionPipeline,
                PipelineType.PLACE_RECOGNITION: PlaceRecognitionPipeline,
                PipelineType.TEXT_ASSOCIATION: TextAssociationPipeline,
                PipelineType.OBJECT_PROPORTION: ObjectProportionPipeline,
                PipelineType.OBJECT_POSITION: ObjectPositionPipeline,
                PipelineType.OBJECT_ABSENCE: ObjectAbsencePipeline,
                PipelineType.OBJECT_ORIENTATION: ObjectOrientationPipeline,
                PipelineType.OBJECT_COUNTING: ObjectCountingPipeline,
            }
            
            pipeline_class = pipeline_classes.get(route)
            if pipeline_class:
                try:
                    pipeline = pipeline_class(gemini_client)
                    result = pipeline.filter(image_input)
                    results.append(result)
                except Exception as e:
                    results.append({
                        "pipeline_type": route.value,
                        "passed": False,
                        "reason": f"处理出错: {str(e)}",
                        "confidence": 0.0,
                        "error": str(e)
                    })
        
        return {
            "image_input": str(image_input),
            "routes": [r.value for r in routes],
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "image_input": str(image_input),
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
    finally:
        # 确保客户端资源被释放
        if gemini_client is not None:
            try:
                gemini_client.close()
            except:
                pass
        # 强制垃圾回收，释放内存
        gc.collect()


def _process_single_item_worker(item: Dict[str, Any], gpu_id: Optional[int] = None) -> Dict[str, Any]:
    """
    独立的工作函数，用于多进程处理单个数据项
    
    这个函数不依赖类实例，避免序列化包含线程锁的对象
    
    Args:
        item: 包含image_input和pipeline_types的数据项
        gpu_id: 可选的GPU ID
        
    Returns:
        处理结果字典
    """
    gemini_client = None
    try:
        # 设置GPU可见性（如果指定）
        if gpu_id is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # 在每个进程中独立创建GeminiClient和Pipeline
        gemini_client = GeminiClient()
        
        image_input = item["image_input"]
        routes = item["pipeline_types"]
        
        # 保存原始数据的所有字段
        original_data = {k: v for k, v in item.items() 
                       if k not in ["image_input", "pipeline_types"]}
        
        if len(routes) == 0:
            return {
                **original_data,
                "error": "No pipeline recognized",
                "timestamp": datetime.now().isoformat()
            }
        
        # 使用第一个pipeline
        pipeline_type = routes[0]
        
        # 如果pipeline_type是字符串，转换为PipelineType枚举
        if isinstance(pipeline_type, str):
            try:
                pipeline_type = PipelineType(pipeline_type)
            except ValueError:
                return {
                    **original_data,
                    "error": f"Invalid pipeline type: {pipeline_type}",
                    "timestamp": datetime.now().isoformat()
                }
        
        # 确保pipeline_type是PipelineType枚举对象
        if not isinstance(pipeline_type, PipelineType):
            return {
                **original_data,
                "error": f"Invalid pipeline type: {pipeline_type}",
                "timestamp": datetime.now().isoformat()
            }
        
        # 创建对应的pipeline
        pipeline_classes = {
            PipelineType.QUESTION: QuestionPipeline,
            PipelineType.CAPTION: CaptionPipeline,
            PipelineType.PLACE_RECOGNITION: PlaceRecognitionPipeline,
            PipelineType.TEXT_ASSOCIATION: TextAssociationPipeline,
            PipelineType.OBJECT_PROPORTION: ObjectProportionPipeline,
            PipelineType.OBJECT_POSITION: ObjectPositionPipeline,
            PipelineType.OBJECT_ABSENCE: ObjectAbsencePipeline,
            PipelineType.OBJECT_ORIENTATION: ObjectOrientationPipeline,
            PipelineType.OBJECT_COUNTING: ObjectCountingPipeline,
        }
        
        pipeline_class = pipeline_classes.get(pipeline_type)
        if not pipeline_class:
            return {
                **original_data,
                "error": f"Pipeline not found: {pipeline_type}",
                "timestamp": datetime.now().isoformat()
            }
        
        # 创建pipeline实例
        pipeline = pipeline_class(gemini_client)
        
        # 调用pipeline进行筛选
        filter_result = pipeline.filter(image_input)
        
        # 合并原始数据和筛选结果
        result = {
            **original_data,
            **filter_result,
            "pipeline_type": pipeline_type.value,
            "timestamp": datetime.now().isoformat()
        }
        result.pop("image_input", None)  # 确保删除image_input
        return result
        
    except Exception as e:
        # 即使出错也保留原始数据
        original_data = {k: v for k, v in item.items() 
                       if k not in ["image_input", "pipeline_types"]}
        result = {
            **original_data,
            "error": f"Unexpected error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }
        result.pop("image_input", None)
        return result
    finally:
        # 确保客户端资源被释放
        if gemini_client is not None:
            try:
                gemini_client.close()
            except:
                pass
        # 强制垃圾回收，释放内存
        gc.collect()


class ImageFilterSystem:
    """图片筛选系统主类"""
    
    def __init__(self, max_workers: int = 4, use_multiprocessing: bool = False, 
                 num_gpus: int = 0, use_async: bool = False):
        """
        初始化筛选系统
        
        Args:
            max_workers: 最大并发工作线程/进程数
            use_multiprocessing: 是否使用多进程（True）或多线程（False）
            num_gpus: GPU数量（用于进程隔离和负载均衡，0表示不使用GPU绑定）
            use_async: 是否使用异步IO（需要安装aiohttp）
        """
        self.router = Router()
        self.gemini_client = GeminiClient()
        self.max_workers = max_workers
        self.use_multiprocessing = use_multiprocessing
        self.num_gpus = num_gpus
        self.use_async = use_async
        
        # 如果使用多GPU，调整worker数量
        if num_gpus > 0:
            # 每个GPU分配多个worker
            workers_per_gpu = max(1, max_workers // num_gpus)
            self.max_workers = num_gpus * workers_per_gpu
            print(f"[INFO] 使用 {num_gpus} 个GPU，每个GPU {workers_per_gpu} 个worker，总计 {self.max_workers} 个worker")
        
        # 初始化各个pipeline
        self.pipelines = {
            PipelineType.QUESTION: QuestionPipeline(self.gemini_client),
            PipelineType.CAPTION: CaptionPipeline(self.gemini_client),
            PipelineType.PLACE_RECOGNITION: PlaceRecognitionPipeline(self.gemini_client),
            PipelineType.TEXT_ASSOCIATION: TextAssociationPipeline(self.gemini_client),
            PipelineType.OBJECT_PROPORTION: ObjectProportionPipeline(self.gemini_client),
            PipelineType.OBJECT_POSITION: ObjectPositionPipeline(self.gemini_client),
            PipelineType.OBJECT_ABSENCE: ObjectAbsencePipeline(self.gemini_client),
            PipelineType.OBJECT_ORIENTATION: ObjectOrientationPipeline(self.gemini_client),
            PipelineType.OBJECT_COUNTING: ObjectCountingPipeline(self.gemini_client),
        }
    
    def process_image(self, image_input, metadata: Dict = None) -> Dict[str, Any]:
        """
        处理单张图片：分流 -> 筛选
        
        Args:
            image_input: 图片,any style
            metadata: 可选的图片元数据
            
        Returns:
            处理结果字典，包含：
            - image_input: 图片路径
            - routes: 分流结果
            - results: 各pipeline的筛选结果列表
            - timestamp: 处理时间
        """
        
        # 分流
        routes = self.router.route(image_input, metadata)
        
        # 进入各个pipeline进行筛选
        results = []
        for route in routes:
            pipeline = self.pipelines.get(route)
            if pipeline:
                try:
                    result = pipeline.filter(image_input)
                    results.append(result)
                except Exception as e:
                    results.append({
                        "pipeline_type": route.value,
                        "passed": False,
                        "reason": f"处理出错: {str(e)}",
                        "confidence": 0.0,
                        "error": str(e)
                    })
        
        return {
            "image_input": str(image_input),
            "routes": [r.value for r in routes],
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
    
    def _process_single_item(self, item: Dict[str, Any], gpu_id: Optional[int] = None) -> Dict[str, Any]:
        """
        处理单个数据项（用于并发处理）
        
        Args:
            item: 包含image_input和pipeline_types的数据项，以及原始数据字段（如index, source_a, source_b等）
            
        Returns:
            处理结果字典，包含原始数据的所有字段和筛选结果，但不包含image_input字段
        """
        try:
            image_input = item["image_input"]
            routes = item["pipeline_types"]
            
            # 保存原始数据的所有字段（除了image_input和pipeline_types，这些会被替换）
            original_data = {k: v for k, v in item.items() 
                           if k not in ["image_input", "pipeline_types"]}
            
            if len(routes) == 0:
                result = {
                    **original_data,  # 保留所有原始字段
                    "error": "No pipeline recognized",
                    "timestamp": datetime.now().isoformat()
                }
                return result
            
            # 使用第一个pipeline
            pipeline_type = routes[0]
            
            # 如果pipeline_type是字符串，转换为PipelineType枚举
            if isinstance(pipeline_type, str):
                try:
                    pipeline_type = PipelineType(pipeline_type)
                except ValueError:
                    result = {
                        **original_data,  # 保留所有原始字段
                        "error": f"Invalid pipeline type: {pipeline_type}",
                        "timestamp": datetime.now().isoformat()
                    }
                    return result
            
            # 确保pipeline_type是PipelineType枚举对象
            if not isinstance(pipeline_type, PipelineType):
                result = {
                    **original_data,  # 保留所有原始字段
                    "error": f"Invalid pipeline type: {pipeline_type}",
                    "timestamp": datetime.now().isoformat()
                }
                return result
            
            # 重新获取pipeline（因为多进程需要）
            if self.use_multiprocessing or gpu_id is not None:
                # 如果指定了GPU ID，设置环境变量用于进程隔离
                if gpu_id is not None:
                    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
                pipeline = self._get_pipeline(pipeline_type)
            else:
                pipeline = self.pipelines.get(pipeline_type)
            
            if pipeline:
                try:
                    # 调用pipeline进行筛选
                    filter_result = pipeline.filter(image_input)
                    
                    # 合并原始数据和筛选结果，删除image_input字段
                    result = {
                        **original_data,  # 保留所有原始字段（index, source_a, source_b等）
                        **filter_result,  # 添加筛选结果字段（passed, basic_score, bonus_score, total_score, reason, confidence等）
                        "pipeline_type": pipeline_type.value,  # 确保有pipeline_type字段
                        "timestamp": datetime.now().isoformat()
                    }
                    # 明确删除image_input字段（如果存在）
                    result.pop("image_input", None)
                    return result
                except Exception as e:
                    result = {
                        **original_data,  # 保留所有原始字段
                        "pipeline_type": pipeline_type.value,
                        "passed": False,
                        "reason": f"处理出错: {str(e)}",
                        "confidence": 0.0,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    }
                    result.pop("image_input", None)  # 确保删除image_input
                    return result
            else:
                result = {
                    **original_data,  # 保留所有原始字段
                    "error": f"Pipeline not found: {pipeline_type}",
                    "timestamp": datetime.now().isoformat()
                }
                result.pop("image_input", None)  # 确保删除image_input
                return result
        except Exception as e:
            # 即使出错也保留原始数据
            original_data = {k: v for k, v in item.items() 
                           if k not in ["image_input", "pipeline_types"]}
            result = {
                **original_data,  # 保留所有原始字段
                "error": f"Unexpected error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
            result.pop("image_input", None)  # 确保删除image_input
            return result
        finally:
            # 强制垃圾回收，释放内存
            gc.collect()
    
    def _process_with_gpu(self, image_input, metadata: Dict, gpu_id: int) -> Dict[str, Any]:
        """在多进程环境中处理图片（绑定GPU）"""
        # 设置GPU可见性
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        return self.process_image(image_input, metadata)
    
    def _process_single_item_with_gpu(self, item: Dict[str, Any], gpu_id: int) -> Dict[str, Any]:
        """在多进程环境中处理单个数据项（绑定GPU）"""
        # 设置GPU可见性
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        return self._process_single_item(item, gpu_id)
    
    def _get_pipeline(self, pipeline_type: PipelineType):
        """获取pipeline实例（用于多进程）"""
        pipeline_classes = {
            PipelineType.QUESTION: QuestionPipeline,
            PipelineType.CAPTION: CaptionPipeline,
            PipelineType.PLACE_RECOGNITION: PlaceRecognitionPipeline,
            PipelineType.TEXT_ASSOCIATION: TextAssociationPipeline,
            PipelineType.OBJECT_PROPORTION: ObjectProportionPipeline,
            PipelineType.OBJECT_POSITION: ObjectPositionPipeline,
            PipelineType.OBJECT_ABSENCE: ObjectAbsencePipeline,
            PipelineType.OBJECT_ORIENTATION: ObjectOrientationPipeline,
            PipelineType.OBJECT_COUNTING: ObjectCountingPipeline,
        }
        pipeline_class = pipeline_classes.get(pipeline_type)
        if pipeline_class:
            # 在多进程模式下，创建新的GeminiClient实例（避免序列化问题）
            # 在多线程模式下，可以共享主进程的client
            if self.use_multiprocessing:
                # 每个进程独立创建client，避免序列化包含线程锁的对象
                gemini_client = GeminiClient()
                return pipeline_class(gemini_client)
            else:
                # 多线程模式下可以共享client
                return pipeline_class(self.gemini_client)
        return None
    
    def process_batch_image(self, image_inputs: List, metadata_list: List[Dict] = None, 
                           use_concurrent: bool = True, save_interval: int = 100,
                           output_path: Optional[Path] = None) -> List[Dict[str, Any]]:
        """
        批量处理图片（支持并发和增量保存）
        
        Args:
            image_inputs: 图片列表 any type
            metadata_list: 可选的元数据列表（与image_inputs一一对应）
            use_concurrent: 是否使用并发处理
            save_interval: 每处理多少张图片保存一次结果（0表示不增量保存）
            output_path: 输出文件路径（用于增量保存）
            
        Returns:
            处理结果列表
        """
        if metadata_list is None:
            metadata_list = [None] * len(image_inputs)
        
        total = len(image_inputs)
        print(f"[INFO] 开始处理 {total} 张图片，并发数: {self.max_workers if use_concurrent else 1}")
        
        if not use_concurrent or total == 1:
            # 串行处理
            results = []
            for idx, (image_input, metadata) in enumerate(zip(image_inputs, metadata_list), 1):
                try:
                    result = self.process_image(image_input, metadata)
                    results.append(result)
                    if idx % 10 == 0:
                        print(f"[进度] {idx}/{total} ({idx*100//total}%)")
                except Exception as e:
                    results.append({
                        "image_input": str(image_input),
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    })
                finally:
                    # 定期垃圾回收
                    if idx % 50 == 0:
                        gc.collect()
            
            return results
        
        # 并发处理
        results = []
        executor_class = ProcessPoolExecutor if self.use_multiprocessing else ThreadPoolExecutor
        
        # 准备任务数据
        tasks = []
        for image_input, metadata in zip(image_inputs, metadata_list):
            tasks.append({
                "image_input": image_input,
                "metadata": metadata
            })
        
        completed = 0
        start_time = time.time()
        
        with executor_class(max_workers=self.max_workers) as executor:
            # 如果使用多GPU，分配任务到不同GPU
            if self.num_gpus > 0 and self.use_multiprocessing:
                # 多GPU + 多进程模式：使用独立函数避免序列化问题
                tasks_per_gpu = len(tasks) // self.num_gpus
                future_to_task = {}
                for gpu_id in range(self.num_gpus):
                    start_idx = gpu_id * tasks_per_gpu
                    if gpu_id == self.num_gpus - 1:
                        end_idx = len(tasks)
                    else:
                        end_idx = (gpu_id + 1) * tasks_per_gpu
                    
                    for task in tasks[start_idx:end_idx]:
                        future = executor.submit(
                            _process_image_worker,  # 使用独立函数
                            task["image_input"],
                            task["metadata"],
                            gpu_id
                        )
                        future_to_task[future] = task
            elif self.use_multiprocessing:
                # 多进程模式（无GPU）：使用独立函数避免序列化问题
                future_to_task = {
                    executor.submit(_process_image_worker, task["image_input"], task["metadata"], None): task
                    for task in tasks
                }
            else:
                # 多线程模式：可以使用类方法（线程安全）
                future_to_task = {
                    executor.submit(self.process_image, task["image_input"], task["metadata"]): task
                    for task in tasks
                }
            
            # 处理完成的任务
            for future in as_completed(future_to_task):
                completed += 1
                try:
                    result = future.result()
                    results.append(result)
                    
                    # 进度报告
                    if completed % 10 == 0 or completed == total:
                        elapsed = time.time() - start_time
                        rate = completed / elapsed if elapsed > 0 else 0
                        eta = (total - completed) / rate if rate > 0 else 0
                        print(f"[进度] {completed}/{total} ({completed*100//total}%) | "
                              f"速度: {rate:.2f} 张/秒 | 预计剩余: {eta:.0f} 秒")
                    
                    # 增量保存
                    if save_interval > 0 and completed % save_interval == 0 and output_path:
                        self._append_results(results[-save_interval:], output_path)
                        print(f"[INFO] 已增量保存 {completed} 个结果到 {output_path}")
                    
                except Exception as e:
                    task = future_to_task[future]
                    results.append({
                        "image_input": str(task["image_input"]),
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    })
                
                finally:
                    # 定期垃圾回收
                    if completed % 50 == 0:
                        gc.collect()
        
        return results
    
    def process_json(self, json_path: Path, use_concurrent: bool = True, 
                    save_interval: int = 100, output_path: Optional[Path] = None,
                    batch_size: int = 1000) -> List[Dict[str, Any]]:
        """
        从JSON文件批量处理图片（支持流式读取和并发处理）
        
        Args:
            json_path: JSON文件路径
            use_concurrent: 是否使用并发处理
            save_interval: 每处理多少张图片保存一次结果（0表示不增量保存）
            output_path: 输出文件路径（用于增量保存）
            batch_size: 批处理大小（每次从JSON读取多少条记录）
            
        Returns:
            处理结果列表
        """
        print(f"[INFO] 开始从JSON文件读取数据: {json_path}")
        
        # 流式读取JSON文件
        route_results = list(self.router.route_from_json(json_path))
        total = len(route_results)
        
        print(f"[INFO] 共找到 {total} 条记录，开始处理，并发数: {self.max_workers if use_concurrent else 1}")
        
        if not use_concurrent or total == 1:
            # 串行处理
            results = []
            for idx, item in enumerate(route_results, 1):
                try:
                    result = self._process_single_item(item)
                    results.append(result)
                    if idx % 10 == 0:
                        print(f"[进度] {idx}/{total} ({idx*100//total}%)")
                except Exception as e:
                    # 即使出错也保留原始数据
                    original_data = {k: v for k, v in item.items() 
                                   if k not in ["image_input", "pipeline_types"]}
                    results.append({
                        **original_data,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    })
                finally:
                    if idx % 50 == 0:
                        gc.collect()
            
            return results
        
        # 并发处理
        results = []
        executor_class = ProcessPoolExecutor if self.use_multiprocessing else ThreadPoolExecutor
        
        completed = 0
        start_time = time.time()
        
        with executor_class(max_workers=self.max_workers) as executor:
            # 如果使用多GPU，分配任务到不同GPU
            if self.num_gpus > 0 and self.use_multiprocessing:
                # 多GPU + 多进程模式：使用独立函数避免序列化问题
                tasks_per_gpu = len(route_results) // self.num_gpus
                future_to_item = {}
                for gpu_id in range(self.num_gpus):
                    start_idx = gpu_id * tasks_per_gpu
                    if gpu_id == self.num_gpus - 1:
                        end_idx = len(route_results)
                    else:
                        end_idx = (gpu_id + 1) * tasks_per_gpu
                    
                    for item in route_results[start_idx:end_idx]:
                        future = executor.submit(
                            _process_single_item_worker,  # 使用独立函数
                            item,
                            gpu_id
                        )
                        future_to_item[future] = item
            elif self.use_multiprocessing:
                # 多进程模式（无GPU）：使用独立函数避免序列化问题
                future_to_item = {
                    executor.submit(_process_single_item_worker, item, None): item
                    for item in route_results
                }
            else:
                # 多线程模式：可以使用类方法（线程安全）
                future_to_item = {
                    executor.submit(self._process_single_item, item): item
                    for item in route_results
                }
            
            # 处理完成的任务
            for future in as_completed(future_to_item):
                completed += 1
                try:
                    result = future.result()
                    results.append(result)
                    
                    # 进度报告
                    if completed % 10 == 0 or completed == total:
                        elapsed = time.time() - start_time
                        rate = completed / elapsed if elapsed > 0 else 0
                        eta = (total - completed) / rate if rate > 0 else 0
                        print(f"[进度] {completed}/{total} ({completed*100//total}%) | "
                              f"速度: {rate:.2f} 条/秒 | 预计剩余: {eta:.0f} 秒")
                    
                    # 增量保存
                    if save_interval > 0 and completed % save_interval == 0 and output_path:
                        self._append_results(results[-save_interval:], output_path)
                        print(f"[INFO] 已增量保存 {completed} 个结果到 {output_path}")
                    
                except Exception as e:
                    item = future_to_item[future]
                    # 即使出错也保留原始数据
                    original_data = {k: v for k, v in item.items() 
                                   if k not in ["image_input", "pipeline_types"]}
                    results.append({
                        **original_data,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    })
                
                finally:
                    if completed % 50 == 0:
                        gc.collect()
        
        return results
    
    def _append_results(self, results: List[Dict[str, Any]], output_path: Path):
        """
        增量追加结果到文件（优化版本：使用追加模式，避免读取整个文件）
        
        Args:
            results: 要追加的结果列表
            output_path: 输出文件路径
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 如果文件不存在，创建新文件并写入初始数组
        if not output_path.exists():
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('[\n')
                # 写入第一个结果（如果有）
                if results:
                    json.dump(results[0], f, ensure_ascii=False, indent=2)
                    for result in results[1:]:
                        f.write(',\n')
                        json.dump(result, f, ensure_ascii=False, indent=2)
                f.write('\n]')
                return
        
        # 文件已存在，需要追加
        # 读取文件内容，找到最后一个]的位置
        try:
            with open(output_path, 'r+', encoding='utf-8') as f:
                # 移动到文件末尾
                f.seek(0, 2)
                file_size = f.tell()
                
                if file_size > 2:  # 文件不为空（至少有"[]"）
                    # 回退到最后一个]之前
                    f.seek(file_size - 1)
                    # 找到最后一个]的位置
                    while f.tell() > 0:
                        char = f.read(1)
                        if char == ']':
                            f.seek(f.tell() - 1)
                            break
                        f.seek(f.tell() - 2)
                    
                    # 删除最后的]
                    f.seek(f.tell() - 1)
                    f.truncate()
                    
                    # 添加逗号和换行
                    f.write(',\n')
                else:
                    # 文件为空，写入初始[
                    f.seek(0)
                    f.write('[\n')
                
                # 追加新结果
                for i, result in enumerate(results):
                    if file_size > 2 and i == 0:
                        # 第一个结果前不需要逗号（已经在上面添加了）
                        pass
                    elif i > 0:
                        f.write(',\n')
                    json.dump(result, f, ensure_ascii=False, indent=2)
                
                # 写入结束的]
                f.write('\n]')
        except Exception as e:
            # 如果追加模式失败，回退到读取-追加-写入模式
            print(f"[WARNING] 追加模式失败，使用回退模式: {e}")
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    existing_results = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                existing_results = []
            
            existing_results.extend(results)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(existing_results, f, ensure_ascii=False, indent=2)
    
    def save_results(self, results: List[Dict[str, Any]], output_path: Path = None):
        """
        保存处理结果到JSON文件
        
        Args:
            results: 处理结果列表
            output_path: 输出文件路径，如果为None则使用默认路径
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = config.OUTPUT_DIR / f"filter_results_{timestamp}.json"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"结果已保存到: {output_path}")


def main():
    """主函数"""
    
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='图片筛选系统')
    parser.add_argument('--image', type=str, help='单张图片路径')
    parser.add_argument('--dir', type=str, help='图片目录路径')
    parser.add_argument('--json', type=str, help='JSON文件路径（包含图片路径和元数据）')
    parser.add_argument('--output', type=str, help='输出文件路径')
    parser.add_argument('--extensions', nargs='+', default=['jpg', 'jpeg', 'png', 'webp'],
                       help='图片文件扩展名')
    parser.add_argument('--workers', type=int, default=4, help='并发工作线程/进程数（默认: 4）')
    parser.add_argument('--no-concurrent', action='store_true', help='禁用并发处理')
    parser.add_argument('--multiprocessing', action='store_true', help='使用多进程而非多线程')
    parser.add_argument('--save-interval', type=int, default=100, 
                       help='增量保存间隔（每处理多少张图片保存一次，0表示不增量保存，默认: 100）')
    parser.add_argument('--num-gpus', type=int, default=0,
                       help='GPU数量（用于多GPU处理，默认: 0）')
    
    args = parser.parse_args()
    
    # 检查API密钥
    if not config.API_KEY:
        print("错误: 未设置API_KEY")
        print("请在.env文件中设置API_KEY，或设置环境变量")
        print("示例: API_KEY=your_api_key_here")
        return
    
    # 初始化系统
    system = ImageFilterSystem(
        max_workers=args.workers,
        use_multiprocessing=args.multiprocessing,
        num_gpus=args.num_gpus
    )
    
    # 根据输入类型处理
    results = []
    use_concurrent = not args.no_concurrent
    save_interval = args.save_interval if args.save_interval > 0 else 0
    output_path = Path(args.output) if args.output else None
    
    if args.json:
        # JSON批量处理模式
        json_path = Path(args.json)
        if not json_path.exists():
            print(f"错误: JSON文件不存在: {json_path}")
            return
        
        print(f"从JSON文件读取数据: {json_path}")
        results = system.process_json(
            json_path, 
            use_concurrent=use_concurrent,
            save_interval=save_interval,
            output_path=output_path
        )
        
    else:
        # 图片路径处理模式（原有逻辑）
        image_paths = []
        
        if args.image:
            # 单张图片
            image_paths.append(Path(args.image))
        elif args.dir:
            # 目录批量处理
            dir_path = Path(args.dir)
            if not dir_path.exists():
                print(f"错误: 目录不存在: {dir_path}")
                return
            
            for ext in args.extensions:
                image_paths.extend(dir_path.glob(f"*.{ext}"))
                image_paths.extend(dir_path.glob(f"*.{ext.upper()}"))
        else:
            # 默认使用input目录
            input_dir = config.INPUT_DIR
            if not input_dir.exists():
                print(f"提示: 输入目录不存在，已创建: {input_dir}")
                input_dir.mkdir(parents=True, exist_ok=True)
                print(f"请将图片放入 {input_dir} 目录后重新运行")
                return
            
            for ext in args.extensions:
                image_paths.extend(input_dir.glob(f"*.{ext}"))
                image_paths.extend(input_dir.glob(f"*.{ext.upper()}"))
        
        if not image_paths:
            print("未找到图片文件")
            return
        
        print(f"找到 {len(image_paths)} 张图片，开始处理...")
        
        # 处理图片
        results = system.process_batch_image(
            image_paths,
            use_concurrent=use_concurrent,
            save_interval=save_interval,
            output_path=output_path
        )
    
    # 如果使用了增量保存，最终结果已经在文件中
    if save_interval > 0 and output_path and output_path.exists():
        print(f"\n所有结果已增量保存到: {output_path}")
    else:
        # 保存结果
        system.save_results(results, output_path)
    
    # 打印摘要
    print("\n处理摘要:")
    print(f"总图片数: {len(results)}")
    success_count = sum(1 for r in results if "error" not in r)
    error_count = len(results) - success_count
    print(f"成功: {success_count}, 失败: {error_count}")
    
    for i, result in enumerate(results[:10], 1):  # 只显示前10个
        if "error" in result:
            # 尝试从原始数据中获取标识信息
            item_id = result.get('id', result.get('sample_index', 'unknown'))
            print(f"  记录 {i} (ID: {item_id}): 处理失败 - {result.get('error')}")
        else:
            # 尝试从原始数据中获取标识信息
            item_id = result.get('id', result.get('sample_index', 'unknown'))
            print(f"  记录 {i} (ID: {item_id})")
            
            # 处理不同的结果格式
            if "results" in result:
                # process_image 返回的格式
                for res in result.get("results", []):
                    status = "✓ 通过" if res.get("passed") else "✗ 未通过"
                    pipeline_name = res.get('pipeline_name', res.get('pipeline_type', 'unknown'))
                    print(f"    - {pipeline_name}: {status} (置信度: {res.get('confidence', 0):.2f})")
            else:
                # process_json 返回的格式
                status = "✓ 通过" if result.get("passed") else "✗ 未通过"
                pipeline_name = result.get('pipeline_name', result.get('pipeline_type', 'unknown'))
                print(f"    - {pipeline_name}: {status} (置信度: {result.get('confidence', 0):.2f})")
    
    if len(results) > 10:
        print(f"  ... 还有 {len(results) - 10} 个结果未显示")


if __name__ == "__main__":
    main()


# 方案1：多进程 + 频繁保存（推荐用于25GB+数据）
# python main.py \
#   --json "/home/zhuxuzhou/test_localization/object_localization/data/match_results/object_localization/finegrained_perception (instance-level)/part-00000.json" \
#   --output "output.json" \
#   --workers 8 \
#   --multiprocessing \
#   --save-interval 50 \
#   > new_output.log 2>&1

# # 方案2：后台运行（推荐）
# nohup python main.py \
#   --json "input.json" \
#   --output "output.json" \
#   --workers 8 \
#   --multiprocessing \
#   --save-interval 100 \
#   > output.log 2>&1 &