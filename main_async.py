"""
异步并行筛选系统主程序
使用 AsyncGeminiClient 和 asyncio 实现高并发筛选
"""
import sys
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
from datetime import datetime
import argparse
import time
import gc

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.router import Router, PipelineType
from src.pipelines import *
from utils.async_client import AsyncGeminiClient
import config


async def process_single_item_async(
    item: Dict[str, Any],
    async_client: AsyncGeminiClient
) -> Dict[str, Any]:
    """
    异步处理单个数据项
    
    Args:
        item: 包含image_input和pipeline_types的数据项
        async_client: 异步Gemini客户端
        
    Returns:
        处理结果字典
    """
    try:
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
        
        # 获取pipeline配置
        pipeline_config = config.PIPELINE_CONFIG.get(pipeline_type.value, {})
        criteria_description = pipeline_config.get("criteria", "")
        question = pipeline_config.get("question", "")
        
        if not criteria_description or not question:
            return {
                **original_data,
                "error": f"Pipeline config missing for {pipeline_type.value}",
                "timestamp": datetime.now().isoformat()
            }
        
        # 使用异步客户端进行筛选
        filter_result = await async_client.filter_image_async(
            image_input=image_input,
            criteria_description=criteria_description,
            question=question,
            temperature=0.3
        )
        
        # 合并原始数据和筛选结果
        result = {
            **original_data,
            **filter_result,
            "pipeline_type": pipeline_type.value,
            "pipeline_name": pipeline_config.get("name", pipeline_type.value),
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


async def process_batch_async(
    items: List[Dict[str, Any]],
    num_gpus: int = 1,
    max_concurrent_per_gpu: int = 10,
    request_delay: float = 0.1,
    save_interval: int = 100,
    output_path: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """
    使用多GPU异步处理批量数据
    
    Args:
        items: 待处理的数据项列表
        num_gpus: GPU数量（用于进程隔离，实际是API并发控制）
        max_concurrent_per_gpu: 每个GPU的最大并发数
        request_delay: 每个请求之间的延迟（秒）
        save_interval: 增量保存间隔
        output_path: 输出文件路径（用于增量保存）
        
    Returns:
        处理结果列表
    """
    # 将任务分配到不同的GPU组
    tasks_per_gpu = len(items) // num_gpus if num_gpus > 1 else len(items)
    gpu_tasks = []
    
    for gpu_id in range(num_gpus):
        start_idx = gpu_id * tasks_per_gpu
        if gpu_id == num_gpus - 1:
            end_idx = len(items)  # 最后一个GPU处理剩余所有任务
        else:
            end_idx = (gpu_id + 1) * tasks_per_gpu
        
        gpu_tasks.append((gpu_id, items[start_idx:end_idx]))
    
    # 为每个GPU创建处理任务
    async def process_gpu_tasks(gpu_id: int, tasks: List[Dict]):
        """处理单个GPU的任务"""
        results = []
        async with AsyncGeminiClient(
            gpu_id=gpu_id,
            max_concurrent=max_concurrent_per_gpu,
            request_delay=request_delay
        ) as client:
            # 创建所有异步任务
            async_tasks = []
            for task in tasks:
                async_task = process_single_item_async(
                    item=task,
                    async_client=client
                )
                async_tasks.append(async_task)
            
            # 等待所有任务完成（使用return_exceptions=True以处理异常）
            task_results = await asyncio.gather(*async_tasks, return_exceptions=True)
            
            # 处理异常结果
            for i, result in enumerate(task_results):
                if isinstance(result, Exception):
                    original_data = {k: v for k, v in tasks[i].items() 
                                   if k not in ["image_input", "pipeline_types"]}
                    results.append({
                        **original_data,
                        "error": str(result),
                        "timestamp": datetime.now().isoformat()
                    })
                else:
                    results.append(result)
        
        return results
    
    # 并发处理所有GPU的任务
    all_results = await asyncio.gather(*[
        process_gpu_tasks(gpu_id, tasks)
        for gpu_id, tasks in gpu_tasks
    ])
    
    # 合并结果
    final_results = []
    for gpu_results in all_results:
        final_results.extend(gpu_results)
    
    return final_results


async def process_json_async(
    json_path: Path,
    num_gpus: int = 1,
    max_concurrent_per_gpu: int = 10,
    request_delay: float = 0.1,
    save_interval: int = 100,
    output_path: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """
    从JSON文件异步批量处理图片
    
    Args:
        json_path: JSON文件路径
        num_gpus: GPU数量
        max_concurrent_per_gpu: 每个GPU的最大并发数
        request_delay: 每个请求之间的延迟（秒）
        save_interval: 增量保存间隔
        output_path: 输出文件路径（用于增量保存）
        
    Returns:
        处理结果列表
    """
    print(f"[INFO] 开始从JSON文件读取数据: {json_path}")
    
    # 使用Router的路由功能（不使用LLM，提高速度）
    route_results = Router.route_from_json(json_path, use_llm=False)
    
    total = len(route_results)
    print(f"[INFO] 共找到 {total} 条记录，开始异步处理")
    print(f"[INFO] GPU数量: {num_gpus}, 每GPU并发数: {max_concurrent_per_gpu}")
    
    if total == 0:
        return []
    
    # 分批处理（避免一次性创建太多任务）
    batch_size = num_gpus * max_concurrent_per_gpu * 2  # 每批处理的任务数
    all_results = []
    completed = 0
    start_time = time.time()
    
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_items = route_results[batch_start:batch_end]
        
        print(f"[INFO] 处理批次 {batch_start // batch_size + 1}: {batch_start}-{batch_end}/{total}")
        
        # 处理当前批次
        batch_results = await process_batch_async(
            items=batch_items,
            num_gpus=num_gpus,
            max_concurrent_per_gpu=max_concurrent_per_gpu,
            request_delay=request_delay,
            save_interval=0,  # 批次内不保存，批次间保存
            output_path=None
        )
        
        all_results.extend(batch_results)
        completed = len(all_results)
        
        # 进度报告
        elapsed = time.time() - start_time
        rate = completed / elapsed if elapsed > 0 else 0
        eta = (total - completed) / rate if rate > 0 else 0
        print(f"[进度] {completed}/{total} ({completed*100//total}%) | "
              f"速度: {rate:.2f} 条/秒 | 预计剩余: {eta:.0f} 秒")
        
        # 增量保存
        if save_interval > 0 and output_path and completed % save_interval == 0:
            _append_results(all_results[-save_interval:], output_path)
            print(f"[INFO] 已增量保存 {completed} 个结果到 {output_path}")
        
        # 定期垃圾回收
        if completed % 50 == 0:
            gc.collect()
    
    return all_results


def _append_results(results: List[Dict[str, Any]], output_path: Path):
    """
    增量追加结果到文件
    
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


def save_results(results: List[Dict[str, Any]], output_path: Path = None):
    """
    保存处理结果到JSON文件
    
    Args:
        results: 处理结果列表
        output_path: 输出文件路径，如果为None则使用默认路径
    """
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = config.OUTPUT_DIR / f"filter_results_async_{timestamp}.json"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"结果已保存到: {output_path}")


async def main_async():
    """异步主函数"""
    
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='异步并行图片筛选系统')
    parser.add_argument('--json', type=str, help='JSON文件路径（包含图片路径和元数据）')
    parser.add_argument('--output', type=str, help='输出文件路径')
    parser.add_argument('--num-gpus', type=int, default=1,
                       help='GPU数量（用于并发控制，默认: 1）')
    parser.add_argument('--concurrency', type=int, default=10,
                       help='每个GPU的最大并发数（默认: 10）')
    parser.add_argument('--request-delay', type=float, default=0.1,
                       help='每个请求之间的延迟（秒，默认: 0.1）')
    parser.add_argument('--save-interval', type=int, default=100,
                       help='增量保存间隔（每处理多少条记录保存一次，0表示不增量保存，默认: 100）')
    
    args = parser.parse_args()
    
    # 检查API密钥
    if not config.API_KEY:
        print("错误: 未设置API_KEY")
        print("请在.env文件中设置API_KEY，或设置环境变量")
        print("示例: API_KEY=your_api_key_here")
        return
    
    # 根据输入类型处理
    results = []
    save_interval = args.save_interval if args.save_interval > 0 else 0
    output_path = Path(args.output) if args.output else None
    
    if args.json:
        # JSON批量处理模式
        json_path = Path(args.json)
        if not json_path.exists():
            print(f"错误: JSON文件不存在: {json_path}")
            return
        
        print(f"从JSON文件读取数据: {json_path}")
        results = await process_json_async(
            json_path=json_path,
            num_gpus=args.num_gpus,
            max_concurrent_per_gpu=args.concurrency,
            request_delay=args.request_delay,
            save_interval=save_interval,
            output_path=output_path
        )
    else:
        print("错误: 必须指定 --json 参数")
        return
    
    # 如果使用了增量保存，最终结果已经在文件中
    if save_interval > 0 and output_path and output_path.exists():
        print(f"\n所有结果已增量保存到: {output_path}")
    else:
        # 保存结果
        save_results(results, output_path)
    
    # 打印摘要
    print("\n处理摘要:")
    print(f"总记录数: {len(results)}")
    success_count = sum(1 for r in results if "error" not in r)
    error_count = len(results) - success_count
    print(f"成功: {success_count}, 失败: {error_count}")


def main():
    """主函数入口（同步包装器）"""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

