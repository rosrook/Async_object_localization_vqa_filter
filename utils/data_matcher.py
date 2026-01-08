"""
数据匹配模块
从重分类文件夹中读取数据,与基准parquet文件匹配,并输出JSON格式
"""
import os
import json
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Any, Dict, Optional, Tuple
import base64
import imghdr
from pathlib import Path
import config


def serialize_value(value: Any) -> Any:
    """
    将PyArrow值序列化为JSON可存储的格式
    """
    if value is None:
        return None
    
    # 处理bytes类型(如图像数据)
    if isinstance(value, bytes):
        return base64.b64encode(value).decode('utf-8')
    
    # 处理PyArrow的binary类型
    if isinstance(value, pa.lib.BinaryScalar):
        return base64.b64encode(value.as_py()).decode('utf-8')
    
    # 处理列表
    if isinstance(value, list):
        return [serialize_value(v) for v in value]
    
    # 处理字典
    if isinstance(value, dict):
        return {k: serialize_value(v) for k, v in value.items()}
    
    # 处理PyArrow结构体
    if hasattr(value, 'as_py'):
        return serialize_value(value.as_py())
    
    # 其他类型直接返回
    return value


def check_has_image(record: Dict[str, Any]) -> bool:
    """
    检查记录中是否包含图像数据
    常见的图像字段: jpg, png, images, image, img
    """
    if record is None:
        return False
        
    image_fields = ['jpg', 'png', 'images', 'image', 'img']
    
    for field in image_fields:
        if field in record:
            value = record[field]
            # 检查是否为非空的bytes或binary数据
            if value is not None:
                if isinstance(value, bytes) and len(value) > 0:
                    return True
                if isinstance(value, str) and is_base64(value):
                    return True
                if isinstance(value, dict) and 'bytes' in value:
                    return True
    
    return False


def is_base64(s: Any, min_len: int = 50) -> bool:
    """
    判断字符串是否可能是 Base64 编码
    增强版本：更严格的验证
    """
    if not isinstance(s, str):
        return False
    
    if len(s) < min_len:
        return False
    
    # Base64字符检查
    import re
    if not re.match(r'^[A-Za-z0-9+/]*={0,2}$', s):
        return False
    
    try:
        decoded = base64.b64decode(s, validate=True)
        # 检查解码后的数据是否看起来像图片（检查常见图片文件头）
        if len(decoded) < 10:
            return False
        
        # 检查常见图片格式的魔数
        image_signatures = [
            b'\xff\xd8\xff',  # JPEG
            b'\x89PNG\r\n\x1a\n',  # PNG
            b'GIF87a',  # GIF
            b'GIF89a',  # GIF
            b'BM',  # BMP
            b'II*\x00',  # TIFF
            b'MM\x00*',  # TIFF
        ]
        
        for sig in image_signatures:
            if decoded.startswith(sig):
                return True
        
        # 如果没有识别出魔数，但长度足够大，也认为可能是图片
        if len(decoded) > 1000:
            return True
            
        return False
    except Exception:
        return False


def find_base64_field(obj: dict) -> Optional[Tuple[str, str]]:
    """
    返回对象中第一个可能的 Base64 字段
    返回: (字段名, base64字符串) 或 None
    """
    if not isinstance(obj, dict):
        return None
    
    # 优先检查常见的图片字段
    priority_fields = ['jpg', 'png', 'image', 'images', 'img']
    
    # 先检查优先字段
    for field in priority_fields:
        if field in obj:
            value = obj[field]
            if isinstance(value, str) and is_base64(value):
                return (field, value)
    
    # 再检查其他字段
    for k, v in obj.items():
        if k not in priority_fields and isinstance(v, str) and is_base64(v):
            return (k, v)
    
    return None


def save_base64_image(img_b64: str, save_dir: str, prefix: str, index: int, 
                      parquet_name: str = None, sid: Any = None) -> Optional[str]:
    """
    将 Base64 编码保存为图片
    自动检测图片类型
    返回图片路径
    
    Args:
        img_b64: Base64编码的图片
        save_dir: 保存目录
        prefix: 前缀 (a/b)
        index: 样本索引
        parquet_name: parquet文件名（用于区分不同文件）
        sid: 样本ID（用于唯一标识）
    """
    try:
        img_bytes = base64.b64decode(img_b64)
    except Exception as e:
        print(f"[ERROR] Base64 decode failed for {prefix}[{index}]: {e}")
        return None

    if len(img_bytes) == 0:
        print(f"[ERROR] Decoded image is empty for {prefix}[{index}]")
        return None

    # 自动检测图片类型
    img_type = imghdr.what(None, h=img_bytes)
    if img_type is None:
        # 如果无法检测类型，尝试从常见格式推断
        if img_bytes.startswith(b'\xff\xd8\xff'):
            img_type = 'jpeg'
        elif img_bytes.startswith(b'\x89PNG'):
            img_type = 'png'
        else:
            print(f"[WARNING] Cannot detect image type for {prefix}[{index}], defaulting to jpg")
            img_type = 'jpg'

    # 构建唯一的图片文件名
    # 格式: {parquet_name}_{sid}_{prefix}.{img_type}
    if parquet_name and sid is not None:
        img_name = f"{parquet_name}_{sid}_{prefix}.{img_type}"
    else:
        # 兜底方案：使用索引
        img_name = f"{prefix}_{index}.{img_type}"
    
    img_path = Path(save_dir) / img_name
    
    try:
        with open(img_path, "wb") as f:
            f.write(img_bytes)
        # 移除详细的图片保存日志
        return str(img_path)
    except Exception as e:
        print(f"[ERROR] Failed to write image to disk for {prefix}[{index}]: {e}")
        return None


def collect_category_dirs(root_dir):
    """
    收集所有类别目录
    返回: [(category, l2category, full_path), ...]
    """
    categories = []
    
    if not os.path.exists(root_dir):
        print(f"[WARNING] Root directory does not exist: {root_dir}")
        return categories
    
    for category in os.listdir(root_dir):
        cat_path = os.path.join(root_dir, category)
        if not os.path.isdir(cat_path):
            continue
        
        for l2category in os.listdir(cat_path):
            l2cat_path = os.path.join(cat_path, l2category)
            if not os.path.isdir(l2cat_path):
                continue
            
            categories.append((category, l2category, l2cat_path))
    
    return categories


def load_benchmark_data(benchmark_file: str) -> Dict[int, Dict[str, Any]]:
    """
    加载基准文件,以index为key建立索引
    """
    print(f"[INFO] Loading benchmark file: {benchmark_file}")
    
    if not os.path.exists(benchmark_file):
        raise FileNotFoundError(f"Benchmark file does not exist: {benchmark_file}")
    
    table = pq.read_table(benchmark_file)
    
    if 'index' not in table.schema.names:
        raise ValueError("Benchmark file must have 'index' column")
    
    # 转换为字典,以index为key
    benchmark_dict = {}
    
    for i in range(table.num_rows):
        row_dict = {}
        for col_name in table.schema.names:
            row_dict[col_name] = table[col_name][i].as_py()
        
        index_value = row_dict['index']
        benchmark_dict[index_value] = row_dict
    
    print(f"[INFO] Loaded {len(benchmark_dict)} records from benchmark")
    return benchmark_dict


def process_category(
    category: str,
    l2category: str,
    cat_path: str,
    benchmark_dict: Dict[int, Dict[str, Any]],
    output_dir: str,
    test_mode: bool = False,
    test_samples: int = 5
):
    """
    处理单个类别目录
    """
    parquet_files = [
        os.path.join(cat_path, f)
        for f in os.listdir(cat_path)
        if f.endswith(".parquet")
    ]

    if not parquet_files:
        print(f"[WARNING] 未找到parquet文件: {cat_path}")
        return

    # 临时图片输出目录（不保存临时图片文件，节省磁盘空间）
    temp_img_dir = None

    for pq_idx, pq_file in enumerate(parquet_files, 1):
        parquet_name = os.path.splitext(os.path.basename(pq_file))[0]
        print(f"[INFO] 处理文件 [{pq_idx}/{len(parquet_files)}]: {parquet_name}")

        try:
            table = pq.read_table(pq_file)
        except Exception as e:
            print(f"[ERROR] Failed to read {pq_file}: {e}")
            continue

        if 'id' not in table.schema.names:
            print(f"[WARNING] No 'id' column in {pq_file}, skipping")
            continue

        total_rows = table.num_rows
        if test_mode:
            process_rows = min(test_samples, total_rows)
            table = table.slice(0, process_rows)
        else:
            process_rows = total_rows

        matched_results = []
        sample_index = 0
        matched_count = 0
        unmatched_count = 0

        for i in range(table.num_rows):
            # 构造 source_a
            source_a = {
                col_name: serialize_value(table[col_name][i].as_py())
                for col_name in table.schema.names
            }

            sid = source_a.get('id')

            # ==== 处理 source_a 图片 ====
            # 如果不需要保存图片文件，跳过图片保存步骤
            # 图片数据已经在source_a的base64字段中，后续处理可以直接使用
            if temp_img_dir:
                try:
                    result = find_base64_field(source_a)
                    if result:
                        field_name, img_b64 = result
                        img_path = save_base64_image(
                            img_b64=img_b64,
                            save_dir=temp_img_dir,
                            prefix="a",
                            index=sample_index,
                            parquet_name=parquet_name,
                            sid=sid
                        )
                        if img_path:
                            source_a["image_path"] = img_path
                except Exception:
                    pass  # 静默处理错误

            # 查找匹配的基准数据
            record_b = benchmark_dict.get(sid)
            source_b = None

            if record_b:
                source_b = {k: serialize_value(v) for k, v in record_b.items()}

                # ==== 处理 source_b 图片 ====
                # 如果不需要保存图片文件，只保留image_path字段指向base64数据
                if temp_img_dir:
                    try:
                        result = find_base64_field(source_b)
                        if result:
                            field_name, img_b64 = result
                            img_path = save_base64_image(
                                img_b64=img_b64,
                                save_dir=temp_img_dir,
                                prefix="b",
                                index=sample_index,
                                parquet_name=parquet_name,
                                sid=sid
                            )
                            if img_path:
                                source_b["image_path"] = img_path
                    except Exception:
                        pass  # 静默处理错误

                matched_count += 1
            else:
                unmatched_count += 1

            result = {
                "sample_index": sample_index,
                "id": sid,
                "source_a": source_a,
                "source_b": source_b,
                "has_image_a": check_has_image(source_a),
                "has_image_b": check_has_image(source_b) if source_b else False
            }

            matched_results.append(result)
            sample_index += 1

        # 输出 JSON
        output_subdir = os.path.join(output_dir, category, l2category)
        os.makedirs(output_subdir, exist_ok=True)
        parquet_name = os.path.splitext(os.path.basename(pq_file))[0]
        filename = f"{parquet_name}_test.json" if test_mode else f"{parquet_name}.json"
        output_file = os.path.join(output_subdir, filename)

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(matched_results, f, ensure_ascii=False, indent=2)
            print(f"[INFO] 已保存 {len(matched_results)} 条记录 (匹配: {matched_count}, 未匹配: {unmatched_count})")
        except Exception as e:
            print(f"[ERROR] Failed to write JSON to disk: {e}")


def match_data(
    recat_root: Optional[str] = None,
    benchmark_file: Optional[str] = None,
    output_dir: Optional[str] = None,
    target_categories: Optional[list] = None,
    test_mode: bool = False,
    test_samples: int = 5,
    test_max_categories: int = 2
) -> str:
    """
    数据匹配主函数
    
    Args:
        recat_root: 重分类后的根目录
        benchmark_file: 基准parquet文件路径
        output_dir: JSON输出目录
        target_categories: 要处理的类别列表，格式: [(category, l2category), ...]，如果为None则处理所有类别
        test_mode: 是否启用测试模式
        test_samples: 测试模式下每个类别处理的样本数
        test_max_categories: 测试模式下最多处理的类别数
        
    Returns:
        输出目录路径
    """
    # 从config读取默认配置
    recat_root = recat_root or config.RECAT_ROOT
    benchmark_file = benchmark_file or config.BENCHMARK_FILE
    output_dir = output_dir or config.MATCH_OUTPUT_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 显示运行模式（简化输出）
    if test_mode:
        print(f"[INFO] 测试模式: 每类别 {test_samples} 个样本，最多 {test_max_categories} 个类别")
    else:
        print(f"[INFO] 生产模式: 处理所有数据")
    
    # 加载基准数据
    benchmark_dict = load_benchmark_data(benchmark_file)
    
    # 收集所有类别
    all_categories = collect_category_dirs(recat_root)
    
    if not all_categories:
        raise RuntimeError(f"No categories found in {recat_root}")
    
    # 过滤目标类别
    if target_categories:
        target_set = set(target_categories)
        categories_to_process = [
            (cat, l2cat, path) for cat, l2cat, path in all_categories
            if (cat, l2cat) in target_set
        ]
    else:
        categories_to_process = all_categories
    
    # 测试模式:限制类别数量
    if test_mode:
        categories_to_process = categories_to_process[:test_max_categories]
    
    print(f"\n[INFO] Will process {len(categories_to_process)} categories")
    
    # 处理每个类别
    for idx, (category, l2category, cat_path) in enumerate(categories_to_process, 1):
        try:
            print(f"[INFO] [{idx}/{len(categories_to_process)}] 处理类别: {category}/{l2category}")
            process_category(
                category, 
                l2category, 
                cat_path, 
                benchmark_dict, 
                output_dir,
                test_mode=test_mode,
                test_samples=test_samples
            )
        except Exception as e:
            print(f"[ERROR] 处理失败 {category}/{l2category}: {e}")
            continue
    
    print(f"\n[INFO] 处理完成！输出目录: {output_dir}")
    
    return output_dir

