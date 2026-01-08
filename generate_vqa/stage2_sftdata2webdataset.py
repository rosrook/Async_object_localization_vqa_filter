""" 
Convert dataset into WebDataset (WDS) format and generate Megatron indexes

支持的输入格式：
1. VQA JSON 格式：从 pipeline.py 生成的 vqa_dataset_successful_*.json 文件
   - 包含 image_base64, full_question, answer, explanation 等字段
   - 自动检测并处理 Base64 编码的图片
   
2. 原有格式：Parquet 文件或包含 image_url 的 JSON 文件

输出：
- WebDataset 格式的 .tar 文件
- Megatron 索引文件 (.nv-meta/**)
"""
import argparse
import json
import os
import yaml
import webdataset as wds
from tqdm import tqdm
import random
import pyarrow.parquet as pq
from megatron.energon.epathlib import EPath
from megatron.energon.flavors import BaseWebdatasetFactory
from megatron.energon.flavors.webdataset import MAIN_FOLDER_NAME
from glob import glob
import logging
import pandas as pd
from multiprocessing import Pool
from io import BytesIO
from PIL import Image
import re
import base64
def sample_loader_template(media: str=None):
    """Returns a template for a sample_loader.py file."""
    # 根据媒体类型决定返回哪些字段
    if media == 'mix':
        return_fields = "video=video if len(video) > 0 else None, image=image if len(image) > 0 else None,"
    elif media == 'video':
        return_fields = "video=video if len(video) > 0 else None,"
    else:  # image 或其他
        return_fields = "image=image if len(image) > 0 else None,"
    
    return "\n".join([
        "def sample_loader(sample: dict) -> dict:",
        "    messages=[]",
        "    system=None",
        "    for message in sample['json']['texts']:",
        "        assert message['role'] in ['system','user','assistant']",
        "        if message['role'] == 'system':",
        "            system=message['content']",
        "            continue",
        "        messages.append(dict(",
        "            role=message['role'],",
        "            content=message['content']",
        "        ))",
        "    video = []",
        "    image = []",
        "    if sample['json']['media'] == 'video':",
        "        for name in sample['json']['name']:",
        "            video.append(sample.get(name))",
        "    elif sample['json']['media'] == 'image':",
        "        for name in sample['json']['name']:",
        "            image.append(sample.get(name))",
        "    return dict(",
        "        __key__=sample['__key__'],",
        "        __restore_key__=sample['__restore_key__'],",
        return_fields,
        "        system=system,",
        "        messages=messages,",
        "    )",
        "def part_filter(part: str) -> bool:",
        "    return True",
    ])

def apply_template(texts, num_img):
    new_text = []
    for it, text in enumerate(texts):
        if text.get('from', None) == 'system':
            continue
        if text.get('from') == 'user' or text.get('from') == 'human':
            text['role'] = 'user'
            text.pop('from')
        elif text.get('from') == 'gpt' or text.get('from') == 'assistant':
            text['role'] = 'assistant'
            text.pop('from')
        if text.get('value') is not None:
            text['content'] = text.pop('value')
        if it ==0:
            if '<image>' not in text['content'] and num_img > 0:
                imgstr = ['<image>']*num_img
                if text['content'].startswith('\n'):
                    text['content'] = text['content'].lstrip('\n')
                text['content'] = ''.join(imgstr) + '\n' + text['content']
        new_text.append(text)
    return new_text

def apply_question_answer_template(question, answer):
    question = question.strip()
    conv = [
        {'role': 'user',
         'content': question},
        {
            'role': 'assistant',
            'content': answer.strip(),
        }
    ]
    return conv


def build_vqa_conversation(vqa_item):
    """
    从 VQA JSON 数据项构建对话格式
    
    Args:
        vqa_item: VQA 数据项，包含 full_question, answer, explanation 等字段
        
    Returns:
        对话列表，格式: [{'role': 'user', 'content': ...}, {'role': 'assistant', 'content': ...}]
    """
    # 获取完整问题（选择题包含选项，填空题就是问题本身）
    full_question = vqa_item.get('full_question', vqa_item.get('question', ''))
    
    # 构建用户消息：<image> + 问题
    user_content = f'<image>\n{full_question}'
    
    # 构建助手消息：答案 + 解释
    answer = vqa_item.get('answer', '')
    explanation = vqa_item.get('explanation', '')
    options = vqa_item.get('options', {})
    
    # 如果答案是选项字母（如 "B"），且存在 options 字典，则获取完整的答案文本
    if answer and options and isinstance(options, dict):
        answer_upper = answer.upper().strip()
        # 检查答案是否只是选项字母（单个字母）
        if len(answer_upper) == 1 and answer_upper in options:
            # 组合选项字母和选项内容：如 "B. The one with striped spiral shell"
            option_text = options[answer_upper]
            answer_text = f"{answer_upper}. {option_text}"
        else:
            # 答案已经是完整文本，直接使用
            answer_text = answer
    else:
        answer_text = answer
    
    if explanation:
        assistant_content = f"{answer_text}\n\nExplanation: {explanation}"
    else:
        assistant_content = answer_text
    
    conv = [
        {'role': 'user', 'content': user_content},
        {'role': 'assistant', 'content': assistant_content.strip()}
    ]
    
    return conv


def decode_base64_image(image_base64):
    """
    解码 Base64 编码的图片
    
    Args:
        image_base64: Base64 编码的图片字符串
        
    Returns:
        bytes: 图片的字节数据，如果解码失败返回 None
    """
    try:
        # 如果包含 data URL 前缀，需要移除
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        img_bytes = base64.b64decode(image_base64)
        return img_bytes
    except Exception as e:
        logging.warning(f"Base64 图片解码失败: {e}")
        return None


def detect_image_format(img_bytes):
    """
    检测图片格式
    
    Args:
        img_bytes: 图片的字节数据
        
    Returns:
        str: 图片格式扩展名 ('jpg' 或 'png')，如果无法检测则返回 'jpg'
    """
    if not img_bytes or len(img_bytes) < 4:
        return 'jpg'  # 默认使用 jpg
    
    # 检查文件头
    # JPEG: FF D8 FF
    if img_bytes[:3] == b'\xff\xd8\xff':
        return 'jpg'
    # PNG: 89 50 4E 47
    elif img_bytes[:4] == b'\x89PNG':
        return 'png'
    # 尝试使用 PIL 检测
    try:
        from PIL import Image
        img = Image.open(BytesIO(img_bytes))
        format_map = {
            'JPEG': 'jpg',
            'PNG': 'png',
            'JPG': 'jpg'
        }
        return format_map.get(img.format, 'jpg')
    except Exception:
        # 如果无法检测，默认使用 jpg
        return 'jpg'


def check_conversation_format(conv):
    if not isinstance(conv, list):
        return False
    allgood = True
    for ii, turn in enumerate(conv):
        role = turn.get('role', '')
        content = turn.get('content', '')
        if role not in ['user', 'assistant']:
            print("role content error")
            allgood = False
        if len(content) < 1:
            print("content empty error")
            allgood = False
    return allgood

def construct_sample_from_row(row, index, media_type, media_bytes, args):
    """从 Parquet 行构建 WebDataset sample"""
    vision_data = {}
    vision_name = []

    if media_type in ['image', 'video']:
        media_key = 'image' if media_type == 'image' else 'video'
        # if media_key in row and row[media_key] is not None:
            # media_bytes = row[media_key]
        if isinstance(media_bytes, bytes):
            vision_data[f"0_{media_key}"] = media_bytes
            vision_name.append(f"0_{media_key}")
        elif isinstance(media_bytes, dict) and 'bytes' in media_bytes:
            vision_data[f"0_{media_key}"] = media_bytes['bytes']
            vision_name.append(f"0_{media_key}")
        elif isinstance(media_bytes, list):
            for i, media_item in enumerate(media_bytes):
                if isinstance(media_item, bytes):
                    vision_data[f"{i}_{media_key}"] = media_item
                    vision_name.append(f"{i}_{media_key}")
                elif isinstance(media_item, dict) and 'bytes' in media_item:
                    vision_data[f"{i}_{media_key}"] =  media_item['bytes']
                    vision_name.append(f"{i}_{media_key}")
        else:
            logging.warning(f"未知的媒体数据格式，跳过{row.get('__source_file__', '')} index {index}")
            return None

    conv = row.get(args.columns_messages, row.get('messages', row.get('texts')))
    if isinstance(conv, dict):
        conv = apply_question_answer_template(conv.get('question', ''), conv.get('answer', ''))

    conv = apply_template(conv,  len(vision_name)) if conv is not None else None
    if conv is None or len(conv) <2 or (conv[0].get('role', None) != 'user' and conv[1].get('role', None) != 'assistant'):
        logging.warning(f"找不到对话内容，跳过{row.get('__source_file__', '')} index {index}")
        return None
    content = {
        "texts": conv,
        "media": media_type,
        "name": vision_name if vision_name else None
    }

    sample = {
        "__source_file__": row.get('__source_file__', ''),
        "__key__": f"{media_type}_{index}",
        **vision_data,
        "json": json.dumps(content).encode("utf-8"),
    }

    ###debug
    for key, val in vision_data.items():
        if not isinstance(val, bytes):
            logging.warning(f"找不到媒体内容，跳过{row.get('__source_file__', '')} index {index}")
            return None

    return sample


from multiprocessing import Process, Queue, Manager
import queue


def parallel_file_reader(file_path_list, batch_size):
    for file_path in file_path_list:
        """单个文件的读取进程"""
        try:
            json_path = file_path
            with open(json_path, 'r') as f:
                json_dat = json.load(f)
            img_path = json_dat.get('image_url', None)
            if img_path is None:
                img_path = os.path.splitext(file_path)[0] + '.jpg'

            img_bytes = open(img_path, 'rb').read()

            row_dict = {'image': img_bytes,
                        'conversations': json_dat}
            row_dict['__source_file__'] = file_path
            output_queue.put(row_dict)
        except Exception as e:
            print(f"读取文件 {file_path} 出错: {e}")

    output_queue.put(None)  # 标记该文件读取完成


def parallel_multi_file_iterator(file_paths, num_workers=4, shuffle_buffer_size=100000):

    # 启动多个读取进程
    processes = []
    files_per_worker = len(file_paths) // num_workers

    for i in range(num_workers):
        start_idx = i * files_per_worker
        end_idx = start_idx + files_per_worker if i < num_workers - 1 else len(file_paths)
        worker_files = file_paths[start_idx:end_idx]

        p = Process(target=parallel_file_reader,
                    args=(worker_files, batch_size))
        p.start()
        processes.append(p)

    # 使用随机缓冲区
    shuffle_buffer = []
    finished_count = 0

    while finished_count < num_workers or shuffle_buffer:
        # 填充缓冲区
        while len(shuffle_buffer) < shuffle_buffer_size and finished_count < num_workers:
            try:
                item = output_queue.get(timeout=0.1)
                if item is None:
                    finished_count += 1
                else:
                    shuffle_buffer.append(item)
            except queue.Empty:
                break

        # 从缓冲区随机取出一个
        if shuffle_buffer:
            idx = random.randint(0, len(shuffle_buffer) - 1)
            yield shuffle_buffer.pop(idx)

    # 等待所有进程结束
    for p in processes:
        p.join()


def vqa_json_file_iterator(json_file_path):
    """
    从 VQA JSON 文件迭代数据
    
    Args:
        json_file_path: VQA JSON 文件路径（包含列表格式的数据）
        
    Yields:
        字典，包含 image (bytes), conversations (对话格式), __source_file__
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            vqa_data = json.load(f)
        
        if not isinstance(vqa_data, list):
            logging.warning(f"VQA JSON 文件应该包含列表，但得到 {type(vqa_data)}: {json_file_path}")
            return
        
        for idx, vqa_item in enumerate(vqa_data):
            # 解码 Base64 图片
            image_base64 = vqa_item.get('image_base64')
            if not image_base64:
                logging.warning(f"跳过缺少 image_base64 的项 {idx} in {json_file_path}")
                continue
            
            img_bytes = decode_base64_image(image_base64)
            if img_bytes is None:
                logging.warning(f"跳过图片解码失败的项 {idx} in {json_file_path}")
                continue
            
            # 构建对话格式
            conv = build_vqa_conversation(vqa_item)
            
            # 构建行数据
            row_dict = {
                'image': img_bytes,
                'conversations': conv,
                '__source_file__': json_file_path
            }
            
            yield row_dict
            
    except Exception as e:
        logging.error(f"读取 VQA JSON 文件失败 {json_file_path}: {e}")


def convert_parquet_to_wds(args):
    """将 Parquet 数据集转换为 WDS 格式"""
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 获取所有输入文件
    if isinstance(args.data_root, list):
        input_files = []
        for path in args.data_root:
            if os.path.isfile(path):
                # 如果是文件，直接添加
                input_files.append(path)
            elif os.path.isdir(path):
                json_files = sorted(glob(os.path.join(path, "*.json")))
                if not json_files:
                    json_files = sorted(glob(os.path.join(path, "**/*.json"), recursive=True))
                input_files.extend(json_files)
    else:
        if os.path.isfile(args.data_root):
            input_files = [args.data_root]
        else:
            input_files = sorted(glob(os.path.join(args.data_root, "*.json")))
            if not input_files:
                input_files = sorted(glob(os.path.join(args.data_root, "**/*.json"), recursive=True))

    print(f"找到 {len(input_files)} 个输入文件")
    random.shuffle(input_files)

    if args.debug_file is not None and args.debug_file > 0:
        input_files = input_files[:args.debug_file]
        print(f"调试模式，仅处理前 {args.debug_file} 个文件")

    # 判断是否使用 VQA JSON 格式（检查第一个文件的内容）
    use_vqa_format = False
    if input_files:
        try:
            with open(input_files[0], 'r', encoding='utf-8') as f:
                first_data = json.load(f)
                # 如果是列表且第一个元素包含 image_base64 和 full_question，则使用 VQA 格式
                if isinstance(first_data, list) and len(first_data) > 0:
                    first_item = first_data[0]
                    if 'image_base64' in first_item and 'full_question' in first_item:
                        use_vqa_format = True
                        print(f"[INFO] 检测到 VQA JSON 格式")
        except Exception as e:
            logging.warning(f"无法检测文件格式，使用默认格式: {e}")

    # 构建数据迭代器
    if use_vqa_format:
        # VQA JSON 格式：直接迭代文件内容
        def vqa_data_iterator():
            for json_file in input_files:
                for item in vqa_json_file_iterator(json_file):
                    yield item
        
        data_iterator = vqa_data_iterator()
    else:
        # 原有格式：使用并行迭代器
        data_iterator = parallel_multi_file_iterator(
            file_paths=input_files,
            num_workers=args.num_workers
        )

    # 写入 WebDataset
    tar = os.path.join(args.output_dir, 'subtaskdata-%d.tar')
    print(f"开始写入 WebDataset 到 {args.output_dir}")

    with wds.ShardWriter(tar, maxcount=args.maxcount, maxsize=args.maxsize) as shard_writer:
        for index, row in enumerate(tqdm(data_iterator, desc="Converting to WDS")):
            try:
                if use_vqa_format:
                    # VQA 格式处理
                    conv = row.get('conversations')
                    if not isinstance(conv, list) or len(conv) < 2:
                        logging.warning(f"对话格式错误，跳过{row.get('__source_file__', '')} index {index}")
                        continue
                    
                    if conv[0].get('role') != 'user' or conv[1].get('role') != 'assistant':
                        logging.warning(f"对话角色错误，跳过{row.get('__source_file__', '')} index {index}")
                        continue
                    
                    if '<image>' not in conv[0].get('content', ''):
                        logging.warning(f"对话内容中缺少图片标记，跳过{row.get('__source_file__', '')} index {index}")
                        continue
                    
                    if not check_conversation_format(conv):
                        logging.warning(f"对话内容格式错误，跳过{row.get('__source_file__', '')} index {index}")
                        continue
                    
                    # 构建样本
                    img_bytes = row.get('image')
                    if not isinstance(img_bytes, bytes):
                        logging.warning(f"图片数据格式错误，跳过{row.get('__source_file__', '')} index {index}")
                        continue
                    
                    # 使用与示例代码一致的字段名格式：0_image
                    vision_data = {"0_image": img_bytes}
                    vision_name = ["0_image"]
                    
                    content = {
                        "texts": conv,
                        "media": "image",
                        "name": vision_name
                    }
                    
                    sample = {
                        "__source_file__": row.get('__source_file__', ''),
                        "__key__": f"image_{index}",
                        **vision_data,
                        "json": json.dumps(content, ensure_ascii=False).encode("utf-8"),
                    }
                    
                    # 验证逻辑（与示例代码保持一致）
                    jsondat = json.loads(sample.get('json', None).decode('utf-8'))
                    video = []
                    image = []
                    allgood = True
                    if jsondat['media'] == 'video':
                        for name in jsondat['name']:
                            if not isinstance(sample.get(name), bytes):
                                allgood = False
                    elif jsondat['media'] == 'image':
                        for name in jsondat['name']:
                            if not isinstance(sample.get(name), bytes):
                                allgood = False
                    else:
                        continue
                    if not allgood:
                        logging.warning(f"not all good{row.get('__source_file__', '')} {index}")
                        continue
                    
                    # 检查图片字段数量
                    num_img2 = 0
                    num_img = len(jsondat['name'])
                    for key in sample.keys():
                        if '_image' in key:
                            num_img2 += 1
                    if num_img != num_img2:
                        logging.warning(f"image data error: expected {num_img}, found {num_img2}")
                        continue
                    
                    # 检查 <image> 标记数量
                    media = jsondat.get('media', '')
                    content_text = jsondat['texts'][0].get('content', '')
                    img_in_content = re.findall(r'<{}>'.format(media), content_text)
                    num_img_in_content = len(img_in_content)
                    if num_img != num_img_in_content:
                        logging.warning(f"image tag error: expected {num_img} tags, found {num_img_in_content}")
                        continue
                    
                    shard_writer.write(sample)
                    
                else:
                    # 原有格式处理
                    convs = row.get('conversations')
                    sample = construct_sample_from_row(row, index, 'image', row.get('image', None), args)
                    if sample is None:
                        continue
                    
                    jsondat = json.loads(sample.get('json', None).decode('utf-8'))
                    conv = jsondat.get('texts', None)
                    
                    if len(conv) < 2 or (conv[0].get('role', None) != 'user' and conv[1].get('role', None) != 'assistant'):
                        logging.warning(f"找不到对话内容，跳过{row.get('__source_file__', '')} index {index}")
                        continue
                    if '<image>' not in conv[0].get('content', '') and jsondat.get('media', None) in ['image', 'video', 'mix']:
                        logging.warning(f"对话内容中缺少图片标记，跳过{row.get('__source_file__', '')} index {index}")
                        continue
                    if not check_conversation_format(conv):
                        logging.warning(f"对话内容格式错误，跳过{row.get('__source_file__', '')} index {index}")
                        continue
                    
                    video = []
                    image = []
                    allgood = True
                    if jsondat['media'] == 'video':
                        for name in jsondat['name']:
                            if not isinstance(sample.get(name), bytes):
                                allgood = False
                    elif jsondat['media'] == 'image':
                        for name in jsondat['name']:
                            if not isinstance(sample.get(name), bytes):
                                allgood = False
                    else:
                        continue
                    if not allgood:
                        logging.warning(f"not all good{row.get('__source_file__', '')} {index}")
                        continue
                    
                    ## check num of media tag
                    num_img2 = 0
                    num_img = len(jsondat['name'])
                    for key in sample.keys():
                        if '_image' in key:
                            num_img2 += 1
                    if num_img != num_img2:
                        print("image data error")
                        continue
                    media = jsondat.get('media', '')
                    content_text = jsondat['texts'][0].get('content', '')
                    img_in_content = re.findall(r'<{}>'.format(media), content_text)
                    num_img_in_content = len(img_in_content)
                    if num_img != num_img_in_content:
                        print("image tag error")
                        continue
                    
                    shard_writer.write(sample)
                
            except Exception as e:
                print(f"处理第{row.get('__source_file__', '')}, {index} 行时出错: {e}")
                import traceback
                traceback.print_exc()
                continue

    # 写入配置文件（对于 VQA 格式，默认使用 image 媒体类型）
    media_type = args.media
    if use_vqa_format and media_type == "mix":
        media_type = "image"  # VQA 格式默认使用 image
    
    # 始终调用 write_config 以生成 Megatron 索引
    write_config(EPath(args.output_dir).absolute(), media_type)

    print(f"数据集成功转换为 WebDataset 格式")

def write_config(path: EPath, media: str=None):
    """写入配置到指定路径"""
    (path / MAIN_FOLDER_NAME).mkdir(exist_ok=True)
    all_tars = list(path.glob("**/*.tar")) + list(path.glob("**/*.tgz"))
    all_tars = [str(p.relative_to(path)) for p in sorted(all_tars)]

    # 根据媒体类型选择类类型
    if media == 'mix':
        class_type = "MultiMixQASample"
    elif media == 'video':
        class_type = "MultiVidQASample"
    else:  # image 或其他
        class_type = "MultiMixQASample"  # 图像类型也使用 MultiMixQASample（如果支持图像）或根据实际情况调整
    
    dataset_definition = {
        "sample_type": {
            "__module__": "aiak_training_llm.data.multimodal",
            "__class__": class_type,
        },
        "part_filter": "sample_loader.py:part_filter",
        "sample_loader": "sample_loader.py:sample_loader"
    }

    with (path / MAIN_FOLDER_NAME / "dataset.yaml").open("w") as f:
        yaml.dump(dataset_definition, f, sort_keys=False)

    with (path / MAIN_FOLDER_NAME / "sample_loader.py").open("w") as f:
        f.write(sample_loader_template(media))

    BaseWebdatasetFactory.prepare_dataset(
        path,
        all_tars,
        split_parts_ratio=[("train", 1.0), ("val", 0), ("test", 0)],
        tar_index_only=False,
        workers=96,
    )

def _add_arguments(parser: argparse.ArgumentParser):
    """添加参数"""
    group = parser.add_argument_group(title='wds')
    group.add_argument('--output_dir', type=str, default='/mnt/tidal-alsh01/dataset/perceptionVLMData/processed_v2.0/datasets--MMMU_EVAL/webdatasets_v2.3/', help='输出目录')
    group.add_argument('--data_root', nargs='+',
                       default=['/mnt/tidal-alsh01/dataset/perceptionVLMData/processed_v2.0/datasets--MMMU_EVAL/image_qa_pair_v2.3/',  ],
                       # default=['/mnt/tidal-alsh01/dataset/perceptionVLMData/processed_v2.0/chemistry',  ],
                       help='Parquet 文件路径或目录')
    # group.add_argument('--parquet_path', nargs='+',
    #                    default='/mnt/tidal-alsh01/dataset/perceptionVLMData/datasets--lmms-lab--LLaVA-OneVision-1.5-Insturct-Data/hme100k/', help='Parquet 文件路径或目录')
    #   '/mnt/tidal-alsh01/dataset/perceptionVLMData/processed_v1.0/datasets--MAmmoTH-VL--MAmmoTH-VL-Instruct-12M
    group.add_argument('--maxcount', type=int, default=128, help='每个 shard 的样本数')
    group.add_argument('--maxsize', type=int, default=3000000000, help='每个 shard 的最大大小')
    group.add_argument('--media', type=str, choices=["mix", "image", "video"], default="mix", help='媒体类型')
    group.add_argument('--columns_messages', type=str, default="conversations", help='消息列名')
    group.add_argument('--shuffle', action='store_true', help='是否 shuffle 数据')
    group.add_argument('--batch_size', type=int, default=20, help='每批读取的行数')
    group.add_argument('--shuffle_buffer_size', type=int, default=100000, help='Shuffle 缓冲区大小')
    group.add_argument('--num_workers', type=int, default=16, help='并行读取的工作进程数')
    group.add_argument('--debug_file', type=int, default=-1, help='并行读取的工作进程数')

    return parser

def parse_args():
    """解析参数"""
    parser = argparse.ArgumentParser()
    _add_arguments(parser)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    """并行读取多个文件"""
    manager = Manager()
    num_workers = args.num_workers
    batch_size = args.batch_size
    output_queue = manager.Queue(maxsize=num_workers * 100)
    # convert_parquet_to_wds 函数内部已经调用了 write_config
    convert_parquet_to_wds(args)
