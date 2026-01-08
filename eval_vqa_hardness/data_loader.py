"""
VQA数据加载器 - 适配自定义数据格式
"""

import json
import base64
from io import BytesIO
from PIL import Image
from typing import List, Dict, Any, Optional


class VQADataLoader:
    """VQA数据加载器 - 适配base64图片格式"""
    
    def __init__(self, data_file: str):
        """
        初始化数据加载器
        
        Args:
            data_file: JSON数据文件路径
        """
        self.data_file = data_file
        self.raw_data = None
        self.processed_data = None
        
        self.load_data()
    
    def load_data(self):
        """加载原始数据"""
        print(f"正在加载数据: {self.data_file}")
        with open(self.data_file, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)
        print(f"成功加载 {len(self.raw_data)} 个样本")
    
    def decode_base64_image(self, base64_str: str) -> Image.Image:
        """
        解码base64图片
        
        Args:
            base64_str: base64编码的图片字符串
            
        Returns:
            PIL Image对象
        """
        # 移除可能的data URI前缀
        if ',' in base64_str:
            base64_str = base64_str.split(',', 1)[1]
        
        image_data = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_data))
        return image.convert('RGB')
    
    def extract_answer_from_options(self, item: Dict) -> List[str]:
        """
        从选项中提取答案
        
        Args:
            item: 单个数据项
            
        Returns:
            答案列表（包含多种可接受的表达）
        """
        answers: List[str] = []
        
        # 方式1: 直接使用answer字段（如 "B" 或完整答案文本）
        raw_answer = item.get('answer')
        if isinstance(raw_answer, str) and raw_answer.strip():
            answers.append(raw_answer.strip())
        
        # 方式2: 使用correct_option字段（选项字母）
        correct_option = item.get('correct_option')
        if isinstance(correct_option, str) and correct_option.strip():
            correct_option = correct_option.strip()
            answers.append(correct_option)
            
            # 如果有有效的options字典，添加对应的文本
            options = item.get('options')
            if isinstance(options, dict) and correct_option in options:
                option_text = options[correct_option]
                if isinstance(option_text, str) and option_text.strip():
                    option_text = option_text.strip()
                    answers.append(option_text)
                    
                    # 添加带选项标识的完整答案（如 "B: Option 2"）
                    answers.append(f"{correct_option}: {option_text}")
        
        # 去重
        answers = list(set(answers))
        
        return answers
    
    def process_item(self, item: Dict, include_options: bool = True) -> Dict[str, Any]:
        """
        处理单个数据项，提取评估所需字段
        
        Args:
            item: 原始数据项
            include_options: 是否在问题中包含选项
            
        Returns:
            处理后的数据项
        """
        processed = {
            'id': item.get('id', item.get('sample_index', 0)),
            'image_base64': item['image_base64'],
            'answers': self.extract_answer_from_options(item),
            'metadata': {
                'question_type': item.get('question_type'),
                'answer_type': item.get('answer_type'),
                'pipeline_intent': item.get('pipeline_intent'),
                'validation_passed': item.get('validation_passed'),
                'validation_score': item.get('validation_score'),
            }
        }
        
        # 选择使用哪个问题字段
        if include_options and 'full_question' in item:
            # 使用包含选项的完整问题
            processed['question'] = item['full_question']
        else:
            # 使用简单问题，不包含选项
            processed['question'] = item.get('question', item.get('full_question'))
        
        return processed
    
    def process_all(self, include_options: bool = True, 
                   max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        处理所有数据
        
        Args:
            include_options: 是否在问题中包含选项
            max_samples: 最大处理样本数
            
        Returns:
            处理后的数据列表
        """
        data_to_process = self.raw_data[:max_samples] if max_samples else self.raw_data
        
        self.processed_data = []
        print(f"正在处理数据（include_options={include_options}）...")
        
        for item in data_to_process:
            try:
                processed = self.process_item(item, include_options=include_options)
                self.processed_data.append(processed)
            except Exception as e:
                print(f"处理样本 {item.get('id', 'unknown')} 时出错: {e}")
                continue
        
        print(f"成功处理 {len(self.processed_data)} 个样本")
        return self.processed_data
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        if not self.raw_data:
            return {}
        
        stats = {
            'total_samples': len(self.raw_data),
            'question_types': {},
            'answer_types': {},
            'validation_passed': 0,
            'avg_validation_score': 0
        }
        
        scores = []
        for item in self.raw_data:
            # 问题类型统计
            q_type = item.get('question_type', 'unknown')
            stats['question_types'][q_type] = stats['question_types'].get(q_type, 0) + 1
            
            # 答案类型统计
            a_type = item.get('answer_type', 'unknown')
            stats['answer_types'][a_type] = stats['answer_types'].get(a_type, 0) + 1
            
            # 验证统计
            if item.get('validation_passed'):
                stats['validation_passed'] += 1
            
            if item.get('validation_score') is not None:
                scores.append(item['validation_score'])
        
        if scores:
            stats['avg_validation_score'] = sum(scores) / len(scores)
        
        return stats
    
    def print_statistics(self):
        """打印数据集统计信息"""
        stats = self.get_statistics()
        
        print("\n" + "=" * 60)
        print("数据集统计信息")
        print("=" * 60)
        print(f"总样本数: {stats['total_samples']}")
        
        print(f"\n问题类型分布:")
        for q_type, count in stats['question_types'].items():
            percentage = count / stats['total_samples'] * 100
            print(f"  {q_type}: {count} ({percentage:.1f}%)")
        
        print(f"\n答案类型分布:")
        for a_type, count in stats['answer_types'].items():
            percentage = count / stats['total_samples'] * 100
            print(f"  {a_type}: {count} ({percentage:.1f}%)")
        
        print(f"\n验证信息:")
        print(f"  通过验证: {stats['validation_passed']}/{stats['total_samples']}")
        print(f"  平均验证分数: {stats['avg_validation_score']:.3f}")
        print("=" * 60 + "\n")
    
    def save_sample_images(self, output_dir: str = "sample_images", n: int = 5):
        """
        保存前n个样本的图片用于检查
        
        Args:
            output_dir: 输出目录
            n: 保存的样本数
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"正在保存前 {n} 个样本图片到 {output_dir}...")
        
        for i, item in enumerate(self.raw_data[:n]):
            try:
                image = self.decode_base64_image(item['image_base64'])
                image_path = os.path.join(output_dir, f"sample_{i}_{item.get('id', i)}.jpg")
                image.save(image_path)
                print(f"  保存: {image_path}")
            except Exception as e:
                print(f"  保存样本 {i} 失败: {e}")
        
        print("完成！")


def main():
    """使用示例"""
    
    # 加载数据
    loader = VQADataLoader("vqa.json")
    
    # 打印统计信息
    loader.print_statistics()
    
    # 处理数据（包含选项）
    processed_data = loader.process_all(
        include_options=True,  # True: 使用full_question, False: 使用简单question
        max_samples=None  # None: 处理全部, 或指定数字
    )
    
    # 查看前3个处理后的样本
    print("\n前3个处理后的样本:")
    print("=" * 60)
    for i, item in enumerate(processed_data[:3]):
        print(f"\n样本 {i+1}:")
        print(f"  ID: {item['id']}")
        print(f"  问题: {item['question'][:100]}...")
        print(f"  答案: {item['answers']}")
        print(f"  问题类型: {item['metadata']['question_type']}")
    
    # 保存几张样本图片检查
    loader.save_sample_images(n=3)


if __name__ == "__main__":
    main()