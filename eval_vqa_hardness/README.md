# VQA数据集评估框架 - 本地模型版

用于评估本地Hugging Face格式的VQA模型在自定义数据集上的准确率，帮助判断数据集难度是否合适。

## 📁 文件结构

```
vqa_evaluation/
├── data_loader.py             # 数据加载器（适配你的数据格式）
├── vqa_evaluator.py           # 主评估程序
├── model_adapters.py          # 不同模型的适配器
├── vqa_analyzer.py            # 结果分析工具
├── run_evaluation.py          # 一键运行评估脚本（推荐）
├── requirements.txt           # 依赖包
├── vqa.json                   # 你的数据集文件（包含image_base64）
├── your_model/                # 你的本地模型目录
│   ├── config.json
│   ├── pytorch_model.bin
│   └── ...
├── sample_images/             # 样本图片（自动生成）
└── evaluation_results.json    # 评估结果（运行后生成）
```

## 🎯 数据格式

你的数据格式已完美适配！每个样本包含：

```json
{
  "id": "12345",
  "question": "Which term matches the picture?",
  "full_question": "Which term matches the picture?\nA: Option 1\nB: Option 2\nC: Option 3",
  "answer": "B",
  "image_base64": "/9j/4AAQSkZJRgABAQAAAQ...",
  "options": {"A": "Option 1", "B": "Option 2", "C": "Option 3"},
  "correct_option": "B",
  ...
}
```

**数据加载器会自动提取：**
- `question` 或 `full_question`（可配置）
- `image_base64`（自动解码为图片）
- `answer`、`correct_option` 和对应的选项文本（作为多个可接受答案）

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

如果使用GPU，确保安装正确的PyTorch版本：
```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. 配置参数

编辑 `run_evaluation.py` 中的 `CONFIG` 字典：

```python
CONFIG = {
    'model_path': "./your_model_path",   # 你的模型路径
    'device': "cuda",                     # "cuda" 或 "cpu"
    'data_file': "vqa.json",             # 你的数据文件
    'include_options': True,              # True: 使用full_question
    'max_samples': None,                  # 快速测试可设为10
    'strict_match': False,                # 答案匹配模式
}
```

### 3. 一键运行评估

```bash
python run_evaluation.py
```

**运行过程：**
1. 加载数据并显示统计信息
2. 自动解码base64图片
3. 加载VQA模型
4. 逐个样本评估
5. 输出结果和难度评估
6. 保存详细结果到JSON

### 4. 查看详细分析

```bash
python vqa_analyzer.py
```

这会显示：
- 不同问题类型的准确率
- 错误样本详细分析
- 准确率分布图表
- 导出错误报告

## 🎯 支持的模型类型

框架提供了常见VQA模型的适配器（`model_adapters.py`）：

1. **BLIP/BLIP-2** - Salesforce的视觉问答模型
2. **LLaVA** - 大型语言和视觉助手
3. **InstructBLIP** - 指令微调的BLIP
4. **Qwen2-VL** - 通义千问视觉语言模型

### 自定义模型适配

如果你的模型不在上述列表中，需要在 `vqa_evaluator.py` 中修改以下方法：

```python
def load_model(self):
    # 根据你的模型加载方式修改
    from transformers import YourProcessor, YourModel
    
    self.processor = YourProcessor.from_pretrained(self.model_path)
    self.model = YourModel.from_pretrained(self.model_path).to(self.device)

def query_model(self, image_path: str, question: str) -> str:
    # 根据你的模型输入输出格式修改
    image = self.load_image(image_path)
    inputs = self.processor(images=image, text=question, return_tensors="pt").to(self.device)
    outputs = self.model.generate(**inputs)
    answer = self.processor.decode(outputs[0], skip_special_tokens=True)
    return answer
```

## 📊 输出说明

### evaluation_results.json

```json
{
  "model_path": "./your_model",
  "device": "cuda",
  "strict_match": false,
  "total_samples": 100,
  "correct_predictions": 78,
  "accuracy": 0.78,
  "results": [
    {
      "id": "sample_001",
      "image": "image_001.jpg",
      "question": "图片中有多少个人？",
      "ground_truth": ["3", "三"],
      "prediction": "图片中有3个人",
      "correct": true
    }
  ]
}
```

### 分析报告

- **accuracy_plot.png**: 不同问题类型的准确率柱状图
- **error_report.txt**: 详细的错误样本报告

## 🎯 如何判断数据难度

根据评估结果判断数据集难度：

| 准确率范围 | 难度评估 | 建议 |
|----------|---------|------|
| > 95% | 过于简单 | 增加难度，添加更复杂的问题 |
| 85-95% | 适中偏易 | 可以适当增加难度 |
| 70-85% | 难度适中 | 适合当前模型训练 |
| 50-70% | 较难 | 适合挑战模型能力上限 |
| < 50% | 非常难 | 可能需要更强的模型或更多训练数据 |

同时关注：
- 不同问题类型的表现差异
- 错误样本的共同特征
- 模型在哪些类型问题上表现薄弱

## ⚙️ 高级配置

### 1. 答案匹配策略

**宽松匹配（默认）**：
- 检查答案是否包含正确答案的任一表达
- 适用于开放式问答

**严格匹配**：
- 要求完全匹配（忽略大小写和标点）
- 适用于选择题或简短答案

```python
results = evaluator.evaluate_dataset(
    ...,
    strict_match=True  # 启用严格匹配
)
```

### 2. 快速测试

评估少量样本以快速测试流程：

```python
results = evaluator.evaluate_dataset(
    ...,
    max_samples=10  # 只评估前10个样本
)
```

### 3. GPU内存优化

如果遇到显存不足：

```python
# 使用float16精度
self.model = AutoModelForVision2Seq.from_pretrained(
    self.model_path,
    torch_dtype=torch.float16,
    device_map="auto"  # 自动分配到多GPU
)

# 或使用量化加载
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

self.model = AutoModelForVision2Seq.from_pretrained(
    self.model_path,
    quantization_config=quantization_config
)
```

### 4. 批处理评估

目前实现是逐样本评估，如需批处理可修改：

```python
def evaluate_batch(self, images, questions, batch_size=8):
    # 实现批处理逻辑
    pass
```

## 🔧 常见问题

### Q1: 模型加载失败

**A:** 检查：
- 模型路径是否正确
- 是否安装了模型所需的依赖（如`sentencepiece`）
- 尝试添加 `trust_remote_code=True`

### Q2: 显存不足

**A:** 尝试：
- 使用CPU：`device="cpu"`
- 使用float16：`torch_dtype=torch.float16`
- 使用量化：`load_in_4bit=True`
- 减小batch_size

### Q3: 答案匹配不准确

**A:** 
- 调整 `check_answer()` 方法
- 使用LLM判断答案等价性
- 添加同义词词典
- 使用语义相似度匹配

### Q4: 评估速度慢

**A:**
- 使用GPU加速
- 减小`max_new_tokens`
- 使用量化模型
- 先用`max_samples`测试小批量

## 📝 扩展建议

- [ ] 支持多模型对比评估
- [ ] 添加基于LLM的智能答案匹配
- [ ] 支持更多问题类型的自动分类
- [ ] 添加难度分析和样本推荐
- [ ] 支持视频问答评估
- [ ] 导出详细的HTML评估报告
- [ ] 支持分布式评估（多GPU/多机）

## 📄 License

MIT License