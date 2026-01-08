# VQA评估框架 - 快速开始指南

## ✅ 代码检查结果

### 代码状态：**可以运行** ✓

经过检查，代码结构良好，已修复一个小bug，现在可以正常运行。

### 修复的问题
- ✅ 修复了 `vqa_evaluator.py` 中错误处理时可能访问不存在的 `image` 字段的问题

### 代码质量
- ✅ 所有文件语法正确，无linter错误
- ✅ 模块化设计良好，职责清晰
- ✅ 数据格式适配正确（适配 pipeline.py 的输出格式）
- ✅ 错误处理完善
- ✅ 支持多种模型类型

## 🚀 快速开始

### 1. 安装依赖

```bash
cd eval_vqa_hardness
pip install -r requirements.txt
```

**如果使用GPU，需要安装对应CUDA版本的PyTorch：**

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. 准备数据文件

确保你有从 `pipeline.py` 生成的 VQA 数据文件，格式如下：

```json
[
  {
    "id": "12345",
    "question": "Which term matches the picture?",
    "full_question": "Which term matches the picture?\nA: Option 1\nB: Option 2",
    "answer": "B",
    "image_base64": "/9j/4AAQSkZJRg...",
    "options": {"A": "Option 1", "B": "Option 2"},
    "correct_option": "B",
    ...
  }
]
```

将数据文件放在 `eval_vqa_hardness` 目录下，例如 `vqa_dataset.json`

### 3. 配置运行参数

编辑 `run_evaluation.py`，修改 `CONFIG` 字典：

```python
CONFIG = {
    # 模型配置
    'model_path': "./your_model_path",  # ⚠️ 修改为你的模型路径
    'device': "cuda",                    # "cuda" 或 "cpu"
    
    # 数据配置
    'data_file': "vqa_dataset.json",     # ⚠️ 修改为你的数据文件路径
    'include_options': True,             # True: 使用full_question, False: 使用question
    
    # 评估配置
    'max_samples': 10,                   # ⚠️ 快速测试：设为10，正式评估：设为None
    'strict_match': False,               # False: 宽松匹配, True: 严格匹配
    
    # 输出配置
    'output_file': "evaluation_results.json",
    'save_sample_images': True,
    'num_sample_images': 3,
}
```

### 4. 运行评估

```bash
python run_evaluation.py
```

**运行过程：**
1. 加载数据并显示统计信息
2. 自动解码base64图片
3. 加载VQA模型
4. 逐个样本评估（显示进度条）
5. 输出结果和难度评估
6. 保存详细结果到JSON

### 5. 查看分析结果

```bash
python vqa_analyzer.py
```

这会显示：
- 不同问题类型的准确率
- 错误样本详细分析
- 准确率分布图表
- 导出错误报告

## 📋 完整运行示例

### 示例1：快速测试（10个样本）

```python
# 在 run_evaluation.py 中设置
CONFIG = {
    'model_path': "./models/blip2",
    'device': "cuda",
    'data_file': "vqa_dataset.json",
    'include_options': True,
    'max_samples': 10,  # 只评估10个样本
    'strict_match': False,
}
```

运行：
```bash
python run_evaluation.py
```

### 示例2：完整评估

```python
CONFIG = {
    'model_path': "./models/blip2",
    'device': "cuda",
    'data_file': "vqa_dataset.json",
    'include_options': True,
    'max_samples': None,  # 评估全部样本
    'strict_match': False,
}
```

### 示例3：使用CPU评估

```python
CONFIG = {
    'model_path': "./models/blip2",
    'device': "cpu",  # 使用CPU
    'data_file': "vqa_dataset.json",
    'max_samples': 5,  # CPU较慢，建议少量测试
}
```

## 🎯 支持的模型类型

框架支持以下模型（通过 `model_adapters.py`）：

1. **BLIP/BLIP-2** - Salesforce的视觉问答模型
2. **LLaVA** - 大型语言和视觉助手
3. **InstructBLIP** - 指令微调的BLIP
4. **Qwen2-VL** - 通义千问视觉语言模型

**如果使用其他模型：**

框架会自动尝试使用 `AutoProcessor` 和 `AutoModelForVision2Seq` 加载，大多数Hugging Face视觉语言模型应该可以直接使用。

如果遇到问题，可以：
1. 检查模型是否支持 `AutoProcessor`
2. 参考 `model_adapters.py` 添加自定义适配器
3. 修改 `vqa_evaluator.py` 中的 `load_model()` 方法

## 📊 输出文件说明

### evaluation_results.json

包含完整的评估结果：
```json
{
  "model_path": "./models/blip2",
  "device": "cuda",
  "strict_match": false,
  "total_samples": 100,
  "correct_predictions": 78,
  "accuracy": 0.78,
  "results": [
    {
      "id": "12345",
      "question": "...",
      "ground_truth": ["B", "Option 2"],
      "prediction": "B",
      "correct": true,
      "metadata": {...}
    }
  ]
}
```

### sample_images/

保存的样本图片（用于检查数据是否正确）

## ⚠️ 常见问题

### Q1: 模型加载失败

**解决方案：**
- 检查模型路径是否正确
- 确保安装了模型所需的依赖（如 `sentencepiece`）
- 尝试添加 `trust_remote_code=True`（代码中已包含）

### Q2: 显存不足（CUDA out of memory）

**解决方案：**
1. 使用CPU：`device="cpu"`
2. 使用float16：修改 `vqa_evaluator.py` 中的 `load_model()`：
   ```python
   self.model = AutoModelForVision2Seq.from_pretrained(
       self.model_path,
       torch_dtype=torch.float16,  # 使用半精度
       device_map="auto"
   )
   ```
3. 减少 `max_samples` 先测试

### Q3: 答案匹配不准确

**解决方案：**
- 调整 `strict_match` 参数
- 检查 `check_answer()` 方法的匹配逻辑
- 查看错误样本，手动调整匹配规则

### Q4: 数据格式不匹配

**解决方案：**
- 确保数据文件是JSON数组格式
- 确保每个样本包含 `image_base64` 字段
- 确保包含 `question` 或 `full_question` 字段
- 确保包含 `answer` 或 `correct_option` 字段

## 📈 如何判断数据难度

根据评估结果的准确率：

| 准确率范围 | 难度评估 | 建议 |
|----------|---------|------|
| > 95% | 过于简单 | 增加难度，添加更复杂的问题 |
| 85-95% | 适中偏易 | 可以适当增加难度 |
| 70-85% | 难度适中 | ✅ 适合当前模型训练 |
| 50-70% | 较难 | 适合挑战模型能力上限 |
| < 50% | 非常难 | 可能需要更强的模型或更多训练数据 |

## 🔧 高级用法

### 自定义答案匹配

修改 `vqa_evaluator.py` 中的 `check_answer()` 方法：

```python
def check_answer(self, prediction: str, ground_truth: List[str], 
                strict: bool = False) -> bool:
    # 自定义匹配逻辑
    ...
```

### 批处理评估

目前是逐样本评估，如需批处理可修改 `evaluate_dataset()` 方法。

### 使用自定义模型适配器

参考 `model_adapters.py` 创建新的适配器类。

## 📝 下一步

1. ✅ 运行快速测试（10个样本）
2. ✅ 检查输出结果
3. ✅ 运行完整评估
4. ✅ 分析结果，判断数据难度
5. ✅ 根据结果调整数据集

## 💡 提示

- 首次运行建议先用 `max_samples=10` 快速测试
- 如果模型很大，建议先用CPU测试流程是否正确
- 查看 `sample_images/` 目录确认图片解码正确
- 使用 `vqa_analyzer.py` 进行详细分析

