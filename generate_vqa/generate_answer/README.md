# VQA答案生成系统

根据生成的问题和图片，自动生成答案的系统。

## 功能说明

### 选择题（Multiple Choice）
1. **生成正确答案**：根据问题和图片生成正确答案
2. **生成错误选项**：生成2-4个（可配置）与正确答案格式、长度相似的错误选项
3. **组合选项**：将所有选项打乱顺序，组合成 A:... B:... C:... 的格式
4. **完整问题**：将题干和选项组合成完整的选择题
5. **答案标识**：answer字段存储正确答案对应的选项字母（如"A"、"B"）

### 填空题（Fill in the Blank）
1. **生成答案**：根据问题和图片直接生成答案
2. **简洁格式**：答案保持简洁，不包含不必要的分析内容

## 系统架构

```
generate_answer/
├── __init__.py              # 模块初始化
├── answer_config.json       # 配置文件
├── answer_generator.py      # 答案生成核心逻辑
├── validator.py            # 答案校验模块（格式检查、VQA验证）
├── main.py                  # 命令行入口
└── README.md               # 本文档
```

## 使用方法

### 基本使用

```bash
# 处理问题文件，生成答案
python generate_vqa/generate_answer/main.py questions.json answers.json

# 指定配置文件
python generate_vqa/generate_answer/main.py questions.json answers.json \
    --config generate_vqa/generate_answer/answer_config.json

# 限制处理样本数（用于测试）
python generate_vqa/generate_answer/main.py questions.json answers.json -n 100
```

## 输入数据格式

输入文件应该是问题生成系统的输出JSON数组，每个元素包含：

```json
{
  "question": "Which term best matches the picture?",
  "question_type": "multiple_choice",
  "image_base64": "base64编码的图片数据",
  "pipeline_name": "question",
  "id": 123,
  ...
}
```

必需字段：
- `question`: 问题文本
- `question_type`: 题型（"multiple_choice" 或 "fill_in_blank"）
- `image_base64`: 图片的base64编码

## 输出数据格式

### 选择题输出

```json
{
  "question": "Which term best matches the picture?",
  "question_type": "multiple_choice",
  "image_base64": "base64编码的图片数据",
  "answer": "B",
  "explanation": "图片中显示的是...",
  "full_question": "Which term best matches the picture?\nA: truck\nB: car\nC: bus\nD: motorcycle",
  "options": {
    "A": "truck",
    "B": "car",
    "C": "bus",
    "D": "motorcycle"
  },
  "correct_option": "B",
  "pipeline_name": "question",
  "id": 123,
  "generated_at": "2024-01-01T12:00:00"
}
```

### 填空题输出

```json
{
  "question": "How many objects are in the picture?",
  "question_type": "fill_in_blank",
  "image_base64": "base64编码的图片数据",
  "answer": "5",
  "explanation": "图片中可以看到5个明显的物体",
  "full_question": "How many objects are in the picture?",
  "pipeline_name": "object_counting",
  "id": 123,
  "generated_at": "2024-01-01T12:00:00"
}
```

## 配置文件说明

`answer_config.json` 包含以下配置：

```json
{
  "multiple_choice": {
    "wrong_options": {
      "min_count": 2,
      "max_count": 4
    }
  },
  "generation_settings": {
    "temperature": 0.7,
    "max_tokens": 512
  }
}
```

- **wrong_options.min_count**: 选择题最少错误选项数（默认2）
- **wrong_options.max_count**: 选择题最多错误选项数（默认4）
- **generation_settings.temperature**: 生成温度参数（默认0.7）
- **generation_settings.max_tokens**: 最大token数（默认512）

## 工作流程

### 选择题生成流程

1. **生成正确答案**
   - 调用MLLM，根据问题和图片生成正确答案
   - 答案保持简洁（通常1-5个词）
   - 解释内容放入explanation字段

2. **生成错误选项**
   - 随机选择2-4个错误选项数量
   - 调用MLLM生成与正确答案格式、长度相似的错误选项
   - 确保选项具有迷惑性但明显错误

3. **组合选项**
   - 将所有选项（正确答案+错误选项）打乱顺序
   - 格式化为 A:... B:... C:... 的形式

4. **构建完整问题**
   - 将题干和选项组合成完整选择题
   - 确定正确答案对应的选项字母

### 填空题生成流程

1. **生成答案**
   - 调用MLLM，根据问题和图片生成答案
   - 答案保持简洁，不包含分析内容
   - 解释内容放入explanation字段

## 校验模块

系统包含完整的校验模块，在答案生成后自动进行校验和修复。

### Step 1: 格式检查与修复

#### 检查项：
- **占位符检查**：检查问题、答案、选项中是否包含未填充的占位符（如`[object]`、`{placeholder}`、`___`等）
- **选项重复检查**：检查选择题的选项是否有重复
- **答案完整性检查**：检查answer字段是否存在且有效，correct_option是否与answer一致
- **自动修复**：尝试自动修复发现的问题（如选项重复、答案不一致等）
- **验证修复结果**：重新检查修复后的结果是否通过所有格式检查

### Step 2: VQA验证

#### 验证项：
- **困惑度分析**：分析问题是否清晰、是否容易产生歧义
  - 清晰度评分（0.0-1.0）
  - 歧义级别（low/medium/high）
  - 清晰度问题列表
  
- **置信度评估**：评估答案的置信度，判断答案是否正确
  - 正确性判断（true/false）
  - 置信度评分（0.0-1.0）
  - 正确性原因说明
  - 其他可能的选项（如果有）
  
- **答案验证**：验证答案是否与图片内容一致
  - 有效性判断（true/false）
  - 验证原因说明
  - 验证问题列表

### 校验报告

每个答案生成结果都包含`validation_report`字段，包含：

```json
{
  "validation_report": {
    "format_check": {
      "passed": true,
      "issues": [],
      "fixes_applied": []
    },
    "vqa_validation": {
      "passed": true,
      "perplexity_analysis": {
        "passed": true,
        "clarity_score": 0.9,
        "ambiguity_level": "low",
        "issues": []
      },
      "confidence_assessment": {
        "passed": true,
        "is_correct": true,
        "confidence": 0.95,
        "correctness_reason": "..."
      },
      "answer_validation": {
        "passed": true,
        "is_valid": true,
        "validation_reason": "...",
        "issues": []
      }
    },
    "fixed_issues": [],
    "validation_passed": true
  }
}
```

## 注意事项

1. **答案简洁性**：answer字段只包含答案本身，不包含分析或解释
2. **解释分离**：所有解释和分析内容都放在explanation字段中
3. **选项格式**：选择题选项会自动打乱顺序，确保随机性
4. **错误处理**：如果生成失败，会在错误文件中记录详细信息
5. **自动修复**：系统会尝试自动修复格式问题，但某些问题可能需要人工处理
6. **校验结果**：即使校验未完全通过，结果仍会保存，但会包含详细的校验报告

## 示例工作流

```bash
# 1. 生成问题
python generate_vqa/generate_question/main.py input.json questions.json

# 2. 生成答案
python generate_vqa/generate_answer/main.py questions.json answers.json

# 3. 分析结果
python utils/analyze_results.py answers.json -o statistics.txt
```

## 故障排除

### 问题：答案生成失败
- 检查图片base64编码是否正确
- 检查问题文本是否完整
- 查看错误文件了解详细错误信息

### 问题：选择题选项数量不对
- 检查配置文件中的min_count和max_count设置
- 查看生成的错误选项是否被正确解析

### 问题：答案包含不必要的分析
- 检查prompt是否正确强调答案简洁性
- 查看explanation字段是否包含了解释内容

