# VQA问题生成系统

基于配置文件的声明式VQA问题生成系统。系统通过读取JSON配置文件来定义各种pipeline的意图、约束和生成规则，然后自动为图像生成符合要求的VQA问题。

## 设计原则

1. **配置驱动**: 所有pipeline定义都在JSON配置文件中，无需修改代码即可添加新pipeline
2. **严格流程**: 遵循6步严格流程，确保生成的问题质量
3. **图像接地**: 所有问题必须基于图像内容，不能依赖外部知识
4. **可扩展性**: 添加新pipeline只需修改配置文件

## 系统架构

```
generate_vqa/
├── question_config.json           # Pipeline配置文件
├── generate_question/            # 问题生成模块目录
│   ├── __init__.py
│   ├── config_loader.py          # 配置加载模块
│   ├── object_selector.py        # 对象选择模块
│   ├── slot_filler.py            # 槽位填充模块
│   ├── question_generator.py     # 问题生成模块
│   ├── validator.py              # 问题验证模块
│   ├── vqa_generator.py         # 主生成器（整合所有模块）
│   └── main.py                   # 命令行入口
├── README.md                      # 英文文档
└── README_CN.md                   # 中文文档
```

## 工作流程

系统对每个图像-pipeline对执行以下6个步骤：

### STEP 1: 加载Pipeline规范
- 从配置文件中读取指定pipeline的配置
- 获取意图、约束、槽位定义等信息

### STEP 2: 对象选择（如果需要）
- 如果pipeline包含`object_grounding`字段：
  - 根据全局策略和pipeline特定约束选择目标对象
  - 如果无法选择合适对象，丢弃该图像-pipeline对

### STEP 3: 槽位填充
- 填充`required_slots`中的必需槽位
  - 如果任何必需槽位无法填充，丢弃样本
- 随机填充`optional_slots`中的可选槽位（增加多样性）

### STEP 4: 问题生成
- 根据pipeline意图、填充的槽位和示例模板生成问题
- 允许改写和变化，但不能改变意图

### STEP 5: 验证
- 验证问题是否符合：
  - Pipeline特定约束
  - 全局约束
  - 图像接地要求
- 如果验证失败，丢弃问题

### STEP 6: 输出
- 只输出通过验证的问题
- 不生成回退或通用问题

## 使用方法

### 基本使用

```bash
# 处理单个文件，使用所有pipeline
python generate_vqa/generate_question/main.py input.json output.json

# 指定配置文件路径
python generate_vqa/generate_question/main.py input.json output.json --config generate_vqa/question_config.json
```

### 高级选项

```bash
# 只使用特定的pipeline
python generate_vqa/main.py input.json output.json \
    --pipelines question object_counting

# 限制处理样本数（用于测试）
python generate_vqa/main.py input.json output.json -n 100

# 组合使用
python generate_vqa/main.py input.json output.json \
    --pipelines object_position object_proportion \
    -n 50
```

## 输入数据格式

输入文件应该是`batch_process.sh`输出的JSON数组，每个元素包含：

```json
{
  "sample_index": 0,
  "id": 123,
  "pipeline_type": "object_counting",
  "pipeline_name": "Object Counting Pipeline",
  "source_a": {
    "id": 123,
    "image_base64": "base64编码的图片数据",
    "image_input": "图片路径或base64",
    ...其他字段
  },
  "source_b": {...}
}
```

系统会从`source_a`中提取图片数据。

### Pipeline类型识别

系统支持从输入数据中自动识别pipeline类型：

- **优先使用 `pipeline_type` 字段**：如 `"object_counting"`，直接对应配置文件中的pipeline名称
- **其次使用 `pipeline_name` 字段**：如 `"Object Counting Pipeline"`，系统会自动映射到对应的pipeline类型
- **也可以从 `source_a` 或 `source_b` 中查找**：如果记录顶层没有，会尝试从这些字段中查找

**重要**：如果记录中指定了`pipeline_type`或`pipeline_name`，系统会**只为该记录使用对应的pipeline生成问题**，而不是为所有pipeline生成。这样可以：
- 减少生成的问题数量（从 记录数 × pipeline数 变为 记录数）
- 利用已有的pipeline类型信息
- 提高数据质量和相关性

如果记录中没有指定pipeline信息，系统会使用传入的`--pipelines`参数（如果指定了），或者使用所有pipeline（向后兼容）。

## 输出数据格式

### 成功结果文件

输出文件是JSON数组，每个元素包含：

```json
{
  "pipeline_name": "question",
  "pipeline_intent": "question",
  "question": "Which term best matches the picture?",
  "question_type": "multiple_choice",
  "answer_type": "single_label",
  "slots": {
    "object_category_granularity": "detailed"
  },
  "selected_object": {
    "name": "car",
    "category": "vehicle",
    "reason": "...",
    "confidence": 0.95
  },
  "validation_reason": "问题符合所有约束",
  "timestamp": "2024-01-01T12:00:00",
  "sample_index": 0,
  "id": 123,
  "source_a_id": 123,
  "image_base64": "base64编码的图片数据"
}
```

### 错误/丢弃数据文件

系统会自动收集所有错误和丢弃的数据，保存到带时间戳的文件中（格式：`{output_filename}_errors_{timestamp}.json`），避免多次运行时重复。

错误文件包含以下信息：

```json
{
  "pipeline_name": "object_counting",
  "error_stage": "object_selection",
  "error_reason": "无法选择对象",
  "sample_index": 0,
  "id": 123,
  "source_a_id": 123,
  "metadata": {
    "record_index": 1,
    "id": 123
  },
  "timestamp": "2024-01-01T12:00:00"
}
```

错误阶段（error_stage）可能的值：
- `config_loading`: Pipeline配置加载失败
- `object_selection`: 对象选择失败
- `slot_filling`: 槽位填充失败
- `question_generation`: 问题生成失败
- `validation`: 问题验证失败
- `data_loading`: 数据加载失败
- `unknown`: 未知错误

## Pipeline配置说明

每个pipeline在`question_config.json`中定义，包含：

- **intent**: Pipeline意图
- **description**: 描述
- **question_intent**: 问题意图
- **answer_type**: 答案类型
- **required_slots**: 必需槽位列表
- **optional_slots**: 可选槽位列表
- **object_grounding**: 对象选择配置（可选）
- **question_constraints**: 问题约束
- **example_template**: 示例模板

### 题型比例配置

系统支持按比例控制选择题和填空题的生成。在`generation_policy`中可以配置题型比例：

```json
"generation_policy": {
  "question_type_ratio": {
    "multiple_choice": 0.7,
    "fill_in_blank": 0.3
  }
}
```

- **multiple_choice**: 选择题比例（默认0.7，即70%）
- **fill_in_blank**: 填空题比例（默认0.3，即30%）

系统会按照配置的比例随机选择题型，并在生成问题时告知MLLM需要生成对应类型的问题。最终输出结果中的`question_type`字段会标识该问题的题型（"multiple_choice"或"fill_in_blank"）。

### 添加新Pipeline

只需在`question_config.json`的`pipelines`部分添加新配置，无需修改代码：

```json
{
  "pipelines": {
    "my_new_pipeline": {
      "name": "My New Pipeline",
      "intent": "my_intent",
      "description": "...",
      "question_intent": "...",
      "answer_type": "...",
      "required_slots": [],
      "optional_slots": [],
      "question_constraints": [...],
      "example_template": "..."
    }
  }
}
```

## 约束和验证

### 全局约束

- 问题必须基于图像（image_grounding_required）
- 不允许通用问题（no_generic_questions）
- 禁止的问题类型：
  - 通用场景描述
  - 基于观点的问题
  - 假设性问题
  - 仅基于常识的问题
  - 无法从图像回答的问题

### 验证规则

1. 问题必须明确引用至少一个视觉实体或区域
2. 问题必须仅基于图像即可回答
3. 如果图像内容改变，答案应该改变

## 注意事项

1. **资源管理**: 系统使用Gemini API，确保API密钥配置正确
2. **处理时间**: 每个图像-pipeline对需要多次API调用，处理大量数据可能需要较长时间
3. **丢弃策略**: 如果对象选择失败或验证失败，样本会被丢弃而不是生成低质量问题
4. **多样性**: 可选槽位随机填充以增加问题多样性

## 示例工作流

```bash
# 1. 处理原始数据（使用现有的main.py）
python main.py --json input_data.json --output filtered_results.json

# 2. 批量处理（如果需要分割）
./utils/batch_process.sh data/chunks data/results 1

# 3. 合并结果
python utils/split_json.py merge data/results/*_result.json merged_results.json

# 4. 生成VQA问题
python generate_vqa/generate_question/main.py merged_results.json vqa_questions.json

# 5. 分析结果
python utils/analyze_results.py vqa_questions.json -o statistics.txt
```

## 故障排除

### 问题：无法选择对象
- 检查图像是否包含合适的对象
- 检查pipeline的`object_grounding.constraints`是否过于严格
- 考虑调整`object_selection_policy.fallback_strategy`

### 问题：验证失败率高
- 检查生成的问题是否真正基于图像
- 检查是否违反了全局约束
- 查看验证原因（`validation_reason`字段）

### 问题：API调用失败
- 检查网络连接
- 检查API密钥配置
- 检查API配额限制

