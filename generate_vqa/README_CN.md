# VQA问题生成系统 - 中文说明

## 系统概述

这是一个基于配置文件的声明式VQA（视觉问答）问题生成系统。系统通过读取JSON配置文件来定义各种pipeline的意图、约束和生成规则，然后自动为图像生成符合要求的VQA问题。

## 核心设计理念

1. **配置驱动（Configuration-Driven）**: 所有pipeline定义都在JSON配置文件中，无需修改代码即可添加新pipeline
2. **严格流程（Strict Pipeline）**: 遵循6步严格流程，确保生成的问题质量
3. **图像接地（Image-Grounded）**: 所有问题必须基于图像内容，不能依赖外部知识
4. **可扩展性（Extensibility）**: 添加新pipeline只需修改配置文件，无需代码改动

## 系统架构

系统由以下模块组成：

```
generate_vqa/
├── question_config.json           # Pipeline配置文件（定义所有pipeline）
├── generate_question/            # 问题生成模块目录
│   ├── __init__.py
│   ├── config_loader.py          # 配置加载模块（读取和解析配置）
│   ├── object_selector.py        # 对象选择模块（从图像中选择目标对象）
│   ├── slot_filler.py            # 槽位填充模块（填充问题模板中的槽位）
│   ├── question_generator.py     # 问题生成模块（生成自然语言问题）
│   ├── validator.py              # 问题验证模块（验证问题是否符合约束）
│   ├── vqa_generator.py         # 主生成器（整合所有模块，实现6步流程）
│   └── main.py                   # 命令行入口
├── README.md                      # 英文文档
└── README_CN.md                   # 中文文档
```

## 6步处理流程详解

系统对每个**图像-pipeline对**执行以下6个步骤：

### STEP 1: 加载Pipeline规范

**目的**: 从配置文件中读取指定pipeline的完整配置

**实现**:
- 使用`ConfigLoader`读取`question_config.json`
- 获取pipeline的意图、描述、槽位定义、约束等信息
- 如果pipeline不存在，跳过该图像-pipeline对

**代码位置**: `vqa_generator.py` 的 `process_image_pipeline_pair()` 方法

```python
pipeline_config = self.config_loader.get_pipeline_config(pipeline_name)
if not pipeline_config:
    return None
```

### STEP 2: 对象选择（如果需要）

**目的**: 如果pipeline需要对象定位，从图像中选择最合适的目标对象

**实现**:
- 检查pipeline配置中是否包含`object_grounding`字段
- 如果包含，使用`ObjectSelector`选择对象
- 选择依据：
  - 全局对象选择策略（`object_selection_policy`）
  - Pipeline特定的约束（`object_grounding.constraints`）
- 如果无法选择合适对象，根据策略决定是否丢弃样本

**代码位置**: `object_selector.py` 的 `select_object()` 方法

**关键逻辑**:
```python
if pipeline_config.get("object_grounding"):
    selected_object = self.object_selector.select_object(...)
    if selected_object is None:
        # 根据fallback_strategy决定是否丢弃
        return None
```

**示例**: 
- `object_proportion` pipeline需要选择对象来计算比例
- `object_position` pipeline需要选择对象来定位位置
- `question` pipeline不需要对象选择（直接识别整个图像）

### STEP 3: 槽位填充

**目的**: 填充问题模板中需要的槽位值

**实现**:
- **必需槽位（required_slots）**: 必须全部填充，否则丢弃样本
  - 从选中对象中提取（如`object`、`objects`）
  - 从图像信息中提取（如`region`、`granularity`）
  - 使用LLM分析（对于复杂槽位）
- **可选槽位（optional_slots）**: 随机填充以增加多样性
  - 50%概率填充每个可选槽位
  - 使用预设的默认值或随机选择

**代码位置**: `slot_filler.py` 的 `fill_slots()` 方法

**槽位类型示例**:
- `object`: 对象名称（从选中对象获取）
- `objects`: 对象复数形式
- `object_category_granularity`: 类别粒度（basic/detailed）
- `spatial_granularity`: 空间粒度（coarse/fine）
- `direction_granularity`: 方向粒度（cardinal/intercardinal/fine）

### STEP 4: 问题生成

**目的**: 根据pipeline意图、填充的槽位和示例模板生成自然语言问题

**实现**:
- 使用`QuestionGenerator`调用Gemini API
- 输入包括：
  - Pipeline意图和描述
  - 填充的槽位
  - 选中的对象信息
  - 示例模板（`example_template`）
  - 问题约束（`question_constraints`）
- 允许改写和变化表面形式，但不能改变意图
- 温度设置为0.7以增加多样性

**代码位置**: `question_generator.py` 的 `generate_question()` 方法

**生成策略**:
- 基于示例模板但允许改写
- 必须明确引用视觉实体
- 不能引入新对象或外部知识

### STEP 5: 验证

**目的**: 验证生成的问题是否符合所有约束条件

**实现**:
- **基本检查**: 问题非空
- **全局约束检查**: 
  - 检查是否违反禁止的问题类型
  - 检查关键词匹配
- **Pipeline约束检查**: 检查pipeline特定约束
- **LLM深度验证**: 使用LLM验证问题是否：
  - 明确引用视觉实体
  - 仅基于图像可回答
  - 符合pipeline意图

**代码位置**: `validator.py` 的 `validate()` 方法

**验证规则**:
1. 问题必须明确引用至少一个视觉实体或区域
2. 问题必须仅基于图像即可回答
3. 如果图像内容改变，答案应该改变
4. 不能依赖外部知识或仅基于常识

### STEP 6: 输出

**目的**: 输出通过验证的问题

**实现**:
- 只输出通过所有验证的问题
- 不生成回退或通用问题
- 如果验证失败，丢弃问题（不输出）

**输出格式**:
```json
{
  "pipeline_name": "question",
  "pipeline_intent": "question",
  "question": "Which term best matches the picture?",
  "answer_type": "single_label",
  "slots": {...},
  "selected_object": {...},
  "validation_reason": "问题符合所有约束",
  "timestamp": "2024-01-01T12:00:00",
  "sample_index": 0,
  "id": 123
}
```

## 代码实现详解

### 1. ConfigLoader (config_loader.py)

**职责**: 加载和解析配置文件

**主要方法**:
- `get_pipeline_config(pipeline_name)`: 获取指定pipeline配置
- `get_global_constraints()`: 获取全局约束
- `get_object_selection_policy()`: 获取对象选择策略
- `list_pipelines()`: 列出所有可用pipeline

### 2. ObjectSelector (object_selector.py)

**职责**: 从图像中选择目标对象

**工作流程**:
1. 检查是否需要对象选择
2. 构建包含约束和策略的prompt
3. 调用Gemini API分析图像
4. 解析返回的JSON，提取对象信息
5. 如果选择失败，返回None

**关键点**: 使用LLM进行对象选择，确保选择的对象符合所有约束条件

### 3. SlotFiller (slot_filler.py)

**职责**: 填充问题模板中的槽位

**槽位解析策略**:
- **从对象中解析**: `object`、`objects`从`selected_object`获取
- **从图像中解析**: `region`、`spatial_granularity`等需要分析图像
- **使用默认值**: 对于有预设值的槽位（如`object_category_granularity`）
- **使用LLM**: 对于复杂槽位，可以调用LLM分析

**多样性策略**: 可选槽位随机填充（50%概率），增加问题多样性

### 4. QuestionGenerator (question_generator.py)

**职责**: 生成自然语言问题

**生成策略**:
- 基于示例模板但允许改写
- 使用温度0.7增加多样性
- 必须遵循pipeline意图和约束
- 不能改变意图或引入新对象

**Prompt构建**: 包含意图、描述、槽位、对象信息、约束等

### 5. QuestionValidator (validator.py)

**职责**: 验证问题质量

**验证层次**:
1. **基本检查**: 问题非空
2. **关键词检查**: 检查禁止的问题类型关键词
3. **LLM深度验证**: 使用LLM验证问题是否真正基于图像

**验证标准**:
- 明确引用视觉实体
- 仅基于图像可回答
- 符合pipeline意图
- 不依赖外部知识

### 6. VQAGenerator (vqa_generator.py)

**职责**: 整合所有模块，实现完整的6步流程

**主要方法**:
- `process_image_pipeline_pair()`: 处理单个图像-pipeline对，返回(结果, 错误信息)
- `process_data_file()`: 批量处理数据文件

**工作流程**:
1. 初始化所有模块（ObjectSelector、SlotFiller等）
2. 加载配置和策略
3. 对每个图像-pipeline对执行6步流程
4. 收集成功结果和错误/丢弃数据
5. 保存成功结果到输出文件
6. 保存错误/丢弃数据到带时间戳的文件（避免重复）

**错误数据收集**:
- 系统会自动收集所有错误和丢弃的数据
- 错误文件命名格式：`{output_filename}_errors_{timestamp}.json`
- 包含错误阶段、错误原因、样本信息等详细信息

## 配置文件结构

`question_config.json`包含以下部分：

### 1. meta: 元数据
```json
{
  "version": "1.1",
  "task": "VQA_question_generation",
  "design_principle": "intent_constrained_and_image_grounded_generation"
}
```

### 2. global_constraints: 全局约束
- `image_grounding_required`: 必须基于图像
- `no_generic_questions`: 不允许通用问题
- `forbidden_question_types`: 禁止的问题类型列表
- `validation_rules`: 验证规则列表

### 3. object_selection_policy: 对象选择策略
- `enabled`: 是否启用
- `general_criteria`: 通用选择标准
- `fallback_strategy`: 选择失败时的策略（discard_image）

### 4. pipelines: Pipeline定义
每个pipeline包含：
- `intent`: Pipeline意图
- `description`: 描述
- `question_intent`: 问题意图
- `answer_type`: 答案类型
- `required_slots`: 必需槽位
- `optional_slots`: 可选槽位
- `object_grounding`: 对象选择配置（可选）
- `question_constraints`: 问题约束
- `example_template`: 示例模板

## 输出数据格式

### 成功结果文件

输出文件是JSON数组，包含所有成功生成的问题。

### 错误/丢弃数据文件

系统会自动收集所有错误和丢弃的数据，保存到带时间戳的文件中（格式：`{output_filename}_errors_{timestamp}.json`），避免多次运行时重复覆盖。

错误文件包含以下信息：
- `pipeline_name`: Pipeline名称
- `error_stage`: 错误发生的阶段
- `error_reason`: 错误原因
- `sample_index`: 样本索引
- `id`: 样本ID
- `source_a_id`: source_a的ID
- `metadata`: 元数据
- `timestamp`: 时间戳

错误阶段（error_stage）可能的值：
- `config_loading`: Pipeline配置加载失败
- `object_selection`: 对象选择失败
- `slot_filling`: 槽位填充失败
- `question_generation`: 问题生成失败
- `validation`: 问题验证失败
- `data_loading`: 数据加载失败
- `unknown`: 未知错误

## 使用示例

### 基本使用

```bash
# 处理单个文件，使用所有pipeline
python generate_vqa/generate_question/main.py input.json output.json

# 运行后会生成两个文件：
# - output.json: 成功生成的问题
# - output_errors_20240101_120000.json: 错误和丢弃的数据（带时间戳）
```

### 高级选项

```bash
# 只使用特定的pipeline
python generate_vqa/generate_question/main.py input.json output.json \
    --pipelines question object_counting

# 限制处理样本数（用于测试）
python generate_vqa/generate_question/main.py input.json output.json -n 100
```

### 完整工作流

```bash
# 1. 处理原始数据
python main.py --json input_data.json --output filtered_results.json

# 2. 批量处理（如果需要）
./utils/batch_process.sh data/chunks data/results 1

# 3. 合并结果
python utils/split_json.py merge data/results/*_result.json merged_results.json

# 4. 生成VQA问题
python generate_vqa/generate_question/main.py merged_results.json vqa_questions.json

# 5. 分析结果
python utils/analyze_results.py vqa_questions.json -o statistics.txt
```

## 添加新Pipeline

只需在`question_config.json`中添加新配置，无需修改代码：

```json
{
  "pipelines": {
    "my_new_pipeline": {
      "name": "My New Pipeline",
      "intent": "my_intent",
      "description": "描述新pipeline的功能",
      "question_intent": "问题意图",
      "answer_type": "答案类型",
      "required_slots": ["object"],
      "optional_slots": ["granularity"],
      "object_grounding": {
        "selection_required": true,
        "selection_strategy": "best_fit",
        "constraints": ["对象必须清晰可见"]
      },
      "question_constraints": [
        "问题必须明确引用对象",
        "不能依赖外部知识"
      ],
      "example_template": "What is the [object] in the image?"
    }
  }
}
```

## 重要原则

1. **配置定义WHAT，模型决定HOW**: JSON定义可以问什么，LLM决定如何表达
2. **所有pipeline共享相同执行逻辑**: 代码逻辑统一，只有配置不同
3. **添加新pipeline无需代码改动**: 只需修改配置文件
4. **丢弃优于低质量**: 如果无法生成高质量问题，宁愿丢弃样本

## 故障排除

### 问题：对象选择失败率高
- **原因**: 图像中可能没有符合约束的对象
- **解决**: 
  - 检查`object_grounding.constraints`是否过于严格
  - 调整`object_selection_policy.fallback_strategy`

### 问题：验证失败率高
- **原因**: 生成的问题可能不够基于图像
- **解决**:
  - 检查问题是否真正引用视觉实体
  - 查看`validation_reason`字段了解失败原因
  - 调整`question_constraints`

### 问题：API调用失败
- **原因**: 网络、API密钥或配额问题
- **解决**:
  - 检查网络连接
  - 检查API密钥配置（在`.env`文件中）
  - 检查API配额限制

## 总结

这个系统实现了完全配置驱动的VQA问题生成，通过严格的6步流程确保问题质量。所有pipeline共享相同的执行逻辑，添加新pipeline只需修改配置文件，无需代码改动。系统优先保证问题质量，如果无法生成高质量问题，宁愿丢弃样本也不生成低质量或通用问题。

