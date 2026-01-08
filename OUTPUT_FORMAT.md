# 输出文件格式说明

## 文件格式

输出文件是JSON格式，包含一个数组，每个元素代表一条处理结果。

## 数据结构

### 基本结构

```json
[
  {
    // 原始数据字段（从输入JSON保留）
    "sample_index": 123,
    "id": 1523,
    "question": "Which term matches the picture?",
    "source_a": { ... },
    "source_b": { ... },
    
    // 筛选结果字段
    "pipeline_type": "question",
    "pipeline_name": "Question Pipeline",
    "passed": true,
    "basic_score": 0.5,
    "bonus_score": 0.3,
    "total_score": 0.8,
    "reason": "详细说明哪些必需和可选标准被满足或违反，以及分数如何确定",
    "confidence": 0.85,
    
    // 元数据字段
    "timestamp": "2024-01-01T12:00:00.123456"
  },
  ...
]
```

### 字段说明

#### 1. 原始数据字段（从输入JSON保留）

这些字段来自输入JSON文件，会被完整保留：

- **`sample_index`** (int, 可选): 样本索引
- **`id`** (int/str, 可选): 样本ID
- **`question`** (str, 可选): 问题文本
- **`source_a`** (dict, 可选): 源数据A，通常包含图片路径等信息
- **`source_b`** (dict, 可选): 源数据B，通常包含问题文本等信息

**注意**: `image_input` 字段会被删除，不包含在输出中。

#### 2. 筛选结果字段

这些字段由Pipeline生成：

- **`pipeline_type`** (str, 必需): Pipeline类型标识符
  - 可能的值: `"question"`, `"caption"`, `"place_recognition"`, `"text_association"`, `"object_proportion"`, `"object_position"`, `"object_absence"`, `"object_orientation"`, `"object_counting"`

- **`pipeline_name`** (str, 必需): Pipeline的显示名称
  - 例如: `"Question Pipeline"`, `"Caption Pipeline"` 等

- **`passed`** (bool, 必需): 是否通过筛选
  - `true`: 所有必需标准都满足
  - `false`: 至少一个必需标准未满足

- **`basic_score`** (float, 必需): 基础分数，范围 0.0-0.6
  - 基于必需标准的评分
  - 如果任何必需标准未满足，分数为 0.0
  - 如果所有必需标准满足，分数从 0.1 开始，最高 0.6

- **`bonus_score`** (float, 必需): 奖励分数，范围 0.0-0.4
  - 基于可选标准的评分
  - 如果没有可选标准，自动加 0.4
  - 如果有可选标准，根据满足程度给予 0.0-0.4 的奖励

- **`total_score`** (float, 必需): 总分，范围 0.0-1.0
  - 计算公式: `total_score = basic_score + bonus_score`
  - 最高为 1.0

- **`reason`** (str, 必需): 详细的筛选原因说明
  - 说明哪些必需和可选标准被满足或违反
  - 解释分数如何确定

- **`confidence`** (float, 必需): 判断的置信度，范围 0.0-1.0
  - 表示模型对判断结果的信心程度

#### 3. 错误情况

如果处理过程中出现错误，会包含以下字段：

- **`error`** (str, 可选): 错误信息
  - 例如: `"No pipeline recognized"`, `"Invalid pipeline type: xxx"`, `"处理出错: xxx"`, `"Unexpected error: xxx"`

**注意**: 即使出现错误，原始数据字段（如 `sample_index`, `id`, `source_a`, `source_b` 等）也会被保留。

#### 4. 元数据字段

- **`timestamp`** (str, 必需): 处理时间戳
  - 格式: ISO 8601 格式，例如 `"2024-01-01T12:00:00.123456"`

## 完整示例

### 成功处理的示例

```json
{
  "sample_index": 123,
  "id": 1523,
  "question": "Which term matches the picture?",
  "source_a": {
    "image_input": "/path/to/image.jpg"
  },
  "source_b": {
    "question": "Which term matches the picture?",
    "answer": "热对流"
  },
  "pipeline_type": "question",
  "pipeline_name": "Question Pipeline",
  "passed": true,
  "basic_score": 0.5,
  "bonus_score": 0.3,
  "total_score": 0.8,
  "reason": "图像中有一个清晰的主要物体（气泡），该物体具有与'热对流'概念对应的可识别特征。物体边界清晰，可与背景区分。满足所有必需标准，基础分数0.5。物体位置居中，视觉突出，满足部分可选标准，奖励分数0.3。",
  "confidence": 0.85,
  "timestamp": "2024-01-01T12:00:00.123456"
}
```

### 未通过筛选的示例

```json
{
  "sample_index": 456,
  "id": 654,
  "question": "Which one is the correct caption of this image?",
  "source_a": {
    "image_input": "/path/to/image2.jpg"
  },
  "source_b": {
    "question": "Which one is the correct caption of this image?",
    "options": ["选项1", "选项2"]
  },
  "pipeline_type": "caption",
  "pipeline_name": "Caption Pipeline",
  "passed": false,
  "basic_score": 0.0,
  "bonus_score": 0.0,
  "total_score": 0.0,
  "reason": "图像不是真实世界的摄影场景（是插图），不满足必需标准'需要是现实场景（照片）'。因此未通过筛选，分数为0.0。",
  "confidence": 0.92,
  "timestamp": "2024-01-01T12:00:05.789012"
}
```

### 处理错误的示例

```json
{
  "sample_index": 789,
  "id": 668,
  "question": "What is the name of the place shown?",
  "source_a": {
    "image_input": "/path/to/image3.jpg"
  },
  "source_b": {
    "question": "What is the name of the place shown?"
  },
  "error": "处理出错: API调用失败: 连接超时",
  "timestamp": "2024-01-01T12:00:10.345678"
}
```

### 路由失败的示例

```json
{
  "sample_index": 999,
  "id": 1499,
  "question": "Unknown question type",
  "source_a": {
    "image_input": "/path/to/image4.jpg"
  },
  "source_b": {
    "question": "Unknown question type"
  },
  "error": "No pipeline recognized",
  "timestamp": "2024-01-01T12:00:15.901234"
}
```

## 注意事项

1. **字段顺序**: JSON中的字段顺序可能不固定，但所有字段都会存在（除非是可选字段）

2. **缺失字段**: 如果输入JSON中某些字段缺失，输出中也会缺失（如 `sample_index`, `id` 等）

3. **image_input字段**: 输入中的 `image_input` 字段会被删除，不包含在输出中（因为可能包含敏感路径或大量base64数据）

4. **数组格式**: 输出文件是一个JSON数组，即使只有一条结果也是数组格式

5. **增量保存**: 如果使用 `--save-interval` 参数，文件会以增量方式追加，最终仍是一个完整的JSON数组

6. **编码**: 所有文本字段使用UTF-8编码，支持中文等多语言字符

## 数据统计

可以通过以下方式统计结果：

- **总记录数**: `len(results)`
- **成功数**: `sum(1 for r in results if "error" not in r)`
- **失败数**: `sum(1 for r in results if "error" in r)`
- **通过数**: `sum(1 for r in results if r.get("passed") == True)`
- **未通过数**: `sum(1 for r in results if r.get("passed") == False)`
- **平均分数**: `sum(r.get("total_score", 0) for r in results if "total_score" in r) / len([r for r in results if "total_score" in r])`

