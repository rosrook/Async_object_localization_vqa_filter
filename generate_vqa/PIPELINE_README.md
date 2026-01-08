# VQA数据集生成完整流程

完整的VQA数据集生成流程，从`batch_process.sh`的输出开始，依次生成问题和答案，最终得到完整的VQA数据集。

## 功能特点

- ✅ **完整流程**：自动执行问题生成 → 答案生成 → 数据整合
- ✅ **分流处理**：支持大文件分批处理（默认每批1000条记录），适合处理10G+数据
- ✅ **分类输出**：自动分类输出三个文件：
  - 最终全部通过可用的vqa数据文件
  - 生成question步骤出错的数据文件
  - answer检验不通过的数据文件
- ✅ **结果追踪**：保存每一步的中间结果，便于追踪和调试
- ✅ **统计信息**：自动生成详细的统计信息和质量报告
- ✅ **错误处理**：完善的错误处理和报告机制
- ✅ **可配置**：支持自定义pipeline、样本数、批次大小等参数

## 使用方法

### 基本使用

```bash
# 从batch_process.sh的输出生成完整VQA数据集
python generate_vqa/pipeline.py input.json output_dir/
```

### 高级选项

```bash
# 指定pipeline和样本数
python generate_vqa/pipeline.py /home/zhuxuzhou/test_localization/object_localization/final_output.json vqa_output_dir/ \
    --pipelines question object_counting \
    -n 10

# 指定配置文件
python generate_vqa/pipeline.py /home/zhuxuzhou/test_localization/object_localization/final_output.json vqa_output_dir/ \
    --question-config generate_vqa/question_config.json \
    --answer-config generate_vqa/generate_answer/answer_config.json \
    -n 10

# 不保存中间结果（节省空间）
python generate_vqa/pipeline.py input.json output_dir/ --no-intermediate

# 大文件分流处理（自定义批次大小）
python generate_vqa/pipeline.py input.json output_dir/ --batch-size 500
```

## 输入格式

输入文件应该是`batch_process.sh`的输出，格式如下：

```json
[
  {
    "sample_index": 0,
    "id": 123,
    "source_a": {
      "id": 123,
      "image_base64": "base64编码的图片数据",
      "image_input": "图片路径或base64",
      ...其他字段
    },
    "source_b": {...}
  },
  ...
]
```

## 输出结构

流程会在输出目录中生成以下文件：

```
output_dir/
├── vqa_dataset_successful_YYYYMMDD_HHMMSS.json      # 最终全部通过可用的VQA数据
├── question_errors_YYYYMMDD_HHMMSS.json             # 生成question步骤出错的数据
├── answer_validation_failed_YYYYMMDD_HHMMSS.json    # answer检验不通过的数据
├── statistics_YYYYMMDD_HHMMSS.json                  # 统计信息
└── intermediate/                                    # 中间结果（如果启用）
    ├── questions/
    │   └── questions_YYYYMMDD_HHMMSS.json
    └── answers/
        └── answers_YYYYMMDD_HHMMSS.json
```

### 三个主要输出文件说明

1. **vqa_dataset_successful_*.json**：最终全部通过可用的VQA数据
   - 包含校验通过（validation_passed=true）的完整VQA数据
   - 包含问题、答案、图片、选项等完整信息
   - 可直接用于训练或评估

2. **question_errors_*.json**：生成question步骤出错的数据
   - 包含在问题生成阶段失败的所有记录
   - 包含错误原因和原始输入数据
   - 可用于问题排查和重新处理

3. **answer_validation_failed_*.json**：answer检验不通过的数据
   - 包含问题生成成功但答案校验失败的数据
   - 包含完整的校验报告和失败原因
   - 可用于质量分析和改进

### 最终数据集格式

```json
{
  "id": 123,
  "sample_index": 0,
  "source_a_id": 123,
  "question": "Which term best matches the picture?",
  "full_question": "Which term best matches the picture?\nA: truck\nB: car\nC: bus",
  "answer": "B",
  "question_type": "multiple_choice",
  "image_base64": "base64编码的图片数据",
  "options": {
    "A": "truck",
    "B": "car",
    "C": "bus"
  },
  "correct_option": "B",
  "explanation": "图片中显示的是...",
  "pipeline_name": "question",
  "pipeline_intent": "question",
  "answer_type": "single_label",
  "validation_passed": true,
  "validation_score": 0.95,
  "generated_at": "2024-01-01T12:00:00"
}
```

### 统计信息格式

```json
{
  "total_samples": 1000,
  "by_question_type": {
    "multiple_choice": 700,
    "fill_in_blank": 300
  },
  "by_pipeline": {
    "question": 200,
    "object_counting": 150,
    ...
  },
  "validation_summary": {
    "passed": 950,
    "failed": 50,
    "average_score": 0.92
  },
  "quality_metrics": {
    "has_explanation": 980,
    "has_image": 1000,
    "complete_options": 700
  },
  "pipeline_info": {
    "input_file": "input.json",
    "timestamp": "20240101_120000",
    "pipeline_names": null,
    "max_samples": null
  },
  "stats": {
    "input_records": 500,
    "questions_generated": 1000,
    "answers_generated": 1000,
    "validation_passed": 950,
    "validation_failed": 50
  }
}
```

## 工作流程

### Step 1: 生成问题

- 读取输入文件（batch_process.sh的输出）
- 调用`generate_question`模块生成问题
- 保存问题文件到`intermediate/questions/`（如果启用）

### Step 2: 生成答案

- 读取生成的问题文件
- 调用`generate_answer`模块生成答案
- 自动进行校验和修复
- 保存答案文件到`intermediate/answers/`（如果启用）

### Step 3: 生成最终数据集

- 整合所有答案数据
- 计算校验评分
- 生成统计信息
- 保存最终数据集和统计文件

## 校验评分说明

每个样本的`validation_score`基于以下指标计算：

- **格式检查**（40%）：检查占位符、选项重复、答案完整性等
- **困惑度分析**（20%）：问题清晰度评分
- **置信度评估**（20%）：答案正确性置信度
- **答案验证**（20%）：答案与图片的一致性验证

总分范围：0.0 - 1.0

## 示例工作流

```bash
# 1. 使用batch_process.sh处理原始数据
./utils/batch_process.sh data/chunks data/results 1

# 2. 合并结果（如果需要）
python utils/split_json.py merge data/results/*_result.json merged_results.json

# 3. 生成完整VQA数据集
python generate_vqa/pipeline.py merged_results.json vqa_output/

# 4. 查看结果
ls vqa_output/
# vqa_dataset_20240101_120000.json
# statistics_20240101_120000.json
# intermediate/
```

## 参数说明

- `input_file`: 输入JSON文件路径（必需）
- `output_dir`: 输出目录路径（必需）
- `--question-config`: 问题生成配置文件路径（可选）
- `--answer-config`: 答案生成配置文件路径（可选）
- `--pipelines`: 要使用的pipeline列表（可选，默认使用所有）
- `-n, --max-samples`: 最大处理输入记录数（可选，默认全部）
- `--batch-size`: 每批处理的记录数（可选，默认1000，用于大文件分流处理）
- `--no-intermediate`: 不保存中间结果（可选，节省磁盘空间）

## 注意事项

1. **磁盘空间**：如果保存中间结果，会占用更多磁盘空间。建议使用`--no-intermediate`选项
2. **处理时间**：完整流程可能需要较长时间，建议先用小样本测试（使用`-n`参数）
3. **API调用**：需要确保Gemini API配置正确且有足够的配额
4. **错误处理**：如果某一步失败，会记录错误信息但不会中断整个流程
5. **大文件处理**：对于10G+数据，系统会自动分流处理。可以通过`--batch-size`调整批次大小
6. **内存使用**：分流处理可以降低内存占用，适合处理超大文件

## 故障排除

### 问题：输入文件格式不正确
- 检查输入文件是否为有效的JSON数组
- 确认文件包含`source_a`字段和`image_base64`字段

### 问题：API调用失败
- 检查网络连接
- 检查API密钥配置
- 检查API配额限制

### 问题：中间结果文件过大
- 使用`--no-intermediate`选项不保存中间结果
- 或者定期清理`intermediate/`目录

