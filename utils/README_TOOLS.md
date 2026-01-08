# 结果处理工具说明

本目录包含两个用于处理输出结果文件的工具脚本。

## 1. 随机采样工具 (`sample_results.py`)

从输出结果文件中随机选择n个样本，用于检查和分析。

### 使用方法

```bash
python utils/sample_results.py <输入文件> <输出文件> -n <采样数量> [选项]
```

### 参数说明

- `输入文件`: 输入JSON文件路径（完整的筛选结果文件）
- `输出文件`: 输出JSON文件路径（采样后的结果）
- `-n, --num-samples`: 采样数量（必需）
- `--seed`: 随机种子（可选，用于确保可重复性）
- `--preserve-order`: 保持原始顺序（可选，按原始索引排序）
- `--exclude-errors`: 排除错误记录（只采样有筛选结果的记录）
- `--only-passed`: 只采样通过的记录（passed=true）
- `--only-failed`: 只采样未通过的记录（passed=false，排除错误记录）

### 使用示例

```bash
# 从results.json中随机采样100条记录
python utils/sample_results.py results.json sample_100.json -n 100

# 使用随机种子确保可重复性
python utils/sample_results.py results.json sample_100.json -n 100 --seed 42

# 保持原始顺序（按原始索引排序）
python utils/sample_results.py results.json sample_100.json -n 100 --preserve-order

# 排除错误记录，只采样有筛选结果的记录
python utils/sample_results.py results.json sample_100.json -n 100 --exclude-errors

# 只采样通过的记录
python utils/sample_results.py results.json passed_samples.json -n 100 --only-passed

# 只采样未通过的记录（排除错误）
python utils/sample_results.py results.json failed_samples.json -n 100 --only-failed
```

### 输出

- 输出文件包含随机选择的n条记录
- 控制台会显示统计信息（通过数、未通过数、错误数、平均分数等）
- **如果存在错误记录，会显示：**
  - 错误类型统计（每种错误出现的次数）
  - 前3个错误记录的详细信息（ID、错误信息、Pipeline类型等）

---

## 2. 分数分流工具 (`split_by_score.py`)

根据分数阈值将结果文件分流到两个文件（高分和低分）。

### 使用方法

```bash
python utils/split_by_score.py <输入文件> <高分文件> <低分文件> [选项]
```

### 参数说明

- `输入文件`: 输入JSON文件路径（完整的筛选结果文件）
- `高分文件`: 高分输出文件路径（>= threshold的记录）
- `低分文件`: 低分输出文件路径（< threshold的记录）
- `--threshold`: 分数阈值（默认: 0.6，范围: 0.0-1.0）
- `--exclude-equal`: 等于阈值的记录归入低分组（默认归入高分组）
- `--include-no-score`: 将无分数记录（错误记录等）也保存到高分文件（默认不保存）

### 使用示例

```bash
# 以0.6为界分流（默认）
python utils/split_by_score.py results.json high_score.json low_score.json

# 自定义阈值0.7
python utils/split_by_score.py results.json high.json low.json --threshold 0.7

# 等于阈值的记录归入低分组
python utils/split_by_score.py results.json high.json low.json --threshold 0.6 --exclude-equal

# 将无分数记录也保存到高分文件
python utils/split_by_score.py results.json high.json low.json --include-no-score
```

### 输出

- **高分文件**: 包含 `total_score >= threshold` 的记录
- **低分文件**: 包含 `total_score < threshold` 的记录
- **无分数记录**: 如果没有分数（错误记录等），默认会单独保存到 `低分文件名_no_score.json`（除非使用 `--include-no-score`）

### 分流规则

- **高分组**: `total_score >= threshold`（如果 `--exclude-equal`，则为 `total_score > threshold`）
- **低分组**: `total_score < threshold`（如果 `--exclude-equal`，则为 `total_score <= threshold`）
- **无分数记录**: 没有 `total_score` 字段的记录（通常是错误记录）

### 统计信息

工具会输出详细的统计信息：
- 高分组和低分组的记录数
- 各组的分数范围
- 各组的平均分数
- 无分数记录数

---

## 完整工作流程示例

```bash
# 1. 处理数据，生成完整结果文件
python main.py --json input_data.json --output results.json

# 2. 从结果中随机采样100条用于检查
python utils/sample_results.py results.json sample_100.json -n 100 --seed 42

# 3. 根据分数0.6分流
python utils/split_by_score.py results.json high_score.json low_score.json --threshold 0.6

# 4. 检查分流结果
python utils/sample_results.py high_score.json high_sample_50.json -n 50
python utils/sample_results.py low_score.json low_sample_50.json -n 50
```

---

## 3. 结果分析工具 (`analyze_results.py`)

分析合并后的JSON结果文件，统计各种指标。

### 使用方法

```bash
python utils/analyze_results.py <输入文件> [选项]
```

### 参数说明

- `输入文件`: 输入JSON文件路径（合并后的结果文件）
- `-o, --output`: 输出文本文件路径（可选，保存统计结果为文本）
- `--csv`: 导出CSV格式文件路径（可选，用于Excel等工具）

### 使用示例

```bash
# 基本统计（输出到控制台）
python utils/analyze_results.py merged_output.json

# 保存统计结果到文本文件
python utils/analyze_results.py merged_output.json -o statistics.txt

# 同时导出CSV格式（方便在Excel中查看）
python utils/analyze_results.py merged_output.json -o statistics.txt --csv statistics.csv
```

### 统计内容

工具会统计以下信息：

#### 整体统计
- 总记录数、有效记录数、错误记录数
- 通过/未通过数量和百分比
- 通过率
- 分数统计（平均、最高、最低）
- 分数分布（高分≥0.6、中分0.3-0.6、低分<0.3）
- 置信度统计

#### 按Pipeline类型统计
- 每个pipeline类型的详细统计
- 通过/未通过数量
- 分数分布
- 通过率

### 输出格式

- **控制台输出**: 格式化的Markdown风格文本
- **文本文件**: 与控制台输出相同
- **CSV文件**: 表格格式，可在Excel中打开

## 注意事项

1. **文件格式**: 输入文件必须是有效的JSON数组格式
2. **内存使用**: 工具会一次性加载整个文件到内存，对于超大文件（>几GB）可能需要较长时间
3. **编码**: 所有文件使用UTF-8编码，支持中文等多语言字符
4. **备份**: 建议在处理大文件前先备份原始文件

