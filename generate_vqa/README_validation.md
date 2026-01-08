# WebDataset 数据验证指南

本文档说明如何验证生成的 WebDataset 数据的正确性。

## 快速开始

### 1. 基本检查（查看样本内容和结构）

```bash
# 检查指定 tar 文件中的 5 个随机样本
python generate_vqa/inspect_webdataset.py /path/to/subtaskdata-00000.tar -n 5

# 检查目录中的第一个 tar 文件
python generate_vqa/inspect_webdataset.py /path/to/output_dir -n 10
```

### 2. 保存图片到本地（可视化验证）

```bash
# 提取并保存 3 个样本的图片到本地
python generate_vqa/inspect_webdataset.py /path/to/subtaskdata-00000.tar -n 3 --save-images

# 指定保存目录
python generate_vqa/inspect_webdataset.py /path/to/subtaskdata-00000.tar -n 5 --save-images -o /path/to/output
```

### 3. 列出所有 tar 文件

```bash
# 查看目录中有哪些 tar 文件
python generate_vqa/inspect_webdataset.py /path/to/output_dir --list-only
```

## 验证内容

检查脚本会验证以下内容：

### ✅ 数据结构验证
- **样本完整性**：每个样本是否包含图片和 JSON 元数据
- **字段名匹配**：JSON 中的 `name` 字段是否与实际图片字段名匹配
- **图片格式**：图片文件头与字段名（jpg/png）是否一致

### ✅ 对话格式验证
- **对话轮次**：是否包含至少 2 轮对话（user + assistant）
- **角色正确性**：第一轮是否为 user，第二轮是否为 assistant
- **图片标记**：user 消息中是否包含 `<image>` 标记
- **内容完整性**：对话内容是否为空

### ✅ 图片数据验证
- **图片可读性**：能否正确解析图片（尺寸、格式、模式）
- **文件大小**：图片大小是否合理
- **格式一致性**：字段名与实际图片格式是否匹配

## 输出说明

### 正常样本示例

```
================================================================================
样本 1/5: image_0
================================================================================

[元数据]
  媒体类型: image
  图片名称: ['jpg']

[对话内容] (共 2 轮)

  轮次 1 [user]:
    <image>
    Which term matches the picture?
    A: car
    B: truck
    ✓ 包含图片标记 <image>

  轮次 2 [assistant]:
    B. truck
    
    Explanation: 图片中显示的是卡车...

[图片信息]
  期望的图片字段名: ['jpg']
  jpg: 45,231 bytes (44.17 KB)
    ✓ 尺寸: 640x480, 格式: JPEG, 模式: RGB
```

### 警告示例

如果发现问题，脚本会显示警告：

```
⚠ WARNING: JSON 中指定的字段名 'jpg' 在样本中不存在
⚠ WARNING: 字段名是 'png' 但文件头不是 PNG 格式
⚠ WARNING: user 消息中未找到 <image> 标记
```

## 使用场景

### 场景 1：转换后立即验证

```bash
# 转换数据
python generate_vqa/stage2_sftdata2webdataset.py input.json output_dir/

# 验证转换结果
python generate_vqa/inspect_webdataset.py output_dir/ -n 10 --save-images
```

### 场景 2：批量检查多个文件

```bash
# 列出所有文件
python generate_vqa/inspect_webdataset.py output_dir/ --list-only

# 逐个检查（可以写脚本自动化）
for tar_file in output_dir/*.tar; do
    echo "Checking $tar_file"
    python generate_vqa/inspect_webdataset.py "$tar_file" -n 3
done
```

### 场景 3：详细检查特定样本

```bash
# 检查更多样本并保存图片
python generate_vqa/inspect_webdataset.py output_dir/subtaskdata-00000.tar -n 20 --save-images
```

## 常见问题

### Q: 如何知道数据是否正确？
A: 检查脚本会输出详细的验证信息。如果所有检查项都通过（没有 WARNING），数据格式就是正确的。

### Q: 图片保存到哪里了？
A: 默认保存在 `input_path/inspected_samples/` 目录。使用 `-o` 参数可以指定其他目录。

### Q: 如何验证所有样本？
A: 使用一个较大的 `-n` 值，或者先查看总样本数，然后设置 `-n` 为总样本数。

### Q: 检查脚本很慢怎么办？
A: 脚本需要读取整个 tar 文件来索引所有样本。对于大文件，建议只检查部分样本（较小的 `-n` 值）。

## 命令行参数

```
positional arguments:
  input_path            WebDataset tar 文件路径或包含 tar 文件的目录

optional arguments:
  -n, --num-samples     要检查的样本数量（默认: 5）
  --save-images         是否保存图片到本地
  -o, --output-dir      保存图片的目录（默认: input_path/inspected_samples）
  --list-only           仅列出目录中的 tar 文件，不检查内容
```

