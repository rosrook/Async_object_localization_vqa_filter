# WebDataset 训练错误排查指南

## ❌ 错误：`len(images)` is less than the number of `<image>` tokens

### 错误信息
```
ValueError: `len(images)` is less than the number of <image> tokens.
RuntimeError: Error processing sample subtaskdata-21.tar/image_2729: `len(images)` is less than the number of <image> tokens.
```

### 原因分析

这个错误表示：
1. ✅ 对话中包含了 `<image>` 标记
2. ❌ 但 `sample_loader` 无法从 WebDataset 样本中提取到图片数据
3. ❌ 导致 `image=None` 或 `image=[]`

### 排查步骤

#### 1. 使用调试脚本检查 WebDataset 数据

```bash
# 检查样本结构
python generate_vqa/debug_webdataset.py /workspace/my_formatted_webdataset/subtaskdata-0.tar

# 检查特定样本
python generate_vqa/debug_webdataset.py /workspace/my_formatted_webdataset/subtaskdata-0.tar image_2729
```

这个脚本会显示：
- 样本中的所有字段名
- JSON 元数据中的 `name` 字段
- 字段名是否匹配
- 图片数据是否存在

#### 2. 检查 sample_loader.py

查看生成的 `sample_loader.py` 文件：

```bash
cat /workspace/my_formatted_webdataset/.nv-meta/sample_loader.py
```

确认：
- ✅ `media == 'image'` 时，是否正确返回 `image` 字段
- ✅ 字段名匹配逻辑是否正确

#### 3. 验证字段名匹配

问题可能是字段名不匹配。检查：

**JSON 中的字段名**：
```json
{
  "media": "image",
  "name": ["jpg"]  // 或 ["png"]
}
```

**样本中的实际字段名**：
- 应该是：`jpg` 或 `png`
- 而不是：`0_image` 或其他

#### 4. 重新生成 WebDataset（如果字段名不匹配）

如果发现字段名不匹配，需要：

1. **删除旧的 WebDataset**：
```bash
rm -rf /workspace/my_formatted_webdataset/.nv-meta/
rm -f /workspace/my_formatted_webdataset/*.tar
```

2. **重新转换数据**：
```bash
python generate_vqa/stage2_sftdata2webdataset.py \
    --data_root /path/to/vqa_dataset_successful_*.json \
    --output_dir /workspace/my_formatted_webdataset \
    --media image
```

3. **验证生成的数据**：
```bash
python generate_vqa/inspect_webdataset.py /workspace/my_formatted_webdataset -n 3 --save-images
```

### 解决方案

#### 方案 1：确保使用最新版本的转换脚本

确保 `stage2_sftdata2webdataset.py` 已更新，并且：
- ✅ 使用标准扩展名（`jpg` 或 `png`）作为字段名
- ✅ `sample_loader` 模板已修复，支持 `media='image'`

#### 方案 2：手动修复 sample_loader.py

如果不想重新生成数据，可以手动修复 `sample_loader.py`：

```bash
# 编辑 sample_loader.py
vim /workspace/my_formatted_webdataset/.nv-meta/sample_loader.py
```

确保 `return dict(...)` 中包含：
```python
image=image if len(image) > 0 else None,
```

完整示例（media='image'）：
```python
def sample_loader(sample: dict) -> dict:
    messages=[]
    system=None
    for message in sample['json']['texts']:
        assert message['role'] in ['system','user','assistant']
        if message['role'] == 'system':
            system=message['content']
            continue
        messages.append(dict(
            role=message['role'],
            content=message['content']
        ))
    video = []
    image = []
    if sample['json']['media'] == 'video':
        for name in sample['json']['name']:
            video.append(sample.get(name))
    elif sample['json']['media'] == 'image':
        for name in sample['json']['name']:
            image.append(sample.get(name))
    return dict(
        __key__=sample['__key__'],
        __restore_key__=sample['__restore_key__'],
        image=image if len(image) > 0 else None,  # 确保这行存在！
        system=system,
        messages=messages,
    )
```

### 验证修复

修复后，验证数据：

```bash
# 1. 检查数据
python generate_vqa/inspect_webdataset.py /workspace/my_formatted_webdataset -n 5

# 2. 检查 sample_loader
cat /workspace/my_formatted_webdataset/.nv-meta/sample_loader.py | grep -A 5 "return dict"

# 3. 重新运行训练（会重新生成索引）
bash train/sft_bench.sh
```

### 常见问题

#### Q: 为什么字段名是 `jpg` 而不是 `0_image`？

A: 为了使用标准的文件扩展名，更符合 WebDataset 的惯例。同时，`sample_loader` 会根据 JSON 中的 `name` 字段来查找对应的字段。

#### Q: 如果我的数据中有些是 jpg，有些是 png 怎么办？

A: 转换脚本会自动检测图片格式，并分别使用 `jpg` 或 `png` 作为字段名。这是正常的，`sample_loader` 可以处理这种情况。

#### Q: 重新生成数据需要多长时间？

A: 取决于数据量。通常几万到几十万样本需要几分钟到几十分钟。

### 预防措施

1. **转换后立即验证**：
```bash
python generate_vqa/inspect_webdataset.py output_dir/ -n 10 --save-images
```

2. **检查 sample_loader**：
```bash
cat output_dir/.nv-meta/sample_loader.py
```

3. **小批量测试训练**：
先设置 `NSTEP=10` 测试训练是否正常，确认无误后再进行完整训练。

