# 训练脚本使用指南

## 📋 快速开始

### 1. 复制模板脚本

```bash
cd train/
cp sft_bench_TEMPLATE.sh train_my_dataset.sh
chmod +x train_my_dataset.sh
```

### 2. 修改关键参数

打开 `train_my_dataset.sh`，修改以下标记为 `[需要修改]` 的部分：

## 🔧 必须修改的参数

### ✅ 1. 实验输出路径

```bash
# 修改实验输出目录
EXP_DIR='/mnt/tidal-alsh01/dataset/perceptionVLM/exps_zixu/vllm/llava_ov/'

# 修改保存路径（建议格式：stage2_sft_数据集名_日期）
SAVE_CKPT_PATH=$EXP_DIR/stage2_sft_vqa_$(date +%m%d)
```

**示例命名**：
- `stage2_sft_vqa_1220` - VQA数据集，12月20日
- `stage2_sft_agriculture_1210` - 农业数据集，12月10日
- `stage2_sft_chemistry_1202` - 化学数据集，12月2日

### ✅ 2. 数据路径

```bash
# 指向你的 WebDataset 目录
# 目录结构应该是：
#   /path/to/webdatasets/
#   ├── .nv-meta/          # Megatron 索引目录
#   ├── subtaskdata-00000.tar
#   ├── subtaskdata-00001.tar
#   └── ...
DATA_PATH=/mnt/tidal-alsh01/dataset/perceptionVLMData/processed_v2.0/YOUR_DATASET/webdatasets/
```

**重要**：确保这个路径下：
1. ✅ 有 `.nv-meta` 目录（由 `stage2_sftdata2webdataset.py` 生成）
2. ✅ 有 `.tar` 文件（WebDataset 数据文件）

### ✅ 3. Batch Size 配置

```bash
# 根据你的 GPU 数量和显存调整
MBS=1          # Micro Batch Size（每个GPU的批次大小）
GBS=8          # Global Batch Size（全局批次大小）
```

**计算公式**：
```
GBS = MBS × GPU数量
```

**示例**：
- 4个GPU，MBS=1 → GBS=4
- 8个GPU，MBS=1 → GBS=8
- 8个GPU，MBS=2 → GBS=16
- 32个GPU，MBS=1 → GBS=32

**老师建议**：`GBS = 4 × GPU数量` 或 `GBS = 8 × GPU数量`

### ✅ 4. 训练步数

```bash
# 计算公式：NSTEP ≈ 样本数量 / GBS
NSTEP=3200
```

**计算公式**：
```
NSTEP = 样本数量 / GBS
```

**示例**：
- 样本数：100,000，GBS=32 → NSTEP=3,125（建议设为3200）
- 样本数：50,000，GBS=16 → NSTEP=3,125（建议设为3200）
- 样本数：10,000，GBS=8 → NSTEP=1,250（建议设为1300）

**建议**：可以稍微设置多一点，比如多 5-10%，以便完整训练一个 epoch

### ✅ 5. 模型检查点路径

```bash
# 选择你的起始检查点
CHECKPOINT_PATH=/path/to/your/checkpoint
```

**选项说明**：

1. **从头开始训练（使用基础模型）**
   ```bash
   CHECKPOINT_PATH=/mnt/tidal-alsh01/dataset/perceptionVLM/models/LLaVA-OneVision-1.5-4B-Base
   ```

2. **使用 Stage0 模型**
   ```bash
   CHECKPOINT_PATH=/mnt/tidal-alsh01/dataset/perceptionVLM/models/LLaVA-OneVision-1.5-4B-stage0
   ```

3. **使用 Stage2 模型（推荐，继续训练）**
   ```bash
   CHECKPOINT_PATH=/mnt/tidal-alsh01/dataset/perceptionVLM/models/LLaVA-OneVision-1.5-4B-stage2_mcore_tp1_pp1
   ```

4. **断点续训（使用之前训练的检查点）**
   ```bash
   CHECKPOINT_PATH=/mnt/tidal-alsh01/dataset/perceptionVLM/exps_zixu/vllm/llava_ov/stage2_sft_previous/iter_0400
   ```

### ✅ 6. GPU 数量（如果是单机）

```bash
# 在第 65 行左右
GPUS_PER_NODE=2    # 修改为你的实际GPU数量（单机训练时）
```

## 🔍 可能需要调整的参数

### 学习率

```bash
--lr 1.0e-5    # 默认学习率，通常不需要修改
```

如果训练不稳定（损失爆炸或下降很慢），可以调整：
- 学习率太大：降低到 `5e-6` 或 `1e-6`
- 学习率太小：增加到 `2e-5` 或 `5e-5`

### 保存间隔

```bash
--save-interval 200    # 每200步保存一次检查点
```

根据训练步数调整：
- NSTEP=3200 → save-interval=200（保存16个检查点）
- NSTEP=1000 → save-interval=100（保存10个检查点）

### 数据加载工作进程数

```bash
--num-workers 16    # 数据加载器工作进程数
```

建议设置为：
- CPU核心数的一半，或
- GPU数量的 2-4 倍

## 🚀 运行训练

### 1. 检查配置

在运行前，确认：

```bash
# 1. 数据路径存在且包含 .nv-meta 和 .tar 文件
ls -la $DATA_PATH/.nv-meta/
ls -la $DATA_PATH/*.tar | head -5

# 2. 检查点路径存在
ls -la $CHECKPOINT_PATH/

# 3. Tokenizer 路径存在
ls -la $TOKENIZER_PATH/
```

### 2. 运行训练

```bash
bash train_my_dataset.sh
```

### 3. 监控训练

**查看日志**：
```bash
# 实时查看日志
tail -f $SAVE_CKPT_PATH/run_*.log

# 查看最新的日志
ls -t $SAVE_CKPT_PATH/run_*.log | head -1 | xargs tail -f
```

**查看 TensorBoard**：
```bash
tensorboard --logdir=$TENSORBOARD_PATH
```

然后在浏览器打开：`http://localhost:6006`

## 📊 训练过程监控

### 关键指标

1. **Loss（损失）**：应该逐步下降
2. **Learning Rate（学习率）**：按余弦曲线衰减
3. **GPU 利用率**：应该接近 100%

### 常见问题

#### ❌ OOM (Out of Memory)

**解决方法**：
- 减小 `MBS`（Micro Batch Size）
- 减小 `SEQ_LEN`（序列长度）
- 检查是否有其他进程占用显存

#### ❌ 数据加载慢

**解决方法**：
- 增加 `--num-workers`
- 检查数据路径的网络速度
- 确保数据在本地或高速存储上

#### ❌ Loss 不下降或爆炸

**解决方法**：
- 降低学习率 `--lr`
- 检查数据质量
- 检查学习率预热设置

## 📁 输出文件结构

训练完成后，`$SAVE_CKPT_PATH` 目录结构：

```
stage2_sft_vqa_1220/
├── iter_0000/          # 检查点（每 save-interval 步保存一次）
├── iter_0200/
├── iter_0400/
├── ...
├── iter_3200/
├── dataloader/         # 数据加载器缓存
├── tensorboard/        # TensorBoard 日志
└── run_2024-12-20_10:30:45_*.log  # 训练日志
```

## 🔄 断点续训

如果训练中断，可以从最近的检查点继续：

```bash
# 1. 找到最新的检查点
ls -d $SAVE_CKPT_PATH/iter_* | sort -V | tail -1

# 2. 修改脚本中的 CHECKPOINT_PATH
CHECKPOINT_PATH=/path/to/latest/checkpoint/iter_0400

# 3. 重新运行脚本
bash train_my_dataset.sh
```

## 📝 完整示例

假设：
- 数据集：VQA，100,000 个样本
- GPU：8 个
- 数据路径：`/data/vqa/webdatasets/`

```bash
# 1. 计算参数
GBS = 8 × 4 = 32  (使用 4×GPU数量)
NSTEP = 100000 / 32 ≈ 3125 → 设为 3200

# 2. 修改脚本
EXP_DIR='/mnt/tidal-alsh01/dataset/perceptionVLM/exps_zixu/vllm/llava_ov/'
SAVE_CKPT_PATH=$EXP_DIR/stage2_sft_vqa_1220
DATA_PATH=/data/vqa/webdatasets/
GBS=32
NSTEP=3200
GPUS_PER_NODE=8

# 3. 运行
bash train_my_dataset.sh
```

## 🆘 需要帮助？

如果遇到问题：
1. 检查日志文件中的错误信息
2. 确认所有路径都存在
3. 确认 GPU 可用：`nvidia-smi`
4. 确认数据格式正确：使用 `inspect_webdataset.py` 检查

