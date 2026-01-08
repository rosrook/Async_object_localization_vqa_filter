#!/bin/bash
# ==============================================================================
# LLaVA-OneVision Stage 2 SFT 训练脚本模板
# ==============================================================================
# 
# 使用说明：
# 1. 复制此文件为你的训练脚本（如：train_my_dataset.sh）
# 2. 修改下面标记为 [需要修改] 的部分
# 3. 运行: bash train_my_dataset.sh
#
# ==============================================================================

# ==============================================================================
# [需要修改] 1. 路径配置
# ==============================================================================
# AIAK 训练框架路径（通常不需要修改，除非你的环境不同）
AIAK_TRAINING_PATH=/workspace/onevision
AIAK_MAGATRON_PATH="${AIAK_MAGATRON_PATH:-${AIAK_TRAINING_PATH%/}/aiak_megatron}"

# [需要修改] 实验输出目录
EXP_DIR='/mnt/tidal-alsh01/dataset/perceptionVLM/models_zhuxuzhou/vllm/llava_ov/'

# [需要修改] 保存检查点的路径（建议命名：stage2_sft_数据集名_日期）
# 例如：stage2_sft_vqa_1220, stage2_sft_agriculture_1210
SAVE_CKPT_PATH=$EXP_DIR/stage2_sft_my_formatted_webdataset_$(date +%m%d)
TENSORBOARD_PATH="${SAVE_CKPT_PATH}/tensorboard"

echo "SAVE_CKPT_PATH: ${SAVE_CKPT_PATH}"
echo "TENSORBOARD_PATH: ${TENSORBOARD_PATH}"

# ==============================================================================
# [需要修改] 2. 训练超参数配置
# ==============================================================================

# 并行配置（通常单机多卡训练不需要修改）
TP=1          # Tensor Parallel (张量并行度，通常单机用1)
PP=1          # Pipeline Parallel (流水线并行度，通常单机用1)

# 序列长度配置
SEQ_LEN=32768  # 序列最大长度（根据你的任务调整，VQA通常用32768）

# [需要修改] Batch Size 配置
MBS=1          # Micro Batch Size (每个GPU的微批次大小，根据GPU显存调整)
# [重要] GBS = MBS * GPU数量
# 例如：MBS=1, GPU数量=8 → GBS=8
#       MBS=2, GPU数量=4 → GBS=8
#       MBS=1, GPU数量=32 → GBS=32
GBS=32          # Global Batch Size (全局批次大小，通常设为 4*GPU数量 或 8*GPU数量)

# [需要修改] 训练步数
# 计算公式：NSTEP ≈ 样本数量 / GBS
# 例如：样本数=100000, GBS=32 → NSTEP=3125
# 建议：稍微多设置一些，比如 NSTEP=3200 或 3300
NSTEP=3200     # 训练迭代步数

# ==============================================================================
# [需要修改] 3. 数据路径配置
# ==============================================================================
# [重要] 这里指定你的 WebDataset 数据路径
# 应该指向包含 .nv-meta 目录和 .tar 文件的目录
# 例如：/path/to/your/webdataset/
#       └── .nv-meta/
#       └── subtaskdata-00000.tar
#       └── subtaskdata-00001.tar
DATA_PATH=/workspace/my_formatted_webdataset

# ==============================================================================
# [需要修改] 4. 模型和Tokenizer路径
# ==============================================================================
# Tokenizer 路径（通常使用 Instruct 版本的 tokenizer）
TOKENIZER_PATH=/mnt/tidal-alsh01/dataset/perceptionVLM/models/LLaVA-OneVision-1.5-4B-Instruct

# [需要修改] 检查点路径（预训练模型或之前的检查点）
# 选项1：使用基础模型（从头开始训练）
# CHECKPOINT_PATH=/mnt/tidal-alsh01/dataset/perceptionVLM/models/LLaVA-OneVision-1.5-4B-Base

# 选项2：使用 Stage0 模型
# CHECKPOINT_PATH=/mnt/tidal-alsh01/dataset/perceptionVLM/models/LLaVA-OneVision-1.5-4B-stage0

# 选项3：使用 Stage2 模型（继续训练）
# CHECKPOINT_PATH=/mnt/tidal-alsh01/dataset/perceptionVLM/models/LLaVA-OneVision-1.5-4B-stage2_mcore_tp1_pp1

# 选项4：使用之前训练的检查点（断点续训）
# CHECKPOINT_PATH=/mnt/tidal-alsh01/dataset/perceptionVLM/exps_zixu/vllm/llava_ov/stage2_sft_previous/iter_0400
CHECKPOINT_PATH=/mnt/tidal-alsh01/dataset/perceptionVLM/models/LLaVA-OneVision-1.5-4B-stage2_mcore_tp1_pp1

# ==============================================================================
# 5. 多节点配置（通常单机训练不需要修改）
# ==============================================================================
if [ $WORLD_SIZE -gt 1 ];
then
   SINGLE_NODE=0
   GPUS_PER_NODE=8
   NNODES=$WORLD_SIZE
   NODE_RANK=$RANK
   MASTER_ADDR=$MASTER_ADDR
   MASTER_PORT=$MASTER_PORT
 else
   SINGLE_NODE=1
   GPUS_PER_NODE=8        # [可能需要修改] 单机GPU数量
   NNODES=1
   NODE_RANK=0
   MASTER_ADDR=localhost
   MASTER_PORT=6003
fi

echo "SINGLE_NODE: ${SINGLE_NODE}"
echo "GPUS_PER_NODE: ${GPUS_PER_NODE}"
echo "NNODES: ${NNODES}"
echo "NODE_RANK: ${NODE_RANK}"
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"

# ==============================================================================
# 6. 创建输出目录
# ==============================================================================
mkdir -p "$SAVE_CKPT_PATH"
mkdir -p "$TENSORBOARD_PATH"
mkdir -p "$SAVE_CKPT_PATH/dataloader"

# [可能需要修改] 如果上面的 GPUS_PER_NODE 设置不正确，可以在这里覆盖
# GPUS_PER_NODE=2

# ==============================================================================
# 7. 分布式训练参数
# ==============================================================================
if [[ $SINGLE_NODE -eq 1 ]]; then
    DISTRIBUTED_ARGS=(
        --nproc_per_node "$GPUS_PER_NODE"
    )
else
    DISTRIBUTED_ARGS=(
        --nproc_per_node "$GPUS_PER_NODE"
        --nnodes "$NNODES"
        --node_rank "$NODE_RANK"
        --master_addr "$MASTER_ADDR"
        --master_port "$MASTER_PORT"
    )
fi

# ==============================================================================
# 8. 模型参数
# ==============================================================================
MODEL_ARGS=(
    --model-name llava-ov-1.5-4b
)

# ==============================================================================
# 9. 数据加载参数
# ==============================================================================
DATA_ARGS=(
    --tokenizer-type HFTokenizer
    --hf-tokenizer-path "$TOKENIZER_PATH"
    --data-path "$DATA_PATH"
    --dataloader-type external
    --split 100,0,0           # 训练/验证/测试集比例：100%训练，0%验证，0%测试
    --num-workers 16          # [可能需要调整] 数据加载器工作进程数（根据CPU核心数调整）
    --chat-template qwen2-vl  # 聊天模板类型
)

# ==============================================================================
# 10. 训练超参数
# ==============================================================================
TRAINING_ARGS=(
    --image-resolution 1000   # 图像分辨率
    --training-phase sft      # 训练阶段：sft (Supervised Fine-Tuning)
    --trainable-modules language_model adapter vision_model  # 可训练模块
    --seq-length "${SEQ_LEN}"
    --max-position-embeddings 32768
    --init-method-std 0.02
    --micro-batch-size "${MBS}"
    --global-batch-size "${GBS}"
    --lr 1.0e-5              # [可能需要调整] 学习率
    --min-lr 1.0e-6          # 最小学习率
    --clip-grad 1.0          # 梯度裁剪
    --weight-decay 0
    --optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.99
    --adam-eps 1e-05
    --norm-epsilon 1e-6
    --train-iters "$NSTEP"
    --lr-decay-iters "$NSTEP"
    --lr-decay-style cosine   # 学习率衰减策略：cosine
    --lr-warmup-fraction 0.002  # 学习率预热比例
    --initial-loss-scale 65536
    --bf16                   # 使用 bfloat16 精度
    --load "$CHECKPOINT_PATH"
    --save "$SAVE_CKPT_PATH"
    --save-interval 200      # [可能需要调整] 保存检查点的间隔（每200步保存一次）
    --ckpt-format torch
    --dataloader-save "${SAVE_CKPT_PATH}/dataloader"

    --ckpt-fully-parallel-load
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
)

# ==============================================================================
# 11. 模型并行参数
# ==============================================================================
MODEL_PARALLEL_ARGS=(
    --attention-backend flash
    --pipeline-model-parallel-size "${PP}"
    --tensor-model-parallel-size "${TP}"
    --use-distributed-optimizer
    --distributed-backend nccl
)

# ==============================================================================
# 12. 日志参数
# ==============================================================================
LOGGING_ARGS=(
    --log-interval 1         # 每1步打印一次日志
    --tensorboard-dir "${TENSORBOARD_PATH}"
    --log-timers-to-tensorboard
)

# WANDB 日志（可选，需要设置环境变量）
if [ -n "${WANDB_API_KEY}" ]; then
    LOGGING_ARGS+=(
        --wandb-project "${WANDB_PROJECT}"
        --wandb-exp-name "${WANDB_NAME}"
    )
fi

# ==============================================================================
# 13. 执行训练
# ==============================================================================
TM=$(date "+%Y-%m-%d_%H:%M:%S")
logfile="${SAVE_CKPT_PATH}/run_${TM}_tp${TP}_pp${PP}_seqlen${SEQ_LEN}_mbs${MBS}_gbs${GBS}_${NSTEP}steps.log"

# 环境变量配置
export OFFLINE_PACKED_DATA='0'
export OFFLINE_PACKING_VQA='0'
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.72
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 运行训练
PYTHONPATH="$AIAK_MAGATRON_PATH:$AIAK_TRAINING_PATH:$PYTHONPATH" \
    torchrun "${DISTRIBUTED_ARGS[@]}" \
    "$AIAK_TRAINING_PATH/aiak_training_llm/train.py" \
    "${MODEL_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    ${IMG_ARGS:+${IMG_ARGS[@]}} \
    "${TRAINING_ARGS[@]}" \
    "${MODEL_PARALLEL_ARGS[@]}" \
    "${LOGGING_ARGS[@]}" \
    2>&1 | tee "$logfile"

echo "训练完成！日志保存在: $logfile"

