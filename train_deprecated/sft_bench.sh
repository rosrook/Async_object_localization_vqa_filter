AIAK_TRAINING_PATH=/mnt/ali-sh-1/usr/zixu/code/llvm/LLaVA-OneVision
AIAK_MAGATRON_PATH="${AIAK_MAGATRON_PATH:-${AIAK_TRAINING_PATH%/}/aiak_megatron}"

EXP_DIR='/mnt/tidal-alsh01/dataset/perceptionVLM/exps_zixu/vllm/llava_ov/'
SAVE_CKPT_PATH=$EXP_DIR/stage2_sft_agricuture_1210
TENSORBOARD_PATH="${SAVE_CKPT_PATH}/tensorboard"
echo "SAVE_CKPT_PATH: ${SAVE_CKPT_PATH}"
echo "TENSORBOARD_PATH: ${TENSORBOARD_PATH}"

TP=1
PP=1
SEQ_LEN=32768
MBS=1
GBS=2
NSTEP=220
#DATA_PATH=/mnt/tidal-alsh01/dataset/perceptionVLMData/datasets--lmms-lab--LLaVA-NeXT-780k-webdataset
#DATA_PATH=/mnt/tidal-alsh01/dataset/perceptionVLMData/processed_v1.0/datasets--lmms-lab--LLaVA-OneVision-1.5-Insturct-4.6Mixture
#DATA_PATH=/mnt/tidal-alsh01/dataset/perceptionVLMData/processed_v1.0/datasets--lmms-lab--LLaVA-OneVision-1.5-Insturct-46Mixture
DATA_PATH=/mnt/tidal-alsh01/dataset/perceptionVLMData/processed_v2.0/datasets--MMMU_EVAL/webdatasets/
DATA_PATH=/mnt/tidal-alsh01/dataset/perceptionVLMData/processed_v2.0/datasets--MMMU_EVAL/webdatasets_v2/
DATA_PATH=/mnt/tidal-alsh01/dataset/perceptionVLMData/processed_v2.0/datasets--MMMU_EVAL/webdatasets_trainset/
DATA_PATH=/mnt/tidal-alsh01/dataset/perceptionVLMData/processed_v2.0/chemistry_webdataset_v3
#TOKENIZER_PATH=/mnt/tidal-alsh01/dataset/perceptionVLM/models/LLaVA-OneVision-1.5-4B-stage0
#CHECKPOINT_PATH=/mnt/tidal-alsh01/dataset/perceptionVLM/models/LLaVA-OneVision-1.5-4B-Base
TOKENIZER_PATH=/mnt/tidal-alsh01/dataset/perceptionVLM/models/LLaVA-OneVision-1.5-4B-Instruct
CHECKPOINT_PATH=/mnt/tidal-alsh01/dataset/perceptionVLM/models/LLaVA-OneVision-1.5-4B-stage2_mcore_tp1_pp1
#CHECKPOINT_PATH=LLaVA-OneVision-1.5-4B-Base_mcore_tp1_pp1
#CHECKPOINT_PATH=/mnt/ali-sh-1/usr/zixu/code/llvm/LLaVA-OneVision/stage_2_instruct_llava_ov_4b_46Mixture


#! /bin/bash
# The script needs to be run on at least 1 nodes.

# --- Multi-node configuration ---
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
   GPUS_PER_NODE=2
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
# --- End of Multi-node configuration ---




mkdir -p "$SAVE_CKPT_PATH"
mkdir -p "$TENSORBOARD_PATH"
mkdir -p "$SAVE_CKPT_PATH/dataloader"
GPUS_PER_NODE=2


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

MODEL_ARGS=(
    --model-name llava-ov-1.5-4b
)

DATA_ARGS=(
    --tokenizer-type HFTokenizer
    --hf-tokenizer-path "$TOKENIZER_PATH"
    --data-path "$DATA_PATH"
    --dataloader-type external
    --split 100,0,0
    --num-workers 16
    --chat-template qwen2-vl
)

TRAINING_ARGS=(
    --image-resolution 1000
    --training-phase sft
    --trainable-modules language_model adapter vision_model
    --seq-length "${SEQ_LEN}"
    --max-position-embeddings 32768
    --init-method-std 0.02
    --micro-batch-size "${MBS}"
    --global-batch-size "${GBS}"
    --lr 1.0e-5
    --min-lr 1.0e-6
    --clip-grad 1.0
    --weight-decay 0
    --optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.99
    --adam-eps 1e-05
    --norm-epsilon 1e-6
    --train-iters "$NSTEP"
    --lr-decay-iters "$NSTEP"
    --lr-decay-style cosine
    --lr-warmup-fraction 0.002
    --initial-loss-scale 65536
    --bf16
    --load "$CHECKPOINT_PATH"
    --save "$SAVE_CKPT_PATH"
    --save-interval 200
    --ckpt-format torch
    --dataloader-save "${SAVE_CKPT_PATH}/dataloader"

    --ckpt-fully-parallel-load
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
)

MODEL_PARALLEL_ARGS=(
    --attention-backend flash
    --pipeline-model-parallel-size "${PP}"
    --tensor-model-parallel-size "${TP}"
    --use-distributed-optimizer
    --distributed-backend nccl
)

LOGGING_ARGS=(
    --log-interval 1
    --tensorboard-dir "${TENSORBOARD_PATH}"
    --log-timers-to-tensorboard
)

if [ -n "${WANDB_API_KEY}" ]; then
    LOGGING_ARGS+=(
        --wandb-project "${WANDB_PROJECT}"
        --wandb-exp-name "${WANDB_NAME}"
    )
fi

TM=$(date "+%Y-%m-%d_%H:%M:%S")
logfile="${SAVE_CKPT_PATH}/run_${TM}_tp${TP}_pp${PP}_seqlen${SEQ_LEN}_mbs${MBS}_gbs${GBS}_${NSTEP}steps.log"

export OFFLINE_PACKED_DATA='0'
export OFFLINE_PACKING_VQA='0'
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.72
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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
