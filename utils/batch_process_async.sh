#!/bin/bash
# 异步并行批量处理分割后的JSON文件
# 用法: ./batch_process_async.sh <输入目录> <输出目录> [num_gpus] [concurrency] [其他参数]
#
# 参数说明:
#   num_gpus: GPU数量（用于并发控制，默认: 1）
#   concurrency: 每个GPU的最大并发数（默认: 10）
#   其他参数: 传递给 main_async.py 的额外参数（如 --request-delay, --save-interval 等）

INPUT_DIR="${1:-data/chunks}"
OUTPUT_DIR="${2:-data/results}"
NUM_GPUS="${3:-1}"
CONCURRENCY="${4:-10}"
EXTRA_ARGS="${@:5}"

if [ ! -d "$INPUT_DIR" ]; then
    echo "错误: 输入目录不存在: $INPUT_DIR"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# 获取所有JSON文件并排序
FILES=($(ls "$INPUT_DIR"/*.json | sort))

TOTAL=${#FILES[@]}
echo "找到 $TOTAL 个文件，开始异步并行处理..."
echo "GPU数量: $NUM_GPUS"
echo "每GPU并发数: $CONCURRENCY"
echo "额外参数: $EXTRA_ARGS"
echo ""

# 处理每个文件
for i in "${!FILES[@]}"; do
    FILE="${FILES[$i]}"
    FILENAME=$(basename "$FILE")
    OUTPUT_FILE="$OUTPUT_DIR/${FILENAME%.json}_result.json"
    
    echo "[$((i+1))/$TOTAL] 处理: $FILENAME"
    echo "  开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    
    # 使用异步版本的主程序
    python main_async.py \
        --json "$FILE" \
        --output "$OUTPUT_FILE" \
        --num-gpus "$NUM_GPUS" \
        --concurrency "$CONCURRENCY" \
        $EXTRA_ARGS
    
    EXIT_CODE=$?
    
    echo "  结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
    
    if [ $EXIT_CODE -ne 0 ]; then
        echo "错误: 处理 $FILENAME 失败 (退出码: $EXIT_CODE)"
        # 可以选择继续或退出
        # exit 1
    else
        echo "✓ 成功处理: $FILENAME"
    fi
    
    echo ""
done

echo "所有文件处理完成！"
echo "结果文件在: $OUTPUT_DIR"
echo ""
echo "合并结果:"
echo "  python utils/split_json.py merge $OUTPUT_DIR/*_result.json merged_final.json"

# 使用示例:
# # 分割为小文件
# python utils/split_json.py split "/path/to/large_file.json" /path/to/chunks/ -s 500
#   
# # 异步批量处理（单GPU，并发10）
# ./utils/batch_process_async.sh /path/to/chunks/ /path/to/results/ 1 10
#   
# # 异步批量处理（多GPU，每GPU并发5，请求延迟0.2秒，每50条保存一次）
# ./utils/batch_process_async.sh /path/to/chunks/ /path/to/results/ 4 5 --request-delay 0.2 --save-interval 50
#   
# # 合并结果
# python utils/split_json.py merge /path/to/results/*_result.json final_output.json

