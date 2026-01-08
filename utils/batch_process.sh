#!/bin/bash
# 批量处理分割后的JSON文件
# 用法: ./batch_process.sh <输入目录> <输出目录> [workers] [其他参数]

INPUT_DIR="${1:-data/chunks}"
OUTPUT_DIR="${2:-data/results}"
WORKERS="${3:-1}"
EXTRA_ARGS="${@:4}"

if [ ! -d "$INPUT_DIR" ]; then
    echo "错误: 输入目录不存在: $INPUT_DIR"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# 获取所有JSON文件并排序
FILES=($(ls "$INPUT_DIR"/*.json | sort))

TOTAL=${#FILES[@]}
echo "找到 $TOTAL 个文件，开始处理..."
echo "并发数: $WORKERS"
echo "额外参数: $EXTRA_ARGS"
echo ""

# 处理每个文件
for i in "${!FILES[@]}"; do
    FILE="${FILES[$i]}"
    FILENAME=$(basename "$FILE")
    OUTPUT_FILE="$OUTPUT_DIR/${FILENAME%.json}_result.json"
    
    echo "[$((i+1))/$TOTAL] 处理: $FILENAME"
    
    python main.py \
        --json "$FILE" \
        --output "$OUTPUT_FILE" \
        --workers "$WORKERS" \
        $EXTRA_ARGS
    
    if [ $? -ne 0 ]; then
        echo "错误: 处理 $FILENAME 失败"
        # 可以选择继续或退出
        # exit 1
    fi
    
    echo ""
done

echo "所有文件处理完成！"
echo "结果文件在: $OUTPUT_DIR"
echo ""
echo "合并结果:"
echo "  python utils/split_json.py merge $OUTPUT_DIR/*_result.json merged_final.json"

# # 分割为小文件
# python utils/split_json.py split "/home/zhuxuzhou/test_localization/object_localization/data/match_results/object_localization/finegrained_perception (instance-level)/part-00000.json" /home/zhuxuzhou/test_localization/object_localization/data/chunks/ -s 500
   
# # 批量处理
# ./utils/batch_process.sh /home/zhuxuzhou/test_localization/object_localization/data/chunks/ /home/zhuxuzhou/test_localization/object_localization/data/results/ 1
   
# # 合并结果
# python utils/split_json.py merge /home/zhuxuzhou/test_localization/object_localization/data/results/*_result.json final_output.json