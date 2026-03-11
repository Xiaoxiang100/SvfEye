#!/bin/bash

# --- I. 配置部分 ---

# 1. GPU 列表
GPU_LIST="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

# 2. 路径配置
MODEL_PATH="/your/model/path/Qwen2.5-VL-3B-Instruct"
ANNO_PATH="/your/data/svfeye_data"
# 3. Benchmark 配置
BENCHMARK="hr-bench_4k"
CONF_THRESHOLD="${CONF_THRESHOLD:-1.00}"  # 优先使用环境变量，否则默认1.00

# 4. 模式选择
MODE_FLAG=""
MERGE_FILE_SUFFIX="_svfeye" # 默认输出文件名后缀
if [[ "$1" == "direct" ]]; then
    MODE_FLAG="--direct-answer"
    MERGE_FILE_SUFFIX="_direct" # 修改输出文件名后缀
    echo "模式: Direct Answer (Baseline)"
else
    echo "模式: svfeye (裁剪)"
fi
# --- II. 执行部分 ---

IFS=',' read -ra GPULIST <<< "$GPU_LIST"
CHUNKS=${#GPULIST[@]}

MODEL_BASENAME=$(basename "$MODEL_PATH")
ANSWERS_DIR="svfeye/eval/answers/${BENCHMARK}/${MODEL_BASENAME}"

mkdir -p ${ANSWERS_DIR}
echo "========================================="
echo "  svfeye 评估脚本启动"
echo "========================================="
echo "模型路径: ${MODEL_PATH}"
echo "数据路径: ${ANNO_PATH}"
echo "Benchmark: ${BENCHMARK}"
echo "输出目录: ${ANSWERS_DIR}"
echo "使用 ${CHUNKS} 块GPU并行处理..."
echo "========================================="

# 启动并行推理任务
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python svfeye/eval/perform_svfeye.py \
        --model-path ${MODEL_PATH} \
        --annotation_path ${ANNO_PATH} \
        --benchmark ${BENCHMARK} \
        --answers-file ${ANSWERS_DIR}/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --conf-threshold ${CONF_THRESHOLD} \
        ${MODE_FLAG} & # <--- 关键修正：在这里使用我们准备好的 MODE_FLAG
done

# 等待所有后台任务完成
echo "所有推理任务已启动，等待完成... (这可能需要一些时间)"
wait
echo "所有推理任务已完成！"

# 根据模式合并结果文件
THRESHOLD_STR=$(printf "%.2f" ${CONF_THRESHOLD} | tr -d '.')
MERGE_FILE=${ANSWERS_DIR}/hr-bench_4k_test_threshold${MERGE_FILE_SUFFIX}_th${THRESHOLD_STR}.jsonl
echo "正在合并结果到: ${MERGE_FILE}"

> "$MERGE_FILE"

for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${ANSWERS_DIR}/${CHUNKS}_${IDX}.jsonl >> "$MERGE_FILE"
done

echo "========================================="
echo "  评估完成！"
echo "========================================="
echo "结果文件已生成: ${MERGE_FILE}"
echo "接下来，请运行以下命令计算最终得分:"
echo "python svfeye/eval/eval_results_vstar.py --answers-file ${MERGE_FILE}"
echo "========================================="