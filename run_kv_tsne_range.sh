#!/bin/bash

#####################################################################
# KV Cache t-SNE 范围分析脚本
#
# 使用方法：
#   bash run_kv_tsne_range.sh <start_id> <end_id> [output_dir] [perplexity]
#
# 示例：
#   bash run_kv_tsne_range.sh 0 5                  # 分析样本 0-5
#   bash run_kv_tsne_range.sh 10 20               # 分析样本 10-20
#   bash run_kv_tsne_range.sh 5 10 ./output 40   # 分析5-10，perplexity=40
#####################################################################

cd /home/shm/document/exp/FusionRAG

# 解析参数
START_ID=${1:-0}
END_ID=${2:-5}
OUTPUT_DIR=${3:-"./kv_tsne_analysis"}
PERPLEXITY=${4:-30}

# 生成样本 ID 列表
SAMPLE_IDS=$(seq ${START_ID} ${END_ID})

# 环境配置
PYTHON_PATH="/home/shm/anaconda3/envs/fusionrag/bin/python"
SCRIPT_PATH="./visualize_kv_tsne.py"

# 数据配置
CACHE_DIR="/mnt/data3/tmp/fusionrag"
DATASET="musique"
MODEL_NAME="Qwen2.5-7B-Instruct"

# 分析配置
CHUNK_ID="1"
MAX_LAYERS="28"
MAX_TOKENS="500"
LAYERS="0 5 11 16 18 22 25 27"

echo "==========================================="
echo "KV Cache t-SNE 范围分析"
echo "==========================================="
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "样本范围: ${START_ID} - ${END_ID}"
echo "样本 IDs: ${SAMPLE_IDS}"
echo "分析层: ${LAYERS}"
echo "t-SNE Perplexity: ${PERPLEXITY}"
echo "输出目录: ${OUTPUT_DIR}"
echo "==========================================="
echo ""

# 创建输出目录
mkdir -p "${OUTPUT_DIR}"

# 运行分析
${PYTHON_PATH} ${SCRIPT_PATH} \
    --cache_dir "${CACHE_DIR}" \
    --dataset "${DATASET}" \
    --model_name "${MODEL_NAME}" \
    --sample_ids ${SAMPLE_IDS} \
    --chunk_id ${CHUNK_ID} \
    --layers ${LAYERS} \
    --max_layers ${MAX_LAYERS} \
    --max_tokens ${MAX_TOKENS} \
    --perplexity ${PERPLEXITY} \
    --output_dir "${OUTPUT_DIR}"

EXIT_CODE=$?

echo ""
echo "==========================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✓ 分析完成！查看: ${OUTPUT_DIR}/"
else
    echo "✗ 分析失败！退出码: ${EXIT_CODE}"
fi
echo "==========================================="

exit ${EXIT_CODE}
