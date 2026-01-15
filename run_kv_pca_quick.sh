#!/bin/bash

#####################################################################
# KV Cache PCA 快速分析脚本
#
# 使用方法：
#   bash run_kv_pca_quick.sh <num_samples> [output_dir]
#
# 示例：
#   bash run_kv_pca_quick.sh 5                  # 分析前5个样本
#   bash run_kv_pca_quick.sh 10 ./my_output    # 分析前10个样本，保存到指定目录
#   bash run_kv_pca_quick.sh 20                # 分析前20个样本
#####################################################################

cd /home/shm/document/exp/FusionRAG

# 解析参数
NUM_SAMPLES=${1:-5}       # 默认分析5个样本
OUTPUT_DIR=${2:-"./kv_pca_analysis"}

# 生成样本 ID 列表 (0 到 NUM_SAMPLES-1)
SAMPLE_IDS=$(seq 0 $((NUM_SAMPLES-1)))

# 环境配置
PYTHON_PATH="/home/shm/anaconda3/envs/fusionrag/bin/python"
SCRIPT_PATH="./visualize_kv_pca.py"

# 数据配置
CACHE_DIR="/mnt/data3/tmp/fusionrag"
DATASET="musique"
MODEL_NAME="Qwen2.5-7B-Instruct"

# 分析配置
CHUNK_ID="1"                 # Chunk ID (1=第一个文档, 0=system prompt)
MAX_LAYERS="28"              # 模型总层数
MAX_TOKENS="500"             # 每层采样的最大token数
LAYERS="0 5 11 16 22 27"     # 分析的层

echo "=========================================="
echo "KV Cache PCA 快速分析"
echo "=========================================="
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "分析样本数: ${NUM_SAMPLES}"
echo "样本 IDs: ${SAMPLE_IDS}"
echo "分析层: ${LAYERS}"
echo "输出目录: ${OUTPUT_DIR}"
echo "=========================================="
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
    --output_dir "${OUTPUT_DIR}"

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✓ 分析完成！"
    echo ""
    echo "生成的可视化文件："
    echo "  1. pca_key_example*_chunk*.png  - Key cache PCA 分布图"
    echo "  2. pca_value_example*_chunk*.png - Value cache PCA 分布图"
    echo "  3. l2_distance_trends.png - 各层L2距离趋势"
    echo "  4. pca_variance_explained.png - PCA方差解释比例"
    echo "  5. summary_statistics.json - 统计摘要"
    echo ""
    echo "查看结果："
    echo "  ls -lh ${OUTPUT_DIR}/"
else
    echo "✗ 分析失败！退出码: ${EXIT_CODE}"
fi
echo "=========================================="

exit ${EXIT_CODE}
