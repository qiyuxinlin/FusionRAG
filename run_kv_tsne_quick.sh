#!/bin/bash

#####################################################################
# KV Cache t-SNE 快速分析脚本
#
# 使用方法：
#   bash run_kv_tsne_quick.sh <num_samples> [output_dir] [perplexity]
#
# 示例：
#   bash run_kv_tsne_quick.sh 5                     # 分析前5个样本，默认perplexity=30
#   bash run_kv_tsne_quick.sh 10 ./my_output      # 分析前10个样本，保存到指定目录
#   bash run_kv_tsne_quick.sh 5 ./output 50       # 分析5个样本，perplexity=50
#
# 参数说明：
#   - num_samples: 分析的样本数量（默认5）
#   - output_dir: 输出目录（默认./kv_tsne_analysis）
#   - perplexity: t-SNE困惑度参数，控制局部/全局结构的平衡（默认30，范围5-50）
#####################################################################

cd /home/shm/document/exp/FusionRAG

# 解析参数
NUM_SAMPLES=${1:-20}         # 默认分析5个样本
OUTPUT_DIR=${2:-"./kv_tsne_analysis"}
PERPLEXITY=${3:-30}         # 默认perplexity=30

# 生成样本 ID 列表 (0 到 NUM_SAMPLES-1)
SAMPLE_IDS=$(seq 0 $((NUM_SAMPLES-1)))

# 环境配置
PYTHON_PATH="/home/shm/anaconda3/envs/fusionrag/bin/python"
SCRIPT_PATH="./visualize_kv_tsne.py"

# 数据配置
CACHE_DIR="/mnt/data3/tmp/fusionrag"
DATASET="musique"
MODEL_NAME="Qwen2.5-7B-Instruct"

# 分析配置
CHUNK_ID="1"                 # Chunk ID (1=第一个文档, 0=system prompt)
MAX_LAYERS="28"              # 模型总层数
MAX_TOKENS="500"             # 每层采样的最大token数
LAYERS="0 5 11 16 18 22 25 27"     # 分析的层（8层均匀分布）

echo "==========================================="
echo "KV Cache t-SNE 快速分析"
echo "==========================================="
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "分析样本数: ${NUM_SAMPLES}"
echo "样本 IDs: ${SAMPLE_IDS}"
echo "分析层: ${LAYERS}"
echo "t-SNE Perplexity: ${PERPLEXITY}"
echo "输出目录: ${OUTPUT_DIR}"
echo "==========================================="

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
    echo "✓ 分析完成！"
    echo ""
    echo "生成的可视化文件："
    echo "  1. tsne_key_example*_chunk*.png  - Key cache t-SNE 分布图"
    echo "  2. tsne_value_example*_chunk*.png - Value cache t-SNE 分布图"
    echo "  3. l2_distance_trends.png - 各层L2距离趋势"
    echo "  4. summary_statistics.json - 统计摘要"
    echo ""
    echo "查看结果："
    echo "  ls -lh ${OUTPUT_DIR}/"
    echo ""
    echo "📊 t-SNE vs PCA 对比："
    echo "  • t-SNE: 更好地保留局部结构，适合发现聚类"
    echo "  • PCA: 保留全局结构，有方差解释率"
else
    echo "✗ 分析失败！退出码: ${EXIT_CODE}"
fi
echo "==========================================="

exit ${EXIT_CODE}
