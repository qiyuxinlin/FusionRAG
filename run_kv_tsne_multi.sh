#!/bin/bash

#####################################################################
# 多方法 KV Cache t-SNE 对比分析脚本
#
# 使用方法：
#   bash run_kv_tsne_multi.sh <方法1> <方法2> [方法3 ...] [样本数] [输出目录]
#
# 示例：
#   # 对比 no_preprocess 和 bge
#   bash /home/shm/document/exp/FusionRAG/run_kv_tsne_multi.sh bge no_preprocess fixed_doc 20
#
#   # 对比 3种方法
#   bash run_kv_tsne_multi.sh bge random repeat_self
#
#   # 对比 2种方法，分析5个样本
#   bash run_kv_tsne_multi.sh bge repeat_self 5
#
# 可用方法：
#   no_preprocess, bge, random, repeat_self, fixed_doc,
#   random_docs, random_text, bge_shuffled
#####################################################################

cd /home/shm/document/exp/FusionRAG

# 环境配置
PYTHON_PATH="/home/shm/anaconda3/envs/fusionrag/bin/python"
SCRIPT_PATH="./visualize_kv_tsne_multi.py"

# 数据配置
CACHE_DIR="/mnt/data3/tmp/fusionrag"
DATASET="musique"
MODEL_NAME="Qwen2.5-7B-Instruct"

# 分析配置
CHUNK_ID="1"
MAX_LAYERS="28"
MAX_TOKENS="500"
PERPLEXITY="30"
# LAYERS="0 5 11 16 18 22 25 27"
LAYERS="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27"
# 解析参数
METHODS=()
NUM_SAMPLES=""
OUTPUT_DIR="/home/shm/document/exp/FusionRAG/kv_tsne_analysis_all/v0"

# 可用方法列表
VALID_METHODS=("no_preprocess" "bge" "random" "repeat_self" "fixed_doc" "random_docs" "random_text" "bge_shuffled")

for arg in "$@"; do
    # 检查是否是数字（样本数）
    if [[ $arg =~ ^[0-9]+$ ]]; then
        NUM_SAMPLES=$arg
    # 检查是否是路径（包含/或.）
    elif [[ $arg == *"/"* ]] || [[ $arg == "."* ]]; then
        OUTPUT_DIR=$arg
    # 检查是否是有效方法
    elif [[ " ${VALID_METHODS[@]} " =~ " ${arg} " ]]; then
        METHODS+=("$arg")
    else
        echo "⚠️  警告: 未知参数 '$arg' (将被忽略)"
    fi
done

# 设置默认值
if [ ${#METHODS[@]} -eq 0 ]; then
    METHODS=("no_preprocess" "bge")
    echo "未指定方法，使用默认: no_preprocess bge"
fi

if [ -z "$NUM_SAMPLES" ]; then
    NUM_SAMPLES=5
fi

if [ -z "$OUTPUT_DIR" ]; then
    METHODS_STR=$(IFS=_; echo "${METHODS[*]}")
    OUTPUT_DIR="./kv_tsne_multi_${METHODS_STR}"
fi

# 生成样本 ID 列表
SAMPLE_IDS=$(seq 0 $((NUM_SAMPLES-1)))

echo "==========================================="
echo "多方法 KV Cache t-SNE 对比分析"
echo "==========================================="
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "对比方法: ${METHODS[@]}"
echo "分析样本数: ${NUM_SAMPLES}"
echo "样本 IDs: ${SAMPLE_IDS}"
echo "分析层: ${LAYERS}"
echo "t-SNE Perplexity: ${PERPLEXITY}"
echo "输出目录: ${OUTPUT_DIR}"
echo "==========================================="
echo ""
echo "⚠️  提示: t-SNE比PCA慢，每个样本每层需要1-2秒"
echo ""

# 检查KV cache是否存在
echo "检查 KV cache 可用性..."
for method in "${METHODS[@]}"; do
    case $method in
        no_preprocess)
            dir_path="${CACHE_DIR}/${MODEL_NAME}/${DATASET}/kv_cache"
            ;;
        bge)
            dir_path="${CACHE_DIR}/${MODEL_NAME}/${DATASET}/preprocess_kv_cache_global_topk10_bge"
            ;;
        random)
            dir_path="${CACHE_DIR}/${MODEL_NAME}/${DATASET}/preprocess_kv_cache_global_topk10_random"
            ;;
        repeat_self)
            dir_path="${CACHE_DIR}/${MODEL_NAME}/${DATASET}/preprocess_kv_cache_global_topk10_repeat_self"
            ;;
        fixed_doc)
            dir_path="${CACHE_DIR}/${MODEL_NAME}/${DATASET}/preprocess_kv_cache_global_topk10_fixed_doc"
            ;;
        random_docs)
            dir_path="${CACHE_DIR}/${MODEL_NAME}/${DATASET}/preprocess_kv_cache_global_topk10_random_docs"
            ;;
        random_text)
            dir_path="${CACHE_DIR}/${MODEL_NAME}/${DATASET}/preprocess_kv_cache_global_topk10_random_text"
            ;;
        bge_shuffled)
            dir_path="${CACHE_DIR}/${MODEL_NAME}/${DATASET}/preprocess_kv_cache_global_topk10_bge_shuffled"
            ;;
    esac

    if [ -d "$dir_path" ]; then
        echo "  ✓ ${method}: ${dir_path}"
    else
        echo "  ✗ ${method}: 目录不存在 ${dir_path}"
    fi
done
echo ""

# 创建输出目录
mkdir -p "${OUTPUT_DIR}"

# 运行分析
${PYTHON_PATH} ${SCRIPT_PATH} \
    --cache_dir "${CACHE_DIR}" \
    --dataset "${DATASET}" \
    --model_name "${MODEL_NAME}" \
    --methods ${METHODS[@]} \
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
    echo "  1. tsne_key_example*_chunk*_*.png  - Key cache t-SNE 对比图"
    echo "  2. tsne_value_example*_chunk*_*.png - Value cache t-SNE 对比图"
    echo "  3. l2_distance_trends_*.png - L2距离趋势对比"
    echo "  4. summary_statistics.json - 统计摘要"
    echo ""
    echo "查看结果："
    echo "  ls -lh ${OUTPUT_DIR}/"
    echo ""
    echo "📊 分析的方法："
    for method in "${METHODS[@]}"; do
        echo "  • ${method}"
    done
    echo ""
    echo "📌 t-SNE vs PCA:"
    echo "  • t-SNE: 保留局部结构，适合发现聚类"
    echo "  • PCA: 保留全局结构，有方差解释率"
else
    echo "✗ 分析失败！退出码: ${EXIT_CODE}"
fi
echo "==========================================="

exit ${EXIT_CODE}
