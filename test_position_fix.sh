#!/bin/bash
# 测试位置映射修复

GPUS="5"
PYTHON_PATH="/home/shm/anaconda3/envs/fusionrag/bin/python"
SCRIPT_PATH="/home/shm/document/exp/FusionRAG/test_fusionrag_reflect_v2.py"
CACHE_DIR="/tmp/test_position_fix"
cd /home/shm/document/exp/FusionRAG

export CUDA_VISIBLE_DEVICES=${GPUS}

# 清空缓存，确保测试首次运行（有 missing chunks）
rm -rf ${CACHE_DIR}

echo "=========================================="
echo "测试位置映射修复（rate=0.1）"
echo "=========================================="
${PYTHON_PATH} ${SCRIPT_PATH} \
    --model_type "qwen" \
    --model_path "/mnt/data/models/Qwen2.5-7B-Instruct" \
    --model_name "Qwen2.5-7B-Instruct" \
    --data_path "./data/result_reflect_optimized.json" \
    --dataset_name "musique" \
    --cache_path "${CACHE_DIR}" \
    --rate 0.1 \
    --preprocess false \
    --recall_method "online_lazy" \
    --reprocess_method "FusionRAG" \
    --revert_rope true \
    --max_samples 3 2>&1 | grep -E "(RECOMPUTE TOKEN DEBUG|Main|Sub|Missing|loaded|Doc.*:|cache_position)"

echo ""
echo "查看调试输出，确认："
echo "  1. cache_position 不会超出 past_len + new_docs 的范围"
echo "  2. Missing chunks 的 cache_position 从 past_len 开始"
echo "  3. 不会覆盖已加载文档的位置"
