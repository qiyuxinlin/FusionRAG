#!/bin/bash
# 测试完整修复

GPUS="5"
PYTHON_PATH="/home/shm/anaconda3/envs/fusionrag/bin/python"
SCRIPT_PATH="/home/shm/document/exp/FusionRAG/test_fusionrag_reflect_v2.py"
CACHE_DIR="/tmp/test_complete_fix"
cd /home/shm/document/exp/FusionRAG

export CUDA_VISIBLE_DEVICES=${GPUS}

# 清空缓存
rm -rf ${CACHE_DIR}

echo "=========================================="
echo "测试完整修复（rate=0.1, 3 samples）"
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
    --max_samples 3 2>&1 | grep -E "(RECOMPUTE|Main|Sub|Missing|Doc.*:|cache_position|Extracting)"

echo ""
echo "=========================================="
echo "对比测试（rate=0.0, 相同 3 samples）"
echo "=========================================="
${PYTHON_PATH} ${SCRIPT_PATH} \
    --model_type "qwen" \
    --model_path "/mnt/data/models/Qwen2.5-7B-Instruct" \
    --model_name "Qwen2.5-7B-Instruct" \
    --data_path "./data/result_reflect_optimized.json" \
    --dataset_name "musique" \
    --cache_path "${CACHE_DIR}" \
    --rate 0.0 \
    --preprocess false \
    --recall_method "online_lazy" \
    --reprocess_method "FusionRAG" \
    --revert_rope true \
    --max_samples 3 2>&1 | grep -E "(Main|Sub)"

echo ""
echo "预期：rate=0.1 的性能应该 >= rate=0.0"
