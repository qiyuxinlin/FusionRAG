#!/bin/bash
# 测试修复后的重算逻辑

GPUS="5"
PYTHON_PATH="/home/shm/anaconda3/envs/fusionrag/bin/python"
SCRIPT_PATH="/home/shm/document/exp/FusionRAG/test_fusionrag_reflect_v2.py"
CACHE_DIR="/tmp/test_fix_cache"
cd /home/shm/document/exp/FusionRAG

export CUDA_VISIBLE_DEVICES=${GPUS}

# 清空缓存
rm -rf ${CACHE_DIR}

echo "=========================================="
echo "测试修复：Rate=0.1 (之前 63.0%，预期提升)"
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
    --max_samples 5 2>&1 | grep -E "(Main Questions|Sub Questions|F1|EM|ONLINE_LAZY mode|Importance-based|Missing docs will be added)"

echo ""
echo "查看日志中的关键信息：应该看到"
echo "  - ONLINE_LAZY mode: X tokens loaded, Y tokens missing"
echo "  - Importance-based selection: Z tokens from loaded docs (rate=0.1)"
echo "  - Missing docs will be added separately (100% coverage)"
