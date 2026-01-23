#!/bin/bash
# 对比 rate=0.0 和 rate>0 的重算 token 差异

GPUS="6"
PYTHON_PATH="/home/shm/anaconda3/envs/fusionrag/bin/python"
SCRIPT_PATH="/home/shm/document/exp/FusionRAG/test_fusionrag_reflect_v2.py"
CACHE_DIR="/tmp/debug_rate_comparison"
cd /home/shm/document/exp/FusionRAG

export CUDA_VISIBLE_DEVICES=${GPUS}

# 清空缓存，确保第一次运行有 missing chunks
# rm -rf ${CACHE_DIR}

echo "========================================"
echo "测试 1: rate=0.0 (baseline)"
echo "========================================"
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
    --max_samples 1 2>&1 | tee /tmp/rate_0.0_debug.log

echo ""
echo "========================================"
echo "测试 2: rate=0.1"
echo "========================================"
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
    --max_samples 1 2>&1 | tee /tmp/rate_0.1_debug.log

echo ""
echo "========================================"
echo "对比分析"
echo "========================================"
echo ""
echo "Rate 0.0 的 cache_position:"
grep "cache_position (kv positions)" /tmp/rate_0.0_debug.log | head -5
echo ""
echo "Rate 0.1 的 cache_position:"
grep "cache_position (kv positions)" /tmp/rate_0.1_debug.log | head -5
echo ""
echo "Rate 0.0 的 Token Distribution:"
grep -A 10 "Token Distribution:" /tmp/rate_0.0_debug.log | head -15
echo ""
echo "Rate 0.1 的 Token Distribution:"
grep -A 10 "Token Distribution:" /tmp/rate_0.1_debug.log | head -15
echo ""
echo "详细日志已保存到："
echo "  /tmp/rate_0.0_debug.log"
echo "  /tmp/rate_0.1_debug.log"
