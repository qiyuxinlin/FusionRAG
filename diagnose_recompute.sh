#!/bin/bash
# 诊断重算问题的测试脚本

GPUS="5"
PYTHON_PATH="/home/shm/anaconda3/envs/fusionrag/bin/python"
SCRIPT_PATH="/home/shm/document/exp/FusionRAG/test_fusionrag_reflect_v2.py"
CACHE_DIR="/tmp/diagnose_cache"
cd /home/shm/document/exp/FusionRAG

export CUDA_VISIBLE_DEVICES=${GPUS}

# 清空缓存，确保有 missing_chunks
rm -rf ${CACHE_DIR}

echo "============================================"
echo "测试1: Rate=0.0 (baseline, 只重算 missing + 问题)"
echo "============================================"
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
    --max_samples 3 2>&1 | grep -E "(Main|Sub|F1|EM|Adjusting budget|Added.*missing|doc_id:|select_time)"

echo ""
echo "============================================"
echo "测试2: Rate=0.1 (低重算率，预期性能下降)"
echo "============================================"
# 复用缓存，减少 missing_chunks 的影响
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
    --max_samples 3 2>&1 | grep -E "(Main|Sub|F1|EM|Adjusting budget|Added.*missing|doc_id:|select_time)"

echo ""
echo "============================================"
echo "测试3: Rate=0.5 (高重算率，预期性能恢复)"
echo "============================================"
${PYTHON_PATH} ${SCRIPT_PATH} \
    --model_type "qwen" \
    --model_path "/mnt/data/models/Qwen2.5-7B-Instruct" \
    --model_name "Qwen2.5-7B-Instruct" \
    --data_path "./data/result_reflect_optimized.json" \
    --dataset_name "musique" \
    --cache_path "${CACHE_DIR}" \
    --rate 0.5 \
    --preprocess false \
    --recall_method "online_lazy" \
    --reprocess_method "FusionRAG" \
    --revert_rope true \
    --max_samples 3 2>&1 | grep -E "(Main|Sub|F1|EM|Adjusting budget|Added.*missing|doc_id:|select_time)"

echo ""
echo "============================================"
echo "分析：检查生成的 KV 缓存"
echo "============================================"
ls -lh ${CACHE_DIR}/Qwen2.5-7B-Instruct/musique/kv_cache/ | grep "doc_" | wc -l
echo "个文档被缓存"
