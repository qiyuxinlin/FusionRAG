#!/bin/bash
# Quick test to verify the script works

GPUS="5"
PYTHON_PATH="/home/shm/anaconda3/envs/fusionrag/bin/python"
SCRIPT_PATH="/home/shm/document/exp/FusionRAG/test_fusionrag_reflect_v2.py"
CACHE_DIR="/tmp/test_script_cache"
cd /home/shm/document/exp/FusionRAG

export CUDA_VISIBLE_DEVICES=${GPUS}

echo "Testing with 1 sample, rate=0.2..."
${PYTHON_PATH} ${SCRIPT_PATH} \
    --model_type "qwen" \
    --model_path "/mnt/data/models/Qwen2.5-7B-Instruct" \
    --model_name "Qwen2.5-7B-Instruct" \
    --data_path "./data/result_reflect_optimized.json" \
    --dataset_name "musique" \
    --cache_path "${CACHE_DIR}" \
    --rate 0.2 \
    --preprocess false \
    --recall_method "online_lazy" \
    --reprocess_method "FusionRAG" \
    --revert_rope true \
    --max_samples 1 \
    2>&1 | head -100

EXIT_CODE=${PIPESTATUS[0]}
echo ""
echo "Exit code: ${EXIT_CODE}"

# Check generated files
if [ -d "${CACHE_DIR}/Qwen2.5-7B-Instruct/musique/kv_cache" ]; then
    echo ""
    echo "Generated KV cache files:"
    ls -lh "${CACHE_DIR}/Qwen2.5-7B-Instruct/musique/kv_cache/" | grep "doc_" | head -10
fi
