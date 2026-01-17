#!/bin/bash

#####################################################################
# Quick Test: Compute steering vectors on 2wikimqa (small sample)
#####################################################################

cd /home/shm/document/exp/FusionRAG

PYTHON_PATH="/home/shm/anaconda3/envs/fusionrag/bin/python"
TRAIN_CACHE_BASE="/mnt/data3/tmp/fusionrag"  # Base directory (without model_name/dataset)
TRAIN_DATASET="2wikimqa"
MODEL_NAME="Qwen2.5-7B-Instruct"
STATS_OUTPUT="./kv_stats/test_2wikimqa_steering.pt"

# Sample selection
# Option 1: Use a few samples for quick test
USE_ALL_SAMPLES="false"  # Set to "true" to auto-scan all samples
SAMPLE_IDS="0 1 2 3 4"  # Only used if USE_ALL_SAMPLES="false"

echo "=========================================="
echo "Quick Test: 2wikimqa → Steering Vectors"
echo "=========================================="
echo "Cache base: ${TRAIN_CACHE_BASE}"
echo "Dataset: ${TRAIN_DATASET}"
echo "Model: ${MODEL_NAME}"
if [ "${USE_ALL_SAMPLES}" = "true" ]; then
    echo "Sample selection: Auto-scan all samples"
else
    echo "Sample IDs: ${SAMPLE_IDS}"
fi
echo "Output: ${STATS_OUTPUT}"
echo ""

# Build command based on USE_ALL_SAMPLES
CMD="${PYTHON_PATH} compute_kv_distribution_stats.py \
    --cache_dir \"${TRAIN_CACHE_BASE}\" \
    --dataset \"${TRAIN_DATASET}\" \
    --model_name \"${MODEL_NAME}\""

if [ "${USE_ALL_SAMPLES}" = "true" ]; then
    CMD="${CMD} --use_all_samples"
else
    CMD="${CMD} --sample_ids ${SAMPLE_IDS}"
fi

CMD="${CMD} \
    --chunk_id 1 \
    --max_layers 28 \
    --output_path \"${STATS_OUTPUT}\" \
    --use_manifold_projection \
    --pca_variance_threshold 0.7"

eval ${CMD}

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Test successful!"
    echo "=========================================="
    echo "Steering vectors saved to: ${STATS_OUTPUT}"
    echo ""
    echo "You can now run the full experiment with:"
    echo "  bash run_cross_dataset_experiment.sh"
else
    echo ""
    echo "✗ Test failed!"
    exit 1
fi
