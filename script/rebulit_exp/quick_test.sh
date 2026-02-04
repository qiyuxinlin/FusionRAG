#!/bin/bash
# KV Cache 重建实验快速测试脚本
# 此脚本运行一个小规模测试（2次实验，K=2）

# Python 环境
PYTHON="/home/shm/anaconda3/envs/fusionrag/bin/python"

# 项目目录
DATA_DIR="/home/shm/document/exp/FusionRAG"

cd "${DATA_DIR}" || exit 1

# 运行快速测试
${PYTHON} script/rebulit_exp/kv_cache_rebuilt_exp.py \
    --K 0 \
    --num_experiments 1 \
    --model_type qwen \
    --model_path /mnt/data/models/Qwen2.5-7B-Instruct \
    --data_path ./data/2wiki_input_rebuilt.json \
    --cache_path ./cache/rebuilt_exp/ \
    --device cuda:0 \
    --max_new_tokens 1024 \
    --repeat_prompt "
Instruction: Repeat the above text word-for-word. Output your response as JSON with key 'repeated_text'." \
    --output_file /home/shm/document/exp/FusionRAG/script/rebulit_exp/results/quick_test.json \
    --revert_rope \
    --bge_model_path /mnt/data/models/bge-m3-FP16 \
    --f1_threshold 0.9 \
    --semantic_threshold 0.85

exit_code=$?

if [ ${exit_code} -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "快速测试成功完成！"
    echo "查看 ./results/quick_test.json 获取结果"
    echo "========================================"
else
    echo ""
    echo "========================================"
    echo "快速测试失败，退出码: ${exit_code}"
    echo "========================================"
fi

exit ${exit_code}
