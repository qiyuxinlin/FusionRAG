#!/bin/bash
# FusionRAG结果对比示例脚本

# 示例1: 基本用法
echo "示例1: 基本对比"
python /home/shm/document/exp/FusionRAG/script/compare_results.py \
  --file1 /home/shm/document/exp/FusionRAG/result/fixed_doc/Qwen2.5-7B-Instruct/musique/FusionRAG_global_topk10_fixed_doc/rate_0.0_revert_rope.csv \
  --file2 /home/shm/document/exp/FusionRAG/result/fixed_doc/Qwen2.5-7B-Instruct/musique/FusionRAG_global_topk10_fixed_doc/rate_0.5_revert_rope.csv \
  --name1 "rate_0.0" \
  --name2 "rate_0.5" \
  --output comparison_rate_0.0_vs_0.5.txt

# 示例2: 比较不同配置
echo -e "\n示例2: 比较不同topk配置"
# python /home/shm/document/exp/FusionRAG/script/compare_results.py \
#   --file1 /path/to/topk5_result.csv \
#   --file2 /path/to/topk10_result.csv \
#   --name1 "TopK-5" \
#   --name2 "TopK-10" \
#   --output comparison_topk.txt

# 示例3: 比较baseline和FusionRAG
echo -e "\n示例3: 比较baseline和FusionRAG"
# python /home/shm/document/exp/FusionRAG/script/compare_results.py \
#   --file1 /path/to/baseline_result.csv \
#   --file2 /path/to/fusionrag_result.csv \
#   --name1 "Baseline" \
#   --name2 "FusionRAG" \
#   --output comparison_baseline_vs_fusionrag.txt
