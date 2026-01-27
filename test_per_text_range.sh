#!/bin/bash

echo "=============================================="
echo "测试为每个文本单独设置token范围"
echo "=============================================="
echo ""

echo "示例1: 第一个文本全部显示，第二个文本只看90-130"
echo "命令: bash run_text_kv_pca.sh --text_file example_texts.txt --start_token 0,90 --end_token -1,130 --layers 0"
bash run_text_kv_pca.sh --text_file example_texts.txt \
    --start_token 0,90 \
    --end_token -1,130 \
    --layers 0 \
    --output_dir ./test_per_text_range_1

echo ""
echo "=============================================="
echo "示例2: 两个文本都从0开始，但第一个到90，第二个到130"
echo "命令: bash run_text_kv_pca.sh --text_file example_texts.txt --start_token 0,0 --end_token 90,130 --layers 0"
bash run_text_kv_pca.sh --text_file example_texts.txt \
    --start_token 0,0 \
    --end_token 90,130 \
    --layers 0 \
    --output_dir ./test_per_text_range_2

echo ""
echo "=============================================="
echo "预期结果:"
echo "  测试1: AI_Tech显示全部(90t)，AI_Tech2只显示90-130(40t)"
echo "  测试2: AI_Tech显示0-90(90t)，AI_Tech2显示0-130(130t)"
echo ""
echo "查看结果:"
echo "  ls -lh ./test_per_text_range_*"
echo "=============================================="
