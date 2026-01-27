#!/bin/bash

echo "=============================================="
echo "测试新的token范围功能"
echo "=============================================="
echo ""

echo "测试1: 全部token（应该看到AI_Tech有90t，AI_Tech2有164t）"
echo "命令: bash run_text_kv_pca.sh --text_file example_texts.txt --layers 0 --output_dir ./test1_full"
bash run_text_kv_pca.sh --text_file example_texts.txt --layers 0 --output_dir ./test1_full

echo ""
echo "=============================================="
echo "测试2: 只看第90-130个token（应该只有AI_Tech2有40t）"
echo "命令: bash run_text_kv_pca.sh --text_file example_texts.txt --start_token 90 --end_token 130 --layers 0 --output_dir ./test2_partial"
bash run_text_kv_pca.sh --text_file example_texts.txt --start_token 90 --end_token 130 --layers 0 --output_dir ./test2_partial

echo ""
echo "=============================================="
echo "测试3: 从第90个token开始（AI_Tech=74t，AI_Tech2=74t）"
echo "命令: bash run_text_kv_pca.sh --text_file example_texts.txt --start_token 90 --layers 0 --output_dir ./test3_from90"
bash run_text_kv_pca.sh --text_file example_texts.txt --start_token 90 --layers 0 --output_dir ./test3_from90

echo ""
echo "=============================================="
echo "查看结果:"
echo "  test1_full - 全部token"
echo "  test2_partial - 只看90-130token"
echo "  test3_from90 - 从90开始"
echo "=============================================="
