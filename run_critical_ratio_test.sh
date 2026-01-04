#!/bin/bash
# 运行临界比例分析

export CUDA_VISIBLE_DEVICES=0

echo "========================================="
echo "临界重算比例分析"
echo "========================================="
echo ""
echo "分析样例 4，子问题 1"
echo "测试比例: 0%, 5%, 10%, 15%, 20%, 25%, 30%, 35%"
echo ""
echo "这个分析会:"
echo "1. 计算 attention 分布"
echo "2. 分析 Top-k concentration"
echo "3. 分析 connected components"
echo "4. 为每个比例显示选中的 tokens"
echo "5. 生成可视化图表"
echo ""
echo "开始分析..."
echo ""

python3 analyze_critical_ratio.py \
    --example_idx 4 \
    --sub_question_idx 1 \
    --device cuda:0

echo ""
echo "分析完成！"
echo "查看结果:"
echo "  - 可视化: ./critical_ratio_analysis_ex4_sub1.png"
