#!/usr/bin/bash
# 测试单个样例在不同重算比例下的答案变化
# 用法: ./run_ratio_test_single_example.sh <example_idx>

EXAMPLE_IDX=${1:-4}  # 默认样例 4

export CUDA_VISIBLE_DEVICES=0

echo "========================================="
echo "测试样例 $EXAMPLE_IDX 的答案变化"
echo "========================================="
echo ""
echo "测试比例: 0%, 10%, 15%, 20%, 25%, 30%"
echo ""

# 测试每个比例
for RATIO in 0.0 0.10 0.15 0.20 0.25 0.30
do
    echo ""
    echo "========================================="
    echo "测试比例: $(python3 -c "print(f'{$RATIO:.0%}')")"
    echo "========================================="
    echo ""

    # 创建临时测试脚本
    cat > /tmp/test_ratio_${RATIO}.py << EOF
import sys
sys.path.insert(0, '/mnt/data/wjh/FusionRAG')

from test_fusionrag_reflect import main, PreprocessScope

main(
    model_type='qwen',
    model_path='/mnt/data/models/Qwen2.5-7B-Instruct',
    data_path='./result_reflect.json',
    cache_path='/mnt/data/reflect/',
    model_name='Qwen2.5-7B-Instruct',
    rate=${RATIO},
    reprocess_method='Oracle',
    preprocess=True,
    preprocess_scope=PreprocessScope.GLOBAL,
    bge_model_path='/mnt/data/models/bge-m3-FP16',
    revert_rope=True,
    device="cuda:0",
    use_multi_gpu=True,
    openai_base_url="https://api.deepseek.com/v1",
    openai_api_key="sk-519d391217894b6e91e7c2ebf2a9f4df",
    openai_model="deepseek-chat",
    max_samples=$((EXAMPLE_IDX + 1)),
    draft_layer_selection='entropy'
)
EOF

    # 运行测试
    timeout 600 python3 /tmp/test_ratio_${RATIO}.py 2>&1 | tee /tmp/test_ratio_${RATIO}.log

    # 提取结果
    echo ""
    echo "结果摘要:"
    grep -A 5 "Main Q: $((EXAMPLE_IDX + 1))" ./result/Oracle_global_topk_10_rate_${RATIO}.csv || echo "未找到结果"
    echo ""

done

echo ""
echo "========================================="
echo "所有测试完成"
echo "========================================="
echo ""
echo "结果文件位于: ./result/Oracle_global_topk_10_rate_*.csv"
echo ""

# 生成汇总报告
echo "生成汇总报告..."

cat > /tmp/summarize_results.py << 'EOF'
import csv
import os

example_idx = ${EXAMPLE_IDX}
ratios = [0.0, 0.10, 0.15, 0.20, 0.25, 0.30]

print("\n" + "="*100)
print(f"样例 {example_idx} 的答案变化汇总")
print("="*100)
print()

for ratio in ratios:
    csv_file = f'./result/Oracle_global_topk_10_rate_{ratio}.csv'
    if not os.path.exists(csv_file):
        continue

    print(f"比例 {ratio:.0%}:")
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row['Main Q']) == example_idx + 1:
                print(f"  子问题 {row['Sub Q']}: F1={float(row['f1']):.4f}, EM={float(row['em']):.2f}, Correct={row['correct']}")
                print(f"    答案: {row['predicted'][:80]}")
    print()
EOF

python3 /tmp/summarize_results.py

echo ""
echo "如需详细分析 attention 分布，运行:"
echo "python3 analyze_critical_ratio.py --example_idx ${EXAMPLE_IDX} --sub_question_idx 0"
