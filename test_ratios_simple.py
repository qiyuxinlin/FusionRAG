#!/usr/bin/env python3
"""
简化版：测试单个样例在多个重算比例下的答案变化
直接调用 test_fusionrag_reflect.main() 多次
"""

import os
import sys
import csv
import json

# 确保在正确的目录
os.chdir('/mnt/data/wjh/FusionRAG')
sys.path.insert(0, '/mnt/data/wjh/FusionRAG')

from test_fusionrag_reflect import main, PreprocessScope


def test_multiple_ratios(
    example_idx=4,
    test_ratios=None,
    device='cuda:0'
):
    """
    测试多个重算比例
    """
    if test_ratios is None:
        test_ratios = [0.0, 0.10, 0.15, 0.20, 0.25, 0.30]

    print("="*100)
    print(f"测试样例 {example_idx} 在不同重算比例下的答案变化")
    print("="*100)
    print(f"测试比例: {[f'{r:.0%}' for r in test_ratios]}")
    print("="*100)
    print()

    results_by_ratio = {}

    for ratio in test_ratios:
        print(f"\n{'='*100}")
        print(f"测试比例: {ratio:.0%}")
        print(f"{'='*100}\n")

        try:
            # 运行测试（只测试到指定样例）
            main(
                model_type='qwen',
                model_path='/mnt/data/models/Qwen2.5-7B-Instruct',
                data_path='./result_reflect.json',
                cache_path='/mnt/data/reflect/',
                model_name='Qwen2.5-7B-Instruct',
                rate=ratio,
                reprocess_method='Oracle',
                preprocess=True,
                preprocess_scope=PreprocessScope.GLOBAL,
                bge_model_path='/mnt/data/models/bge-m3-FP16',
                revert_rope=True,
                device=device,
                use_multi_gpu=True,
                openai_base_url="https://api.deepseek.com/v1",
                openai_api_key="sk-519d391217894b6e91e7c2ebf2a9f4df",
                openai_model="deepseek-chat",
                max_samples=example_idx + 1,  # 只测试到这个样例
                draft_layer_selection='entropy'
            )

            # 读取结果
            csv_file = f'./result/Oracle_global_topk_10_rate_{ratio}_revert_rope.csv'
            if os.path.exists(csv_file):
                with open(csv_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    # 找到对应的样例
                    example_rows = [r for r in rows if int(r['Main Q']) == example_idx + 1]
                    results_by_ratio[ratio] = example_rows

                    print(f"\n样例 {example_idx + 1} 的结果:")
                    for row in example_rows:
                        print(f"  子问题 {row['Sub Q']}: F1={float(row['f1']):.4f}, EM={float(row['em']):.2f}, Correct={row['correct']}")
                        print(f"    答案: {row['predicted'][:80]}")
            else:
                print(f"⚠️  未找到结果文件: {csv_file}")

        except Exception as e:
            print(f"❌ 测试比例 {ratio:.0%} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 生成汇总报告
    print(f"\n{'='*100}")
    print(f"样例 {example_idx + 1} 的答案变化汇总")
    print(f"{'='*100}\n")

    # 假设每个比例都测试了相同数量的子问题
    if len(results_by_ratio) > 0:
        first_ratio = list(results_by_ratio.keys())[0]
        num_sub_questions = len(results_by_ratio[first_ratio])

        for sub_q_idx in range(num_sub_questions):
            print(f"\n子问题 {sub_q_idx + 1}:")

            # 获取标准答案（从第一个比例的结果中）
            if sub_q_idx < len(results_by_ratio[first_ratio]):
                ground_truth = results_by_ratio[first_ratio][sub_q_idx]['ground_truth']
                print(f"  标准答案: {ground_truth}")

            print(f"\n  {'Ratio':<8} {'F1':<10} {'EM':<8} {'Correct':<10} Answer")
            print(f"  {'-'*90}")

            # 收集这个子问题在不同比例下的表现
            sub_q_results = []

            for ratio in sorted(results_by_ratio.keys()):
                if sub_q_idx < len(results_by_ratio[ratio]):
                    row = results_by_ratio[ratio][sub_q_idx]
                    f1 = float(row['f1'])
                    em = float(row['em'])
                    correct = row['correct'].lower() == 'true'
                    answer = row['predicted'][:60]
                    correct_mark = "✓" if correct else "✗"

                    print(f"  {ratio:<8.0%} {f1:<10.4f} {em:<8.2f} {correct_mark:<10} {answer}")

                    sub_q_results.append({
                        'ratio': ratio,
                        'f1': f1,
                        'em': em,
                        'correct': correct,
                        'answer': row['predicted']
                    })

            # 找出临界比例
            critical_ratio = None
            for i in range(len(sub_q_results) - 1):
                if not sub_q_results[i]['correct'] and sub_q_results[i+1]['correct']:
                    critical_ratio = sub_q_results[i+1]['ratio']
                    break

            if critical_ratio is not None:
                print(f"\n  ⚠️  临界比例: {critical_ratio:.0%} (从这个比例开始答对)")
            else:
                all_correct = all(r['correct'] for r in sub_q_results)
                all_wrong = all(not r['correct'] for r in sub_q_results)
                if all_correct:
                    print(f"\n  ✓ 所有比例都能答对（包括 0%）")
                elif all_wrong:
                    print(f"\n  ✗ 所有比例都答错（包括 30%）")

    # 保存汇总结果
    output_file = f'./ratio_test_summary_ex{example_idx}.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_by_ratio, f, indent=2, ensure_ascii=False)

    print(f"\n\n详细结果已保存到: {output_file}")

    print("\n建议的下一步：")
    print(f"1. 运行 analyze_critical_ratio.py 分析 attention 分布特征")
    print(f"2. 对比不同临界比例的样例，找出规律")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--example_idx', type=int, default=4, help='Example index to test')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')

    args = parser.parse_args()

    test_multiple_ratios(
        example_idx=args.example_idx,
        device=args.device
    )
