#!/usr/bin/env python3
"""
批量测试多个样例，使用 DraftModel 方法
找出更多有临界比例的样例
"""

import os
import sys
import csv
import json

os.chdir('/mnt/data/wjh/FusionRAG')
sys.path.insert(0, '/mnt/data/wjh/FusionRAG')

from test_fusionrag_reflect import main, PreprocessScope


def batch_test_examples(
    example_indices=None,
    test_ratios=None,
    device='cuda:0'
):
    """
    批量测试多个样例
    """
    if example_indices is None:
        example_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 测试前 10 个样例

    if test_ratios is None:
        test_ratios = [0.0, 0.10, 0.15, 0.20, 0.25, 0.30]

    print("="*100)
    print("批量测试 - DraftModel 方法")
    print("="*100)
    print(f"样例范围: {min(example_indices)} - {max(example_indices)}")
    print(f"测试比例: {[f'{r:.0%}' for r in test_ratios]}")
    print("="*100)
    print()

    all_results = {}

    for ratio in test_ratios:
        print(f"\n{'='*100}")
        print(f"测试比例: {ratio:.0%}")
        print(f"{'='*100}\n")

        try:
            # 运行测试
            main(
                model_type='qwen',
                model_path='/mnt/data/models/Qwen2.5-7B-Instruct',
                draft_model_path='/mnt/data/models/Qwen2.5-3B-Instruct',  # DraftModel
                data_path='./result_reflect.json',
                cache_path='/mnt/data/reflect/',
                model_name='Qwen2.5-7B-Instruct',
                rate=ratio,
                reprocess_method='DraftModel',  # 使用 DraftModel 方法
                preprocess=True,
                preprocess_scope=PreprocessScope.GLOBAL,
                bge_model_path='/mnt/data/models/bge-m3-FP16',
                revert_rope=True,
                device=device,
                use_multi_gpu=True,
                openai_base_url="https://api.deepseek.com/v1",
                openai_api_key="sk-519d391217894b6e91e7c2ebf2a9f4df",
                openai_model="deepseek-chat",
                max_samples=max(example_indices) + 1,
                draft_layer_selection='entropy',
                use_entropy_selection=True
            )

            # 读取结果
            csv_file = f'/mnt/data/reflect/Qwen2.5-7B-Instruct/results/DraftModel_global_topk_10_rate_{ratio}_revert_rope.csv'
            if os.path.exists(csv_file):
                with open(csv_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    all_results[ratio] = rows
                    print(f"✓ 读取到 {len(rows)} 个子问题的结果")
            else:
                print(f"⚠️  未找到结果文件: {csv_file}")

        except Exception as e:
            print(f"❌ 测试比例 {ratio:.0%} 时出错: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 分析结果
    print(f"\n{'='*100}")
    print("分析所有问题的临界比例")
    print(f"{'='*100}\n")

    if len(all_results) == 0:
        print("❌ 没有可用的结果数据")
        return

    # 获取所有子问题
    first_ratio = list(all_results.keys())[0]
    all_questions = all_results[first_ratio]
    num_questions = len(all_questions)

    critical_cases = []
    gradual_cases = []
    always_correct = []
    always_wrong = []

    for q_idx in range(num_questions):
        row_0 = all_results[first_ratio][q_idx]
        main_q = row_0['Main Question']
        sub_q = row_0['Sub Question']
        ground_truth = row_0['Ground Truth']

        # 收集这个问题在不同比例下的表现
        results = []
        for ratio in sorted(all_results.keys()):
            if q_idx < len(all_results[ratio]):
                row = all_results[ratio][q_idx]
                f1 = float(row['F1'])
                em = float(row['EM'])
                correct = row['Correct'].lower() == 'true'
                answer = row['Predicted']
                results.append({
                    'ratio': ratio,
                    'f1': f1,
                    'em': em,
                    'correct': correct,
                    'answer': answer
                })

        # 找出临界比例
        critical_ratio = None
        transition_idx = -1
        for i in range(len(results) - 1):
            if not results[i]['correct'] and results[i+1]['correct']:
                critical_ratio = results[i+1]['ratio']
                transition_idx = i
                break

        # 分类
        if critical_ratio is not None:
            critical_cases.append({
                'q_idx': q_idx,
                'main_question': main_q,
                'sub_question': sub_q,
                'ground_truth': ground_truth,
                'critical_ratio': critical_ratio,
                'before_answer': results[transition_idx]['answer'],
                'after_answer': results[transition_idx + 1]['answer'],
                'results': results
            })
        else:
            all_correct_flag = all(r['correct'] for r in results)
            all_wrong_flag = all(not r['correct'] for r in results)

            if all_correct_flag:
                always_correct.append({
                    'q_idx': q_idx,
                    'main_question': main_q,
                    'sub_question': sub_q
                })
            elif all_wrong_flag:
                always_wrong.append({
                    'q_idx': q_idx,
                    'main_question': main_q,
                    'sub_question': sub_q,
                    'results': results
                })
            else:
                gradual_cases.append({
                    'q_idx': q_idx,
                    'main_question': main_q,
                    'sub_question': sub_q,
                    'results': results
                })

    # 汇总统计
    print(f"总共 {num_questions} 个子问题")
    print(f"  - 有明确临界比例: {len(critical_cases)}")
    print(f"  - F1 逐渐提升: {len(gradual_cases)}")
    print(f"  - 总是答对: {len(always_correct)}")
    print(f"  - 总是答错: {len(always_wrong)}")
    print()

    # 详细显示有临界比例的问题
    if len(critical_cases) > 0:
        print("="*100)
        print("有明确临界比例的问题：")
        print("="*100)
        for case in critical_cases:
            print(f"\nQ{case['q_idx']}:")
            print(f"  主问题: {case['main_question'][:70]}...")
            print(f"  子问题: {case['sub_question'][:70]}...")
            print(f"  标准答案: {case['ground_truth'][:70]}...")
            print(f"  临界比例: {case['critical_ratio']:.0%}")
            prev_ratio = case['critical_ratio'] - 0.05
            print(f"    {prev_ratio:.0%}: {case['before_answer'][:60]}")
            print(f"    {case['critical_ratio']:.0%}: {case['after_answer'][:60]}")

    # 保存详细结果
    output_file = './batch_test_draft_model_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'test_ratios': test_ratios,
            'num_questions': num_questions,
            'critical_cases': critical_cases,
            'gradual_cases': gradual_cases,
            'always_correct': always_correct,
            'always_wrong': always_wrong,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n详细结果已保存到: {output_file}")

    # 输出建议
    print(f"\n{'='*100}")
    print("建议的下一步")
    print(f"{'='*100}")

    if len(critical_cases) > 0:
        print("\n✓ 找到了有临界比例的问题！")
        print(f"  关键问题索引: {[c['q_idx'] for c in critical_cases]}")
        print(f"\n  对每个临界问题，运行 analyze_critical_ratio.py 分析 attention 特征")
        print(f"  示例：")
        for case in critical_cases[:3]:  # 只显示前 3 个
            # 需要找到对应的 example_idx 和 sub_question_idx
            print(f"    python analyze_critical_ratio.py --example_idx <X> --sub_question_idx <Y>")
    else:
        print("\n⚠️  没有找到明确的临界比例")
        print("  建议：")
        print("  1. 测试更密集的比例点（如每 5% 一个点）")
        print("  2. 扩大测试范围到更多样例")

    return {
        'critical_cases': critical_cases,
        'gradual_cases': gradual_cases,
        'always_correct': always_correct,
        'always_wrong': always_wrong,
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--max_examples', type=int, default=10, help='Max number of examples to test')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use')

    args = parser.parse_args()

    example_indices = list(range(args.max_examples))

    results = batch_test_examples(
        example_indices=example_indices,
        device=args.device
    )
