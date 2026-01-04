#!/usr/bin/env python3
"""
测试重算比例与答案变化的相关性

对指定的几个测试样例：
1. 测试 0%, 5%, 10%, 15%, 20%, 25%, 30% 的重算比例
2. 记录每个比例下模型生成的答案和 F1/EM 分数
3. 分析 attention 分布特征
4. 找出规律：什么样的 attention 特征需要多高的重算比例才能答对
"""

import os
import sys
import json
import numpy as np
from test_fusionrag_reflect import main as fusionrag_main, PreprocessScope

# 要测试的比例列表
TEST_RATIOS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

# 要测试的样例（主问题索引）
TEST_EXAMPLES = [4, 5, 6, 7, 8]


def analyze_single_example_with_ratios(
    example_idx,
    model_path='/mnt/data/models/Qwen2.5-7B-Instruct',
    data_path='./result_reflect.json',
    cache_path='/mnt/data/reflect/',
    device='cuda:0',
    use_multi_gpu=True
):
    """
    对单个样例测试多个重算比例
    """
    print(f"\n{'='*100}")
    print(f"分析样例 {example_idx} 的重算比例 vs 答案变化")
    print(f"{'='*100}\n")

    results = []

    for ratio in TEST_RATIOS:
        print(f"\n{'─'*100}")
        print(f"测试比例: {ratio:.0%}")
        print(f"{'─'*100}\n")

        # 运行测试
        # 只测试这一个样例（max_samples = example_idx + 1）
        try:
            fusionrag_main(
                model_type='qwen',
                model_path=model_path,
                data_path=data_path,
                cache_path=cache_path,
                model_name='Qwen2.5-7B-Instruct',
                rate=ratio,
                reprocess_method='Oracle',  # 使用 Oracle 方法，固定比例
                preprocess=True,
                preprocess_scope=PreprocessScope.GLOBAL,
                bge_model_path='/mnt/data/models/bge-m3-FP16',
                revert_rope=True,
                device=device,
                use_multi_gpu=use_multi_gpu,
                openai_base_url="https://api.deepseek.com/v1",
                openai_api_key="sk-519d391217894b6e91e7c2ebf2a9f4df",
                openai_model="deepseek-chat",
                max_samples=example_idx + 1,  # 只测试到这个样例
                draft_layer_selection='entropy'
            )

            # 读取结果
            result_file = f'./result/Oracle_global_topk_10_rate_{ratio}.csv'
            if os.path.exists(result_file):
                import csv
                with open(result_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    # 找到对应的样例（最后一个主问题）
                    example_rows = [r for r in rows if int(r['Main Q']) == example_idx + 1]

                    results.append({
                        'ratio': ratio,
                        'sub_questions': example_rows
                    })

        except Exception as e:
            print(f"❌ 测试比例 {ratio:.0%} 时出错: {e}")
            continue

    # 分析结果
    print(f"\n{'='*100}")
    print(f"样例 {example_idx} 的分析结果")
    print(f"{'='*100}\n")

    # 保存详细结果
    output_file = f'./ratio_correlation_ex{example_idx}.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"详细结果已保存到: {output_file}")

    # 打印摘要
    print("\n重算比例 vs 答案质量:")
    print(f"{'Ratio':<10} {'Sub Q':<8} {'F1':<10} {'EM':<8} {'Correct':<10} Answer")
    print("─" * 100)

    for result in results:
        ratio = result['ratio']
        for sub_q in result['sub_questions']:
            print(f"{ratio:<10.0%} {sub_q['Sub Q']:<8} {float(sub_q['f1']):<10.4f} {float(sub_q['em']):<8.2f} {sub_q['correct']:<10} {sub_q['predicted'][:50]}")

    return results


def analyze_multiple_examples():
    """
    分析多个样例
    """
    print("="*100)
    print("重算比例 vs 答案变化 相关性分析")
    print("="*100)
    print(f"测试样例: {TEST_EXAMPLES}")
    print(f"测试比例: {[f'{r:.0%}' for r in TEST_RATIOS]}")
    print("="*100)

    all_results = {}

    for example_idx in TEST_EXAMPLES:
        try:
            results = analyze_single_example_with_ratios(example_idx)
            all_results[example_idx] = results
        except Exception as e:
            print(f"❌ 分析样例 {example_idx} 时出错: {e}")
            continue

    # 综合分析
    print(f"\n{'='*100}")
    print("综合分析：Attention 特征 vs 临界重算比例")
    print(f"{'='*100}\n")

    # 对每个样例，找出临界比例（从不正确变为正确的比例）
    critical_ratios = {}

    for example_idx, results in all_results.items():
        print(f"\n样例 {example_idx}:")

        for sub_q_idx in range(len(results[0]['sub_questions'])):
            # 提取这个子问题在不同比例下的表现
            sub_q_results = []
            for result in results:
                if sub_q_idx < len(result['sub_questions']):
                    sub_q = result['sub_questions'][sub_q_idx]
                    sub_q_results.append({
                        'ratio': result['ratio'],
                        'f1': float(sub_q['f1']),
                        'em': float(sub_q['em']),
                        'correct': sub_q['correct'].lower() == 'true',
                        'answer': sub_q['predicted']
                    })

            # 找出临界比例
            critical_ratio = None
            for i in range(len(sub_q_results) - 1):
                if not sub_q_results[i]['correct'] and sub_q_results[i+1]['correct']:
                    critical_ratio = sub_q_results[i+1]['ratio']
                    break

            if critical_ratio is not None:
                key = f"ex{example_idx}_sub{sub_q_idx}"
                critical_ratios[key] = critical_ratio
                print(f"  子问题 {sub_q_idx + 1}: 临界比例 = {critical_ratio:.0%}")
            else:
                # 检查是否一直都对或一直都错
                all_correct = all(r['correct'] for r in sub_q_results)
                all_wrong = all(not r['correct'] for r in sub_q_results)
                if all_correct:
                    print(f"  子问题 {sub_q_idx + 1}: 0% 就能答对")
                elif all_wrong:
                    print(f"  子问题 {sub_q_idx + 1}: 30% 仍无法答对")
                else:
                    print(f"  子问题 {sub_q_idx + 1}: 答案质量逐步提升，无明确临界点")

    # 保存综合结果
    summary_file = './ratio_correlation_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            'test_ratios': TEST_RATIOS,
            'test_examples': TEST_EXAMPLES,
            'critical_ratios': critical_ratios,
            'all_results': all_results
        }, f, indent=2, ensure_ascii=False)

    print(f"\n综合分析已保存到: {summary_file}")

    print("\n建议的下一步：")
    print("1. 对有临界比例的样例，使用 analyze_critical_ratio.py 分析其 attention 特征")
    print("2. 对比不同样例的 attention 特征（concentration, components, spread, gini）")
    print("3. 建立 attention 特征 → 临界比例 的映射关系")
    print("4. 设计基于实证的动态比例算法")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--example_idx', type=int, help='Single example to test')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--use_multi_gpu', action='store_true', default=True)

    args = parser.parse_args()

    if args.example_idx is not None:
        # 测试单个样例
        analyze_single_example_with_ratios(
            example_idx=args.example_idx,
            device=args.device,
            use_multi_gpu=args.use_multi_gpu
        )
    else:
        # 测试多个样例
        analyze_multiple_examples()
