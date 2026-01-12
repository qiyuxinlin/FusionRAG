#!/usr/bin/env python3
"""
分析两个数据集的文本块数量和长度
"""

import json
import numpy as np
from collections import defaultdict

def analyze_dataset(filepath, dataset_name):
    """分析单个数据集"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"\n{'='*80}")
    print(f"数据集: {dataset_name}")
    print(f"文件: {filepath}")
    print(f"{'='*80}")

    # 统计信息
    num_main_questions = len(data)
    all_chunk_counts = []  # 每个子问题的文本块数量
    all_chunk_lengths = []  # 所有文本块的长度（字符数）
    all_chunk_token_lengths = []  # 估算的 token 数
    sub_question_counts = []  # 每个主问题的子问题数量

    for item in data:
        # 子问题在 intermediate_context 中
        sub_questions = item.get('intermediate_context', [])
        sub_question_counts.append(len(sub_questions))

        for sub_q_info in sub_questions:
            # 文本块在 'retrieve docs' 字段
            chunks = sub_q_info.get('retrieve docs', [])
            all_chunk_counts.append(len(chunks))

            for chunk in chunks:
                if isinstance(chunk, str):
                    chunk_text = chunk
                elif isinstance(chunk, dict):
                    chunk_text = chunk.get('text', chunk.get('content', chunk.get('paragraph', '')))
                else:
                    continue

                char_len = len(chunk_text)
                # 估算 token 数 (英文约 4 字符/token)
                token_len = len(chunk_text) // 4

                all_chunk_lengths.append(char_len)
                all_chunk_token_lengths.append(token_len)

    # 输出统计结果
    print(f"\n--- 基本信息 ---")
    print(f"主问题数量: {num_main_questions}")
    print(f"子问题总数: {len(all_chunk_counts)}")
    print(f"每个主问题的子问题数: {np.mean(sub_question_counts):.2f} ± {np.std(sub_question_counts):.2f}")

    print(f"\n--- 文本块数量（每个子问题）---")
    print(f"总文本块数: {sum(all_chunk_counts)}")
    print(f"平均: {np.mean(all_chunk_counts):.2f}")
    print(f"标准差: {np.std(all_chunk_counts):.2f}")
    print(f"最小: {np.min(all_chunk_counts)}")
    print(f"最大: {np.max(all_chunk_counts)}")
    print(f"中位数: {np.median(all_chunk_counts):.2f}")
    print(f"分位数 [25%, 50%, 75%, 90%]: {np.percentile(all_chunk_counts, [25, 50, 75, 90])}")

    if all_chunk_lengths:
        print(f"\n--- 文本块长度（字符数）---")
        print(f"平均: {np.mean(all_chunk_lengths):.2f}")
        print(f"标准差: {np.std(all_chunk_lengths):.2f}")
        print(f"最小: {np.min(all_chunk_lengths)}")
        print(f"最大: {np.max(all_chunk_lengths)}")
        print(f"中位数: {np.median(all_chunk_lengths):.2f}")
        print(f"分位数 [25%, 50%, 75%, 90%]: {np.percentile(all_chunk_lengths, [25, 50, 75, 90])}")

        print(f"\n--- 文本块长度（估算 token 数）---")
        print(f"平均: {np.mean(all_chunk_token_lengths):.2f}")
        print(f"中位数: {np.median(all_chunk_token_lengths):.2f}")

        # 文本块长度分布
        print(f"\n--- 文本块长度分布 ---")
        bins = [0, 100, 200, 500, 1000, 2000, 5000, 10000, float('inf')]
        bin_labels = ['0-100', '100-200', '200-500', '500-1k', '1k-2k', '2k-5k', '5k-10k', '>10k']
        hist, _ = np.histogram(all_chunk_lengths, bins=bins)
        for label, count in zip(bin_labels, hist):
            pct = count / len(all_chunk_lengths) * 100
            bar = '█' * int(pct / 2)
            print(f"  {label:>10}: {count:>5} ({pct:>5.1f}%) {bar}")

    return {
        'name': dataset_name,
        'num_main_questions': num_main_questions,
        'num_sub_questions': len(all_chunk_counts),
        'chunk_counts': all_chunk_counts,
        'chunk_lengths': all_chunk_lengths,
    }


def main():
    datasets = [
        ('/mnt/data/wjh/FusionRAG/data/2wikimqa_reflect.json', '2WikiMQA'),
        ('/mnt/data/wjh/FusionRAG/data/result_reflect.json', 'MuSiQue'),
    ]

    results = []
    for filepath, name in datasets:
        try:
            result = analyze_dataset(filepath, name)
            results.append(result)
        except Exception as e:
            print(f"Error analyzing {name}: {e}")
            import traceback
            traceback.print_exc()

    # 对比分析
    if len(results) == 2:
        print(f"\n{'='*80}")
        print("对比分析")
        print(f"{'='*80}")

        r1, r2 = results
        print(f"\n{'指标':<30} {r1['name']:<20} {r2['name']:<20} {'差异':<15}")
        print("-" * 85)

        # 子问题数
        print(f"{'子问题总数':<30} {r1['num_sub_questions']:<20} {r2['num_sub_questions']:<20}")

        # 文本块数量
        avg1 = np.mean(r1['chunk_counts'])
        avg2 = np.mean(r2['chunk_counts'])
        print(f"{'每个子问题文本块数(平均)':<30} {avg1:<20.2f} {avg2:<20.2f} {avg1/avg2:.2f}x")

        # 文本块长度
        if r1['chunk_lengths'] and r2['chunk_lengths']:
            len1 = np.mean(r1['chunk_lengths'])
            len2 = np.mean(r2['chunk_lengths'])
            print(f"{'文本块长度(平均字符)':<30} {len1:<20.2f} {len2:<20.2f} {len1/len2:.2f}x")

            med1 = np.median(r1['chunk_lengths'])
            med2 = np.median(r2['chunk_lengths'])
            print(f"{'文本块长度(中位数)':<30} {med1:<20.2f} {med2:<20.2f} {med1/med2:.2f}x")

        # 总 token 数估算
        total1 = sum(r1['chunk_lengths']) // 2
        total2 = sum(r2['chunk_lengths']) // 2
        print(f"{'总 token 数(估算)':<30} {total1:<20} {total2:<20} {total1/total2:.2f}x")

        print(f"\n--- 结论 ---")
        if avg1 > avg2 * 1.2:
            print(f"  - {r1['name']} 每个子问题的文本块数量更多 ({avg1:.1f} vs {avg2:.1f})")
        elif avg2 > avg1 * 1.2:
            print(f"  - {r2['name']} 每个子问题的文本块数量更多 ({avg2:.1f} vs {avg1:.1f})")

        if r1['chunk_lengths'] and r2['chunk_lengths']:
            if len1 < len2 * 0.8:
                print(f"  - {r1['name']} 的文本块更短 ({len1:.0f} vs {len2:.0f} 字符)")
            elif len2 < len1 * 0.8:
                print(f"  - {r2['name']} 的文本块更短 ({len2:.0f} vs {len1:.0f} 字符)")


if __name__ == '__main__':
    main()
