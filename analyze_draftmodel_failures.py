#!/usr/bin/env python3
"""
分析 DraftModel 失败但 rate=1 成功的子问题的特征
"""

import json
import pandas as pd
import numpy as np
from collections import defaultdict


def extract_sub_question(query_text):
    """从 'Intermediate query1: xxx' 格式中提取真正的问题"""
    if ':' in query_text:
        return query_text.split(':', 1)[1].strip()
    return query_text.strip()


def load_dataset(filepath):
    """加载数据集并构建索引"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 构建索引: (main_question, sub_question) -> chunk_info
    index = {}
    for item in data:
        main_q = item.get('question', '')
        for sub_q_info in item.get('intermediate_context', []):
            # 子问题在 'query' 字段，格式为 "Intermediate query1: xxx"
            raw_query = sub_q_info.get('query', sub_q_info.get('question', ''))
            sub_q = extract_sub_question(raw_query)
            chunks = sub_q_info.get('retrieve docs', [])

            # 提取文本块内容
            chunk_texts = []
            for chunk in chunks:
                if isinstance(chunk, str):
                    chunk_texts.append(chunk)
                elif isinstance(chunk, dict):
                    chunk_texts.append(chunk.get('text', chunk.get('content', chunk.get('paragraph', ''))))

            index[(main_q, sub_q)] = {
                'chunks': chunk_texts,
                'num_chunks': len(chunk_texts),
                'chunk_lengths': [len(t) for t in chunk_texts],
                'total_length': sum(len(t) for t in chunk_texts),
            }

    return index


def analyze_csv(csv_path, dataset_index, dataset_name):
    """分析 CSV 文件中的失败案例"""
    df = pd.read_csv(csv_path)

    print(f"\n{'='*80}")
    print(f"分析文件: {csv_path}")
    print(f"数据集: {dataset_name}")
    print(f"{'='*80}")

    # 转换 Correct 列
    df['Correct_bool'] = df['Correct'].astype(str).str.lower().isin(['true', 'yes', '1'])

    # 检查是否有 Rate1_Correct 列
    if 'Rate1_Correct' not in df.columns:
        print("警告: CSV 中没有 Rate1_Correct 列，无法进行对比分析")
        return None

    df['Rate1_Correct_bool'] = df['Rate1_Correct'].astype(str).str.lower().isin(['true', 'yes', '1'])

    # 筛选: DraftModel 错，rate=1 对
    failures = df[~df['Correct_bool'] & df['Rate1_Correct_bool']]
    # 筛选: DraftModel 对
    successes = df[df['Correct_bool']]

    print(f"\n总子问题数: {len(df)}")
    print(f"DraftModel 正确: {df['Correct_bool'].sum()} ({df['Correct_bool'].mean()*100:.1f}%)")
    print(f"Rate=1 正确: {df['Rate1_Correct_bool'].sum()} ({df['Rate1_Correct_bool'].mean()*100:.1f}%)")
    print(f"DraftModel 错但 Rate=1 对: {len(failures)} ({len(failures)/len(df)*100:.1f}%)")

    # 收集特征
    def collect_features(subset, name):
        features = {
            'num_chunks': [],
            'total_length': [],
            'avg_chunk_length': [],
            'max_chunk_length': [],
            'min_chunk_length': [],
        }
        matched = 0

        for _, row in subset.iterrows():
            main_q = row['Main Question']
            sub_q = row['Sub Question']

            # 查找匹配
            key = (main_q, sub_q)
            if key in dataset_index:
                info = dataset_index[key]
                matched += 1

                features['num_chunks'].append(info['num_chunks'])
                features['total_length'].append(info['total_length'])

                if info['chunk_lengths']:
                    features['avg_chunk_length'].append(np.mean(info['chunk_lengths']))
                    features['max_chunk_length'].append(np.max(info['chunk_lengths']))
                    features['min_chunk_length'].append(np.min(info['chunk_lengths']))

        print(f"\n--- {name} ({matched}/{len(subset)} 匹配到数据集) ---")

        if matched == 0:
            print("  无匹配数据")
            return None

        stats = {}
        for key, values in features.items():
            if values:
                stats[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'median': np.median(values),
                    'min': np.min(values),
                    'max': np.max(values),
                }
                print(f"  {key}:")
                print(f"    平均: {stats[key]['mean']:.1f} ± {stats[key]['std']:.1f}")
                print(f"    中位数: {stats[key]['median']:.1f}")
                print(f"    范围: [{stats[key]['min']:.0f}, {stats[key]['max']:.0f}]")

        return stats

    failure_stats = collect_features(failures, "DraftModel 失败案例")
    success_stats = collect_features(successes, "DraftModel 成功案例")

    # 对比分析
    if failure_stats and success_stats:
        print(f"\n--- 对比分析 (失败 vs 成功) ---")
        for key in failure_stats.keys():
            f_mean = failure_stats[key]['mean']
            s_mean = success_stats[key]['mean']
            diff_pct = (f_mean - s_mean) / s_mean * 100 if s_mean != 0 else 0
            direction = "↑" if diff_pct > 0 else "↓"
            print(f"  {key}: 失败={f_mean:.1f}, 成功={s_mean:.1f}, 差异={diff_pct:+.1f}% {direction}")

    # 打印失败案例详情
    print(f"\n--- 失败案例详情 (前 10 个) ---")
    for i, (_, row) in enumerate(failures.head(10).iterrows()):
        main_q = row['Main Question']
        sub_q = row['Sub Question']
        key = (main_q, sub_q)

        print(f"\n{i+1}. Sub-Q: {sub_q[:70]}...")
        print(f"   Ground Truth: {row['Ground Truth']}")
        print(f"   DraftModel 预测: {row['Predicted']}")
        print(f"   Rate=1 预测: {row.get('Rate1_Predicted', 'N/A')}")

        if key in dataset_index:
            info = dataset_index[key]
            print(f"   文本块数: {info['num_chunks']}")
            print(f"   总长度: {info['total_length']} 字符 (~{info['total_length']//4} tokens)")
            print(f"   块长度: {info['chunk_lengths'][:5]}{'...' if len(info['chunk_lengths']) > 5 else ''}")

    return {
        'failure_stats': failure_stats,
        'success_stats': success_stats,
        'num_failures': len(failures),
        'num_successes': len(successes),
    }


def main():
    # 加载数据集
    datasets = {
        '2wikimqa': '/mnt/data/wjh/FusionRAG/data/2wikimqa_reflect.json',
        'musique': '/mnt/data/wjh/FusionRAG/data/result_reflect.json',
    }

    # CSV 文件 (需要包含 Rate1_Correct 列的对比结果)
    csv_files = [
        ('/mnt/data/reflect/Qwen2.5-7B-Instruct/2wikimqa/results/DraftModel_global_topk_10_rate_0.3_draft_Qwen2.5-3B-Instruct.csv', '2wikimqa'),
        ('/mnt/data/reflect/Qwen2.5-7B-Instruct/musique/results/DraftModel_global_topk_10_rate_0.3_draft_Qwen2.5-3B-Instruct.csv', 'musique'),
    ]

    # 加载数据集索引
    dataset_indices = {}
    for name, path in datasets.items():
        try:
            print(f"加载数据集: {name}...")
            dataset_indices[name] = load_dataset(path)
            print(f"  索引大小: {len(dataset_indices[name])}")
        except Exception as e:
            print(f"  加载失败: {e}")

    # 分析每个 CSV
    results = []
    for csv_path, dataset_name in csv_files:
        try:
            if dataset_name in dataset_indices:
                result = analyze_csv(csv_path, dataset_indices[dataset_name], dataset_name)
                if result:
                    results.append((dataset_name, result))
        except Exception as e:
            print(f"分析 {csv_path} 时出错: {e}")
            import traceback
            traceback.print_exc()

    # 总结
    if results:
        print(f"\n{'='*80}")
        print("总结")
        print(f"{'='*80}")

        for dataset_name, result in results:
            if result['failure_stats'] and result['success_stats']:
                print(f"\n{dataset_name}:")
                print(f"  失败案例特征趋势:")

                f_total = result['failure_stats'].get('total_length', {}).get('mean', 0)
                s_total = result['success_stats'].get('total_length', {}).get('mean', 0)
                if f_total > s_total * 1.1:
                    print(f"    - 文本总长度更长 ({f_total:.0f} vs {s_total:.0f})")
                elif f_total < s_total * 0.9:
                    print(f"    - 文本总长度更短 ({f_total:.0f} vs {s_total:.0f})")

                f_chunks = result['failure_stats'].get('num_chunks', {}).get('mean', 0)
                s_chunks = result['success_stats'].get('num_chunks', {}).get('mean', 0)
                if f_chunks > s_chunks * 1.1:
                    print(f"    - 文本块数量更多 ({f_chunks:.1f} vs {s_chunks:.1f})")
                elif f_chunks < s_chunks * 0.9:
                    print(f"    - 文本块数量更少 ({f_chunks:.1f} vs {s_chunks:.1f})")


if __name__ == '__main__':
    main()
