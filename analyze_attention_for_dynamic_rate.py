#!/usr/bin/env python3
"""
分析 DraftModel attention 特征，用于设计动态重算比例算法。

核心思路：
1. 收集困难问题（rate=0.3错，rate=1对）和简单问题（rate=0.3就对）的 attention 特征
2. 找出能区分两类问题的特征
3. 设计基于这些特征的动态 rate 预测算法
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple

project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

RESULTS_DIR = "/mnt/data/reflect/Qwen2.5-7B-Instruct/results"


def load_hard_and_easy_questions():
    """
    加载困难问题和简单问题列表。
    困难问题：rate=0.3错，rate=1对
    简单问题：rate=0.3和rate=1都对
    """
    draft_03 = pd.read_csv(os.path.join(RESULTS_DIR, "DraftModel_global_topk_10_rate_0.3.csv"))

    # 转换为布尔值
    draft_03['Correct'] = draft_03['Correct'].apply(lambda x: x == True or x == 'True')
    draft_03['Rate1_Correct'] = draft_03['Rate1_Correct'].apply(lambda x: x == True or x == 'True')

    # 分类问题
    hard_questions = draft_03[(draft_03['Correct'] == False) & (draft_03['Rate1_Correct'] == True)]
    easy_questions = draft_03[(draft_03['Correct'] == True) & (draft_03['Rate1_Correct'] == True)]

    # 提取关键信息
    hard_list = []
    for idx, row in hard_questions.iterrows():
        hard_list.append({
            'idx': idx,
            'main_question': row['Main Question'],
            'sub_question': row['Sub Question'],
            'ground_truth': row['Ground Truth'],
            'predicted_03': row['Predicted'],
            'predicted_1': row['Rate1_Predicted'],
        })

    easy_list = []
    for idx, row in easy_questions.iterrows():
        easy_list.append({
            'idx': idx,
            'main_question': row['Main Question'],
            'sub_question': row['Sub Question'],
        })

    print(f"Hard questions: {len(hard_list)}")
    print(f"Easy questions: {len(easy_list)}")

    return hard_list, easy_list


def compute_attention_features(attention_scores: Dict, system_len: int, doc_len: int, query_len: int) -> Dict:
    """
    从 attention 分布中提取特征，用于预测所需的重算比例。

    Args:
        attention_scores: {layer_idx: [num_heads, seq_len, seq_len]}
        system_len: system prompt 长度
        doc_len: 文档长度
        query_len: query 长度

    Returns:
        特征字典
    """
    features = {}

    total_len = system_len + doc_len + query_len
    doc_start = system_len
    doc_end = system_len + doc_len
    query_start = system_len + doc_len

    # 收集所有层的 query→doc attention
    all_layer_attention = []
    layer_entropies = []

    for layer_idx in sorted(attention_scores.keys()):
        layer_attn = attention_scores[layer_idx]  # [num_heads, seq_len, seq_len]

        # 提取 query→doc attention
        query_to_doc = layer_attn[:, query_start:total_len, doc_start:doc_end]  # [num_heads, query_len, doc_len]

        # 对所有 heads 和 query positions 平均
        doc_attention_avg = query_to_doc.mean(axis=(0, 1))  # [doc_len]
        all_layer_attention.append(doc_attention_avg)

        # 计算熵
        p = doc_attention_avg / (doc_attention_avg.sum() + 1e-10)
        p = np.clip(p, 1e-10, 1.0)
        entropy = -np.sum(p * np.log(p))
        layer_entropies.append(entropy)

    # 聚合 attention（平均所有层）
    aggregated_attn = np.mean(all_layer_attention, axis=0)

    # =========================================================================
    # Feature 1: Attention 熵（越高表示 attention 越分散，可能需要更高 rate）
    # =========================================================================
    p_agg = aggregated_attn / (aggregated_attn.sum() + 1e-10)
    p_agg = np.clip(p_agg, 1e-10, 1.0)
    features['attention_entropy'] = -np.sum(p_agg * np.log(p_agg))
    features['min_layer_entropy'] = min(layer_entropies)
    features['max_layer_entropy'] = max(layer_entropies)
    features['entropy_range'] = features['max_layer_entropy'] - features['min_layer_entropy']

    # =========================================================================
    # Feature 2: Coverage ratio（达到 X% attention 需要的 token 比例）
    # =========================================================================
    sorted_indices = np.argsort(aggregated_attn)[::-1]
    sorted_attn = aggregated_attn[sorted_indices]
    cumsum = np.cumsum(sorted_attn) / sorted_attn.sum()

    # 达到不同覆盖率需要的 token 数
    for coverage in [0.5, 0.7, 0.85, 0.9, 0.95]:
        count = np.searchsorted(cumsum, coverage) + 1
        features[f'coverage_{int(coverage*100)}_ratio'] = count / doc_len

    # =========================================================================
    # Feature 3: Attention 集中度（Gini 系数）
    # =========================================================================
    sorted_attn_asc = np.sort(aggregated_attn)
    n = len(sorted_attn_asc)
    index = np.arange(1, n + 1)
    gini = ((2 * index - n - 1) * sorted_attn_asc).sum() / (n * sorted_attn_asc.sum() + 1e-10)
    features['gini_coefficient'] = gini

    # =========================================================================
    # Feature 4: 高 attention 位置的分布特征
    # =========================================================================
    mean_attn = np.mean(aggregated_attn)
    std_attn = np.std(aggregated_attn)

    # 高 attention 位置 (> mean + 1*std)
    high_positions = np.where(aggregated_attn > mean_attn + std_attn)[0]
    features['high_attn_count'] = len(high_positions)
    features['high_attn_ratio'] = len(high_positions) / doc_len

    if len(high_positions) > 1:
        # 高 attention 位置的跨度
        features['high_attn_span'] = (high_positions.max() - high_positions.min()) / doc_len
        # 高 attention 位置的间隔
        gaps = np.diff(sorted(high_positions))
        features['high_attn_max_gap'] = gaps.max() if len(gaps) > 0 else 0
        features['high_attn_mean_gap'] = gaps.mean() if len(gaps) > 0 else 0
    else:
        features['high_attn_span'] = 0
        features['high_attn_max_gap'] = 0
        features['high_attn_mean_gap'] = 0

    # =========================================================================
    # Feature 5: 连通分量分析
    # =========================================================================
    threshold_positions = list(np.where(aggregated_attn > mean_attn + 0.5 * std_attn)[0])
    components = find_connected_components(threshold_positions, max_gap=2)
    features['num_components'] = len(components)
    if components:
        component_sizes = [len(c) for c in components]
        features['max_component_size'] = max(component_sizes)
        features['mean_component_size'] = np.mean(component_sizes)
    else:
        features['max_component_size'] = 0
        features['mean_component_size'] = 0

    # =========================================================================
    # Feature 6: 统计特征
    # =========================================================================
    features['attention_mean'] = np.mean(aggregated_attn)
    features['attention_std'] = np.std(aggregated_attn)
    features['attention_max'] = np.max(aggregated_attn)
    features['attention_skewness'] = ((aggregated_attn - mean_attn) ** 3).mean() / (std_attn ** 3 + 1e-10)
    features['attention_kurtosis'] = ((aggregated_attn - mean_attn) ** 4).mean() / (std_attn ** 4 + 1e-10) - 3

    return features


def find_connected_components(positions: List[int], max_gap: int = 2) -> List[List[int]]:
    """找到连通分量"""
    if not positions:
        return []

    positions = sorted(set(positions))
    components = []
    current_component = [positions[0]]

    for i in range(1, len(positions)):
        if positions[i] - positions[i-1] <= max_gap:
            current_component.append(positions[i])
        else:
            components.append(current_component)
            current_component = [positions[i]]

    components.append(current_component)
    return components


def analyze_features_difference(hard_features: List[Dict], easy_features: List[Dict]):
    """
    分析困难问题和简单问题的特征差异。
    """
    print("\n" + "="*80)
    print("Feature Analysis: Hard vs Easy Questions")
    print("="*80)

    if not hard_features or not easy_features:
        print("Not enough data for analysis")
        return

    # 收集所有特征名
    all_feature_names = set(hard_features[0].keys())

    # 计算每个特征的统计量
    results = []
    for feature_name in sorted(all_feature_names):
        hard_values = [f[feature_name] for f in hard_features]
        easy_values = [f[feature_name] for f in easy_features]

        hard_mean = np.mean(hard_values)
        easy_mean = np.mean(easy_values)
        diff = hard_mean - easy_mean
        diff_pct = (hard_mean / (easy_mean + 1e-10) - 1) * 100

        results.append({
            'feature': feature_name,
            'hard_mean': hard_mean,
            'easy_mean': easy_mean,
            'diff': diff,
            'diff_pct': diff_pct,
            'hard_std': np.std(hard_values),
            'easy_std': np.std(easy_values),
        })

    # 按差异百分比排序
    results.sort(key=lambda x: abs(x['diff_pct']), reverse=True)

    print(f"\n{'Feature':<25} {'Hard Mean':<12} {'Easy Mean':<12} {'Diff %':<10} {'Hard Std':<10}")
    print("-" * 80)
    for r in results:
        print(f"{r['feature']:<25} {r['hard_mean']:<12.4f} {r['easy_mean']:<12.4f} "
              f"{r['diff_pct']:>+8.1f}% {r['hard_std']:<10.4f}")

    # 找出最具区分性的特征
    print("\n" + "="*80)
    print("Most Discriminative Features (for dynamic rate prediction):")
    print("="*80)

    discriminative = [r for r in results if abs(r['diff_pct']) > 10]
    for i, r in enumerate(discriminative[:10]):
        direction = "higher" if r['diff'] > 0 else "lower"
        print(f"{i+1}. {r['feature']}: Hard questions have {direction} values "
              f"({r['hard_mean']:.4f} vs {r['easy_mean']:.4f}, {r['diff_pct']:+.1f}%)")

    return results


def main():
    """
    主函数：收集 attention 特征并分析。
    """
    hard_list, easy_list = load_hard_and_easy_questions()

    print(f"\nHard questions details:")
    for q in hard_list:
        print(f"  [{q['idx']}] {q['sub_question'][:60]}...")

    # 保存问题列表供后续使用
    with open("hard_questions.json", "w") as f:
        json.dump(hard_list, f, indent=2, ensure_ascii=False)

    print("\n" + "="*80)
    print("Next Steps:")
    print("="*80)
    print("1. Run DraftModel prefill on these questions to collect attention features")
    print("2. Compare features between hard and easy questions")
    print("3. Design dynamic rate prediction algorithm based on discriminative features")


if __name__ == "__main__":
    main()
