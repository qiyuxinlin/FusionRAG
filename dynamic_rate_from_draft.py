#!/usr/bin/env python3
"""
基于 DraftModel attention 分布动态计算重算比例。

设计思路：
1. 大多数问题（~82%）在低rate下就能正确回答
2. 困难问题的特点是attention分布分散，信息分布在多个位置
3. 通过分析attention特征来预测所需的重算比例

核心特征：
- attention_entropy: 越高表示attention越分散，需要更高rate
- coverage_85_ratio: 达到85%覆盖率需要的token比例，越高需要更高rate
- num_components: 连通分量数量，越多表示信息分散
- gini_coefficient: 越低表示attention越分散
"""

import numpy as np
from typing import Dict, List, Tuple


def compute_dynamic_rate_from_draft_attention(
    draft_attention_scores: Dict,
    system_len: int,
    doc_len: int,
    query_len: int,
    min_rate: float = 0.05,
    max_rate: float = 0.30,
) -> Tuple[float, Dict]:
    """
    根据 DraftModel 的 attention 分布动态计算最优重算比例。

    Args:
        draft_attention_scores: {layer_idx: attention_matrix [num_heads, seq_len, seq_len]}
        system_len: system prompt 长度
        doc_len: 文档长度
        query_len: query 长度
        min_rate: 最小重算比例（默认5%）
        max_rate: 最大重算比例（默认30%）

    Returns:
        rate: 推荐的重算比例
        features: 计算过程中的特征信息
    """
    total_len = system_len + doc_len + query_len
    doc_start = system_len
    doc_end = system_len + doc_len
    query_start = system_len + doc_len

    # =========================================================================
    # Step 1: 提取各层的 query→doc attention 并计算特征
    # =========================================================================
    all_layer_attention = []
    layer_entropies = []

    for layer_idx in sorted(draft_attention_scores.keys()):
        layer_attn = draft_attention_scores[layer_idx]

        # 提取 query→doc attention
        query_to_doc = layer_attn[:, query_start:total_len, doc_start:doc_end]
        doc_attention_avg = query_to_doc.mean(axis=(0, 1))
        all_layer_attention.append(doc_attention_avg)

        # 计算层熵
        p = doc_attention_avg / (doc_attention_avg.sum() + 1e-10)
        p = np.clip(p, 1e-10, 1.0)
        entropy = -np.sum(p * np.log(p))
        layer_entropies.append(entropy)

    # 聚合attention（使用熵最低的几层，attention更集中的层更可靠）
    sorted_layer_indices = np.argsort(layer_entropies)[:4]  # 选择熵最低的4层
    selected_attentions = [all_layer_attention[i] for i in sorted_layer_indices]
    aggregated_attn = np.mean(selected_attentions, axis=0)

    # =========================================================================
    # Step 2: 计算关键特征
    # =========================================================================
    features = {}

    # Feature 1: 聚合attention的熵
    p_agg = aggregated_attn / (aggregated_attn.sum() + 1e-10)
    p_agg = np.clip(p_agg, 1e-10, 1.0)
    features['attention_entropy'] = -np.sum(p_agg * np.log(p_agg))
    features['max_entropy'] = np.log(doc_len)  # 均匀分布的熵（最大值）
    features['normalized_entropy'] = features['attention_entropy'] / features['max_entropy']

    # Feature 2: Coverage ratios
    sorted_indices = np.argsort(aggregated_attn)[::-1]
    sorted_attn = aggregated_attn[sorted_indices]
    cumsum = np.cumsum(sorted_attn) / (sorted_attn.sum() + 1e-10)

    for coverage in [0.5, 0.7, 0.85, 0.9, 0.95]:
        count = np.searchsorted(cumsum, coverage) + 1
        features[f'coverage_{int(coverage*100)}_ratio'] = count / doc_len

    # Feature 3: Gini coefficient
    sorted_attn_asc = np.sort(aggregated_attn)
    n = len(sorted_attn_asc)
    index = np.arange(1, n + 1)
    gini = ((2 * index - n - 1) * sorted_attn_asc).sum() / (n * sorted_attn_asc.sum() + 1e-10)
    features['gini_coefficient'] = gini

    # Feature 4: 高attention位置分布
    mean_attn = np.mean(aggregated_attn)
    std_attn = np.std(aggregated_attn)

    high_positions = np.where(aggregated_attn > mean_attn + std_attn)[0]
    features['high_attn_ratio'] = len(high_positions) / doc_len

    # 连通分量分析
    threshold_positions = list(np.where(aggregated_attn > mean_attn + 0.5 * std_attn)[0])
    components = find_connected_components(threshold_positions, max_gap=2)
    features['num_components'] = len(components)

    if len(high_positions) > 1:
        features['high_attn_span'] = (high_positions.max() - high_positions.min()) / doc_len
    else:
        features['high_attn_span'] = 0

    # =========================================================================
    # Step 3: 动态计算rate
    # =========================================================================
    # 基础rate
    base_rate = min_rate

    # 根据特征增加rate
    rate_adjustments = []

    # 调整1: 基于熵（attention分散程度）
    # 归一化熵 > 0.6 时增加rate
    if features['normalized_entropy'] > 0.6:
        entropy_adjustment = (features['normalized_entropy'] - 0.6) * 0.3
        rate_adjustments.append(('entropy', entropy_adjustment))

    # 调整2: 基于覆盖率
    # 如果达到85%覆盖需要超过15%的token，增加rate
    if features['coverage_85_ratio'] > 0.15:
        coverage_adjustment = (features['coverage_85_ratio'] - 0.15) * 0.5
        rate_adjustments.append(('coverage', coverage_adjustment))

    # 调整3: 基于Gini系数
    # Gini < 0.7 表示attention不够集中，需要更高rate
    if features['gini_coefficient'] < 0.7:
        gini_adjustment = (0.7 - features['gini_coefficient']) * 0.2
        rate_adjustments.append(('gini', gini_adjustment))

    # 调整4: 基于连通分量数量
    # 超过5个连通分量表示信息分散
    if features['num_components'] > 5:
        component_adjustment = min((features['num_components'] - 5) * 0.02, 0.1)
        rate_adjustments.append(('components', component_adjustment))

    # 调整5: 基于高attention位置跨度
    # 跨度超过50%表示信息分布在文档的多个区域
    if features['high_attn_span'] > 0.5:
        span_adjustment = (features['high_attn_span'] - 0.5) * 0.15
        rate_adjustments.append(('span', span_adjustment))

    # 计算最终rate
    total_adjustment = sum(adj for _, adj in rate_adjustments)
    final_rate = min(max(base_rate + total_adjustment, min_rate), max_rate)

    features['base_rate'] = base_rate
    features['total_adjustment'] = total_adjustment
    features['rate_adjustments'] = rate_adjustments
    features['final_rate'] = final_rate

    return final_rate, features


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


def predict_rate_simple(features: Dict) -> float:
    """
    简化版的rate预测（用于调试）。

    基于经验规则：
    - 如果attention高度集中（gini > 0.8, 覆盖率低），使用低rate
    - 如果attention分散（gini < 0.6, 覆盖率高），使用高rate
    """
    # 简单的线性组合
    score = 0.0

    # 覆盖率权重最高
    score += features['coverage_85_ratio'] * 0.4

    # Gini系数（反向，越低需要越高rate）
    score += (1 - features['gini_coefficient']) * 0.3

    # 归一化熵
    score += features['normalized_entropy'] * 0.2

    # 连通分量（归一化到0-1）
    score += min(features['num_components'] / 20, 1) * 0.1

    # 映射到0.05-0.30范围
    rate = 0.05 + score * 0.25

    return min(max(rate, 0.05), 0.30)


if __name__ == "__main__":
    # 测试示例
    print("Dynamic Rate from Draft Model Attention")
    print("="*60)

    # 模拟一些特征值来测试算法
    test_cases = [
        {"name": "Easy question (concentrated attention)",
         "normalized_entropy": 0.4, "coverage_85_ratio": 0.08,
         "gini_coefficient": 0.85, "num_components": 2, "high_attn_span": 0.2},
        {"name": "Medium question",
         "normalized_entropy": 0.6, "coverage_85_ratio": 0.15,
         "gini_coefficient": 0.7, "num_components": 5, "high_attn_span": 0.4},
        {"name": "Hard question (scattered attention)",
         "normalized_entropy": 0.8, "coverage_85_ratio": 0.25,
         "gini_coefficient": 0.5, "num_components": 10, "high_attn_span": 0.7},
    ]

    for case in test_cases:
        rate = predict_rate_simple(case)
        print(f"\n{case['name']}:")
        print(f"  entropy={case['normalized_entropy']:.2f}, coverage85={case['coverage_85_ratio']:.2f}")
        print(f"  gini={case['gini_coefficient']:.2f}, components={case['num_components']}")
        print(f"  -> Predicted rate: {rate:.3f}")
