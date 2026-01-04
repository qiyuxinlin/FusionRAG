#!/usr/bin/env python3
"""
Dynamic Recomputation Ratio Calculation

基于 draft model attention 分析的动态重算比例计算方法

核心思想：
1. 不使用单一的熵或累积覆盖率
2. 综合考虑多个因素：
   - Top-k coverage: 需要多少比例才能覆盖 80-90% attention
   - Spread factor: 高 attention 位置的分散程度
   - Component factor: 连通分量数量（信息片段数量）
   - Minimum safety buffer: 至少保证能捕获 long-tail 信息

目标：确保动态比例既不会过高浪费计算，也不会过低遗漏关键信息
"""

import numpy as np


def compute_dynamic_ratio_from_attention(
    draft_attention_scores,
    system_len,
    doc_len,
    query_len,
    base_ratio=0.3,
    min_ratio=0.20,  # 提高最小比例，确保安全边界
    max_ratio=0.50
):
    """
    根据 draft model 的 attention 分布动态计算重算比例

    策略说明：
    1. 计算达到不同覆盖率（80%, 85%, 90%）所需的 token 比例
    2. 分析 attention 的分散程度（连通分量）
    3. 计算 attention 的集中度（Gini 系数）
    4. 综合这些因素，确定一个合理的动态比例

    核心原则：
    - 如果 attention 高度集中 + 少量连通分量 → 可以降低比例（但不低于 min_ratio）
    - 如果 attention 分散 + 多个连通分量 → 需要提高比例
    - 始终确保有足够的 safety buffer 捕获 long-tail tokens

    Args:
        draft_attention_scores: {layer_idx: attention_matrix [num_heads, seq_len, seq_len]}
        system_len: system prompt 长度
        doc_len: 文档长度
        query_len: query 长度
        base_ratio: 参考基准比例
        min_ratio: 最小比例（默认 0.20，确保 long-tail）
        max_ratio: 最大比例

    Returns:
        dynamic_ratio: 计算得到的动态比例
        analysis: 详细分析结果字典
    """
    print(f"\n{'='*100}")
    print("Computing Dynamic Recomputation Ratio from Draft Model Attention")
    print(f"{'='*100}\n")

    # =========================================================================
    # Step 1: 提取 query→doc attention
    # =========================================================================
    total_len = system_len + doc_len + query_len
    doc_start = system_len
    doc_end = system_len + doc_len
    query_start = system_len + doc_len

    layer_entropy = {}
    layer_attention = {}

    for layer_idx in sorted(draft_attention_scores.keys()):
        layer_attn = draft_attention_scores[layer_idx]  # [num_heads, seq_len, seq_len]

        # 提取 query→doc attention
        query_to_doc = layer_attn[:, query_start:total_len, doc_start:doc_end]  # [num_heads, query_len, doc_len]

        # 对所有 heads 和 query positions 平均
        doc_attention_avg = query_to_doc.mean(axis=(0, 1))  # [doc_len]
        layer_attention[layer_idx] = doc_attention_avg

        # 计算熵
        p = doc_attention_avg / (doc_attention_avg.sum() + 1e-10)
        p = np.clip(p, 1e-10, 1.0)
        entropy = -np.sum(p * np.log(p))
        layer_entropy[layer_idx] = entropy

    # =========================================================================
    # Step 2: 选择低熵层（attention 更集中的层）
    # =========================================================================
    top_k = min(4, len(layer_entropy))
    sorted_layers = sorted(layer_entropy.items(), key=lambda x: x[1])
    layers_to_use = [layer_idx for layer_idx, _ in sorted_layers[:top_k]]

    print(f"Selected {top_k} layers with lowest entropy for analysis:")
    for i, (layer_idx, entropy) in enumerate(sorted_layers[:top_k]):
        print(f"  Layer {layer_idx}: entropy = {entropy:.4f}")

    # 聚合选中层的 attention
    multi_layer_attn = np.stack([layer_attention[i] for i in layers_to_use]).mean(axis=0)  # [doc_len]

    # =========================================================================
    # Step 3: 计算关键特征
    # =========================================================================

    # 特征 1: Coverage-based ratio
    # 计算达到不同覆盖率所需的 token 比例
    sorted_attn = np.sort(multi_layer_attn)[::-1]
    cumsum_attn = np.cumsum(sorted_attn) / (np.sum(multi_layer_attn) + 1e-10)

    coverage_thresholds = [0.80, 0.85, 0.90]
    coverage_ratios = {}

    for threshold in coverage_thresholds:
        tokens_needed = np.searchsorted(cumsum_attn, threshold) + 1
        ratio = tokens_needed / doc_len
        coverage_ratios[threshold] = ratio

    print(f"\nCoverage Analysis:")
    for threshold, ratio in coverage_ratios.items():
        print(f"  {threshold*100:.0f}% coverage requires: {ratio*100:.1f}% of tokens")

    # 使用 85% 覆盖率作为基准
    coverage_based_ratio = coverage_ratios[0.85]

    # 特征 2: 连通分量分析
    # 找到高 attention 位置并分析连通性
    mean_attn = np.mean(multi_layer_attn)
    std_attn = np.std(multi_layer_attn)
    threshold = mean_attn + 0.5 * std_attn

    high_attn_positions = list(np.where(multi_layer_attn > threshold)[0])
    components = find_connected_components(high_attn_positions, max_gap=2)
    num_components = len(components)

    # 归一化分量数量（假设 1-20 个分量）
    normalized_components = min(max(num_components - 1, 0) / 19.0, 1.0)

    print(f"\nConnected Components Analysis:")
    print(f"  High attention positions: {len(high_attn_positions)}")
    print(f"  Number of connected components: {num_components}")
    print(f"  Normalized component factor: {normalized_components:.4f}")

    # 特征 3: Spread factor
    # 衡量高 attention 位置在文档中的分散程度
    if len(high_attn_positions) > 0:
        positions_array = np.array(high_attn_positions)
        position_span = positions_array.max() - positions_array.min() if len(positions_array) > 1 else 0
        spread_ratio = position_span / doc_len
    else:
        spread_ratio = 0.0

    print(f"\nSpread Analysis:")
    print(f"  Position span: {position_span if len(high_attn_positions) > 0 else 0} tokens")
    print(f"  Spread ratio: {spread_ratio:.4f}")

    # 特征 4: Gini 系数（集中度）
    sorted_attn_gini = np.sort(multi_layer_attn)
    n = len(sorted_attn_gini)
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * sorted_attn_gini)) / (n * np.sum(sorted_attn_gini)) - (n + 1) / n
    # Gini: 0 (平等) → 1 (不平等/集中)

    print(f"\nConcentration Analysis:")
    print(f"  Gini coefficient: {gini:.4f} (higher = more concentrated)")

    # =========================================================================
    # Step 4: 综合计算动态比例
    # =========================================================================

    # 策略：
    # 1. 从 coverage_based_ratio 开始
    # 2. 根据连通分量数量调整：分量多 → 提高比例
    # 3. 根据分散程度调整：分散 → 提高比例
    # 4. 根据 Gini 系数调整：集中 → 可以降低比例
    # 5. 确保在 [min_ratio, max_ratio] 范围内

    # 基础比例：使用 85% coverage
    ratio = coverage_based_ratio

    # 调整 1: 连通分量因子
    # 如果分量多（信息分散），增加比例
    component_adjustment = normalized_components * 0.10  # 最多增加 10%
    ratio += component_adjustment

    # 调整 2: 分散度因子
    # 如果高 attention 位置很分散，增加比例
    spread_adjustment = spread_ratio * 0.08  # 最多增加 8%
    ratio += spread_adjustment

    # 调整 3: Gini 系数因子
    # 如果 Gini 很高（很集中），可以略微降低比例
    # 但这个调整应该保守，因为集中不代表可以忽略 long-tail
    if gini > 0.7:  # 高度集中
        gini_adjustment = -0.03
    elif gini < 0.5:  # 比较分散
        gini_adjustment = 0.03
    else:
        gini_adjustment = 0.0
    ratio += gini_adjustment

    # 调整 4: Safety buffer
    # 始终至少保证 min_ratio
    ratio = max(ratio, min_ratio)

    # 调整 5: 上限
    ratio = min(ratio, max_ratio)

    # 最终调整：四舍五入到 0.05 的倍数，便于理解和使用
    dynamic_ratio = round(ratio * 20) / 20  # 0.05 的倍数

    # =========================================================================
    # Step 5: 输出分析结果
    # =========================================================================

    print(f"\n{'='*100}")
    print("Dynamic Ratio Calculation Summary")
    print(f"{'='*100}\n")

    print(f"Factor Contributions:")
    print(f"  1. Coverage-based ratio (85%):        {coverage_based_ratio:.4f}")
    print(f"  2. Component adjustment:              +{component_adjustment:.4f}")
    print(f"  3. Spread adjustment:                 +{spread_adjustment:.4f}")
    print(f"  4. Gini adjustment:                   {gini_adjustment:+.4f}")
    print(f"  ---")
    print(f"  Raw computed ratio:                   {ratio:.4f}")
    print(f"  After range constraint [{min_ratio}, {max_ratio}]: {min(max(ratio, min_ratio), max_ratio):.4f}")
    print(f"  FINAL DYNAMIC RATIO (rounded):        {dynamic_ratio:.4f} ({dynamic_ratio*100:.0f}%)\n")

    print(f"Comparison with base ratio:")
    print(f"  Base ratio:    {base_ratio:.2%}")
    print(f"  Dynamic ratio: {dynamic_ratio:.2%}")
    if dynamic_ratio < base_ratio:
        print(f"  → Reduced by {(base_ratio - dynamic_ratio)*100:.1f}% (attention is concentrated)")
    elif dynamic_ratio > base_ratio:
        print(f"  → Increased by {(dynamic_ratio - base_ratio)*100:.1f}% (attention is spread/fragmented)")
    else:
        print(f"  → Same as base ratio")

    print(f"\n{'='*100}\n")

    # 保存详细分析结果
    analysis = {
        'doc_len': doc_len,
        'selected_layers': layers_to_use,
        'layer_entropies': {str(k): float(v) for k, v in layer_entropy.items()},
        'coverage_analysis': {
            f'{int(k*100)}%': float(v) for k, v in coverage_ratios.items()
        },
        'coverage_based_ratio': float(coverage_based_ratio),
        'num_components': num_components,
        'normalized_components': float(normalized_components),
        'spread_ratio': float(spread_ratio),
        'gini_coefficient': float(gini),
        'adjustments': {
            'component': float(component_adjustment),
            'spread': float(spread_adjustment),
            'gini': float(gini_adjustment),
        },
        'raw_ratio': float(ratio),
        'dynamic_ratio': float(dynamic_ratio),
        'base_ratio': float(base_ratio),
        'min_ratio': float(min_ratio),
        'max_ratio': float(max_ratio),
    }

    return dynamic_ratio, analysis


def find_connected_components(positions, max_gap=2):
    """
    找到位置列表中的连通分量（相邻 token 群组）

    Args:
        positions: token 位置列表
        max_gap: 最大允许间隔，小于等于此值认为连通

    Returns:
        连通分量列表
    """
    if not positions:
        return []

    positions = sorted(positions)
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


if __name__ == '__main__':
    # 测试代码
    print("Dynamic Ratio Calculation Module")
    print("This module is imported by per_head_generation.py")
