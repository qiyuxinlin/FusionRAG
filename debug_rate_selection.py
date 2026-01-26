#!/usr/bin/env python3
"""
Debug script: 分析为什么 rate=0.99 性能反而差于 rate=0.8/0.9

核心问题：
- rate=0.8: Main Acc = 0.8333
- rate=0.9: Main Acc = 0.8333
- rate=0.99: Main Acc = 0.7759 (下降 6.89%)

理论上 rate 越高应该性能越好（更接近完全前向传播），
但实验结果显示 rate=0.99 反而更差。
"""

import numpy as np
import torch

def find_connected_components(positions, max_gap=2):
    """找到连通分量"""
    if not positions:
        return []

    positions = sorted(positions)
    components = []
    current = [positions[0]]

    for i in range(1, len(positions)):
        if positions[i] - positions[i-1] <= max_gap:
            current.append(positions[i])
        else:
            components.append(current)
            current = [positions[i]]
    components.append(current)

    return components


def smart_query_selection(attention_scores, doc_len, target_ratio, system_len=0,
                         device='cpu', threshold_factor=0.5, verbose=False):
    """
    复制原始的 smart_query_selection 逻辑，加入 debug 输出
    """
    if isinstance(attention_scores, torch.Tensor):
        attention_scores = attention_scores.float().cpu().numpy()

    target_count = int(doc_len * target_ratio)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Smart Query Selection Debug")
        print(f"{'='*60}")
        print(f"doc_len: {doc_len}")
        print(f"target_ratio: {target_ratio}")
        print(f"target_count: {target_count}")
        print(f"threshold_factor: {threshold_factor}")

    # Step 1: 找到高 attention 位置
    mean_attn = np.mean(attention_scores)
    std_attn = np.std(attention_scores)
    threshold = mean_attn + threshold_factor * std_attn

    high_attn_positions = list(np.where(attention_scores > threshold)[0])

    if verbose:
        print(f"\nStep 1: 高 attention 位置")
        print(f"  mean: {mean_attn:.6f}, std: {std_attn:.6f}")
        print(f"  threshold: {threshold:.6f}")
        print(f"  high_attn_positions: {len(high_attn_positions)} 个")

    # Step 2: 连通分量分析
    components = find_connected_components(high_attn_positions, max_gap=2)

    if verbose:
        print(f"\nStep 2: 连通分量分析")
        print(f"  找到 {len(components)} 个连通分量")
        for i, comp in enumerate(components[:5]):  # 只显示前5个
            print(f"    分量 {i}: {len(comp)} 个位置")

    # Step 3: 计算每个分量的总 attention
    component_scores = []
    for comp in components:
        total_score = sum(attention_scores[p] for p in comp)
        component_scores.append((comp, total_score))

    # Step 4: 按总 attention 排序
    component_scores.sort(key=lambda x: x[1], reverse=True)

    # Step 5: 贪心选择分量 + 上下文扩展 (±1)
    selected = set()

    for i, (comp, total_score) in enumerate(component_scores):
        # 扩展分量边界 (±1)
        extended_comp = set()
        for p in comp:
            for offset in range(-1, 2):
                new_p = p + offset
                if 0 <= new_p < doc_len:
                    extended_comp.add(new_p)

        # 检查是否会超过目标 (允许 10% 余量)
        new_positions = extended_comp - selected
        before_len = len(selected)

        if len(selected) + len(new_positions) <= target_count * 1.1:
            selected.update(extended_comp)
            if verbose and i < 5:
                print(f"  分量 {i}: 添加 {len(extended_comp)} 个位置 (扩展后), 累计: {before_len} -> {len(selected)}")
        else:
            if verbose and i < 5:
                print(f"  分量 {i}: 跳过 (会超过限制 {target_count * 1.1:.0f})")

    if verbose:
        print(f"\nStep 5 结束: 已选择 {len(selected)} 个位置 (目标: {target_count})")

    # Step 6: 补充到目标数量
    if len(selected) < target_count:
        sorted_indices = np.argsort(attention_scores)[::-1]
        added = 0
        for pos in sorted_indices:
            if pos not in selected:
                selected.add(int(pos))
                added += 1
                if len(selected) >= target_count:
                    break
        if verbose:
            print(f"\nStep 6: 补充了 {added} 个位置 -> {len(selected)}")

    # Step 7: 如果超过目标，移除最低分的位置
    removed = 0
    while len(selected) > target_count:
        min_pos = min(selected, key=lambda p: attention_scores[p])
        selected.remove(min_pos)
        removed += 1

    if verbose and removed > 0:
        print(f"\nStep 7: 移除了 {removed} 个低分位置 -> {len(selected)}")

    # 转换为全局索引
    selected_global = [p + system_len for p in sorted(selected)]

    if verbose:
        print(f"\n最终选择: {len(selected_global)} 个位置")
        print(f"{'='*60}\n")

    return selected_global, selected  # 返回局部索引用于分析


def analyze_selection_distribution(selected_local, doc_len):
    """分析选择的分布特性"""
    selected_arr = np.array(sorted(selected_local))

    # 计算覆盖率
    coverage = len(selected_local) / doc_len

    # 计算间隙
    if len(selected_arr) > 1:
        gaps = np.diff(selected_arr)
        avg_gap = np.mean(gaps)
        max_gap = np.max(gaps)
        gap_std = np.std(gaps)
    else:
        avg_gap = max_gap = gap_std = 0

    # 计算连续性
    continuous_segments = 0
    if len(selected_arr) > 0:
        current_start = selected_arr[0]
        for i in range(1, len(selected_arr)):
            if selected_arr[i] - selected_arr[i-1] > 1:
                continuous_segments += 1
        continuous_segments += 1  # 最后一段

    return {
        'coverage': coverage,
        'avg_gap': avg_gap,
        'max_gap': max_gap,
        'gap_std': gap_std,
        'n_segments': continuous_segments,
        'segment_density': len(selected_local) / max(continuous_segments, 1)
    }


def simulate_attention_pattern(doc_len, pattern_type='realistic'):
    """
    模拟不同类型的 attention 分布

    pattern_type:
        - 'realistic': 少数峰值 + 长尾
        - 'uniform': 均匀分布
        - 'sparse': 极度稀疏的峰值
    """
    if pattern_type == 'realistic':
        # 模拟真实的 attention: 少数位置高值，大部分位置低值
        attention = np.random.exponential(0.001, doc_len)
        # 添加几个峰值
        n_peaks = max(3, doc_len // 100)
        peak_positions = np.random.choice(doc_len, n_peaks, replace=False)
        for pos in peak_positions:
            # 峰值周围有邻域效应
            for offset in range(-5, 6):
                p = pos + offset
                if 0 <= p < doc_len:
                    attention[p] += 0.01 * np.exp(-abs(offset) / 2)

    elif pattern_type == 'uniform':
        attention = np.random.uniform(0.0005, 0.002, doc_len)

    elif pattern_type == 'sparse':
        attention = np.random.exponential(0.0005, doc_len)
        n_peaks = max(2, doc_len // 200)
        peak_positions = np.random.choice(doc_len, n_peaks, replace=False)
        for pos in peak_positions:
            attention[pos] += 0.05

    return attention


def main():
    """
    主分析函数：模拟不同 rate 下的 token 选择行为
    """
    print("="*80)
    print("分析 Rate=0.99 性能下降问题")
    print("="*80)

    # 模拟一个典型的文档长度 (2wikimqa 平均每个文档约 150-300 tokens)
    doc_len = 1000  # 假设有 5 个文档，每个 200 tokens

    print(f"\n模拟设置:")
    print(f"  文档总长度: {doc_len} tokens")
    print(f"  模拟 attention pattern: realistic (少数峰值 + 长尾)")

    # 模拟 attention 分布
    np.random.seed(42)
    attention_scores = simulate_attention_pattern(doc_len, 'realistic')

    print(f"\nAttention 统计:")
    print(f"  Mean: {np.mean(attention_scores):.6f}")
    print(f"  Std:  {np.std(attention_scores):.6f}")
    print(f"  Max:  {np.max(attention_scores):.6f}")
    print(f"  Min:  {np.min(attention_scores):.6f}")

    # 测试不同的 rate
    rates = [0.5, 0.8, 0.9, 0.95, 0.99]
    threshold_factor = 0.5  # 与配置文件一致

    print(f"\n{'='*80}")
    print(f"不同 Rate 下的选择行为分析 (threshold_factor={threshold_factor})")
    print(f"{'='*80}")

    results = []

    for rate in rates:
        print(f"\n\n{'#'*80}")
        print(f"# Rate = {rate}")
        print(f"{'#'*80}")

        selected_global, selected_local = smart_query_selection(
            attention_scores,
            doc_len,
            rate,
            system_len=0,
            threshold_factor=threshold_factor,
            verbose=True
        )

        # 分析选择的分布
        dist = analyze_selection_distribution(selected_local, doc_len)

        print(f"\n分布分析:")
        print(f"  覆盖率: {dist['coverage']*100:.2f}%")
        print(f"  平均间隙: {dist['avg_gap']:.2f} tokens")
        print(f"  最大间隙: {dist['max_gap']:.0f} tokens")
        print(f"  间隙标准差: {dist['gap_std']:.2f}")
        print(f"  连续段数: {dist['n_segments']}")
        print(f"  段密度: {dist['segment_density']:.2f} tokens/段")

        results.append({
            'rate': rate,
            'n_selected': len(selected_local),
            **dist
        })

    # 总结对比
    print(f"\n\n{'='*80}")
    print("总结：Rate 对比")
    print(f"{'='*80}")
    print(f"{'Rate':<8} {'Selected':<10} {'Coverage':<12} {'Avg Gap':<12} {'Max Gap':<12} {'Segments':<10}")
    print("-"*80)

    for r in results:
        print(f"{r['rate']:<8.2f} {r['n_selected']:<10} {r['coverage']*100:<11.2f}% "
              f"{r['avg_gap']:<12.2f} {r['max_gap']:<12.0f} {r['n_segments']:<10}")

    print("\n" + "="*80)
    print("关键发现:")
    print("="*80)

    # 分析 rate=0.99 vs rate=0.9
    r99 = results[-1]
    r90 = results[-2]

    print(f"\nRate 0.90 vs 0.99 对比:")
    print(f"  选择数量差异: {r99['n_selected'] - r90['n_selected']} tokens")
    print(f"  间隙变化: {r99['avg_gap']:.2f} vs {r90['avg_gap']:.2f}")
    print(f"  段数变化: {r99['n_segments']} vs {r90['n_segments']}")

    print("\n可能的问题:")
    print("  1. 连通分量算法的边界效应:")
    print("     - 当 rate 接近 1 时，Step 5 的 10% 余量限制可能导致")
    print("       某些重要连通分量被跳过")
    print("     - Step 6 的补充策略只按单个 token 的 attention 值排序，")
    print("       可能破坏了连通性和上下文完整性")
    print()
    print("  2. 阈值固定导致的问题:")
    print("     - threshold_factor=0.5 对所有 rate 都相同")
    print("     - 高 rate 时应该降低阈值，包含更多候选位置")
    print("     - 但现在阈值固定，导致 Step 5 选不够，Step 6 随机补充")
    print()
    print("  3. 10% 余量的反作用:")
    print(f"     - Rate=0.90: 目标={doc_len*0.90:.0f}, 上限={doc_len*0.90*1.1:.0f}")
    print(f"     - Rate=0.99: 目标={doc_len*0.99:.0f}, 上限={doc_len*0.99*1.1:.0f}")
    print("     - 余量空间: 0.90 有 90 tokens, 0.99 只有 9.9 tokens")
    print("     - 连通分量扩展后更容易超过 0.99 的上限！")

    print("\n" + "="*80)
    print("建议的修复方案:")
    print("="*80)
    print("1. 动态调整 threshold_factor:")
    print("   - 低 rate (< 0.5): threshold_factor = 0.5")
    print("   - 中 rate (0.5-0.8): threshold_factor = 0.3")
    print("   - 高 rate (> 0.8): threshold_factor = 0.1")
    print()
    print("2. 动态调整余量:")
    print("   - 低 rate: 允许 10% 余量")
    print("   - 高 rate (> 0.9): 允许 5% 余量或固定余量 (如 50 tokens)")
    print()
    print("3. 改进 Step 6 的补充策略:")
    print("   - 优先补充已选位置的邻居 (±2 范围内)")
    print("   - 然后按 attention 值排序补充")
    print()
    print("4. 考虑直接 topk 策略:")
    print("   - 当 rate > 0.95 时，直接使用 attention topk 选择")
    print("   - 不使用连通分量分析，避免边界效应")


if __name__ == '__main__':
    main()
