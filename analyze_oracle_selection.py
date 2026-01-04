#!/usr/bin/env python3
"""
Oracle Token Selection Analysis

分析 Oracle + 固定30% vs Oracle + 动态选择 的算法差异
基于理论分析，找出动态选择效果不好的原因，并提出改进方案
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os

def analyze_cumulative_coverage(attention_scores, epsilon=0.1):
    """
    分析累积覆盖率方法的问题
    """
    scores = np.array(attention_scores)
    scores = scores / scores.sum()  # 归一化

    # 排序
    sorted_indices = np.argsort(scores)[::-1]
    sorted_scores = scores[sorted_indices]
    cumsum = np.cumsum(sorted_scores)

    # 找到覆盖 (1-ε) 需要的 token 数量
    coverage_threshold = 1.0 - epsilon
    tokens_needed = np.searchsorted(cumsum, coverage_threshold) + 1

    return {
        'tokens_needed': tokens_needed,
        'ratio_needed': tokens_needed / len(scores),
        'actual_coverage': cumsum[tokens_needed - 1] if tokens_needed <= len(scores) else 1.0,
        'top_indices': sorted_indices[:tokens_needed].tolist(),
        'missed_indices': sorted_indices[tokens_needed:].tolist(),
        'missed_weight': 1.0 - cumsum[tokens_needed - 1] if tokens_needed <= len(scores) else 0.0
    }


def simulate_attention_distributions():
    """
    模拟不同类型的 attention 分布，分析动态选择的表现
    """
    np.random.seed(42)
    doc_len = 1000

    distributions = {}

    # 1. 高度集中分布 (少数 token 占大部分权重)
    scores = np.random.exponential(scale=0.1, size=doc_len)
    # 人为增加几个 token 的权重
    scores[100:105] += 10
    scores[500:503] += 5
    distributions['concentrated'] = scores / scores.sum()

    # 2. 中等集中分布
    scores = np.random.exponential(scale=0.5, size=doc_len)
    scores[100:120] += 2
    scores[400:420] += 2
    scores[700:720] += 2
    distributions['moderate'] = scores / scores.sum()

    # 3. 分散分布 (接近均匀)
    scores = np.random.uniform(0.8, 1.2, size=doc_len)
    distributions['spread'] = scores / scores.sum()

    # 4. 多峰分布 (多个重要区域)
    scores = np.random.exponential(scale=0.1, size=doc_len)
    for peak_center in [100, 300, 500, 700, 900]:
        for offset in range(-10, 11):
            idx = peak_center + offset
            if 0 <= idx < doc_len:
                scores[idx] += 3 * np.exp(-abs(offset) / 3)
    distributions['multi_peak'] = scores / scores.sum()

    return distributions


def compare_selection_methods(attention_scores, fixed_rate=0.3, epsilon=0.1, min_rate=0.05, max_rate=0.5):
    """
    比较固定比例和动态选择方法
    """
    scores = np.array(attention_scores)
    doc_len = len(scores)
    scores = scores / scores.sum()

    # 方法1: 固定比例
    fixed_budget = int(fixed_rate * doc_len)
    sorted_indices = np.argsort(scores)[::-1]
    fixed_selected = set(sorted_indices[:fixed_budget].tolist())
    fixed_coverage = sum(scores[i] for i in fixed_selected)

    # 方法2: 动态选择 (基于累积覆盖率)
    sorted_scores = scores[sorted_indices]
    cumsum = np.cumsum(sorted_scores)
    coverage_threshold = 1.0 - epsilon

    tokens_needed = np.searchsorted(cumsum, coverage_threshold) + 1
    dynamic_rate = tokens_needed / doc_len
    dynamic_rate = max(min_rate, min(max_rate, dynamic_rate))

    dynamic_budget = int(dynamic_rate * doc_len)
    dynamic_selected = set(sorted_indices[:dynamic_budget].tolist())
    dynamic_coverage = sum(scores[i] for i in dynamic_selected)

    return {
        'fixed_rate': fixed_rate,
        'fixed_budget': fixed_budget,
        'fixed_coverage': fixed_coverage,
        'dynamic_rate': dynamic_rate,
        'dynamic_budget': dynamic_budget,
        'dynamic_coverage': dynamic_coverage,
        'only_in_fixed': len(fixed_selected - dynamic_selected),
        'only_in_dynamic': len(dynamic_selected - fixed_selected),
        'overlap': len(fixed_selected & dynamic_selected),
    }


def analyze_why_dynamic_fails():
    """
    分析动态选择失败的核心原因
    """
    print("="*80)
    print("Why Dynamic Selection Might Fail: Theoretical Analysis")
    print("="*80)

    print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│                     The Core Problem with Cumulative Coverage                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Assumption: "High attention = Important information"                       │
│                                                                             │
│  Reality: Attention only measures how much the model "looks at" tokens      │
│           when processing the query, NOT the importance of information      │
│                                                                             │
│  Example Scenario:                                                          │
│  ────────────────                                                           │
│  Question: "In what year was Company X founded?"                            │
│  Document: "Company X, founded by John Smith in 1985, is a leading..."     │
│                                                                             │
│  Attention pattern:                                                         │
│    "Company X"    → Very high attention (model focuses on entity)           │
│    "founded by"   → Medium attention                                        │
│    "John Smith"   → Medium attention                                        │
│    "1985"        → LOW attention (but THIS is the answer!)                  │
│    "leading"      → Low attention                                           │
│                                                                             │
│  If we select based on cumulative coverage:                                 │
│    - 90% coverage might only need tokens around "Company X"                 │
│    - The actual answer "1985" gets dropped!                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
""")

    print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Why Fixed 30% Works Better                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Fixed 30% acts as a "safety buffer":                                       │
│                                                                             │
│  1. Captures Long-Tail Tokens                                               │
│     - Even if a token has low attention, if it's in top 30%,               │
│       it gets selected                                                      │
│     - These long-tail tokens often contain critical facts                   │
│                                                                             │
│  2. More Robust to Distribution Variations                                  │
│     - Dynamic selection can give 5% for concentrated attention             │
│     - Fixed 30% always gives 30%, regardless of distribution               │
│                                                                             │
│  3. Covers More Context                                                     │
│     - RAG answers often require multiple pieces of information             │
│     - 30% rate ensures broader coverage across the document                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
""")


def propose_improvements():
    """
    提出改进方案
    """
    print("="*80)
    print("Proposed Improvements for Dynamic Selection")
    print("="*80)

    print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Improvement 1: Higher Minimum Rate                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Current: min_rate = 5%                                                     │
│  Proposed: min_rate = 15-20%                                                │
│                                                                             │
│  Rationale:                                                                 │
│  - Even with very concentrated attention, we need a safety buffer           │
│  - 15-20% ensures we capture most long-tail important tokens               │
│                                                                             │
│  Code change in compute_dynamic_budget():                                   │
│    min_rate = 0.15  # Changed from 0.05                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│              Improvement 2: Entropy-Adaptive Rate Calculation               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Idea: Use attention entropy to adjust selection rate                       │
│                                                                             │
│  - Low entropy (concentrated): Use coverage-based rate + buffer            │
│  - High entropy (spread): Fall back to higher fixed rate                   │
│                                                                             │
│  Formula:                                                                   │
│    normalized_entropy = entropy / log(doc_len)                             │
│    if normalized_entropy < 0.3:  # Concentrated                            │
│        rate = max(coverage_rate, 0.15) + 0.05  # Add buffer               │
│    elif normalized_entropy < 0.6:  # Moderate                              │
│        rate = max(coverage_rate, 0.20)                                     │
│    else:  # Spread                                                         │
│        rate = 0.30  # Fall back to fixed rate                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│              Improvement 3: Position-Aware Token Selection                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Idea: Tokens near high-attention tokens are also important                 │
│                                                                             │
│  Method:                                                                    │
│  1. Find high-attention regions (top-k tokens)                             │
│  2. Expand selection to include ±N nearby tokens                           │
│  3. This captures context around important entities                        │
│                                                                             │
│  Example:                                                                   │
│    If "Company X" has high attention at position 50                        │
│    Also select positions 45-55 to capture surrounding context             │
│    This might include the year "1985" nearby                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│              Improvement 4: Hybrid Coverage + Fixed Minimum                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Simplest practical fix:                                                    │
│                                                                             │
│    dynamic_rate = max(                                                      │
│        coverage_based_rate,   # From cumulative coverage                   │
│        0.20,                  # Minimum safety buffer                      │
│        entropy_based_rate     # Higher if attention is spread              │
│    )                                                                        │
│                                                                             │
│  This ensures:                                                              │
│  - At least 20% tokens selected (safety buffer)                            │
│  - More tokens when coverage suggests (respects concentration)             │
│  - Even more when entropy is high (captures spread information)            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
""")


def run_simulation():
    """
    运行模拟实验，展示不同分布下的选择差异
    """
    print("="*80)
    print("Simulation: Comparing Selection Methods Across Different Distributions")
    print("="*80)

    distributions = simulate_attention_distributions()

    results = []
    for name, scores in distributions.items():
        comparison = compare_selection_methods(scores, fixed_rate=0.3, epsilon=0.1)

        # 计算熵
        entropy = -np.sum(scores * np.log(scores + 1e-10))
        max_entropy = np.log(len(scores))
        normalized_entropy = entropy / max_entropy

        print(f"\n{name.upper()} Distribution:")
        print(f"  Normalized Entropy: {normalized_entropy:.4f}")
        print(f"  Fixed 30%: {comparison['fixed_budget']} tokens, coverage={comparison['fixed_coverage']:.4f}")
        print(f"  Dynamic:   {comparison['dynamic_budget']} tokens ({comparison['dynamic_rate']*100:.1f}%), coverage={comparison['dynamic_coverage']:.4f}")
        print(f"  Only in Fixed (missed by dynamic): {comparison['only_in_fixed']} tokens")

        results.append({
            'name': name,
            'entropy': normalized_entropy,
            **comparison
        })

    return results


def visualize_problem(output_dir='./oracle_analysis'):
    """
    可视化动态选择的问题
    """
    os.makedirs(output_dir, exist_ok=True)

    # 创建一个示例：高度集中的 attention 分布
    np.random.seed(42)
    doc_len = 200

    # 模拟 attention 分布
    scores = np.random.exponential(scale=0.05, size=doc_len)
    # 添加几个高 attention 峰
    scores[20:25] += 2  # "Company X"
    scores[40:45] += 0.5  # "founded by"
    scores[60:65] += 0.5  # "John Smith"
    scores[80:82] += 0.1  # "1985" - the answer, but low attention!

    scores = scores / scores.sum()

    # 计算选择结果
    sorted_indices = np.argsort(scores)[::-1]
    sorted_scores = scores[sorted_indices]
    cumsum = np.cumsum(sorted_scores)

    # 动态选择 (90% 覆盖)
    tokens_needed_90 = np.searchsorted(cumsum, 0.9) + 1
    dynamic_rate = max(0.05, min(0.5, tokens_needed_90 / doc_len))
    dynamic_budget = int(dynamic_rate * doc_len)
    dynamic_selected = set(sorted_indices[:dynamic_budget])

    # 固定 30%
    fixed_budget = int(0.3 * doc_len)
    fixed_selected = set(sorted_indices[:fixed_budget])

    # 创建图表
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))

    x = np.arange(doc_len)

    # 图1: Attention 分布
    ax1 = axes[0]
    ax1.bar(x, scores, alpha=0.7, color='steelblue')
    ax1.axhline(y=np.mean(scores), color='red', linestyle='--', label='Mean')
    ax1.set_title('Attention Distribution (Simulated)')
    ax1.set_ylabel('Attention Score')
    ax1.legend()

    # 标注关键区域
    ax1.annotate('"Company X"\n(high attn)', xy=(22, scores[22]),
                 xytext=(22, scores[22]+0.02), fontsize=8, ha='center')
    ax1.annotate('"1985"\n(answer, low attn!)', xy=(81, scores[81]),
                 xytext=(81, scores[81]+0.02), fontsize=8, ha='center', color='red')

    # 图2: 动态选择
    ax2 = axes[1]
    colors = ['green' if i in dynamic_selected else 'lightgray' for i in range(doc_len)]
    ax2.bar(x, scores, color=colors, alpha=0.7)
    ax2.set_title(f'Dynamic Selection ({dynamic_budget} tokens, {dynamic_rate*100:.1f}%)')
    ax2.set_ylabel('Attention Score')

    # 标注 answer 位置是否被选中
    answer_pos = 80
    if answer_pos in dynamic_selected:
        ax2.annotate('Answer SELECTED', xy=(answer_pos, scores[answer_pos]),
                     xytext=(answer_pos, scores[answer_pos]+0.01), fontsize=10, color='green')
    else:
        ax2.annotate('Answer MISSED!', xy=(answer_pos, scores[answer_pos]),
                     xytext=(answer_pos, scores[answer_pos]+0.01), fontsize=10, color='red', fontweight='bold')

    # 图3: 固定 30%
    ax3 = axes[2]
    colors = ['blue' if i in fixed_selected else 'lightgray' for i in range(doc_len)]
    ax3.bar(x, scores, color=colors, alpha=0.7)
    ax3.set_title(f'Fixed 30% Selection ({fixed_budget} tokens)')
    ax3.set_xlabel('Token Position')
    ax3.set_ylabel('Attention Score')

    if answer_pos in fixed_selected:
        ax3.annotate('Answer SELECTED', xy=(answer_pos, scores[answer_pos]),
                     xytext=(answer_pos, scores[answer_pos]+0.01), fontsize=10, color='green', fontweight='bold')
    else:
        ax3.annotate('Answer MISSED!', xy=(answer_pos, scores[answer_pos]),
                     xytext=(answer_pos, scores[answer_pos]+0.01), fontsize=10, color='red')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'dynamic_selection_problem.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nVisualization saved to {output_path}")
    print(f"Answer token (position {answer_pos}) in dynamic selection: {answer_pos in dynamic_selected}")
    print(f"Answer token (position {answer_pos}) in fixed 30%: {answer_pos in fixed_selected}")


def main():
    print("="*80)
    print("Oracle Token Selection Analysis")
    print("="*80)

    # 1. 分析为什么动态选择会失败
    analyze_why_dynamic_fails()

    # 2. 运行模拟实验
    run_simulation()

    # 3. 提出改进方案
    propose_improvements()

    # 4. 可视化问题
    visualize_problem()

    print("\n" + "="*80)
    print("Summary: Key Recommendations")
    print("="*80)
    print("""
1. **Immediate Fix**: Increase min_rate from 5% to 15-20%
   - Simple one-line change in compute_dynamic_budget()
   - Ensures safety buffer for long-tail important tokens

2. **Better Fix**: Use entropy-adaptive rate
   - Low entropy (concentrated): coverage_rate + buffer
   - High entropy (spread): fixed 30%
   - Combines benefits of both methods

3. **Advanced Fix**: Position-aware selection
   - Expand selection to include neighbors of high-attention tokens
   - Captures context around important entities
""")


if __name__ == '__main__':
    main()
