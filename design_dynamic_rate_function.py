#!/usr/bin/env python3
"""
设计动态重算比例函数

目标：根据 attention 特征，输出 0% ~ 30% 之间的 rate
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# 读取数据
with open('./low_rate_analysis_features.json', 'r') as f:
    features = json.load(f)

df = pd.DataFrame(features)

# 转换类型
for col in ['top5_ratio', 'top10_ratio', 'top20_ratio', 'top50_ratio', 'top100_ratio']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

print("="*80)
print("设计动态重算比例函数")
print("="*80)

# ============================================================
# 方案1: 基于 peak_sharpness 的线性映射
# ============================================================
def rate_formula_v1(peak_sharpness, min_rate=0.05, max_rate=0.30):
    """
    线性映射：peak_sharpness 从 [1.0, 2.0] 映射到 [min_rate, max_rate]
    """
    # 归一化到 [0, 1]
    normalized = (peak_sharpness - 1.0) / (2.0 - 1.0)
    normalized = np.clip(normalized, 0, 1)

    rate = min_rate + (max_rate - min_rate) * normalized
    return rate

# ============================================================
# 方案2: 基于 peak_sharpness 的 sigmoid 映射
# ============================================================
def rate_formula_v2(peak_sharpness, min_rate=0.05, max_rate=0.30, center=1.3, steepness=5):
    """
    Sigmoid 映射：在 center 附近快速变化
    """
    sigmoid = 1 / (1 + np.exp(-steepness * (peak_sharpness - center)))
    rate = min_rate + (max_rate - min_rate) * sigmoid
    return rate

# ============================================================
# 方案3: 组合 peak_sharpness 和 max_attn
# ============================================================
def rate_formula_v3(peak_sharpness, max_attn, min_rate=0.05, max_rate=0.30):
    """
    组合特征：peak_sharpness 和 max_attn 都考虑
    """
    # 归一化 peak_sharpness: [1.0, 2.0] -> [0, 1]
    ps_norm = np.clip((peak_sharpness - 1.0) / 1.0, 0, 1)

    # 归一化 max_attn: [0.02, 0.08] -> [0, 1]
    ma_norm = np.clip((max_attn - 0.02) / 0.06, 0, 1)

    # 组合：两者取平均
    combined = 0.6 * ps_norm + 0.4 * ma_norm

    rate = min_rate + (max_rate - min_rate) * combined
    return rate

# ============================================================
# 方案4: 基于 cov_85 (覆盖率)
# ============================================================
def rate_formula_v4(cov_85, min_rate=0.05, max_rate=0.30):
    """
    基于覆盖率：cov_85 越高，需要的 rate 越高
    但加一个缩放因子
    """
    # cov_85 通常在 0.1 ~ 0.35 之间
    # 直接使用 cov_85 作为 rate，但限制在 [min_rate, max_rate]
    rate = np.clip(cov_85 * 1.2, min_rate, max_rate)
    return rate

# ============================================================
# 方案5: 简单公式 - 基于 peak_sharpness 的平方
# ============================================================
def rate_formula_v5(peak_sharpness, min_rate=0.05, max_rate=0.30):
    """
    平方映射：对高 peak_sharpness 更敏感
    """
    normalized = (peak_sharpness - 1.0) / 1.0
    normalized = np.clip(normalized, 0, 1)

    # 平方使得高值更突出
    rate = min_rate + (max_rate - min_rate) * (normalized ** 2)
    return rate

# ============================================================
# 测试各个方案
# ============================================================
print("\n" + "="*80)
print("各方案计算的 rate 分布")
print("="*80)

# 计算各方案的 rate
df['rate_v1'] = df['peak_sharpness'].apply(rate_formula_v1)
df['rate_v2'] = df['peak_sharpness'].apply(rate_formula_v2)
df['rate_v3'] = df.apply(lambda x: rate_formula_v3(x['peak_sharpness'], x['max_attn']), axis=1)
df['rate_v4'] = df['cov_85'].apply(rate_formula_v4)
df['rate_v5'] = df['peak_sharpness'].apply(rate_formula_v5)

stayed_correct = df[df['category'] == 'stayed_correct']
became_wrong = df[df['category'] == 'became_wrong']

print("\n各方案的 rate 统计:")
print(f"\n{'方案':<10} {'Stayed Correct mean':<22} {'Became Wrong mean':<22} {'差异':<10}")
print("-" * 70)

for version in ['rate_v1', 'rate_v2', 'rate_v3', 'rate_v4', 'rate_v5']:
    sc_mean = stayed_correct[version].mean()
    bw_mean = became_wrong[version].mean()
    diff = bw_mean - sc_mean
    print(f"{version:<10} {sc_mean:.4f}                   {bw_mean:.4f}                   {diff:+.4f}")

# ============================================================
# 模拟策略效果
# ============================================================
print("\n" + "="*80)
print("模拟策略效果")
print("="*80)

# 假设：rate >= 某阈值时，became_wrong 会变成正确
# 我们需要找到 became_wrong 需要多少 rate 才能答对
# 由于我们没有这个数据，假设 became_wrong 需要 rate >= 0.20 才能答对

def simulate_strategy(df, rate_col, required_rate_for_wrong=0.20):
    """
    模拟策略效果

    假设：
    - stayed_correct: 无论 rate 多少都对
    - became_wrong: 需要 rate >= required_rate_for_wrong 才能对
    """
    correct = 0
    total = 0
    total_rate = 0

    for _, row in df.iterrows():
        total += 1
        rate = row[rate_col]
        total_rate += rate

        if row['category'] == 'stayed_correct':
            correct += 1
        elif row['category'] == 'became_wrong':
            if rate >= required_rate_for_wrong:
                correct += 1
        elif row['category'] == 'became_correct':
            correct += 1
        # stayed_wrong 不计入正确

    accuracy = correct / total
    avg_rate = total_rate / total

    return accuracy, avg_rate

print("\n假设: became_wrong 需要 rate >= 0.20 才能答对")
print(f"\n{'方案':<10} {'预期正确率':<15} {'平均 rate':<15} {'vs 固定0.3节省':<15}")
print("-" * 60)

# Baseline
baseline_acc = (len(stayed_correct) + len(df[df['category'] == 'became_correct'])) / len(df)
print(f"{'固定0.05':<10} {baseline_acc:.2%}            0.050            -")

for version in ['rate_v1', 'rate_v2', 'rate_v3', 'rate_v4', 'rate_v5']:
    acc, avg_rate = simulate_strategy(df, version, required_rate_for_wrong=0.20)
    saving = (0.30 - avg_rate) / 0.30 * 100
    print(f"{version:<10} {acc:.2%}            {avg_rate:.3f}            {saving:.1f}%")

print(f"{'固定0.30':<10} {1.0:.2%}            0.300            0.0%")

# ============================================================
# 查看各方案对 became_wrong 的 rate 分配
# ============================================================
print("\n" + "="*80)
print("各方案对 became_wrong 案例的 rate 分配")
print("="*80)

print(f"\n{'问题':<45} {'v1':<8} {'v2':<8} {'v3':<8} {'v4':<8} {'v5':<8}")
print("-" * 90)

for _, row in became_wrong.iterrows():
    query = row['query'][:42] + "..."
    print(f"{query:<45} {row['rate_v1']:.3f}   {row['rate_v2']:.3f}   {row['rate_v3']:.3f}   {row['rate_v4']:.3f}   {row['rate_v5']:.3f}")

# ============================================================
# 可视化
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 各方案的 rate 分布
for idx, version in enumerate(['rate_v1', 'rate_v2', 'rate_v3', 'rate_v4', 'rate_v5']):
    ax = axes[idx // 3, idx % 3]

    ax.hist(stayed_correct[version], bins=10, alpha=0.6, label='Stayed Correct', color='green')
    ax.hist(became_wrong[version], bins=5, alpha=0.6, label='Became Wrong', color='red')
    ax.axvline(x=0.20, color='black', linestyle='--', label='Threshold (0.20)')
    ax.set_xlabel('Rate')
    ax.set_ylabel('Count')
    ax.set_title(f'{version}')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 0.35)

# 最后一个图：peak_sharpness vs rate 的关系
ax = axes[1, 2]
x = np.linspace(1.0, 2.2, 100)
ax.plot(x, [rate_formula_v1(xi) for xi in x], label='v1 (linear)', linewidth=2)
ax.plot(x, [rate_formula_v2(xi) for xi in x], label='v2 (sigmoid)', linewidth=2)
ax.plot(x, [rate_formula_v5(xi) for xi in x], label='v5 (quadratic)', linewidth=2)
ax.scatter(stayed_correct['peak_sharpness'], stayed_correct['rate_v1'],
           c='green', alpha=0.5, s=50, label='Stayed Correct')
ax.scatter(became_wrong['peak_sharpness'], became_wrong['rate_v1'],
           c='red', alpha=0.8, s=80, marker='X', label='Became Wrong')
ax.set_xlabel('Peak Sharpness')
ax.set_ylabel('Rate')
ax.set_title('Rate Functions')
ax.legend(fontsize=8)
ax.set_xlim(0.9, 2.2)
ax.set_ylim(0, 0.35)

plt.tight_layout()
plt.savefig('./analysis_dynamic_rate_functions.png', dpi=150, bbox_inches='tight')
print(f"\n图表已保存: analysis_dynamic_rate_functions.png")
plt.close()

# ============================================================
# 推荐的公式
# ============================================================
print("\n" + "="*80)
print("推荐公式")
print("="*80)

print("""
基于分析，推荐使用 方案1 (线性映射) 或 方案3 (组合特征):

【方案1: 简单线性映射】
def compute_dynamic_rate(peak_sharpness, min_rate=0.05, max_rate=0.30):
    normalized = (peak_sharpness - 1.0) / 1.0
    normalized = max(0, min(1, normalized))
    rate = min_rate + (max_rate - min_rate) * normalized
    return rate

【方案3: 组合 peak_sharpness 和 max_attn】
def compute_dynamic_rate(peak_sharpness, max_attn, min_rate=0.05, max_rate=0.30):
    ps_norm = max(0, min(1, (peak_sharpness - 1.0) / 1.0))
    ma_norm = max(0, min(1, (max_attn - 0.02) / 0.06))
    combined = 0.6 * ps_norm + 0.4 * ma_norm
    rate = min_rate + (max_rate - min_rate) * combined
    return rate

公式物理意义:
- peak_sharpness 越高 → attention 越集中 → 容易漏掉关键 token → 需要更高 rate
- max_attn 越高 → 单个 token 权重越大 → 该 token 越关键 → 需要更高 rate
""")
