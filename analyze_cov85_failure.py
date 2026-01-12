#!/usr/bin/env python3
"""
深入分析 cov_85 指标为什么不能用来确定重算比例

核心问题：
- cov_85 的物理意义是 "达到 85% attention 覆盖需要多少比例的 token"
- 直觉上，cov_85 越高 → attention 越分散 → 需要更多 token 才能覆盖重要信息 → 需要更高的 rate
- 但实验显示 cov_85 在 stayed_correct 和 became_wrong 之间分离度很低 (0.16)

本分析要回答：
1. 两组的 cov_85 分布到底差多少？
2. 为什么 cov_85 无法区分两组？
3. 有没有反直觉的案例？
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

# 分组
stayed_correct = df[df['category'] == 'stayed_correct']
became_wrong = df[df['category'] == 'became_wrong']

print("="*80)
print("深入分析：为什么 cov_85 不能用来确定重算比例？")
print("="*80)

# ============================================================
# 1. cov_85 的基本统计
# ============================================================
print("\n" + "="*80)
print("1. cov_85 基本统计")
print("="*80)

print(f"\n样本数量:")
print(f"  stayed_correct: {len(stayed_correct)}")
print(f"  became_wrong: {len(became_wrong)}")

print(f"\n[stayed_correct] cov_85 统计:")
print(f"  mean: {stayed_correct['cov_85'].mean():.4f}")
print(f"  std:  {stayed_correct['cov_85'].std():.4f}")
print(f"  min:  {stayed_correct['cov_85'].min():.4f}")
print(f"  25%:  {stayed_correct['cov_85'].quantile(0.25):.4f}")
print(f"  50%:  {stayed_correct['cov_85'].quantile(0.50):.4f}")
print(f"  75%:  {stayed_correct['cov_85'].quantile(0.75):.4f}")
print(f"  max:  {stayed_correct['cov_85'].max():.4f}")

print(f"\n[became_wrong] cov_85 统计:")
print(f"  mean: {became_wrong['cov_85'].mean():.4f}")
print(f"  std:  {became_wrong['cov_85'].std():.4f}")
print(f"  min:  {became_wrong['cov_85'].min():.4f}")
print(f"  25%:  {became_wrong['cov_85'].quantile(0.25):.4f}")
print(f"  50%:  {became_wrong['cov_85'].quantile(0.50):.4f}")
print(f"  75%:  {became_wrong['cov_85'].quantile(0.75):.4f}")
print(f"  max:  {became_wrong['cov_85'].max():.4f}")

# 计算分离度 (Cohen's d)
mean_sc = stayed_correct['cov_85'].mean()
mean_bw = became_wrong['cov_85'].mean()
std_pooled = np.sqrt((stayed_correct['cov_85'].var() + became_wrong['cov_85'].var()) / 2)
cohens_d = (mean_bw - mean_sc) / std_pooled if std_pooled > 0 else 0

print(f"\n分离度 (Cohen's d): {cohens_d:.4f}")
print(f"  解释: < 0.2 极小, 0.2-0.5 小, 0.5-0.8 中等, > 0.8 大")

# ============================================================
# 2. 分布重叠分析
# ============================================================
print("\n" + "="*80)
print("2. 分布重叠分析")
print("="*80)

# 计算分位数重叠
sc_q25, sc_q75 = stayed_correct['cov_85'].quantile([0.25, 0.75])
bw_q25, bw_q75 = became_wrong['cov_85'].quantile([0.25, 0.75])

print(f"\nstayed_correct IQR: [{sc_q25:.4f}, {sc_q75:.4f}]")
print(f"became_wrong IQR:   [{bw_q25:.4f}, {bw_q75:.4f}]")

# 重叠区间
overlap_low = max(sc_q25, bw_q25)
overlap_high = min(sc_q75, bw_q75)
if overlap_low < overlap_high:
    print(f"IQR 重叠区间: [{overlap_low:.4f}, {overlap_high:.4f}]")
    print(f"  → 两组的中间 50% 数据大量重叠！")
else:
    print(f"IQR 无重叠")

# ============================================================
# 3. 具体案例分析
# ============================================================
print("\n" + "="*80)
print("3. 具体案例分析")
print("="*80)

# 找出反直觉的案例
# 情况A: cov_85 高但答对了 (stayed_correct)
# 情况B: cov_85 低但答错了 (became_wrong)

print("\n[情况A] cov_85 高但 rate=0.05 也答对了 (stayed_correct)")
print("-"*70)
high_cov_correct = stayed_correct.nlargest(5, 'cov_85')
for _, row in high_cov_correct.iterrows():
    print(f"  cov_85={row['cov_85']:.3f}  Query: {row['query'][:60]}...")

print("\n[情况B] cov_85 低但 rate=0.05 答错了 (became_wrong)")
print("-"*70)
low_cov_wrong = became_wrong.nsmallest(3, 'cov_85')
for _, row in low_cov_wrong.iterrows():
    print(f"  cov_85={row['cov_85']:.3f}  Query: {row['query'][:60]}...")

# 最关键的对比
print("\n[最关键对比] cov_85 相近但结果不同")
print("-"*70)

# 找 cov_85 接近 0.20 的案例
threshold = 0.20
tolerance = 0.03

similar_cov_correct = stayed_correct[
    (stayed_correct['cov_85'] > threshold - tolerance) &
    (stayed_correct['cov_85'] < threshold + tolerance)
]
similar_cov_wrong = became_wrong[
    (became_wrong['cov_85'] > threshold - tolerance) &
    (became_wrong['cov_85'] < threshold + tolerance)
]

print(f"\ncov_85 在 [{threshold-tolerance:.2f}, {threshold+tolerance:.2f}] 范围内:")
print(f"  stayed_correct: {len(similar_cov_correct)} 个")
print(f"  became_wrong:   {len(similar_cov_wrong)} 个")

if len(similar_cov_correct) > 0 and len(similar_cov_wrong) > 0:
    print("\n  stayed_correct 案例:")
    for _, row in similar_cov_correct.head(3).iterrows():
        print(f"    cov_85={row['cov_85']:.3f}  {row['query'][:55]}...")
    print("\n  became_wrong 案例:")
    for _, row in similar_cov_wrong.head(3).iterrows():
        print(f"    cov_85={row['cov_85']:.3f}  {row['query'][:55]}...")

# ============================================================
# 4. 根本原因分析
# ============================================================
print("\n" + "="*80)
print("4. 根本原因分析")
print("="*80)

print("""
cov_85 失效的根本原因:

1. cov_85 衡量的是 "attention 分布的集中程度"
   - cov_85 低 → attention 集中在少数 token
   - cov_85 高 → attention 分散在很多 token

2. 但 "是否需要高 rate" 取决于完全不同的因素:
   - 关键信息是否在被选中的 top-k token 中
   - 关键信息是否有冗余（多处出现）
   - 问题答案的类型（事实 vs 推理）

3. 关键洞察: attention 集中 ≠ 关键信息被选中

   反例1: attention 很集中 (cov_85 低)，但关键 token 没在 top-5%
   → 虽然 attention 集中，但集中在了错误的位置
   → 需要更高的 rate 来覆盖真正重要的 token

   反例2: attention 很分散 (cov_85 高)，但关键信息多处出现
   → 即使 attention 分散，top-5% 也能覆盖足够的关键信息
   → 不需要高 rate

4. 本质问题: cov_85 是 attention 分布的统计量，但无法告诉我们:
   - "哪些 token 是真正重要的"（需要知道答案才能判断）
   - "top-k 选择是否包含了这些重要 token"
""")

# ============================================================
# 5. 验证: 对比 cov_85 和实际 top-5% 覆盖
# ============================================================
print("\n" + "="*80)
print("5. 数值验证")
print("="*80)

# cov_85 和 cov_90 的关系
print("\n各覆盖率指标的两组均值:")
for cov in ['cov_50', 'cov_70', 'cov_80', 'cov_85', 'cov_90', 'cov_95']:
    sc_mean = stayed_correct[cov].mean()
    bw_mean = became_wrong[cov].mean()
    diff = bw_mean - sc_mean
    print(f"  {cov}: stayed_correct={sc_mean:.3f}, became_wrong={bw_mean:.3f}, diff={diff:+.3f}")

print("\n结论: 所有覆盖率指标的两组差异都很小（< 0.03）！")

# ============================================================
# 6. 可视化
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 6.1 cov_85 分布对比
ax = axes[0, 0]
ax.hist(stayed_correct['cov_85'], bins=12, alpha=0.6, label='Stayed Correct', color='green', density=True)
ax.hist(became_wrong['cov_85'], bins=6, alpha=0.6, label='Became Wrong', color='red', density=True)
ax.axvline(stayed_correct['cov_85'].mean(), color='green', linestyle='--', linewidth=2, label=f'SC mean={mean_sc:.3f}')
ax.axvline(became_wrong['cov_85'].mean(), color='red', linestyle='--', linewidth=2, label=f'BW mean={mean_bw:.3f}')
ax.set_xlabel('cov_85 (ratio needed for 85% coverage)')
ax.set_ylabel('Density')
ax.set_title('cov_85 Distribution: Nearly Complete Overlap')
ax.legend(fontsize=9)

# 6.2 cov_85 vs rate=0.05 (5%) 的关系
ax = axes[0, 1]
ax.axhline(y=0.05, color='blue', linestyle='--', linewidth=2, label='rate=0.05 (5%)')
ax.scatter(stayed_correct.index, stayed_correct['cov_85'], c='green', alpha=0.6, s=60, label='Stayed Correct')
ax.scatter(became_wrong.index, became_wrong['cov_85'], c='red', alpha=0.8, s=100, marker='X', label='Became Wrong')
ax.set_xlabel('Sample Index')
ax.set_ylabel('cov_85')
ax.set_title('cov_85 vs rate=0.05: Even Low cov_85 Can Fail')
ax.legend(fontsize=9)

# 6.3 箱线图
ax = axes[1, 0]
data_to_plot = [stayed_correct['cov_85'].values, became_wrong['cov_85'].values]
bp = ax.boxplot(data_to_plot, labels=['Stayed Correct', 'Became Wrong'], patch_artist=True)
bp['boxes'][0].set_facecolor('lightgreen')
bp['boxes'][1].set_facecolor('lightcoral')
ax.set_ylabel('cov_85')
ax.set_title(f'cov_85 Boxplot (Cohen\'s d = {cohens_d:.2f})')

# 6.4 多覆盖率对比
ax = axes[1, 1]
coverages = ['cov_50', 'cov_70', 'cov_80', 'cov_85', 'cov_90', 'cov_95']
sc_means = [stayed_correct[c].mean() for c in coverages]
bw_means = [became_wrong[c].mean() for c in coverages]
x = np.arange(len(coverages))
width = 0.35
ax.bar(x - width/2, sc_means, width, label='Stayed Correct', color='green', alpha=0.7)
ax.bar(x + width/2, bw_means, width, label='Became Wrong', color='red', alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(coverages)
ax.set_ylabel('Mean Coverage Ratio')
ax.set_title('All Coverage Metrics: Minimal Difference')
ax.legend()

plt.tight_layout()
plt.savefig('./analysis_cov85_failure.png', dpi=150, bbox_inches='tight')
print(f"\n图表已保存: analysis_cov85_failure.png")
plt.close()

# ============================================================
# 7. 最终结论
# ============================================================
print("\n" + "="*80)
print("7. 最终结论")
print("="*80)

print("""
为什么 cov_85 不能用来确定重算比例？

【统计原因】
- 两组 cov_85 均值差异仅 0.007（0.206 vs 0.213）
- 分离度 Cohen's d = 0.16，属于"几乎没有区分能力"
- 两组的 IQR 高度重叠

【理论原因】
cov_85 衡量的是: "达到 85% attention 覆盖需要多少比例的 token"

这个指标假设:
  "attention 覆盖 = 信息覆盖"

但这个假设是错误的！

真实情况:
1. attention 可能集中在非关键 token 上
2. 关键信息可能在 attention 较低的 token 中
3. 不同问题的"关键信息"定义不同（答案、上下文、推理线索...）

【直观解释】
想象一个问题："谁是彼得的儿子？"

情况A (cov_85 低但答错):
- attention 集中在 "彼得" 这个词（占 30%）
- 但真正的答案 "约翰" 在 attention 较低的位置
- top-5% 选中了 "彼得"，但漏掉了 "约翰"
- → 需要更高 rate

情况B (cov_85 高但答对):
- attention 分散在多个地方
- 但 "约翰" 这个答案出现了 3 次，都有一定 attention
- top-5% 至少选中了其中一个 "约翰"
- → 低 rate 就够了

【根本矛盾】
要知道是否需要高 rate，我们需要知道 "关键 token 是否被选中"
但要知道哪些是关键 token，我们需要知道正确答案
而知道正确答案后，就不需要预测 rate 了！

这是一个信息论的死结：
  - 我们只能获取 attention 分布（Draft Model 的输出）
  - 但 attention 分布不包含 "哪些 token 包含正确答案" 的信息
  - 所以无法预测 "是否需要更高 rate"
""")
