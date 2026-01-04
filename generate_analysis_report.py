#!/usr/bin/env python3
"""
生成动态重算比例分析报告的图表
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (12, 8)

# 读取分析结果
with open('./low_rate_analysis_features.json', 'r') as f:
    attention_features = json.load(f)

with open('./first_token_confidence_analysis.json', 'r') as f:
    confidence_features = json.load(f)

# 转换为 DataFrame
attn_df = pd.DataFrame(attention_features)
conf_df = pd.DataFrame(confidence_features)

# 转换类型
for col in ['top5_ratio', 'top10_ratio', 'top20_ratio', 'top50_ratio', 'top100_ratio']:
    if col in attn_df.columns:
        attn_df[col] = attn_df[col].astype(float)

# 分组
attn_stayed_correct = attn_df[attn_df['category'] == 'stayed_correct']
attn_became_wrong = attn_df[attn_df['category'] == 'became_wrong']

conf_stayed_correct = conf_df[conf_df['category'] == 'stayed_correct']
conf_became_wrong = conf_df[conf_df['category'] == 'became_wrong']

print(f"Attention analysis: stayed_correct={len(attn_stayed_correct)}, became_wrong={len(attn_became_wrong)}")
print(f"Confidence analysis: stayed_correct={len(conf_stayed_correct)}, became_wrong={len(conf_became_wrong)}")

# ============================================================
# 图1: Attention 特征对比 - 箱线图
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

features_to_plot = [
    ('cov_85', 'Coverage@85%'),
    ('gini', 'Gini Coefficient'),
    ('max_attn', 'Max Attention'),
    ('peak_sharpness', 'Peak Sharpness'),
    ('top5_ratio', 'Top-5 Ratio'),
    ('norm_entropy', 'Normalized Entropy'),
]

for idx, (feat, title) in enumerate(features_to_plot):
    ax = axes[idx // 3, idx % 3]

    data_sc = attn_stayed_correct[feat].dropna()
    data_bw = attn_became_wrong[feat].dropna()

    bp = ax.boxplot([data_sc, data_bw], labels=['Stayed Correct', 'Became Wrong'])
    ax.set_title(title)
    ax.set_ylabel(feat)

    # 添加均值点
    means = [data_sc.mean(), data_bw.mean()]
    ax.scatter([1, 2], means, color='red', marker='D', s=50, zorder=3, label='Mean')

    # 计算分离度
    pooled_std = np.sqrt((data_sc.std()**2 + data_bw.std()**2) / 2)
    separation = abs(data_bw.mean() - data_sc.mean()) / pooled_std if pooled_std > 0 else 0
    ax.text(0.95, 0.95, f'Sep: {separation:.3f}', transform=ax.transAxes,
            ha='right', va='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat'))

plt.suptitle('Attention Distribution Features: Stayed Correct vs Became Wrong\n(rate=0.05)', fontsize=14)
plt.tight_layout()
plt.savefig('./analysis_attention_features_boxplot.png', dpi=150, bbox_inches='tight')
print("Saved: analysis_attention_features_boxplot.png")
plt.close()

# ============================================================
# 图2: 首 Token 置信度对比 - 箱线图
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

conf_features = [
    ('top1_prob', 'Top-1 Probability'),
    ('top5_prob', 'Top-5 Probability'),
    ('norm_entropy', 'Normalized Entropy'),
]

for idx, (feat, title) in enumerate(conf_features):
    ax = axes[idx]

    data_sc = conf_stayed_correct[feat].dropna()
    data_bw = conf_became_wrong[feat].dropna()

    bp = ax.boxplot([data_sc, data_bw], labels=['Stayed Correct', 'Became Wrong'])
    ax.set_title(title)
    ax.set_ylabel(feat)

    means = [data_sc.mean(), data_bw.mean()]
    ax.scatter([1, 2], means, color='red', marker='D', s=50, zorder=3)

    pooled_std = np.sqrt((data_sc.std()**2 + data_bw.std()**2) / 2)
    separation = abs(data_bw.mean() - data_sc.mean()) / pooled_std if pooled_std > 0 else 0
    ax.text(0.95, 0.95, f'Sep: {separation:.3f}', transform=ax.transAxes,
            ha='right', va='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat'))

plt.suptitle('Draft Model First Token Confidence: Stayed Correct vs Became Wrong\n(Full Attention)', fontsize=14)
plt.tight_layout()
plt.savefig('./analysis_first_token_confidence_boxplot.png', dpi=150, bbox_inches='tight')
print("Saved: analysis_first_token_confidence_boxplot.png")
plt.close()

# ============================================================
# 图3: 分离度汇总条形图
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))

# Attention 特征分离度
attn_features_sep = {}
for feat in ['cov_85', 'cov_90', 'gini', 'max_attn', 'peak_sharpness', 'top5_ratio', 'top10_ratio', 'norm_entropy']:
    if feat not in attn_df.columns:
        continue
    data_sc = attn_stayed_correct[feat].dropna()
    data_bw = attn_became_wrong[feat].dropna()
    pooled_std = np.sqrt((data_sc.std()**2 + data_bw.std()**2) / 2)
    separation = abs(data_bw.mean() - data_sc.mean()) / pooled_std if pooled_std > 0 else 0
    attn_features_sep[f'Attn: {feat}'] = separation

# 置信度特征分离度
conf_features_sep = {}
for feat in ['top1_prob', 'top5_prob', 'norm_entropy']:
    data_sc = conf_stayed_correct[feat].dropna()
    data_bw = conf_became_wrong[feat].dropna()
    pooled_std = np.sqrt((data_sc.std()**2 + data_bw.std()**2) / 2)
    separation = abs(data_bw.mean() - data_sc.mean()) / pooled_std if pooled_std > 0 else 0
    conf_features_sep[f'Conf: {feat}'] = separation

all_sep = {**attn_features_sep, **conf_features_sep}
sorted_sep = sorted(all_sep.items(), key=lambda x: x[1], reverse=True)

names = [x[0] for x in sorted_sep]
values = [x[1] for x in sorted_sep]
colors = ['steelblue' if 'Attn' in n else 'coral' for n in names]

bars = ax.barh(range(len(names)), values, color=colors)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names)
ax.set_xlabel('Separation Score (Cohen\'s d)')
ax.set_title('Feature Separation Scores: Ability to Distinguish Stayed Correct vs Became Wrong')
ax.axvline(x=0.5, color='red', linestyle='--', label='Medium effect size (0.5)')
ax.axvline(x=0.8, color='darkred', linestyle='--', label='Large effect size (0.8)')
ax.legend()

# 添加数值标签
for bar, val in zip(bars, values):
    ax.text(val + 0.02, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('./analysis_separation_scores.png', dpi=150, bbox_inches='tight')
print("Saved: analysis_separation_scores.png")
plt.close()

# ============================================================
# 图4: 散点图 - 关键特征的分布
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图: peak_sharpness vs cov_85
ax = axes[0]
ax.scatter(attn_stayed_correct['cov_85'], attn_stayed_correct['peak_sharpness'],
           c='green', alpha=0.6, s=80, label='Stayed Correct')
ax.scatter(attn_became_wrong['cov_85'], attn_became_wrong['peak_sharpness'],
           c='red', alpha=0.8, s=100, marker='X', label='Became Wrong')
ax.set_xlabel('Coverage@85%')
ax.set_ylabel('Peak Sharpness')
ax.set_title('Attention Features Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# 右图: top1_prob vs query (scatter)
ax = axes[1]
ax.scatter(range(len(conf_stayed_correct)), conf_stayed_correct['top1_prob'],
           c='green', alpha=0.6, s=80, label='Stayed Correct')
ax.scatter(range(len(conf_stayed_correct), len(conf_stayed_correct) + len(conf_became_wrong)),
           conf_became_wrong['top1_prob'],
           c='red', alpha=0.8, s=100, marker='X', label='Became Wrong')
ax.set_xlabel('Question Index')
ax.set_ylabel('Top-1 Probability')
ax.set_title('First Token Confidence Distribution')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('./analysis_scatter_plots.png', dpi=150, bbox_inches='tight')
print("Saved: analysis_scatter_plots.png")
plt.close()

# ============================================================
# 输出统计摘要
# ============================================================
print("\n" + "="*80)
print("STATISTICAL SUMMARY")
print("="*80)

print("\n[Attention Features]")
print(f"Sample sizes: stayed_correct={len(attn_stayed_correct)}, became_wrong={len(attn_became_wrong)}")
print("\nFeature              Stayed Correct       Became Wrong         Separation")
print("-" * 75)
for feat in ['cov_85', 'gini', 'peak_sharpness', 'max_attn', 'top5_ratio']:
    if feat not in attn_df.columns:
        continue
    sc_mean = attn_stayed_correct[feat].mean()
    sc_std = attn_stayed_correct[feat].std()
    bw_mean = attn_became_wrong[feat].mean()
    bw_std = attn_became_wrong[feat].std()
    pooled_std = np.sqrt((sc_std**2 + bw_std**2) / 2)
    sep = abs(bw_mean - sc_mean) / pooled_std if pooled_std > 0 else 0
    print(f"{feat:<20} {sc_mean:.4f} ± {sc_std:.4f}    {bw_mean:.4f} ± {bw_std:.4f}    {sep:.4f}")

print("\n[First Token Confidence]")
print(f"Sample sizes: stayed_correct={len(conf_stayed_correct)}, became_wrong={len(conf_became_wrong)}")
print("\nFeature              Stayed Correct       Became Wrong         Separation")
print("-" * 75)
for feat in ['top1_prob', 'top5_prob', 'norm_entropy']:
    sc_mean = conf_stayed_correct[feat].mean()
    sc_std = conf_stayed_correct[feat].std()
    bw_mean = conf_became_wrong[feat].mean()
    bw_std = conf_became_wrong[feat].std()
    pooled_std = np.sqrt((sc_std**2 + bw_std**2) / 2)
    sep = abs(bw_mean - sc_mean) / pooled_std if pooled_std > 0 else 0
    print(f"{feat:<20} {sc_mean:.4f} ± {sc_std:.4f}    {bw_mean:.4f} ± {bw_std:.4f}    {sep:.4f}")

print("\n" + "="*80)
print("CONCLUSION: All features have low separation scores (< 0.5)")
print("Neither attention distribution nor first token confidence can reliably")
print("distinguish questions that need higher recomputation rate.")
print("="*80)
