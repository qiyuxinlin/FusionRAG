#!/usr/bin/env python3
"""
验证基于 peak_sharpness 的动态 rate 策略

策略：
- 如果 peak_sharpness > 阈值 → 使用高 rate
- 否则 → 使用低 rate
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
print("基于 Peak Sharpness 的动态 Rate 策略验证")
print("="*80)

# 分组
stayed_correct = df[df['category'] == 'stayed_correct']
became_wrong = df[df['category'] == 'became_wrong']

print(f"\n样本分布:")
print(f"  stayed_correct: {len(stayed_correct)}")
print(f"  became_wrong: {len(became_wrong)}")

# ============================================================
# 分析阈值
# ============================================================
print("\n" + "="*80)
print("Peak Sharpness 分布分析")
print("="*80)

print(f"\nStayed Correct:")
print(f"  mean: {stayed_correct['peak_sharpness'].mean():.4f}")
print(f"  std:  {stayed_correct['peak_sharpness'].std():.4f}")
print(f"  min:  {stayed_correct['peak_sharpness'].min():.4f}")
print(f"  max:  {stayed_correct['peak_sharpness'].max():.4f}")
print(f"  25%:  {stayed_correct['peak_sharpness'].quantile(0.25):.4f}")
print(f"  50%:  {stayed_correct['peak_sharpness'].quantile(0.50):.4f}")
print(f"  75%:  {stayed_correct['peak_sharpness'].quantile(0.75):.4f}")

print(f"\nBecame Wrong:")
print(f"  mean: {became_wrong['peak_sharpness'].mean():.4f}")
print(f"  std:  {became_wrong['peak_sharpness'].std():.4f}")
print(f"  min:  {became_wrong['peak_sharpness'].min():.4f}")
print(f"  max:  {became_wrong['peak_sharpness'].max():.4f}")
print(f"  25%:  {became_wrong['peak_sharpness'].quantile(0.25):.4f}")
print(f"  50%:  {became_wrong['peak_sharpness'].quantile(0.50):.4f}")
print(f"  75%:  {became_wrong['peak_sharpness'].quantile(0.75):.4f}")

# ============================================================
# 测试不同阈值
# ============================================================
print("\n" + "="*80)
print("不同阈值下的分类效果")
print("="*80)

print("\n策略: if peak_sharpness > threshold → 预测需要高 rate")
print("\n阈值      | 正确识别 became_wrong | 误判 stayed_correct | 精确率 | 召回率 | F1")
print("-" * 85)

best_f1 = 0
best_threshold = 0

for threshold in np.arange(1.0, 2.0, 0.05):
    # 预测: peak_sharpness > threshold → 需要高 rate (预测为 became_wrong)
    predicted_high_rate = df['peak_sharpness'] > threshold

    # 实际 became_wrong
    actual_became_wrong = df['category'] == 'became_wrong'

    # True Positive: 正确识别 became_wrong
    tp = (predicted_high_rate & actual_became_wrong).sum()

    # False Positive: 误判 stayed_correct 为需要高 rate
    fp = (predicted_high_rate & ~actual_became_wrong).sum()

    # False Negative: 漏判 became_wrong
    fn = (~predicted_high_rate & actual_became_wrong).sum()

    # True Negative: 正确识别 stayed_correct
    tn = (~predicted_high_rate & ~actual_became_wrong).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"{threshold:.2f}      | {tp}/{len(became_wrong)} ({tp/len(became_wrong)*100:5.1f}%)          | {fp}/{len(stayed_correct)} ({fp/len(stayed_correct)*100:5.1f}%)          | {precision:.3f}  | {recall:.3f}  | {f1:.3f}")

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"\n最佳阈值: {best_threshold:.2f} (F1={best_f1:.3f})")

# ============================================================
# 使用最佳阈值模拟策略
# ============================================================
print("\n" + "="*80)
print(f"使用阈值 {best_threshold:.2f} 的策略模拟")
print("="*80)

# 假设:
# - peak_sharpness > threshold → 使用 rate=0.3 (高)
# - peak_sharpness <= threshold → 使用 rate=0.05 (低)

predicted_high_rate = df['peak_sharpness'] > best_threshold

# 统计
need_high_rate = predicted_high_rate.sum()
can_use_low_rate = (~predicted_high_rate).sum()

print(f"\n预测结果:")
print(f"  需要高 rate (>阈值): {need_high_rate} ({need_high_rate/len(df)*100:.1f}%)")
print(f"  可用低 rate (<=阈值): {can_use_low_rate} ({can_use_low_rate/len(df)*100:.1f}%)")

# 分析策略效果
# 对于 became_wrong:
#   - 如果预测为高 rate → 假设会答对 (因为 rate=1 时是对的)
#   - 如果预测为低 rate → 仍然答错
# 对于 stayed_correct:
#   - 无论预测什么都会答对

correct_after_strategy = 0
wrong_after_strategy = 0

for _, row in df.iterrows():
    predicted_high = row['peak_sharpness'] > best_threshold

    if row['category'] == 'stayed_correct':
        # 无论如何都对
        correct_after_strategy += 1
    elif row['category'] == 'became_wrong':
        if predicted_high:
            # 正确识别，使用高 rate，假设会答对
            correct_after_strategy += 1
        else:
            # 漏判，使用低 rate，仍然答错
            wrong_after_strategy += 1
    elif row['category'] == 'stayed_wrong':
        # 无论如何都错
        wrong_after_strategy += 1
    else:  # became_correct
        correct_after_strategy += 1

print(f"\n策略效果模拟:")
print(f"  原始 rate=0.05 正确率: {len(stayed_correct)}/{len(df)} = {len(stayed_correct)/len(df)*100:.1f}%")
print(f"  策略后预期正确率: {correct_after_strategy}/{len(df)} = {correct_after_strategy/len(df)*100:.1f}%")
print(f"  提升: +{(correct_after_strategy - len(stayed_correct))/len(df)*100:.1f}%")

# 计算平均 rate
# 假设 高 rate = 0.3, 低 rate = 0.05
high_rate_count = need_high_rate
low_rate_count = can_use_low_rate
avg_rate = (high_rate_count * 0.3 + low_rate_count * 0.05) / len(df)

print(f"\n计算开销:")
print(f"  使用高 rate (0.3): {high_rate_count} 次")
print(f"  使用低 rate (0.05): {low_rate_count} 次")
print(f"  平均 rate: {avg_rate:.3f}")
print(f"  对比固定 rate=0.3: 节省 {(0.3 - avg_rate) / 0.3 * 100:.1f}% 重算量")

# ============================================================
# 可视化
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图: peak_sharpness 分布
ax = axes[0]
ax.hist(stayed_correct['peak_sharpness'], bins=15, alpha=0.6, label='Stayed Correct', color='green')
ax.hist(became_wrong['peak_sharpness'], bins=8, alpha=0.6, label='Became Wrong', color='red')
ax.axvline(x=best_threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold={best_threshold:.2f}')
ax.set_xlabel('Peak Sharpness')
ax.set_ylabel('Count')
ax.set_title('Peak Sharpness Distribution')
ax.legend()

# 右图: max_attn 分布
ax = axes[1]
ax.hist(stayed_correct['max_attn'], bins=15, alpha=0.6, label='Stayed Correct', color='green')
ax.hist(became_wrong['max_attn'], bins=8, alpha=0.6, label='Became Wrong', color='red')
ax.set_xlabel('Max Attention')
ax.set_ylabel('Count')
ax.set_title('Max Attention Distribution')
ax.legend()

plt.tight_layout()
plt.savefig('./analysis_peak_sharpness_threshold.png', dpi=150, bbox_inches='tight')
print(f"\n图表已保存: analysis_peak_sharpness_threshold.png")
plt.close()

# ============================================================
# 尝试组合特征
# ============================================================
print("\n" + "="*80)
print("组合特征策略: peak_sharpness 和 max_attn")
print("="*80)

# 计算组合分数
df['combined_score'] = df['peak_sharpness'] * df['max_attn'] * 100

print("\n组合分数 = peak_sharpness * max_attn * 100")
print(f"\nStayed Correct: {stayed_correct['peak_sharpness'].values * stayed_correct['max_attn'].values * 100}")
print(f"  mean: {(stayed_correct['peak_sharpness'] * stayed_correct['max_attn'] * 100).mean():.4f}")

print(f"\nBecame Wrong:")
print(f"  mean: {(became_wrong['peak_sharpness'] * became_wrong['max_attn'] * 100).mean():.4f}")

print("\n组合分数阈值测试:")
print("阈值      | 正确识别 became_wrong | 误判 stayed_correct | F1")
print("-" * 65)

for threshold in np.arange(3.0, 8.0, 0.5):
    predicted_high_rate = df['combined_score'] > threshold
    actual_became_wrong = df['category'] == 'became_wrong'

    tp = (predicted_high_rate & actual_became_wrong).sum()
    fp = (predicted_high_rate & ~actual_became_wrong).sum()
    fn = (~predicted_high_rate & actual_became_wrong).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"{threshold:.1f}       | {tp}/{len(became_wrong)} ({tp/len(became_wrong)*100:5.1f}%)          | {fp}/{len(stayed_correct)} ({fp/len(stayed_correct)*100:5.1f}%)          | {f1:.3f}")

# ============================================================
# 详细查看分类结果
# ============================================================
print("\n" + "="*80)
print(f"阈值 {best_threshold:.2f} 下的具体分类结果")
print("="*80)

print("\n[正确识别的 became_wrong] (peak_sharpness > 阈值)")
for _, row in became_wrong[became_wrong['peak_sharpness'] > best_threshold].iterrows():
    print(f"  {row['query'][:50]}...")
    print(f"    peak_sharpness: {row['peak_sharpness']:.3f}, max_attn: {row['max_attn']:.4f}")

print("\n[漏判的 became_wrong] (peak_sharpness <= 阈值)")
for _, row in became_wrong[became_wrong['peak_sharpness'] <= best_threshold].iterrows():
    print(f"  {row['query'][:50]}...")
    print(f"    peak_sharpness: {row['peak_sharpness']:.3f}, max_attn: {row['max_attn']:.4f}")

print("\n[误判的 stayed_correct] (peak_sharpness > 阈值)")
for _, row in stayed_correct[stayed_correct['peak_sharpness'] > best_threshold].iterrows():
    print(f"  {row['query'][:50]}...")
    print(f"    peak_sharpness: {row['peak_sharpness']:.3f}, max_attn: {row['max_attn']:.4f}")
