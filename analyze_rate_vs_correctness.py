#!/usr/bin/env python3
"""
分析：什么样的 attention 特征能预测 "低 rate 答错，高 rate 答对"？

核心问题：
- 我们需要找到一个指标，能区分出"需要高 rate"的问题
- 不是所有问题都需要高 rate，只有 token selection 问题才需要
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

# 加载不同 rate 的结果
results_dir = Path('/mnt/data/reflect/Qwen2.5-7B-Instruct/results/musique')

def load_csv(filename):
    """加载 CSV 结果"""
    filepath = results_dir / filename
    if filepath.exists():
        df = pd.read_csv(filepath)
        return df
    return None

# 加载不同配置的结果
configs = {
    'rate_1': 'DraftModel_global_topk_10_rate_1.csv',
    'rate_0.3': 'DraftModel_global_topk_10_rate_0.3_draft_Qwen2.5-3B-Instruct.csv',
    'rate_0.2': 'DraftModel_global_topk_10_rate_0.2_draft_Qwen2.5-3B-Instruct.csv',
}

dfs = {}
for name, filename in configs.items():
    df = load_csv(filename)
    if df is not None:
        dfs[name] = df
        print(f"Loaded {name}: {len(df)} rows, accuracy: {df['Correct'].mean()*100:.2f}%")

if len(dfs) < 2:
    print("Need at least rate_1 and one other rate to compare")
    exit()

# 找出关键问题类型
print("\n" + "="*80)
print("分析：什么问题需要高 rate？")
print("="*80)

# 合并数据
df_rate1 = dfs['rate_1']
df_rate02 = dfs.get('rate_0.2')
df_rate03 = dfs.get('rate_0.3')

# 创建合并 key
df_rate1['key'] = df_rate1['Main Question'] + '|||' + df_rate1['Sub Question']
if df_rate02 is not None:
    df_rate02['key'] = df_rate02['Main Question'] + '|||' + df_rate02['Sub Question']
if df_rate03 is not None:
    df_rate03['key'] = df_rate03['Main Question'] + '|||' + df_rate03['Sub Question']

# 分析 rate=0.2 vs rate=1
if df_rate02 is not None:
    print("\n--- Rate=0.2 vs Rate=1 分析 ---")

    merged = df_rate1.merge(df_rate02, on='key', suffixes=('_r1', '_r02'))

    # 转换 Correct 列为布尔值
    merged['Correct_r1'] = merged['Correct_r1'].astype(str).str.lower().isin(['true', 'yes', '1'])
    merged['Correct_r02'] = merged['Correct_r02'].astype(str).str.lower().isin(['true', 'yes', '1'])

    # 分类问题
    # Type A: rate=1 对，rate=0.2 也对 → 不需要高 rate
    type_a = merged[(merged['Correct_r1'] == True) & (merged['Correct_r02'] == True)]
    # Type B: rate=1 对，rate=0.2 错 → 需要高 rate（token selection 问题）
    type_b = merged[(merged['Correct_r1'] == True) & (merged['Correct_r02'] == False)]
    # Type C: rate=1 错，rate=0.2 也错 → rate 再高也没用
    type_c = merged[(merged['Correct_r1'] == False) & (merged['Correct_r02'] == False)]
    # Type D: rate=1 错，rate=0.2 对 → 奇怪情况（噪声）
    type_d = merged[(merged['Correct_r1'] == False) & (merged['Correct_r02'] == True)]

    print(f"\nType A (都对，不需要高rate): {len(type_a)} ({len(type_a)/len(merged)*100:.1f}%)")
    print(f"Type B (r1对r0.2错，需要高rate): {len(type_b)} ({len(type_b)/len(merged)*100:.1f}%)")
    print(f"Type C (都错，rate再高也没用): {len(type_c)} ({len(type_c)/len(merged)*100:.1f}%)")
    print(f"Type D (r1错r0.2对，噪声): {len(type_d)} ({len(type_d)/len(merged)*100:.1f}%)")

    # 关键发现
    print(f"\n关键发现:")
    print(f"  - 真正需要高 rate 的问题只有 {len(type_b)} 个 ({len(type_b)/len(merged)*100:.1f}%)")
    print(f"  - 如果我们能准确识别这些问题，只对它们用高 rate")
    print(f"  - 其他问题用低 rate，就能节省计算")

    # 保存 Type B 问题用于后续分析
    type_b_questions = type_b[['key', 'Sub Question_r1', 'Ground Truth_r1', 'Predicted_r1', 'Predicted_r02']].copy()
    type_b_questions.to_csv('/mnt/data/wjh/FusionRAG/type_b_questions.csv', index=False)
    print(f"\n  Type B 问题已保存到 type_b_questions.csv")

# 如果有 rate=0.3 的结果，也分析一下
if df_rate03 is not None:
    print("\n--- Rate=0.3 vs Rate=1 分析 ---")

    merged = df_rate1.merge(df_rate03, on='key', suffixes=('_r1', '_r03'))
    merged['Correct_r1'] = merged['Correct_r1'].astype(str).str.lower().isin(['true', 'yes', '1'])
    merged['Correct_r03'] = merged['Correct_r03'].astype(str).str.lower().isin(['true', 'yes', '1'])

    type_a = merged[(merged['Correct_r1'] == True) & (merged['Correct_r03'] == True)]
    type_b = merged[(merged['Correct_r1'] == True) & (merged['Correct_r03'] == False)]
    type_c = merged[(merged['Correct_r1'] == False) & (merged['Correct_r03'] == False)]
    type_d = merged[(merged['Correct_r1'] == False) & (merged['Correct_r03'] == True)]

    print(f"\nType A (都对): {len(type_a)} ({len(type_a)/len(merged)*100:.1f}%)")
    print(f"Type B (r1对r0.3错): {len(type_b)} ({len(type_b)/len(merged)*100:.1f}%)")
    print(f"Type C (都错): {len(type_c)} ({len(type_c)/len(merged)*100:.1f}%)")
    print(f"Type D (r1错r0.3对): {len(type_d)} ({len(type_d)/len(merged)*100:.1f}%)")

# 分析 rate=0.2 vs rate=0.3 的改善
if df_rate02 is not None and df_rate03 is not None:
    print("\n--- Rate=0.2 vs Rate=0.3 改善分析 ---")

    merged = df_rate02.merge(df_rate03, on='key', suffixes=('_r02', '_r03'))
    merged['Correct_r02'] = merged['Correct_r02'].astype(str).str.lower().isin(['true', 'yes', '1'])
    merged['Correct_r03'] = merged['Correct_r03'].astype(str).str.lower().isin(['true', 'yes', '1'])

    # 哪些问题通过提高 rate 从 0.2 到 0.3 后答对了？
    improved = merged[(merged['Correct_r02'] == False) & (merged['Correct_r03'] == True)]
    degraded = merged[(merged['Correct_r02'] == True) & (merged['Correct_r03'] == False)]

    print(f"\n提高 rate 0.2→0.3 后:")
    print(f"  - 改善 (r0.2错→r0.3对): {len(improved)} 个问题")
    print(f"  - 退化 (r0.2对→r0.3错): {len(degraded)} 个问题")
    print(f"  - 净改善: {len(improved) - len(degraded)} 个问题")

    if len(improved) > 0:
        print(f"\n改善的问题 (这些是真正需要更高 rate 的):")
        for _, row in improved.head(10).iterrows():
            print(f"  - {row['Sub Question_r02'][:60]}...")

print("\n" + "="*80)
print("结论")
print("="*80)
print("""
要设计有效的动态 rate 策略，我们需要：
1. 识别 Type B 问题（rate=1 对，低rate错）
2. 分析这些问题的 attention 特征
3. 找到能区分 Type A 和 Type B 的指标

下一步：用 3B 模型计算 Type A 和 Type B 问题的 attention 特征，
看看是否有显著差异。
""")
