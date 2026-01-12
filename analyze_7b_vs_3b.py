#!/usr/bin/env python3
"""
分析 7B vs 3B draft model 的 token 选择差异
找出 7B 答对但 3B 答错的案例，分析 token 选择差异
"""
import sys
sys.path.insert(0, '/mnt/data/wjh/FusionRAG')

import pandas as pd
import torch
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
# from per_head_generation import compute_draft_model_attention, entropy_layer_selection

# 加载 CSV 结果
csv_7b = '/mnt/data/reflect/Qwen2.5-7B-Instruct/results/DraftModel_global_topk_10_rate_0.3_draft_Qwen2.5-7B-Instruct.csv'
csv_3b = '/mnt/data/reflect/Qwen2.5-7B-Instruct/results/DraftModel_global_topk_10_rate_0.3_draft_Qwen2.5-3B-Instruct.csv'

df_7b = pd.read_csv(csv_7b)
df_3b = pd.read_csv(csv_3b)

print(f"7B 结果: {len(df_7b)} 条")
print(f"3B 结果: {len(df_3b)} 条")

# 找出 7B 答对但 3B 答错的案例
df_7b['key'] = df_7b['Main Question'] + ' | ' + df_7b['Sub Question']
df_3b['key'] = df_3b['Main Question'] + ' | ' + df_3b['Sub Question']

# Merge
merged = df_7b.merge(df_3b, on='key', suffixes=('_7b', '_3b'))
print(f"匹配的案例: {len(merged)} 条")

# 7B 正确，3B 错误
diff_cases = merged[(merged['Correct_7b'] == True) & (merged['Correct_3b'] == False)]
print(f"\n7B 答对但 3B 答错的案例: {len(diff_cases)} 条")

# 3B 正确，7B 错误
diff_cases_rev = merged[(merged['Correct_7b'] == False) & (merged['Correct_3b'] == True)]
print(f"3B 答对但 7B 答错的案例: {len(diff_cases_rev)} 条")

print("\n" + "="*80)
print("7B 答对但 3B 答错的案例详情:")
print("="*80)

for idx, row in diff_cases.iterrows():
    print(f"\n【案例 {idx+1}】")
    print(f"主问题: {row['Main Question_7b']}")
    print(f"子问题: {row['Sub Question_7b']}")
    print(f"标准答案: {row['Ground Truth_7b']}")
    print(f"7B 预测: {row['Predicted_7b']} (正确)")
    print(f"3B 预测: {row['Predicted_3b']} (错误)")
    print("-" * 40)

# 保存差异案例到 JSON 供后续分析
diff_cases_list = []
for idx, row in diff_cases.iterrows():
    diff_cases_list.append({
        'main_question': row['Main Question_7b'],
        'sub_question': row['Sub Question_7b'],
        'ground_truth': row['Ground Truth_7b'],
        'predicted_7b': row['Predicted_7b'],
        'predicted_3b': row['Predicted_3b'],
        'f1_7b': row['F1_7b'],
        'f1_3b': row['F1_3b'],
    })

with open('diff_cases_7b_vs_3b.json', 'w', encoding='utf-8') as f:
    json.dump(diff_cases_list, f, ensure_ascii=False, indent=2)
print(f"\n差异案例已保存到 diff_cases_7b_vs_3b.json")
