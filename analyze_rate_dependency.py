#!/usr/bin/env python3
"""
分析不同重算比例下的结果差异，找出问题难度与所需rate的关系。
"""

import pandas as pd
import os
from collections import defaultdict

RESULTS_DIR = "/mnt/data/reflect/Qwen2.5-7B-Instruct/results"

def load_csv_results(filepath):
    """加载CSV结果文件"""
    df = pd.read_csv(filepath)
    return df

def compare_rates():
    """比较不同rate下的结果"""

    # 加载DraftModel的结果
    draft_03 = load_csv_results(os.path.join(RESULTS_DIR, "DraftModel_global_topk_10_rate_0.3.csv"))
    draft_02 = load_csv_results(os.path.join(RESULTS_DIR, "DraftModel_global_topk_10_rate_0.2.csv"))

    # Rate=1 baseline (从rate_0.3的文件中提取，因为它包含Rate1的结果)
    print("="*80)
    print("DraftModel Rate=0.3 vs Rate=1 分析")
    print("="*80)

    # 检查列名
    print("\n列名:", list(draft_03.columns))

    # 比较 rate=0.3 的 Correct 和 Rate1_Correct
    if 'Correct' in draft_03.columns and 'Rate1_Correct' in draft_03.columns:
        # 转换为布尔值
        draft_03['Correct'] = draft_03['Correct'].apply(lambda x: x == True or x == 'True')
        draft_03['Rate1_Correct'] = draft_03['Rate1_Correct'].apply(lambda x: x == True or x == 'True')

        # 分类问题
        both_correct = draft_03[(draft_03['Correct'] == True) & (draft_03['Rate1_Correct'] == True)]
        both_wrong = draft_03[(draft_03['Correct'] == False) & (draft_03['Rate1_Correct'] == False)]
        only_rate1_correct = draft_03[(draft_03['Correct'] == False) & (draft_03['Rate1_Correct'] == True)]
        only_rate03_correct = draft_03[(draft_03['Correct'] == True) & (draft_03['Rate1_Correct'] == False)]

        print(f"\n总问题数: {len(draft_03)}")
        print(f"两种rate都答对: {len(both_correct)} ({len(both_correct)/len(draft_03)*100:.1f}%)")
        print(f"两种rate都答错: {len(both_wrong)} ({len(both_wrong)/len(draft_03)*100:.1f}%)")
        print(f"只有Rate=1答对: {len(only_rate1_correct)} ({len(only_rate1_correct)/len(draft_03)*100:.1f}%)")
        print(f"只有Rate=0.3答对: {len(only_rate03_correct)} ({len(only_rate03_correct)/len(draft_03)*100:.1f}%)")

        print("\n" + "="*80)
        print("需要更高重算比例的问题（Rate=0.3错，Rate=1对）:")
        print("="*80)
        for idx, row in only_rate1_correct.iterrows():
            print(f"\n问题 {idx}:")
            print(f"  主问题: {row['Main Question'][:80]}...")
            print(f"  子问题: {row['Sub Question'][:80]}...")
            print(f"  Rate=0.3 预测: {str(row['Predicted'])[:80]}...")
            print(f"  Rate=1.0 预测: {str(row['Rate1_Predicted'])[:80]}...")
            print(f"  标准答案: {str(row['Ground Truth'])[:80]}...")

        return only_rate1_correct

def compare_02_vs_03():
    """比较rate=0.2和rate=0.3"""
    print("\n" + "="*80)
    print("DraftModel Rate=0.2 vs Rate=0.3 分析")
    print("="*80)

    draft_02 = load_csv_results(os.path.join(RESULTS_DIR, "DraftModel_global_topk_10_rate_0.2.csv"))
    draft_03 = load_csv_results(os.path.join(RESULTS_DIR, "DraftModel_global_topk_10_rate_0.3.csv"))

    # 使用Sub Question作为key来匹配
    draft_02['Correct'] = draft_02['Correct'].apply(lambda x: x == True or x == 'True')
    draft_03['Correct'] = draft_03['Correct'].apply(lambda x: x == True or x == 'True')

    # 合并数据
    merged = draft_02.merge(draft_03, on=['Main Question', 'Sub Question'],
                            suffixes=('_02', '_03'))

    # 分类
    improved_03 = merged[(merged['Correct_02'] == False) & (merged['Correct_03'] == True)]
    regressed_03 = merged[(merged['Correct_02'] == True) & (merged['Correct_03'] == False)]

    print(f"\n总匹配问题数: {len(merged)}")
    print(f"Rate=0.2错 → Rate=0.3对: {len(improved_03)} 个问题")
    print(f"Rate=0.2对 → Rate=0.3错: {len(regressed_03)} 个问题")

    if len(improved_03) > 0:
        print("\n需要>=0.3 rate的问题:")
        for idx, row in improved_03.iterrows():
            print(f"  - {row['Sub Question'][:60]}...")

    return improved_03, regressed_03

def check_oracle_small_samples():
    """检查Oracle的小样本测试结果"""
    print("\n" + "="*80)
    print("Oracle 不同rate测试结果分析（小样本）")
    print("="*80)

    rates = ['0.0', '0.1', '0.15', '0.2', '0.25', '0.3']
    results = {}

    for rate in rates:
        filepath = os.path.join(RESULTS_DIR, f"Oracle_global_topk_10_rate_{rate}_revert_rope.csv")
        if os.path.exists(filepath):
            df = load_csv_results(filepath)
            df['Correct'] = df['Correct'].apply(lambda x: x == True or x == 'True')
            results[rate] = df
            print(f"\nRate={rate}: {len(df)} 个样本, 正确率 {df['Correct'].mean()*100:.1f}%")

    # 比较每个问题在不同rate下的表现
    if len(results) > 1:
        print("\n各问题在不同rate下的表现:")
        base_rate = '0.3'
        if base_rate in results:
            for idx, row in results[base_rate].iterrows():
                sub_q = row['Sub Question'][:50]
                status = []
                for rate in rates:
                    if rate in results:
                        match = results[rate][results[rate]['Sub Question'] == row['Sub Question']]
                        if len(match) > 0:
                            correct = match.iloc[0]['Correct']
                            status.append(f"r{rate}:{'✓' if correct else '✗'}")
                print(f"  {sub_q}... -> {' '.join(status)}")

if __name__ == "__main__":
    hard_questions = compare_rates()
    print("\n")
    improved, regressed = compare_02_vs_03()
    check_oracle_small_samples()
