#!/usr/bin/env python3
"""
分析除 attention 之外的其他特征，看哪些能区分 Easy 和 Medium 问题。

候选特征：
1. 问题类型（who, what, when, where, why, how）
2. 问题长度
3. 文档数量和长度
4. Query-Document 语义相似度
5. 答案在文档中的分布
"""

import os
import sys
import numpy as np
import pandas as pd
import json
import re
from collections import Counter

sys.path.insert(0, '/mnt/data/wjh/FusionRAG')

RESULTS_DIR = "/mnt/data/reflect/Qwen2.5-7B-Instruct/results"


def load_and_categorize():
    """加载结果并分类"""
    files = {
        'rate_0.05': 'DraftModel_global_topk_10_rate_0.05_revert_rope.csv',
        'rate_0.3': 'DraftModel_global_topk_10_rate_0.3.csv',
    }

    results = {}
    for key, filename in files.items():
        filepath = os.path.join(RESULTS_DIR, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            results[key] = df

    df_005 = results.get('rate_0.05')
    df_03 = results.get('rate_0.3')

    if df_005 is None or df_03 is None:
        print("Missing result files!")
        return None, None

    min_rows = min(len(df_005), len(df_03))

    categories = {'easy': [], 'medium': [], 'hard': []}
    seen = set()

    for i in range(min_rows):
        sub_q = str(df_005.iloc[i]['Sub Question'])
        q_key = sub_q[:80]
        if q_key in seen:
            continue
        seen.add(q_key)

        correct_005 = df_005.iloc[i]['Correct']
        correct_03 = df_03.iloc[i]['Correct']

        info = {
            'idx': i,
            'sub_q': sub_q,
            'main_q': str(df_005.iloc[i]['Main Question']),
            'ground_truth': str(df_005.iloc[i]['Ground Truth']),
        }

        if correct_005:
            categories['easy'].append(info)
        elif correct_03:
            categories['medium'].append(info)
        else:
            categories['hard'].append(info)

    return categories, results


def extract_question_type(question):
    """提取问题类型"""
    q_lower = question.lower().strip()

    # 常见疑问词
    if q_lower.startswith('who ') or ' who ' in q_lower:
        return 'who'
    elif q_lower.startswith('what ') or ' what ' in q_lower:
        return 'what'
    elif q_lower.startswith('when ') or ' when ' in q_lower:
        return 'when'
    elif q_lower.startswith('where ') or ' where ' in q_lower:
        return 'where'
    elif q_lower.startswith('why ') or ' why ' in q_lower:
        return 'why'
    elif q_lower.startswith('how ') or ' how ' in q_lower:
        return 'how'
    elif q_lower.startswith('which ') or ' which ' in q_lower:
        return 'which'
    else:
        return 'other'


def extract_relation_keywords(question):
    """提取关系关键词"""
    q_lower = question.lower()

    relation_keywords = [
        'sibling', 'brother', 'sister', 'parent', 'child', 'son', 'daughter',
        'spouse', 'wife', 'husband', 'married', 'mother', 'father',
        'grandfather', 'grandmother', 'uncle', 'aunt', 'cousin', 'nephew', 'niece',
        'born', 'died', 'death', 'birth', 'founded', 'created', 'established',
        'member', 'part of', 'belongs to', 'located', 'headquarter',
        'crowned', 'elected', 'appointed', 'succeeded'
    ]

    found = []
    for kw in relation_keywords:
        if kw in q_lower:
            found.append(kw)

    return found


def analyze_question_features(categories):
    """分析问题特征"""
    print("\n" + "=" * 70)
    print("QUESTION TYPE ANALYSIS")
    print("=" * 70)

    for cat in ['easy', 'medium', 'hard']:
        questions = categories[cat]
        if not questions:
            continue

        # 问题类型分布
        q_types = [extract_question_type(q['sub_q']) for q in questions]
        type_counts = Counter(q_types)

        print(f"\n{cat.upper()} (n={len(questions)}):")
        print("  Question types:")
        for qtype, count in type_counts.most_common():
            pct = count / len(questions) * 100
            print(f"    {qtype}: {count} ({pct:.1f}%)")

        # 问题长度
        q_lens = [len(q['sub_q'].split()) for q in questions]
        print(f"  Question length: mean={np.mean(q_lens):.1f}, std={np.std(q_lens):.1f}")

        # 关系关键词
        all_keywords = []
        for q in questions:
            all_keywords.extend(extract_relation_keywords(q['sub_q']))
        if all_keywords:
            kw_counts = Counter(all_keywords)
            print(f"  Top relation keywords: {kw_counts.most_common(5)}")


def analyze_with_original_data(categories):
    """结合原始数据分析文档特征"""
    print("\n" + "=" * 70)
    print("DOCUMENT FEATURES ANALYSIS")
    print("=" * 70)

    with open('/mnt/data/wjh/FusionRAG/result_reflect.json') as f:
        data = json.load(f)

    # 构建问题到数据的映射
    sub_q_to_data = {}
    for i, item in enumerate(data):
        intermediate = item.get('intermediate_context', [])
        for sub_idx, sub_q in enumerate(intermediate):
            query = sub_q.get('query', '')
            if query.startswith("Intermediate query"):
                colon_pos = query.find(":")
                if colon_pos != -1:
                    query = query[colon_pos + 1:].strip()
            sub_q_to_data[query[:80]] = sub_q

    for cat in ['easy', 'medium']:
        questions = categories[cat]
        if not questions:
            continue

        doc_counts = []
        doc_lens = []
        answer_positions = []  # 答案在文档中的位置

        for q in questions:
            sub_q = q['sub_q']
            ground_truth = q['ground_truth']

            # 查找对应数据
            found_data = None
            for key, sub_q_data in sub_q_to_data.items():
                if key in sub_q or sub_q[:60] in key:
                    found_data = sub_q_data
                    break

            if found_data:
                docs = found_data.get('retrieve docs', [])[:10]
                doc_counts.append(len(docs))

                total_len = sum(len(doc.split()) for doc in docs)
                doc_lens.append(total_len)

                # 查找答案在哪个文档中
                gt_lower = ground_truth.lower()[:50]
                for idx, doc in enumerate(docs):
                    if gt_lower in doc.lower():
                        answer_positions.append(idx)
                        break
                else:
                    answer_positions.append(-1)  # 答案不在文档中

        print(f"\n{cat.upper()} (matched={len(doc_counts)}):")
        if doc_counts:
            print(f"  Doc count: mean={np.mean(doc_counts):.1f}")
            print(f"  Total doc length (words): mean={np.mean(doc_lens):.0f}, std={np.std(doc_lens):.0f}")

            # 答案位置分析
            found_count = sum(1 for p in answer_positions if p >= 0)
            print(f"  Answer found in docs: {found_count}/{len(answer_positions)} ({found_count/len(answer_positions)*100:.1f}%)")

            if [p for p in answer_positions if p >= 0]:
                valid_positions = [p for p in answer_positions if p >= 0]
                print(f"  Answer position (when found): mean={np.mean(valid_positions):.1f}")


def compare_specific_features(categories):
    """比较特定特征"""
    print("\n" + "=" * 70)
    print("FEATURE COMPARISON: Easy vs Medium")
    print("=" * 70)

    easy_qs = categories['easy']
    medium_qs = categories['medium']

    # 问题类型失败率
    print("\n1. Question Type Failure Rate:")
    all_qs = easy_qs + medium_qs
    type_stats = {}

    for q in all_qs:
        qtype = extract_question_type(q['sub_q'])
        if qtype not in type_stats:
            type_stats[qtype] = {'easy': 0, 'medium': 0}

    for q in easy_qs:
        qtype = extract_question_type(q['sub_q'])
        type_stats[qtype]['easy'] += 1

    for q in medium_qs:
        qtype = extract_question_type(q['sub_q'])
        type_stats[qtype]['medium'] += 1

    print(f"  {'Type':<10} {'Easy':>8} {'Medium':>8} {'Failure%':>10}")
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*10}")

    sorted_types = sorted(type_stats.items(),
                         key=lambda x: x[1]['medium']/(x[1]['easy']+x[1]['medium']+0.01),
                         reverse=True)

    for qtype, stats in sorted_types:
        total = stats['easy'] + stats['medium']
        failure_rate = stats['medium'] / total * 100 if total > 0 else 0
        print(f"  {qtype:<10} {stats['easy']:>8} {stats['medium']:>8} {failure_rate:>9.1f}%")

    # 关系关键词失败率
    print("\n2. Relation Keyword Failure Rate:")

    keyword_stats = {}
    for q in all_qs:
        keywords = extract_relation_keywords(q['sub_q'])
        for kw in keywords:
            if kw not in keyword_stats:
                keyword_stats[kw] = {'easy': 0, 'medium': 0}

    for q in easy_qs:
        keywords = extract_relation_keywords(q['sub_q'])
        for kw in keywords:
            keyword_stats[kw]['easy'] += 1

    for q in medium_qs:
        keywords = extract_relation_keywords(q['sub_q'])
        for kw in keywords:
            keyword_stats[kw]['medium'] += 1

    # 只显示出现次数 >= 3 的关键词
    filtered_keywords = {k: v for k, v in keyword_stats.items()
                        if v['easy'] + v['medium'] >= 3}

    if filtered_keywords:
        sorted_kws = sorted(filtered_keywords.items(),
                           key=lambda x: x[1]['medium']/(x[1]['easy']+x[1]['medium']+0.01),
                           reverse=True)

        print(f"  {'Keyword':<15} {'Easy':>6} {'Medium':>8} {'Failure%':>10}")
        print(f"  {'-'*15} {'-'*6} {'-'*8} {'-'*10}")

        for kw, stats in sorted_kws[:15]:
            total = stats['easy'] + stats['medium']
            failure_rate = stats['medium'] / total * 100 if total > 0 else 0
            print(f"  {kw:<15} {stats['easy']:>6} {stats['medium']:>8} {failure_rate:>9.1f}%")

    # 问题长度分析
    print("\n3. Question Length Analysis:")
    easy_lens = [len(q['sub_q'].split()) for q in easy_qs]
    medium_lens = [len(q['sub_q'].split()) for q in medium_qs]

    print(f"  Easy:   mean={np.mean(easy_lens):.1f}, std={np.std(easy_lens):.1f}")
    print(f"  Medium: mean={np.mean(medium_lens):.1f}, std={np.std(medium_lens):.1f}")

    # 分离度
    all_lens = easy_lens + medium_lens
    sep = abs(np.mean(easy_lens) - np.mean(medium_lens)) / (np.std(all_lens) + 0.01)
    print(f"  Separation: {sep:.3f}")


def main():
    categories, results = load_and_categorize()
    if categories is None:
        return

    total = sum(len(v) for v in categories.values())
    print(f"Total questions (deduplicated): {total}")
    print(f"  Easy: {len(categories['easy'])} ({len(categories['easy'])/total*100:.1f}%)")
    print(f"  Medium: {len(categories['medium'])} ({len(categories['medium'])/total*100:.1f}%)")
    print(f"  Hard: {len(categories['hard'])} ({len(categories['hard'])/total*100:.1f}%)")

    analyze_question_features(categories)
    analyze_with_original_data(categories)
    compare_specific_features(categories)

    # 总结建议
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    print("""
基于以上分析，可以考虑的额外特征：

1. 问题类型 (Question Type)
   - 某些类型（如 relation/sibling）可能更需要重算

2. 关系关键词 (Relation Keywords)
   - 包含 "sibling", "spouse", "child" 等词的问题可能更难

3. 问题长度 (Question Length)
   - 可能与难度相关

4. 答案位置 (Answer Position)
   - 答案不在 top 文档中的问题可能需要更多重算

建议的混合公式：
  base_rate = gini_based_rate  # 基于 attention 的动态 rate

  # 基于问题特征的调整
  if has_relation_keywords:
      rate *= 1.2  # 增加 20%
  if question_type in ['who', 'when']:  # 需要精确匹配的类型
      rate *= 1.1

  rate = clip(rate, min_rate, max_rate)
""")


if __name__ == "__main__":
    main()
