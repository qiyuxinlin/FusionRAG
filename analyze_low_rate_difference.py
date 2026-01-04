#!/usr/bin/env python3
"""
分析 rate=0.05 能答对 vs 不能答对的例子的 attention 特征差异

目标：找出什么样的 attention 特征决定了问题是否需要高重算比例
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import json
from collections import defaultdict

sys.path.insert(0, '/mnt/data/wjh/FusionRAG')

from transformers import AutoTokenizer, AutoConfig
from test_fusionrag_reflect import load_model, prepare_reflect_data
from ktransformers.util.utils import (
    compute_draft_model_attention,
    entropy_layer_selection,
)


def compute_attention_features(attention_scores, doc_len):
    """
    计算 attention 分布的各种特征
    """
    # 归一化
    attn = np.array(attention_scores)
    if attn.sum() > 0:
        attn = attn / attn.sum()

    features = {}

    # 1. Coverage metrics - 达到 X% 覆盖需要多少比例的 token
    sorted_attn = np.sort(attn)[::-1]
    cumsum = np.cumsum(sorted_attn)

    for threshold in [0.50, 0.70, 0.80, 0.85, 0.90, 0.95]:
        tokens_needed = np.searchsorted(cumsum, threshold) + 1
        features[f'cov_{int(threshold*100)}'] = tokens_needed / doc_len

    # 2. Top-k concentration
    features['top5_ratio'] = sorted_attn[:5].sum()
    features['top10_ratio'] = sorted_attn[:10].sum()
    features['top20_ratio'] = sorted_attn[:20].sum()
    features['top50_ratio'] = sorted_attn[:50].sum()
    features['top100_ratio'] = sorted_attn[:100].sum() if len(sorted_attn) >= 100 else sorted_attn.sum()

    # 3. Statistical measures
    features['max_attn'] = float(attn.max())
    features['mean_attn'] = float(attn.mean())
    features['std_attn'] = float(attn.std())

    # 4. Gini coefficient (attention 集中度)
    n = len(attn)
    if n > 0 and attn.sum() > 0:
        sorted_attn_gini = np.sort(attn)
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_attn_gini)) / (n * np.sum(sorted_attn_gini)) - (n + 1) / n
        features['gini'] = float(gini)
    else:
        features['gini'] = 0.0

    # 5. Normalized entropy
    if attn.sum() > 0:
        attn_prob = attn / attn.sum()
        attn_prob = attn_prob[attn_prob > 0]  # 避免 log(0)
        entropy = -np.sum(attn_prob * np.log(attn_prob))
        max_entropy = np.log(len(attn))
        features['norm_entropy'] = float(entropy / max_entropy) if max_entropy > 0 else 0
    else:
        features['norm_entropy'] = 0.0

    # 6. Sparsity (比例低于平均值的 token 数量)
    mean_val = attn.mean()
    features['sparsity'] = float((attn < mean_val).sum() / len(attn))

    # 7. Peak sharpness (最大值与第二大值的比值)
    if len(sorted_attn) >= 2:
        features['peak_sharpness'] = float(sorted_attn[0] / sorted_attn[1]) if sorted_attn[1] > 0 else float('inf')
    else:
        features['peak_sharpness'] = 1.0

    # 8. Document length
    features['doc_len'] = doc_len

    return features


def analyze_questions(
    model,
    tokenizer,
    questions_data,
    system_tensor,
    results_df,
    device='cuda:0'
):
    """
    分析每个问题的 attention 特征
    """
    all_features = []

    # 创建问题到结果的映射
    question_results = {}
    for _, row in results_df.iterrows():
        sub_q = row['Sub Question']
        question_results[sub_q] = {
            'correct_005': row['Correct'],
            'correct_1': row['Rate1_Correct'],
        }

    for q_idx, q_data in enumerate(questions_data):
        main_question = q_data['main_question']
        doc_tensors = q_data['doc_tensors']

        for sub_idx, sub_q_info in enumerate(q_data['sub_questions']):
            sub_question = sub_q_info['query']

            # 检查是否在结果中
            result_info = question_results.get(sub_question)
            if result_info is None:
                continue

            print(f"\nProcessing: {sub_question[:50]}...")

            # 准备输入
            doc_chunk_ids = sub_q_info['chunk_ids']
            sub_q_doc_tensors = [doc_tensors[chunk_id - 1] for chunk_id in doc_chunk_ids]

            question_text = f"<|im_end|>\n<|im_start|>user\nQuestion: {sub_q_info['query']}<|im_end|>\n<|im_start|>assistant\nAnswer: "
            question_tokens = tokenizer.encode(question_text, add_special_tokens=False)
            question_tensor = torch.tensor(question_tokens, dtype=torch.long)

            system_len = system_tensor.shape[0]
            doc_len = sum(t.shape[0] for t in sub_q_doc_tensors)

            # 计算 attention
            all_tokens = [system_tensor] + sub_q_doc_tensors + [question_tensor]
            full_input = torch.cat(all_tokens).unsqueeze(0).to(device)
            query_start = system_len + doc_len

            try:
                oracle_attention = compute_draft_model_attention(model, full_input, query_start, device)
            except Exception as e:
                print(f"  Error computing attention: {e}")
                continue

            # 提取文档部分的 attention
            text_block1_len = sub_q_doc_tensors[0].shape[0]
            effective_doc_len = doc_len - text_block1_len
            selection_start = system_len + text_block1_len

            # 聚合所有层的 attention
            layer_attention_dict = {}
            for layer_idx, layer_attn in oracle_attention.items():
                query_to_doc = layer_attn[:, :, selection_start:selection_start + effective_doc_len]
                doc_attention_avg = query_to_doc.mean(axis=(0, 1))
                layer_attention_dict[layer_idx] = torch.tensor(doc_attention_avg, device=device)

            # 熵选层
            active_layers, _ = entropy_layer_selection(layer_attention_dict, top_k=4, return_entropy=True)

            # 聚合选定层的 attention
            layer_attentions = [layer_attention_dict[idx] for idx in active_layers]
            aggregated_attn = torch.stack(layer_attentions).mean(dim=0).cpu().numpy()

            # 计算特征
            features = compute_attention_features(aggregated_attn, effective_doc_len)
            features['query'] = sub_question
            features['main_question'] = main_question
            features['correct_005'] = result_info['correct_005']
            features['correct_1'] = result_info['correct_1']

            # 分类
            if result_info['correct_1'] and result_info['correct_005']:
                features['category'] = 'stayed_correct'
            elif result_info['correct_1'] and not result_info['correct_005']:
                features['category'] = 'became_wrong'
            elif not result_info['correct_1'] and not result_info['correct_005']:
                features['category'] = 'stayed_wrong'
            else:
                features['category'] = 'became_correct'

            all_features.append(features)
            print(f"  Category: {features['category']}, cov_85: {features['cov_85']:.3f}, gini: {features['gini']:.3f}")

    return all_features


def compare_groups(features_list):
    """
    比较不同组的特征差异
    """
    df = pd.DataFrame(features_list)

    print("\n" + "="*100)
    print("GROUP COMPARISON: stayed_correct vs became_wrong")
    print("="*100)

    stayed_correct = df[df['category'] == 'stayed_correct']
    became_wrong = df[df['category'] == 'became_wrong']

    print(f"\nSample sizes:")
    print(f"  stayed_correct: {len(stayed_correct)}")
    print(f"  became_wrong: {len(became_wrong)}")

    # 比较各个特征
    feature_cols = ['cov_50', 'cov_70', 'cov_80', 'cov_85', 'cov_90', 'cov_95',
                    'top5_ratio', 'top10_ratio', 'top20_ratio', 'top50_ratio',
                    'max_attn', 'gini', 'norm_entropy', 'sparsity', 'peak_sharpness', 'doc_len']

    print(f"\n{'Feature':<20} {'Stayed Correct':<20} {'Became Wrong':<20} {'Diff':<15} {'Separation':<15}")
    print("-" * 90)

    separation_scores = {}

    for col in feature_cols:
        if col not in df.columns:
            continue

        sc_mean = stayed_correct[col].mean()
        sc_std = stayed_correct[col].std()
        bw_mean = became_wrong[col].mean()
        bw_std = became_wrong[col].std()

        diff = bw_mean - sc_mean

        # 计算分离度 (用于判断这个特征能否区分两组)
        pooled_std = np.sqrt((sc_std**2 + bw_std**2) / 2)
        separation = abs(diff) / pooled_std if pooled_std > 0 else 0
        separation_scores[col] = separation

        print(f"{col:<20} {sc_mean:>8.4f} ± {sc_std:<8.4f} {bw_mean:>8.4f} ± {bw_std:<8.4f} {diff:>+10.4f}  {separation:>10.4f}")

    # 找出最有区分度的特征
    print("\n" + "="*100)
    print("MOST DISCRIMINATIVE FEATURES (sorted by separation score)")
    print("="*100)

    sorted_features = sorted(separation_scores.items(), key=lambda x: x[1], reverse=True)
    for feat, score in sorted_features[:10]:
        print(f"  {feat}: {score:.4f}")

    return df, separation_scores


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--max_examples', type=int, default=None)
    args = parser.parse_args()

    print("="*100)
    print("Analyzing rate=0.05 correct vs incorrect examples")
    print("="*100)

    # 读取 rate=0.05 结果
    results_df = pd.read_csv('/mnt/data/reflect/Qwen2.5-7B-Instruct/results/DraftModel_global_topk_10_rate_0.05_revert_rope.csv')
    print(f"Loaded {len(results_df)} results")

    # 加载模型
    model_path = '/mnt/data/models/Qwen2.5-7B-Instruct'
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model, device_map = load_model('qwen', model_path, config, args.device, use_multi_gpu=True)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # 加载数据
    max_questions = args.max_examples if args.max_examples else None
    questions_data, system_tensor, _, _ = prepare_reflect_data(
        '/mnt/data/wjh/FusionRAG/result_reflect.json',
        tokenizer,
        '/mnt/data/models/bge-m3',
        'qwen',
        10,
        max_main_questions=max_questions,
        preprocess=False
    )

    # 分析问题
    features_list = analyze_questions(
        model, tokenizer, questions_data, system_tensor, results_df, args.device
    )

    # 保存原始数据
    with open('./low_rate_analysis_features.json', 'w') as f:
        json.dump(features_list, f, indent=2, ensure_ascii=False, default=str)

    # 比较各组
    df, separation_scores = compare_groups(features_list)

    # 保存分析结果
    analysis_result = {
        'features': features_list,
        'separation_scores': separation_scores
    }
    with open('./low_rate_analysis_result.json', 'w') as f:
        json.dump(analysis_result, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nResults saved to ./low_rate_analysis_features.json and ./low_rate_analysis_result.json")


if __name__ == '__main__':
    main()
