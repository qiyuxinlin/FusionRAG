#!/usr/bin/env python3
"""
分析 Type A 和 Type B 问题的 attention 特征差异

目标：找到能区分"不需要高rate"和"需要高rate"的指标
"""

import json
import pandas as pd
import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from test_fusionrag_reflect import prepare_reflect_data
from pathlib import Path

def compute_attention_features(attn_weights, doc_start, doc_end):
    """计算 attention 分布特征"""
    num_layers = attn_weights.shape[0]
    query_start = doc_end

    q2d_attn = attn_weights[:, :, query_start:, doc_start:doc_end]

    if q2d_attn.shape[2] == 0 or q2d_attn.shape[3] == 0:
        return None

    # 对 heads 和 query positions 取平均
    layer_attn = q2d_attn.mean(dim=(1, 2))  # [num_layers, doc_len]
    avg_attn = layer_attn.mean(dim=0)  # [doc_len]

    features = {}
    doc_len = len(avg_attn)

    # 1. 归一化熵
    p = avg_attn / (avg_attn.sum() + 1e-10)
    p = torch.clamp(p, min=1e-10)
    entropy = -(p * torch.log(p)).sum()
    max_entropy = np.log(doc_len)
    features['normalized_entropy'] = (entropy / max_entropy).item()

    # 2. Gini 系数
    sorted_attn = torch.sort(avg_attn)[0]
    n = len(sorted_attn)
    index = torch.arange(1, n + 1, dtype=torch.float32, device=sorted_attn.device)
    gini = ((2 * index - n - 1) * sorted_attn).sum() / (n * sorted_attn.sum() + 1e-10)
    features['gini'] = gini.item()

    # 3. Peak strength
    features['peak_strength'] = avg_attn.max().item()

    # 4. Coverage ratios
    sorted_desc, _ = torch.sort(avg_attn, descending=True)
    cumsum = torch.cumsum(sorted_desc, dim=0)
    total = cumsum[-1]

    for target in [0.5, 0.8, 0.9, 0.95]:
        coverage = (cumsum >= target * total).nonzero(as_tuple=True)[0]
        if len(coverage) > 0:
            k = coverage[0].item() + 1
        else:
            k = doc_len
        features[f'coverage_{int(target*100)}'] = k / doc_len

    # 5. Top-k 集中度
    for k in [10, 50, 100]:
        if doc_len >= k:
            features[f'top{k}_concentration'] = sorted_desc[:k].sum().item() / total.item()
        else:
            features[f'top{k}_concentration'] = 1.0

    # 6. 层间一致性
    top_k = min(50, doc_len)
    layer_top_tokens = []
    for i in range(num_layers):
        top_indices = torch.topk(layer_attn[i], top_k).indices
        layer_top_tokens.append(set(top_indices.tolist()))

    consistency_scores = []
    for i in range(num_layers - 1):
        intersection = len(layer_top_tokens[i] & layer_top_tokens[i+1])
        union = len(layer_top_tokens[i] | layer_top_tokens[i+1])
        if union > 0:
            consistency_scores.append(intersection / union)

    features['layer_consistency'] = np.mean(consistency_scores) if consistency_scores else 0.5

    return features


def main():
    device = "cuda:0"

    # 加载分类结果
    results_dir = Path('/mnt/data/reflect/Qwen2.5-7B-Instruct/results/musique')

    df_rate1 = pd.read_csv(results_dir / 'DraftModel_global_topk_10_rate_1.csv')
    df_rate02 = pd.read_csv(results_dir / 'DraftModel_global_topk_10_rate_0.2_draft_Qwen2.5-3B-Instruct.csv')

    df_rate1['key'] = df_rate1['Main Question'] + '|||' + df_rate1['Sub Question']
    df_rate02['key'] = df_rate02['Main Question'] + '|||' + df_rate02['Sub Question']

    merged = df_rate1.merge(df_rate02, on='key', suffixes=('_r1', '_r02'))
    merged['Correct_r1'] = merged['Correct_r1'].astype(str).str.lower().isin(['true', 'yes', '1'])
    merged['Correct_r02'] = merged['Correct_r02'].astype(str).str.lower().isin(['true', 'yes', '1'])

    # 分类
    type_a = merged[(merged['Correct_r1'] == True) & (merged['Correct_r02'] == True)]
    type_b = merged[(merged['Correct_r1'] == True) & (merged['Correct_r02'] == False)]

    print(f"Type A (不需要高rate): {len(type_a)}")
    print(f"Type B (需要高rate): {len(type_b)}")

    # 加载 3B 模型
    print("\nLoading 3B model...")
    model = AutoModelForCausalLM.from_pretrained(
        '/mnt/data/models/Qwen2.5-3B-Instruct',
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="eager"
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained('/mnt/data/models/Qwen2.5-3B-Instruct', trust_remote_code=True)

    # 加载数据集
    print("Loading dataset...")
    questions_data, system_tensor, _, _ = prepare_reflect_data(
        './data/result_reflect.json',
        tokenizer,
        '/mnt/data/models/bge-m3-FP16',
        'qwen',
        topk=10,
        max_main_questions=200,
        preprocess=False
    )
    system_len = system_tensor.shape[0]

    # 建立问题到数据的映射
    question_to_data = {}
    for q_data in questions_data:
        for sub_q_info in q_data['sub_questions']:
            question_to_data[sub_q_info['query']] = (q_data, sub_q_info)

    # 计算 attention 特征
    def compute_features_for_questions(df, label):
        features_list = []
        for idx, (_, row) in enumerate(df.iterrows()):
            sub_q = row['Sub Question_r1']
            if sub_q not in question_to_data:
                continue

            q_data, sub_q_info = question_to_data[sub_q]

            # Build input
            doc_chunk_ids = sub_q_info['chunk_ids']
            doc_tensors = q_data['doc_tensors']
            sub_q_doc_tensors = [doc_tensors[chunk_id - 1] for chunk_id in doc_chunk_ids]

            question_text = f"<|im_end|>\n<|im_start|>user\nQuestion: {sub_q_info['query']}<|im_end|>\n<|im_start|>assistant\nAnswer: "
            question_tokens = tokenizer.encode(question_text, add_special_tokens=False)
            question_tensor = torch.tensor(question_tokens, dtype=torch.long)

            all_tokens = [system_tensor] + sub_q_doc_tensors + [question_tensor]
            full_input = torch.cat(all_tokens).unsqueeze(0).to(device)

            doc_start = system_len
            doc_end = system_len + sum(t.shape[0] for t in sub_q_doc_tensors)

            # Get attention - 只保留后半部分层以节省内存
            try:
                with torch.no_grad():
                    outputs = model(full_input, output_attentions=True, use_cache=False)
                    # 只取后半部分层
                    num_layers = len(outputs.attentions)
                    attentions = outputs.attentions[num_layers//2:]

                attn_weights = torch.stack([a.squeeze(0) for a in attentions])
                features = compute_attention_features(attn_weights, doc_start, doc_end)

                if features:
                    features['type'] = label
                    features['question'] = sub_q[:50]
                    features_list.append(features)

                # 清理内存
                del outputs, attentions, attn_weights
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"  Error processing question {idx}: {e}")
                torch.cuda.empty_cache()
                continue

        return features_list

    print("\n计算 Type A 特征...")
    # 随机采样 Type A（因为太多了）
    type_a_sample = type_a.sample(n=min(50, len(type_a)), random_state=42)
    features_a = compute_features_for_questions(type_a_sample, 'A')
    print(f"  计算了 {len(features_a)} 个 Type A 问题")

    print("\n计算 Type B 特征...")
    features_b = compute_features_for_questions(type_b, 'B')
    print(f"  计算了 {len(features_b)} 个 Type B 问题")

    # 比较特征
    print("\n" + "="*80)
    print("Type A vs Type B 特征比较")
    print("="*80)

    feature_names = ['normalized_entropy', 'gini', 'peak_strength',
                     'coverage_50', 'coverage_80', 'coverage_90', 'coverage_95',
                     'top10_concentration', 'top50_concentration', 'top100_concentration',
                     'layer_consistency']

    print(f"\n{'Feature':<25} {'Type A (不需要高rate)':<25} {'Type B (需要高rate)':<25} {'差异':<15} {'Effect Size':<12}")
    print("-" * 105)

    significant_features = []

    for feat in feature_names:
        vals_a = [f[feat] for f in features_a if feat in f]
        vals_b = [f[feat] for f in features_b if feat in f]

        if vals_a and vals_b:
            mean_a = np.mean(vals_a)
            std_a = np.std(vals_a)
            mean_b = np.mean(vals_b)
            std_b = np.std(vals_b)
            diff = mean_b - mean_a

            pooled_std = np.sqrt((std_a**2 + std_b**2) / 2)
            effect_size = abs(diff) / pooled_std if pooled_std > 0 else 0

            marker = "***" if effect_size > 0.8 else ("**" if effect_size > 0.5 else ("*" if effect_size > 0.3 else ""))

            print(f"{feat:<25} {mean_a:>8.4f} ± {std_a:>6.4f}     {mean_b:>8.4f} ± {std_b:>6.4f}     {diff:>+8.4f}     {effect_size:>6.3f} {marker}")

            if effect_size > 0.3:
                significant_features.append((feat, effect_size, diff, mean_a, mean_b))

    print("\n" + "="*80)
    print("显著特征（effect size > 0.3）")
    print("="*80)

    if significant_features:
        significant_features.sort(key=lambda x: x[1], reverse=True)
        for feat, effect, diff, mean_a, mean_b in significant_features:
            direction = "Type B 更高" if diff > 0 else "Type B 更低"
            print(f"\n  {feat}:")
            print(f"    Effect size: {effect:.3f}")
            print(f"    Type A 均值: {mean_a:.4f}")
            print(f"    Type B 均值: {mean_b:.4f}")
            print(f"    差异: {diff:+.4f} ({direction})")

            # 建议阈值
            threshold = (mean_a + mean_b) / 2
            print(f"    建议阈值: {threshold:.4f}")
    else:
        print("  没有找到显著区分特征！")
        print("  这说明 attention 特征可能无法有效预测是否需要高 rate")

    # 保存结果
    all_features = features_a + features_b
    with open('/mnt/data/wjh/FusionRAG/type_ab_attention_features.json', 'w') as f:
        json.dump(all_features, f, indent=2)
    print(f"\n特征数据已保存到 type_ab_attention_features.json")


if __name__ == '__main__':
    main()
