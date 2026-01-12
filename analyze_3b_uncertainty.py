#!/usr/bin/env python3
"""
分析 3B 模型的 attention 不确定性特征

目标：找出能区分 "3B 能答对" 和 "3B 答不对" 的 attention 特征
用于设计动态重算比例的算法
"""

import json
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from test_fusionrag_reflect import prepare_reflect_data


def compute_attention_features(attn_weights, doc_start, doc_end):
    """
    计算 attention 分布的各种特征

    Args:
        attn_weights: [num_layers, num_heads, seq_len, seq_len]
        doc_start: 文档开始位置
        doc_end: 文档结束位置

    Returns:
        dict: 各种特征
    """
    num_layers, num_heads, seq_len, _ = attn_weights.shape

    # 只看 query (最后几个 token) 对 doc 的 attention
    query_start = doc_end  # query 在 doc 之后

    # 提取 query -> doc 的 attention
    # Shape: [num_layers, num_heads, query_len, doc_len]
    q2d_attn = attn_weights[:, :, query_start:, doc_start:doc_end]

    if q2d_attn.shape[2] == 0 or q2d_attn.shape[3] == 0:
        return None

    # 对 heads 和 query positions 取平均，得到每层的 doc attention 分布
    # Shape: [num_layers, doc_len]
    layer_attn = q2d_attn.mean(dim=(1, 2))

    features = {}

    # 1. 熵 (Entropy) - 衡量分布的不确定性
    # 高熵 = 分散 = 不确定
    eps = 1e-10
    entropy_per_layer = -(layer_attn * torch.log(layer_attn + eps)).sum(dim=-1)
    max_entropy = np.log(layer_attn.shape[-1])  # 均匀分布的熵
    normalized_entropy = entropy_per_layer / max_entropy

    features['entropy_mean'] = normalized_entropy.mean().item()
    features['entropy_max'] = normalized_entropy.max().item()
    features['entropy_min'] = normalized_entropy.min().item()
    features['entropy_std'] = normalized_entropy.std().item()

    # 2. 集中度 (Gini coefficient)
    # 高 Gini = 集中 = 确定
    def gini(x):
        x = x.flatten()
        x = torch.sort(x)[0]
        n = len(x)
        cumsum = torch.cumsum(x, dim=0)
        return (2 * torch.sum((torch.arange(1, n+1, device=x.device) * x)) / (n * torch.sum(x)) - (n + 1) / n).item()

    gini_per_layer = [gini(layer_attn[i]) for i in range(num_layers)]
    features['gini_mean'] = np.mean(gini_per_layer)
    features['gini_max'] = np.max(gini_per_layer)
    features['gini_min'] = np.min(gini_per_layer)

    # 3. Top-k coverage - 需要多少 token 才能覆盖 80% 的 attention
    avg_attn = layer_attn.mean(dim=0)  # [doc_len]
    sorted_attn, _ = torch.sort(avg_attn, descending=True)
    cumsum = torch.cumsum(sorted_attn, dim=0)
    total = cumsum[-1]

    coverage_80 = (cumsum >= 0.8 * total).nonzero(as_tuple=True)[0]
    if len(coverage_80) > 0:
        tokens_for_80 = coverage_80[0].item() + 1
    else:
        tokens_for_80 = len(avg_attn)

    features['coverage_80_ratio'] = tokens_for_80 / len(avg_attn)

    coverage_90 = (cumsum >= 0.9 * total).nonzero(as_tuple=True)[0]
    if len(coverage_90) > 0:
        tokens_for_90 = coverage_90[0].item() + 1
    else:
        tokens_for_90 = len(avg_attn)
    features['coverage_90_ratio'] = tokens_for_90 / len(avg_attn)

    # 4. Peak strength - 最高 attention 的强度
    features['peak_strength'] = avg_attn.max().item()
    features['top10_strength'] = sorted_attn[:10].sum().item() if len(sorted_attn) >= 10 else sorted_attn.sum().item()

    # 5. 分散度 - 高 attention tokens 之间的距离
    threshold = avg_attn.mean() + avg_attn.std()
    high_attn_positions = (avg_attn > threshold).nonzero(as_tuple=True)[0]

    if len(high_attn_positions) > 1:
        positions = high_attn_positions.float()
        spread = (positions.max() - positions.min()) / len(avg_attn)
        features['spread'] = spread.item()
    else:
        features['spread'] = 0.0

    # 6. 层间一致性 - 不同层选择的 token 是否一致
    # 低一致性 = 不确定
    layer_top_tokens = []
    for i in range(num_layers):
        top_k = min(50, layer_attn.shape[-1])
        top_indices = torch.topk(layer_attn[i], top_k).indices
        layer_top_tokens.append(set(top_indices.tolist()))

    # 计算相邻层的 IoU
    layer_consistency = []
    for i in range(num_layers - 1):
        intersection = len(layer_top_tokens[i] & layer_top_tokens[i+1])
        union = len(layer_top_tokens[i] | layer_top_tokens[i+1])
        if union > 0:
            layer_consistency.append(intersection / union)

    features['layer_consistency'] = np.mean(layer_consistency) if layer_consistency else 0.0

    return features


def main():
    device = "cuda:0"

    # Load test results
    with open('/mnt/data/wjh/FusionRAG/test_3b_direct_results.json', 'r') as f:
        test_results = json.load(f)

    print(f"Loaded {len(test_results)} test results")

    # Separate correct and wrong cases
    correct_cases = [r for r in test_results if r['is_correct_direct']]
    wrong_cases = [r for r in test_results if not r['is_correct_direct']]

    print(f"  Correct (3B direct can answer): {len(correct_cases)}")
    print(f"  Wrong (3B direct cannot answer): {len(wrong_cases)}")

    # Load 3B model
    print("\nLoading 3B model...")
    config = AutoConfig.from_pretrained('/mnt/data/models/Qwen2.5-3B-Instruct', trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        '/mnt/data/models/Qwen2.5-3B-Instruct',
        config=config,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="eager"  # Required for output_attentions=True
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained('/mnt/data/models/Qwen2.5-3B-Instruct', trust_remote_code=True)

    # Load dataset
    print("\nLoading dataset...")
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

    # Analyze attention features for each case
    all_features = {'correct': [], 'wrong': []}

    print("\n" + "=" * 80)
    print("Analyzing Attention Features")
    print("=" * 80)

    for result in test_results:
        question = result['question']
        is_correct = result['is_correct_direct']
        category = 'correct' if is_correct else 'wrong'

        # Find question in dataset
        found = False
        for q_data in questions_data:
            for sub_q_info in q_data['sub_questions']:
                if sub_q_info['query'] == question:
                    found = True

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

                    # Get attention
                    with torch.no_grad():
                        outputs = model(
                            full_input,
                            output_attentions=True,
                            use_cache=False
                        )
                        attentions = outputs.attentions  # tuple of [batch, heads, seq, seq]

                    # Stack attention from all layers
                    attn_weights = torch.stack([a.squeeze(0) for a in attentions])  # [layers, heads, seq, seq]

                    # Compute features
                    features = compute_attention_features(attn_weights, doc_start, doc_end)

                    if features:
                        features['question'] = question[:50]
                        features['is_correct'] = is_correct
                        all_features[category].append(features)

                        print(f"\n{'✓' if is_correct else '✗'} {question[:60]}...")
                        print(f"  Entropy: {features['entropy_mean']:.3f} (mean), {features['entropy_max']:.3f} (max)")
                        print(f"  Gini: {features['gini_mean']:.3f}")
                        print(f"  Coverage 80%: {features['coverage_80_ratio']:.3f}")
                        print(f"  Layer consistency: {features['layer_consistency']:.3f}")
                        print(f"  Spread: {features['spread']:.3f}")

                    break
            if found:
                break

    # Compare features between correct and wrong cases
    print("\n" + "=" * 80)
    print("FEATURE COMPARISON: Correct vs Wrong")
    print("=" * 80)

    feature_names = ['entropy_mean', 'entropy_max', 'gini_mean', 'coverage_80_ratio',
                     'coverage_90_ratio', 'layer_consistency', 'spread', 'peak_strength']

    print(f"\n{'Feature':<25} {'Correct (n={})'.format(len(all_features['correct'])):<20} {'Wrong (n={})'.format(len(all_features['wrong'])):<20} {'Diff':<10}")
    print("-" * 75)

    discriminative_features = []

    for feat in feature_names:
        correct_vals = [f[feat] for f in all_features['correct'] if feat in f]
        wrong_vals = [f[feat] for f in all_features['wrong'] if feat in f]

        if correct_vals and wrong_vals:
            correct_mean = np.mean(correct_vals)
            wrong_mean = np.mean(wrong_vals)
            diff = wrong_mean - correct_mean

            # 计算 t-test 的 effect size
            pooled_std = np.sqrt((np.var(correct_vals) + np.var(wrong_vals)) / 2)
            if pooled_std > 0:
                effect_size = abs(diff) / pooled_std
            else:
                effect_size = 0

            marker = "***" if effect_size > 0.8 else ("**" if effect_size > 0.5 else ("*" if effect_size > 0.2 else ""))

            print(f"{feat:<25} {correct_mean:>8.4f} ± {np.std(correct_vals):>6.4f}  {wrong_mean:>8.4f} ± {np.std(wrong_vals):>6.4f}  {diff:>+8.4f} {marker}")

            if effect_size > 0.3:
                discriminative_features.append((feat, effect_size, diff))

    print("\n" + "=" * 80)
    print("DISCRIMINATIVE FEATURES (effect size > 0.3)")
    print("=" * 80)

    if discriminative_features:
        discriminative_features.sort(key=lambda x: x[1], reverse=True)
        for feat, effect, diff in discriminative_features:
            direction = "higher when wrong" if diff > 0 else "lower when wrong"
            print(f"  {feat}: effect size = {effect:.3f} ({direction})")
    else:
        print("  No strong discriminative features found")

    # 保存结果
    output = {
        'correct_features': all_features['correct'],
        'wrong_features': all_features['wrong'],
        'summary': {
            feat: {
                'correct_mean': np.mean([f[feat] for f in all_features['correct'] if feat in f]),
                'wrong_mean': np.mean([f[feat] for f in all_features['wrong'] if feat in f])
            } for feat in feature_names
        }
    }

    with open('/mnt/data/wjh/FusionRAG/attention_uncertainty_analysis.json', 'w') as f:
        json.dump(output, f, indent=2, default=lambda x: x if not isinstance(x, np.floating) else float(x))

    print(f"\nResults saved to attention_uncertainty_analysis.json")

    # 提出动态 rate 建议
    print("\n" + "=" * 80)
    print("DYNAMIC RATE STRATEGY RECOMMENDATION")
    print("=" * 80)

    if discriminative_features:
        best_feat = discriminative_features[0][0]
        best_diff = discriminative_features[0][2]

        print(f"""
基于分析，推荐使用 '{best_feat}' 作为动态 rate 的依据：

策略：
  - 当 {best_feat} {'较高' if best_diff > 0 else '较低'} 时，3B 模型更可能答错
  - 此时应该 提高重算比例，让大模型看到更多 context

实现思路：
  1. 计算每个问题的 {best_feat}
  2. 设定阈值（如 median）
  3. 高于阈值：rate = 0.3 (更多重算)
     低于阈值：rate = 0.15 (较少重算)
  4. 或者用线性映射：rate = base_rate + k * ({best_feat} - threshold)
""")


if __name__ == '__main__':
    main()
