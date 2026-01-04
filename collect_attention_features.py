#!/usr/bin/env python3
"""
收集 DraftModel attention 特征，用于分析困难/简单问题的差异。

运行方式：
CUDA_VISIBLE_DEVICES=0 python collect_attention_features.py
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F

project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

from ktransformers.util.utils import rotate_half

RESULTS_DIR = "/mnt/data/reflect/Qwen2.5-7B-Instruct/results"
DRAFT_MODEL_PATH = "/mnt/data/models/Qwen2.5-3B-Instruct"


def load_draft_model(device="cuda:0"):
    """加载 DraftModel"""
    print(f"Loading draft model from {DRAFT_MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(DRAFT_MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        DRAFT_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map=device
    )
    model.eval()
    print(f"Draft model loaded: {model.config.num_hidden_layers} layers, "
          f"{model.config.num_attention_heads} heads")
    return model, tokenizer


def compute_draft_attention(model, tokenizer, input_text: str, device="cuda:0"):
    """
    运行 DraftModel prefill，收集 attention 分布。

    Returns:
        attention_scores: {layer_idx: [num_heads, seq_len, seq_len]}
    """
    config = model.config
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // num_heads

    # Tokenize
    input_ids = tokenizer(input_text, return_tensors="pt")["input_ids"].to(device)
    seq_len = input_ids.shape[1]

    layer_attention_scores = {}

    with torch.no_grad():
        inputs_embeds = model.model.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

        for layer_idx in range(num_layers):
            layer = model.model.layers[layer_idx]

            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)

            bsz, q_len, _ = hidden_states.size()

            query_states = layer.self_attn.q_proj(hidden_states)
            key_states = layer.self_attn.k_proj(hidden_states)
            value_states = layer.self_attn.v_proj(hidden_states)

            query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

            cos, sin = layer.self_attn.rotary_emb(value_states, position_ids)
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
            query_states = (query_states * cos) + (rotate_half(query_states) * sin)
            key_states = (key_states * cos) + (rotate_half(key_states) * sin)

            n_rep = num_heads // num_kv_heads
            key_states_expanded = key_states.repeat_interleave(n_rep, dim=1)
            value_states_expanded = value_states.repeat_interleave(n_rep, dim=1)

            attn_weights = torch.matmul(query_states.float(), key_states_expanded.float().transpose(2, 3)) / (head_dim ** 0.5)
            causal_mask = torch.triu(torch.ones(q_len, q_len, device=device), diagonal=1).bool()
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
            attn_weights = F.softmax(attn_weights, dim=-1)

            # 只保存后半部分层
            if layer_idx >= num_layers // 2:
                layer_attention_scores[layer_idx] = attn_weights[0].cpu().float().numpy()

            attn_output = torch.matmul(attn_weights.to(value_states_expanded.dtype), value_states_expanded)
            attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
            attn_output = layer.self_attn.o_proj(attn_output)

            hidden_states = residual + attn_output
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states

    return layer_attention_scores, seq_len


def compute_attention_features(attention_scores: Dict, system_len: int, doc_len: int, query_len: int) -> Dict:
    """从 attention 分布中提取特征"""
    features = {}

    total_len = system_len + doc_len + query_len
    doc_start = system_len
    doc_end = system_len + doc_len
    query_start = system_len + doc_len

    all_layer_attention = []
    layer_entropies = []

    for layer_idx in sorted(attention_scores.keys()):
        layer_attn = attention_scores[layer_idx]

        # 提取 query→doc attention
        query_to_doc = layer_attn[:, query_start:total_len, doc_start:doc_end]
        doc_attention_avg = query_to_doc.mean(axis=(0, 1))
        all_layer_attention.append(doc_attention_avg)

        # 计算熵
        p = doc_attention_avg / (doc_attention_avg.sum() + 1e-10)
        p = np.clip(p, 1e-10, 1.0)
        entropy = -np.sum(p * np.log(p))
        layer_entropies.append(entropy)

    aggregated_attn = np.mean(all_layer_attention, axis=0)

    # Feature 1: Attention 熵
    p_agg = aggregated_attn / (aggregated_attn.sum() + 1e-10)
    p_agg = np.clip(p_agg, 1e-10, 1.0)
    features['attention_entropy'] = -np.sum(p_agg * np.log(p_agg))
    features['min_layer_entropy'] = min(layer_entropies)
    features['max_layer_entropy'] = max(layer_entropies)
    features['entropy_range'] = features['max_layer_entropy'] - features['min_layer_entropy']

    # Feature 2: Coverage ratio
    sorted_indices = np.argsort(aggregated_attn)[::-1]
    sorted_attn = aggregated_attn[sorted_indices]
    cumsum = np.cumsum(sorted_attn) / (sorted_attn.sum() + 1e-10)

    for coverage in [0.5, 0.7, 0.85, 0.9, 0.95]:
        count = np.searchsorted(cumsum, coverage) + 1
        features[f'coverage_{int(coverage*100)}_ratio'] = count / doc_len

    # Feature 3: Gini 系数
    sorted_attn_asc = np.sort(aggregated_attn)
    n = len(sorted_attn_asc)
    index = np.arange(1, n + 1)
    gini = ((2 * index - n - 1) * sorted_attn_asc).sum() / (n * sorted_attn_asc.sum() + 1e-10)
    features['gini_coefficient'] = gini

    # Feature 4: 高 attention 位置分布
    mean_attn = np.mean(aggregated_attn)
    std_attn = np.std(aggregated_attn)

    high_positions = np.where(aggregated_attn > mean_attn + std_attn)[0]
    features['high_attn_count'] = len(high_positions)
    features['high_attn_ratio'] = len(high_positions) / doc_len

    if len(high_positions) > 1:
        features['high_attn_span'] = (high_positions.max() - high_positions.min()) / doc_len
        gaps = np.diff(sorted(high_positions))
        features['high_attn_max_gap'] = gaps.max() if len(gaps) > 0 else 0
        features['high_attn_mean_gap'] = gaps.mean() if len(gaps) > 0 else 0
    else:
        features['high_attn_span'] = 0
        features['high_attn_max_gap'] = 0
        features['high_attn_mean_gap'] = 0

    # Feature 5: 连通分量
    threshold_positions = list(np.where(aggregated_attn > mean_attn + 0.5 * std_attn)[0])
    components = find_connected_components(threshold_positions, max_gap=2)
    features['num_components'] = len(components)
    if components:
        component_sizes = [len(c) for c in components]
        features['max_component_size'] = max(component_sizes)
        features['mean_component_size'] = np.mean(component_sizes)
    else:
        features['max_component_size'] = 0
        features['mean_component_size'] = 0

    # Feature 6: 统计特征
    features['attention_mean'] = np.mean(aggregated_attn)
    features['attention_std'] = np.std(aggregated_attn)
    features['attention_max'] = np.max(aggregated_attn)
    features['attention_cv'] = std_attn / (mean_attn + 1e-10)  # 变异系数

    return features


def find_connected_components(positions: List[int], max_gap: int = 2) -> List[List[int]]:
    """找到连通分量"""
    if not positions:
        return []
    positions = sorted(set(positions))
    components = []
    current_component = [positions[0]]
    for i in range(1, len(positions)):
        if positions[i] - positions[i-1] <= max_gap:
            current_component.append(positions[i])
        else:
            components.append(current_component)
            current_component = [positions[i]]
    components.append(current_component)
    return components


def load_test_data():
    """加载测试数据"""
    data_file = "/mnt/data/wjh/FusionRAG/result_reflect.json"
    with open(data_file, 'r') as f:
        data = json.load(f)
    return data


def load_hard_and_easy_indices():
    """加载困难和简单问题的索引"""
    draft_03 = pd.read_csv(os.path.join(RESULTS_DIR, "DraftModel_global_topk_10_rate_0.3.csv"))
    draft_03['Correct'] = draft_03['Correct'].apply(lambda x: x == True or x == 'True')
    draft_03['Rate1_Correct'] = draft_03['Rate1_Correct'].apply(lambda x: x == True or x == 'True')

    hard_indices = draft_03[(draft_03['Correct'] == False) & (draft_03['Rate1_Correct'] == True)].index.tolist()
    easy_indices = draft_03[(draft_03['Correct'] == True) & (draft_03['Rate1_Correct'] == True)].index.tolist()

    return hard_indices, easy_indices, draft_03


def main():
    device = "cuda:0"

    # 加载模型
    draft_model, tokenizer = load_draft_model(device)

    # 加载困难/简单问题索引
    hard_indices, easy_indices, results_df = load_hard_and_easy_indices()
    print(f"\nHard questions: {len(hard_indices)}")
    print(f"Easy questions: {len(easy_indices)}")

    # 只采样部分简单问题（为节省时间）
    np.random.seed(42)
    sampled_easy = np.random.choice(easy_indices, min(30, len(easy_indices)), replace=False).tolist()

    # 加载原始数据
    data = load_test_data()

    # 收集特征
    hard_features = []
    easy_features = []

    # 需要重建问题到数据的映射
    # 从CSV中获取问题信息
    all_test_indices = hard_indices + sampled_easy

    print("\n" + "="*80)
    print("Collecting attention features...")
    print("="*80)

    for i, idx in enumerate(all_test_indices):
        row = results_df.iloc[idx]
        main_q = row['Main Question']
        sub_q = row['Sub Question']

        # 查找对应的数据
        found = False
        for example in data:
            if example['main_question'] == main_q:
                for sub in example['sub_questions']:
                    if sub['sub_question'] == sub_q:
                        # 找到了
                        found = True

                        # 构建输入文本
                        system_prompt = "You are a helpful assistant."
                        docs_text = "\n".join([doc['content'] for doc in sub['retrieve_documents'][:10]])
                        query_text = sub_q

                        full_text = f"{system_prompt}\n\n{docs_text}\n\nQuestion: {query_text}\nAnswer:"

                        # Tokenize 计算长度
                        tokens = tokenizer(full_text, return_tensors="pt")["input_ids"][0]
                        system_tokens = tokenizer(system_prompt, return_tensors="pt")["input_ids"][0]
                        docs_tokens = tokenizer(docs_text, return_tensors="pt")["input_ids"][0]

                        system_len = len(system_tokens)
                        doc_len = len(docs_tokens)
                        query_len = len(tokens) - system_len - doc_len

                        # 计算 attention
                        try:
                            attn_scores, seq_len = compute_draft_attention(draft_model, tokenizer, full_text, device)
                            features = compute_attention_features(attn_scores, system_len, doc_len, query_len)
                            features['idx'] = idx
                            features['is_hard'] = idx in hard_indices
                            features['sub_question'] = sub_q[:50]

                            if idx in hard_indices:
                                hard_features.append(features)
                            else:
                                easy_features.append(features)

                            print(f"[{i+1}/{len(all_test_indices)}] idx={idx}, hard={idx in hard_indices}, "
                                  f"entropy={features['attention_entropy']:.2f}, "
                                  f"cov85={features['coverage_85_ratio']:.3f}")
                        except Exception as e:
                            print(f"Error processing idx={idx}: {e}")

                        break
                if found:
                    break

    # 保存特征
    all_features = hard_features + easy_features
    with open("attention_features.json", "w") as f:
        json.dump(all_features, f, indent=2)

    # 分析差异
    analyze_features_difference(hard_features, easy_features)


def analyze_features_difference(hard_features: List[Dict], easy_features: List[Dict]):
    """分析困难和简单问题的特征差异"""
    print("\n" + "="*80)
    print("Feature Analysis: Hard vs Easy Questions")
    print("="*80)

    if not hard_features or not easy_features:
        print("Not enough data")
        return

    # 获取数值特征
    feature_names = [k for k in hard_features[0].keys() if isinstance(hard_features[0][k], (int, float))]

    results = []
    for name in feature_names:
        if name == 'idx' or name == 'is_hard':
            continue
        hard_vals = [f[name] for f in hard_features]
        easy_vals = [f[name] for f in easy_features]

        hard_mean = np.mean(hard_vals)
        easy_mean = np.mean(easy_vals)
        diff_pct = (hard_mean / (easy_mean + 1e-10) - 1) * 100

        results.append({
            'feature': name,
            'hard_mean': hard_mean,
            'easy_mean': easy_mean,
            'diff_pct': diff_pct,
        })

    results.sort(key=lambda x: abs(x['diff_pct']), reverse=True)

    print(f"\n{'Feature':<25} {'Hard Mean':<12} {'Easy Mean':<12} {'Diff %':<10}")
    print("-" * 60)
    for r in results:
        print(f"{r['feature']:<25} {r['hard_mean']:<12.4f} {r['easy_mean']:<12.4f} {r['diff_pct']:>+8.1f}%")

    # 推荐用于动态预测的特征
    print("\n" + "="*80)
    print("Recommended features for dynamic rate prediction:")
    print("="*80)
    for r in results[:5]:
        direction = "HIGHER" if r['diff_pct'] > 0 else "LOWER"
        print(f"  - {r['feature']}: Hard questions have {direction} values ({r['diff_pct']:+.1f}%)")


if __name__ == "__main__":
    main()
