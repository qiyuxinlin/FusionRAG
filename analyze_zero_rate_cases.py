#!/usr/bin/env python3
"""
分析哪些问题在 rate=0（不重算）时也能答对，
以及这些问题的 attention 特征，用于设计动态 rate 算法。
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from collections import defaultdict

sys.path.insert(0, '/mnt/data/wjh/FusionRAG')
from ktransformers.util.utils import rotate_half

DRAFT_MODEL_PATH = "/mnt/data/models/Qwen2.5-3B-Instruct"


def load_results_and_compare():
    """加载 rate=0 和 rate=0.3 以及 rate=1 的结果进行对比"""
    results_dir = "/mnt/data/reflect/Qwen2.5-7B-Instruct/results"

    # 读取各种 rate 的结果
    rate_files = {
        'rate_0': 'Oracle_global_topk_10_rate_0.0_revert_rope.csv',
        'rate_0.1': 'Oracle_global_topk_10_rate_0.1_revert_rope.csv',
        'rate_0.15': 'Oracle_global_topk_10_rate_0.15_revert_rope.csv',
        'rate_0.2': 'Oracle_global_topk_10_rate_0.2_revert_rope.csv',
        'rate_0.3': 'Oracle_global_topk_10_rate_0.3_revert_rope.csv',
        'draft_0.3': 'DraftModel_global_topk_10_rate_0.3.csv',
    }

    results = {}
    for key, filename in rate_files.items():
        filepath = os.path.join(results_dir, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            results[key] = df
            print(f"Loaded {key}: {len(df)} rows, {df['Correct'].sum()}/{len(df)} correct ({df['Correct'].mean()*100:.1f}%)")

    return results


def analyze_correct_cases(results):
    """分析在各 rate 下正确与否的情况"""
    if 'rate_0' not in results or 'rate_0.3' not in results:
        print("Missing required result files!")
        return None

    df_0 = results['rate_0']
    df_03 = results['rate_0.3']

    # 确保行数一致
    min_rows = min(len(df_0), len(df_03))

    # 分类
    categories = {
        'easy': [],      # rate=0 就能答对
        'medium': [],    # rate=0 错，rate>0 才对
        'hard': [],      # 都错
    }

    for i in range(min_rows):
        correct_0 = df_0.iloc[i]['Correct']
        correct_03 = df_03.iloc[i]['Correct']

        sub_q = df_0.iloc[i]['Sub Question']
        main_q = df_0.iloc[i]['Main Question']

        info = {
            'idx': i,
            'sub_q': sub_q,
            'main_q': main_q,
            'correct_0': correct_0,
            'correct_03': correct_03,
        }

        if correct_0:
            categories['easy'].append(info)
        elif correct_03:
            categories['medium'].append(info)
        else:
            categories['hard'].append(info)

    print(f"\n=== Case Categories ===")
    print(f"Easy (rate=0 correct): {len(categories['easy'])} ({len(categories['easy'])/min_rows*100:.1f}%)")
    print(f"Medium (need recompute): {len(categories['medium'])} ({len(categories['medium'])/min_rows*100:.1f}%)")
    print(f"Hard (always wrong): {len(categories['hard'])} ({len(categories['hard'])/min_rows*100:.1f}%)")

    return categories


def compute_draft_attention_features(model, tokenizer, text, device="cuda:0"):
    """计算 draft model 的 attention 特征"""
    config = model.config
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // num_heads

    input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
    seq_len = input_ids.shape[1]

    layer_attentions = []

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

            if hasattr(layer.self_attn, 'rotary_emb'):
                rotary_emb = layer.self_attn.rotary_emb
            elif hasattr(model.model, 'rotary_emb'):
                rotary_emb = model.model.rotary_emb
            else:
                rotary_emb = model.model.layers[0].self_attn.rotary_emb
            cos, sin = rotary_emb(value_states, position_ids)
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

            if layer_idx >= num_layers // 2:
                layer_attentions.append(attn_weights[0].cpu().float().numpy())

            attn_output = torch.matmul(attn_weights.to(value_states_expanded.dtype), value_states_expanded)
            attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
            attn_output = layer.self_attn.o_proj(attn_output)

            hidden_states = residual + attn_output
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states

    return layer_attentions, seq_len


def extract_attention_features(layer_attentions, system_len, doc_len, query_len):
    """从 attention 中提取特征"""
    total_len = system_len + doc_len + query_len
    doc_start = system_len
    doc_end = system_len + doc_len
    query_start = system_len + doc_len

    # 收集各层的 attention 和熵
    layer_doc_attentions = []
    layer_entropies = []

    for layer_attn in layer_attentions:
        # query→doc attention
        query_to_doc = layer_attn[:, query_start:total_len, doc_start:doc_end]
        doc_attention_avg = query_to_doc.mean(axis=(0, 1))
        layer_doc_attentions.append(doc_attention_avg)

        # 计算该层的熵
        p = doc_attention_avg / (doc_attention_avg.sum() + 1e-10)
        p = np.clip(p, 1e-10, 1.0)
        entropy = -np.sum(p * np.log(p))
        layer_entropies.append(entropy)

    # 选择熵最低的 4 层
    sorted_indices = np.argsort(layer_entropies)[:4]
    selected_attentions = [layer_doc_attentions[i] for i in sorted_indices]
    aggregated_attn = np.mean(selected_attentions, axis=0)

    features = {}

    # 1. 覆盖率特征
    sorted_idx = np.argsort(aggregated_attn)[::-1]
    sorted_attn = aggregated_attn[sorted_idx]
    cumsum = np.cumsum(sorted_attn) / (sorted_attn.sum() + 1e-10)

    for coverage in [0.5, 0.7, 0.8, 0.85, 0.9, 0.95]:
        count = np.searchsorted(cumsum, coverage) + 1
        features[f'cov_{int(coverage*100)}'] = count / doc_len

    # 2. Top-k 集中度
    for k in [5, 10, 20]:
        if k <= len(aggregated_attn):
            features[f'top{k}_ratio'] = sorted_attn[:k].sum() / (sorted_attn.sum() + 1e-10)

    # 3. Gini 系数
    sorted_attn_asc = np.sort(aggregated_attn)
    n = len(sorted_attn_asc)
    index = np.arange(1, n + 1)
    gini = ((2 * index - n - 1) * sorted_attn_asc).sum() / (n * sorted_attn_asc.sum() + 1e-10)
    features['gini'] = gini

    # 4. 归一化熵
    p = aggregated_attn / (aggregated_attn.sum() + 1e-10)
    p = np.clip(p, 1e-10, 1.0)
    entropy = -np.sum(p * np.log(p))
    max_entropy = np.log(doc_len) if doc_len > 0 else 1
    features['norm_entropy'] = entropy / max_entropy

    # 5. 最大 attention 值
    features['max_attn'] = float(np.max(aggregated_attn))

    # 6. 最大与第二大的比例（越大越集中）
    if len(sorted_attn) >= 2:
        features['max_to_second'] = sorted_attn[0] / (sorted_attn[1] + 1e-10)

    # 7. 选择层的最低熵（熵越低，分布越集中）
    features['min_layer_entropy'] = float(np.min(layer_entropies))
    features['mean_layer_entropy'] = float(np.mean(sorted(layer_entropies)[:4]))

    return features


def main():
    device = "cuda:0"

    # 加载并比较结果
    print("Loading results...")
    results = load_results_and_compare()

    categories = analyze_correct_cases(results)
    if categories is None:
        return

    # 加载测试数据
    with open('/mnt/data/wjh/FusionRAG/result_reflect.json') as f:
        data = json.load(f)

    # 加载 draft model
    print("\nLoading draft model...")
    tokenizer = AutoTokenizer.from_pretrained(DRAFT_MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        DRAFT_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map=device
    )
    model.eval()
    print(f"Model loaded: {model.config.num_hidden_layers} layers")

    # 为每个类别收集 attention 特征
    category_features = {
        'easy': [],
        'medium': [],
    }

    # 构建子问题到数据的映射
    sub_q_to_data = {}
    for i, item in enumerate(data):
        intermediate = item.get('intermediate_context', [])
        for sub_idx, sub_q in enumerate(intermediate):
            query = sub_q.get('query', '')
            if query.startswith("Intermediate query"):
                colon_pos = query.find(":")
                if colon_pos != -1:
                    query = query[colon_pos + 1:].strip()
            sub_q_to_data[query[:50]] = (i, sub_idx, sub_q)

    # 分析 easy 和 medium 类别的 attention 特征
    max_samples_per_category = 30

    for category in ['easy', 'medium']:
        print(f"\n=== Analyzing {category} cases ===")
        samples = categories[category][:max_samples_per_category]

        for sample in samples:
            sub_q = sample['sub_q']

            # 查找对应的数据
            found = False
            for key, (i, sub_idx, sub_q_data) in sub_q_to_data.items():
                if key in sub_q or sub_q in key:
                    docs = sub_q_data.get('retrieve docs', [])[:10]
                    if not docs:
                        continue

                    # 构建输入
                    system_prompt = "You are a helpful assistant."
                    docs_text = "\n".join(docs)
                    full_text = f"{system_prompt}\n\n{docs_text}\n\nQuestion: {sub_q}\nAnswer:"

                    # 计算长度
                    system_tokens = tokenizer(system_prompt, return_tensors="pt")["input_ids"][0]
                    docs_tokens = tokenizer(docs_text, return_tensors="pt")["input_ids"][0]
                    full_tokens = tokenizer(full_text, return_tensors="pt")["input_ids"][0]

                    system_len = len(system_tokens)
                    doc_len = len(docs_tokens)
                    query_len = len(full_tokens) - system_len - doc_len

                    if doc_len < 100:
                        continue

                    try:
                        layer_attentions, seq_len = compute_draft_attention_features(
                            model, tokenizer, full_text, device
                        )
                        features = extract_attention_features(
                            layer_attentions, system_len, doc_len, query_len
                        )
                        features['doc_len'] = doc_len
                        features['query'] = sub_q[:50]
                        category_features[category].append(features)
                        found = True
                        print(f"  [{len(category_features[category])}] doc_len={doc_len}, gini={features['gini']:.3f}, top10={features['top10_ratio']:.3f}")
                        break
                    except Exception as e:
                        print(f"  Error: {e}")
                        continue

            if len(category_features[category]) >= max_samples_per_category:
                break

    # 统计分析
    print("\n" + "=" * 70)
    print("Feature Comparison: Easy vs Medium")
    print("=" * 70)

    feature_keys = ['gini', 'norm_entropy', 'top10_ratio', 'cov_85', 'cov_90',
                    'max_attn', 'max_to_second', 'min_layer_entropy']

    for key in feature_keys:
        easy_vals = [f[key] for f in category_features['easy'] if key in f]
        medium_vals = [f[key] for f in category_features['medium'] if key in f]

        if easy_vals and medium_vals:
            print(f"\n{key}:")
            print(f"  Easy:   mean={np.mean(easy_vals):.4f}, std={np.std(easy_vals):.4f}, "
                  f"min={np.min(easy_vals):.4f}, max={np.max(easy_vals):.4f}")
            print(f"  Medium: mean={np.mean(medium_vals):.4f}, std={np.std(medium_vals):.4f}, "
                  f"min={np.min(medium_vals):.4f}, max={np.max(medium_vals):.4f}")

            # 计算分离度
            easy_mean, medium_mean = np.mean(easy_vals), np.mean(medium_vals)
            overall_std = np.std(easy_vals + medium_vals)
            if overall_std > 0:
                sep = abs(easy_mean - medium_mean) / overall_std
                print(f"  Separation: {sep:.3f} (higher is better)")

    # 保存结果
    output = {
        'easy': category_features['easy'],
        'medium': category_features['medium'],
    }
    with open('zero_rate_analysis.json', 'w') as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 70)
    print("Recommendations for rate=0 detection")
    print("=" * 70)

    # 计算阈值建议
    if category_features['easy'] and category_features['medium']:
        easy_gini = [f['gini'] for f in category_features['easy']]
        medium_gini = [f['gini'] for f in category_features['medium']]

        easy_top10 = [f['top10_ratio'] for f in category_features['easy']]
        medium_top10 = [f['top10_ratio'] for f in category_features['medium']]

        print(f"""
如果 Gini 系数 > {np.percentile(easy_gini, 75):.3f} 且 top10_ratio > {np.percentile(easy_top10, 75):.3f}:
    → attention 非常集中，可能不需要重算 (rate=0)

如果 Gini 系数 < {np.percentile(medium_gini, 25):.3f} 且 top10_ratio < {np.percentile(medium_top10, 25):.3f}:
    → attention 分散，需要重算 (rate > 0)

建议的公式:
    concentration = gini * top10_ratio
    if concentration > threshold:
        rate = 0  # 不需要重算
    else:
        rate = coverage_90_ratio * scale_factor
""")


if __name__ == "__main__":
    main()
