#!/usr/bin/env python3
"""
分析跨文档的 attention 分布，寻找更优雅的动态 rate 公式。

理论假设：
- 如果 attention 集中在单个文档 → 答案在该文档中 → 不需要太多重算
- 如果 attention 分散在多个文档 → 需要跨文档推理 → 需要更多重算

指标候选：
1. 跨文档 attention 熵
2. Top-1 文档占总 attention 的比例
3. 有效文档数（attention 占比 > threshold 的文档数）
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

sys.path.insert(0, '/mnt/data/wjh/FusionRAG')
from ktransformers.util.utils import rotate_half

DRAFT_MODEL_PATH = "/mnt/data/models/Qwen2.5-3B-Instruct"
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
        return None, None

    min_rows = min(len(df_005), len(df_03))

    categories = {'easy': [], 'medium': []}
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
        }

        if correct_005:
            categories['easy'].append(info)
        elif correct_03:
            categories['medium'].append(info)

    return categories, results


def compute_draft_attention_per_doc(model, tokenizer, full_text, doc_boundaries, device="cuda:0"):
    """
    计算 draft model 的 attention，并按文档分解。

    Args:
        doc_boundaries: list of (start, end) token positions for each document
    """
    config = model.config
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // num_heads

    input_ids = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=4096)["input_ids"].to(device)
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


def extract_cross_doc_features(layer_attentions, system_len, doc_boundaries, query_start, total_len):
    """
    提取跨文档的 attention 特征。

    Args:
        doc_boundaries: list of (start, end) relative to doc region
    """
    # 选择熵最低的 4 层
    layer_entropies = []
    layer_doc_attentions = []

    doc_region_start = system_len
    doc_region_end = query_start

    for layer_attn in layer_attentions:
        # query → doc attention
        query_to_doc = layer_attn[:, query_start:total_len, doc_region_start:doc_region_end]
        doc_attention_avg = query_to_doc.mean(axis=(0, 1))
        layer_doc_attentions.append(doc_attention_avg)

        p = doc_attention_avg / (doc_attention_avg.sum() + 1e-10)
        p = np.clip(p, 1e-10, 1.0)
        entropy = -np.sum(p * np.log(p))
        layer_entropies.append(entropy)

    sorted_indices = np.argsort(layer_entropies)[:4]
    selected_attentions = [layer_doc_attentions[i] for i in sorted_indices]
    aggregated_attn = np.mean(selected_attentions, axis=0)

    # 计算每个文档的 attention 总和
    doc_attention_sums = []
    for doc_start, doc_end in doc_boundaries:
        if doc_end <= len(aggregated_attn):
            doc_attn = aggregated_attn[doc_start:doc_end].sum()
            doc_attention_sums.append(doc_attn)

    if not doc_attention_sums:
        return None

    doc_attention_sums = np.array(doc_attention_sums)
    doc_attention_sums = doc_attention_sums / (doc_attention_sums.sum() + 1e-10)

    features = {}

    # 1. Top-1 文档占比
    features['top1_doc_ratio'] = float(np.max(doc_attention_sums))

    # 2. Top-2 文档占比
    if len(doc_attention_sums) >= 2:
        features['top2_doc_ratio'] = float(np.sort(doc_attention_sums)[-2:].sum())
    else:
        features['top2_doc_ratio'] = features['top1_doc_ratio']

    # 3. 跨文档熵（文档级别的 attention 分布熵）
    p_doc = doc_attention_sums / (doc_attention_sums.sum() + 1e-10)
    p_doc = np.clip(p_doc, 1e-10, 1.0)
    cross_doc_entropy = -np.sum(p_doc * np.log(p_doc))
    max_entropy = np.log(len(doc_attention_sums)) if len(doc_attention_sums) > 1 else 1
    features['cross_doc_entropy'] = float(cross_doc_entropy / max_entropy)

    # 4. 有效文档数（attention > 10% 的文档数）
    effective_docs = np.sum(doc_attention_sums > 0.1)
    features['effective_docs'] = int(effective_docs)

    # 5. Gini 系数（文档级别）
    sorted_doc_attn = np.sort(doc_attention_sums)
    n = len(sorted_doc_attn)
    index = np.arange(1, n + 1)
    gini = ((2 * index - n - 1) * sorted_doc_attn).sum() / (n * sorted_doc_attn.sum() + 1e-10)
    features['doc_gini'] = float(gini)

    return features


def main():
    device = "cuda:0"

    categories, results = load_and_categorize()
    if categories is None:
        return

    print(f"Easy: {len(categories['easy'])}, Medium: {len(categories['medium'])}")

    with open('/mnt/data/wjh/FusionRAG/result_reflect.json') as f:
        data = json.load(f)

    print("\nLoading draft model...")
    tokenizer = AutoTokenizer.from_pretrained(DRAFT_MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        DRAFT_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map=device
    )
    model.eval()

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
            sub_q_to_data[query[:80]] = (sub_q, query)

    category_features = {'easy': [], 'medium': []}
    max_samples = 30

    for category in ['easy', 'medium']:
        print(f"\n=== Analyzing {category} cases ===")
        samples = categories[category]

        for sample in samples:
            if len(category_features[category]) >= max_samples:
                break

            sub_q = sample['sub_q']

            found_key = None
            for key in sub_q_to_data.keys():
                if key in str(sub_q) or str(sub_q)[:60] in key:
                    found_key = key
                    break

            if found_key is None:
                continue

            sub_q_data, full_query = sub_q_to_data[found_key]
            docs = sub_q_data.get('retrieve docs', [])[:10]
            if len(docs) < 2:
                continue

            # 构建输入并记录文档边界
            system_prompt = "You are a helpful assistant."
            system_tokens = tokenizer(system_prompt, return_tensors="pt")["input_ids"][0]
            system_len = len(system_tokens)

            # 逐个文档 tokenize 以获取边界
            doc_boundaries = []
            current_pos = 0
            docs_text_parts = []

            for doc in docs:
                doc_tokens = tokenizer(doc, return_tensors="pt")["input_ids"][0]
                doc_len = len(doc_tokens)
                doc_boundaries.append((current_pos, current_pos + doc_len))
                current_pos += doc_len
                docs_text_parts.append(doc)

            docs_text = "\n".join(docs)
            query_text = f"\n\nQuestion: {full_query}\nAnswer:"
            full_text = f"{system_prompt}\n\n{docs_text}{query_text}"

            full_tokens = tokenizer(full_text, return_tensors="pt")["input_ids"][0]
            total_len = len(full_tokens)

            # 重新计算准确的边界
            docs_combined_tokens = tokenizer(docs_text, return_tensors="pt")["input_ids"][0]
            doc_region_len = len(docs_combined_tokens)
            query_start = system_len + doc_region_len

            try:
                layer_attentions, seq_len = compute_draft_attention_per_doc(
                    model, tokenizer, full_text, doc_boundaries, device
                )
                features = extract_cross_doc_features(
                    layer_attentions, system_len, doc_boundaries, query_start, total_len
                )

                if features:
                    features['num_docs'] = len(docs)
                    category_features[category].append(features)
                    print(f"  [{len(category_features[category])}] "
                          f"top1_doc={features['top1_doc_ratio']:.3f}, "
                          f"cross_entropy={features['cross_doc_entropy']:.3f}, "
                          f"effective_docs={features['effective_docs']}")

            except Exception as e:
                print(f"  Error: {e}")
                continue

    # 统计分析
    print("\n" + "=" * 70)
    print("CROSS-DOC FEATURE COMPARISON")
    print("=" * 70)

    feature_keys = ['top1_doc_ratio', 'top2_doc_ratio', 'cross_doc_entropy', 'effective_docs', 'doc_gini']

    for key in feature_keys:
        easy_vals = [f[key] for f in category_features['easy'] if key in f]
        medium_vals = [f[key] for f in category_features['medium'] if key in f]

        if easy_vals and medium_vals:
            easy_mean = np.mean(easy_vals)
            medium_mean = np.mean(medium_vals)
            overall_std = np.std(easy_vals + medium_vals)

            print(f"\n{key}:")
            print(f"  Easy:   mean={easy_mean:.4f} ± {np.std(easy_vals):.4f}")
            print(f"  Medium: mean={medium_mean:.4f} ± {np.std(medium_vals):.4f}")

            if overall_std > 0:
                sep = abs(easy_mean - medium_mean) / overall_std
                direction = "↑" if easy_mean > medium_mean else "↓"
                print(f"  Separation: {sep:.3f} (Easy {direction})")

    # 建议公式
    print("\n" + "=" * 70)
    print("SUGGESTED FORMULA")
    print("=" * 70)

    if category_features['easy'] and category_features['medium']:
        easy_top1 = np.mean([f['top1_doc_ratio'] for f in category_features['easy']])
        medium_top1 = np.mean([f['top1_doc_ratio'] for f in category_features['medium']])
        easy_cross = np.mean([f['cross_doc_entropy'] for f in category_features['easy']])
        medium_cross = np.mean([f['cross_doc_entropy'] for f in category_features['medium']])

        print(f"""
跨文档特征分析:
  Easy:   top1_doc_ratio={easy_top1:.3f}, cross_doc_entropy={easy_cross:.3f}
  Medium: top1_doc_ratio={medium_top1:.3f}, cross_doc_entropy={medium_cross:.3f}

建议的优雅公式:
  # 基于跨文档 attention 分布
  rate = min_rate + cross_doc_entropy * (max_rate - min_rate)

  物理意义:
  - cross_doc_entropy 低 → attention 集中在单个文档 → rate 低
  - cross_doc_entropy 高 → attention 分散在多个文档 → rate 高

  或者：
  rate = max_rate * (1 - top1_doc_ratio)

  物理意义:
  - top1_doc_ratio 高 → 答案在 top 文档 → rate 低
  - top1_doc_ratio 低 → 需要多文档 → rate 高
""")


if __name__ == "__main__":
    main()
