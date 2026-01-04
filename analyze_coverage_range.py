#!/usr/bin/env python3
"""
分析 coverage 的分布范围，设计能产生更大动态范围的公式。
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


def compute_draft_attention(model, tokenizer, text, device="cuda:0"):
    """计算 draft model 的 attention"""
    config = model.config
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // num_heads

    input_ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)["input_ids"].to(device)
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

    return layer_attentions


def extract_coverage_features(layer_attentions, system_len, doc_len, query_len):
    """提取各种 coverage 特征"""
    total_len = system_len + doc_len + query_len
    doc_start = system_len
    doc_end = system_len + doc_len
    query_start = system_len + doc_len

    layer_doc_attentions = []
    layer_entropies = []

    for layer_attn in layer_attentions:
        query_to_doc = layer_attn[:, query_start:total_len, doc_start:doc_end]
        doc_attention_avg = query_to_doc.mean(axis=(0, 1))
        layer_doc_attentions.append(doc_attention_avg)

        p = doc_attention_avg / (doc_attention_avg.sum() + 1e-10)
        p = np.clip(p, 1e-10, 1.0)
        entropy = -np.sum(p * np.log(p))
        layer_entropies.append(entropy)

    sorted_indices = np.argsort(layer_entropies)[:4]
    selected_attentions = [layer_doc_attentions[i] for i in sorted_indices]
    aggregated_attn = np.mean(selected_attentions, axis=0)

    sorted_idx = np.argsort(aggregated_attn)[::-1]
    sorted_attn = aggregated_attn[sorted_idx]
    cumsum = np.cumsum(sorted_attn) / (sorted_attn.sum() + 1e-10)

    features = {}
    for coverage in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]:
        count = np.searchsorted(cumsum, coverage) + 1
        features[f'cov_{int(coverage*100)}'] = count / doc_len

    # Top-k 占比
    for k in [1, 3, 5, 10]:
        if k <= len(aggregated_attn):
            features[f'top{k}_pct'] = float(sorted_attn[:k].sum())

    return features


def main():
    device = "cuda:0"

    with open('/mnt/data/wjh/FusionRAG/result_reflect.json') as f:
        data = json.load(f)

    print("Loading draft model...")
    tokenizer = AutoTokenizer.from_pretrained(DRAFT_MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        DRAFT_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map=device
    )
    model.eval()

    all_features = []
    max_samples = 60

    for i, item in enumerate(data):
        if len(all_features) >= max_samples:
            break

        intermediate = item.get('intermediate_context', [])
        for sub_idx, sub_q in enumerate(intermediate):
            if len(all_features) >= max_samples:
                break

            query = sub_q.get('query', '')
            if query.startswith("Intermediate query"):
                colon_pos = query.find(":")
                if colon_pos != -1:
                    query = query[colon_pos + 1:].strip()

            docs = sub_q.get('retrieve docs', [])[:10]
            if not docs:
                continue

            system_prompt = "You are a helpful assistant."
            docs_text = "\n".join(docs)
            full_text = f"{system_prompt}\n\n{docs_text}\n\nQuestion: {query}\nAnswer:"

            system_tokens = tokenizer(system_prompt, return_tensors="pt")["input_ids"][0]
            docs_tokens = tokenizer(docs_text, return_tensors="pt")["input_ids"][0]
            full_tokens = tokenizer(full_text, return_tensors="pt")["input_ids"][0]

            system_len = len(system_tokens)
            doc_len = len(docs_tokens)
            query_len = len(full_tokens) - system_len - doc_len

            if doc_len < 100:
                continue

            try:
                layer_attentions = compute_draft_attention(model, tokenizer, full_text, device)
                features = extract_coverage_features(layer_attentions, system_len, doc_len, query_len)
                all_features.append(features)
                print(f"[{len(all_features)}] cov50={features['cov_50']:.3f}, cov70={features['cov_70']:.3f}, "
                      f"cov80={features['cov_80']:.3f}, cov90={features['cov_90']:.3f}")
            except Exception as e:
                print(f"Error: {e}")

    print("\n" + "=" * 70)
    print("COVERAGE DISTRIBUTION STATISTICS")
    print("=" * 70)

    for key in ['cov_50', 'cov_60', 'cov_70', 'cov_80', 'cov_85', 'cov_90', 'cov_95']:
        vals = [f[key] for f in all_features if key in f]
        if vals:
            print(f"\n{key}:")
            print(f"  min={min(vals):.3f}, max={max(vals):.3f}")
            print(f"  mean={np.mean(vals):.3f}, std={np.std(vals):.3f}")
            print(f"  percentiles: 10%={np.percentile(vals, 10):.3f}, "
                  f"25%={np.percentile(vals, 25):.3f}, "
                  f"50%={np.percentile(vals, 50):.3f}, "
                  f"75%={np.percentile(vals, 75):.3f}, "
                  f"90%={np.percentile(vals, 90):.3f}")

    print("\n" + "=" * 70)
    print("TOP-K ATTENTION STATISTICS")
    print("=" * 70)

    for key in ['top1_pct', 'top3_pct', 'top5_pct', 'top10_pct']:
        vals = [f[key] for f in all_features if key in f]
        if vals:
            print(f"\n{key}:")
            print(f"  min={min(vals):.3f}, max={max(vals):.3f}")
            print(f"  mean={np.mean(vals):.3f}, std={np.std(vals):.3f}")

    print("\n" + "=" * 70)
    print("FORMULA SUGGESTIONS")
    print("=" * 70)

    cov70_vals = [f['cov_70'] for f in all_features]
    cov80_vals = [f['cov_80'] for f in all_features]

    print(f"""
Coverage 分布分析:
  cov_70: mean={np.mean(cov70_vals):.3f}, range=[{min(cov70_vals):.3f}, {max(cov70_vals):.3f}]
  cov_80: mean={np.mean(cov80_vals):.3f}, range=[{min(cov80_vals):.3f}, {max(cov80_vals):.3f}]

建议的公式选项:

1. 使用 cov_70 (范围更小，能产生更低的 rate):
   rate = cov_70 * 1.0
   预期 rate 范围: [{min(cov70_vals):.2f}, {max(cov70_vals):.2f}]

2. 使用归一化的 coverage:
   norm_cov = (cov_80 - min) / (max - min)
   rate = min_rate + norm_cov * (max_rate - min_rate)

3. 使用 top-k 占比的反向映射:
   # top10_pct 越高 → attention 越集中 → rate 越低
   rate = max_rate * (1 - top10_pct)
""")

    # 模拟不同公式的 rate 分布
    print("\n" + "=" * 70)
    print("SIMULATED RATE DISTRIBUTIONS")
    print("=" * 70)

    min_rate, max_rate = 0.05, 0.35

    # 公式 1: cov_90 * 0.7
    rates_1 = [min(max(f['cov_90'] * 0.7, min_rate), max_rate) for f in all_features]
    print(f"\n公式 1: rate = cov_90 * 0.7")
    print(f"  Rate 分布: mean={np.mean(rates_1):.3f}, min={min(rates_1):.3f}, max={max(rates_1):.3f}")
    print(f"  低于 0.15 的比例: {sum(1 for r in rates_1 if r < 0.15) / len(rates_1) * 100:.1f}%")

    # 公式 2: cov_70
    rates_2 = [min(max(f['cov_70'], min_rate), max_rate) for f in all_features]
    print(f"\n公式 2: rate = cov_70")
    print(f"  Rate 分布: mean={np.mean(rates_2):.3f}, min={min(rates_2):.3f}, max={max(rates_2):.3f}")
    print(f"  低于 0.15 的比例: {sum(1 for r in rates_2 if r < 0.15) / len(rates_2) * 100:.1f}%")

    # 公式 3: (1 - top10_pct) * max_rate
    rates_3 = [min(max((1 - f['top10_pct']) * max_rate, min_rate), max_rate) for f in all_features]
    print(f"\n公式 3: rate = (1 - top10_pct) * max_rate")
    print(f"  Rate 分布: mean={np.mean(rates_3):.3f}, min={min(rates_3):.3f}, max={max(rates_3):.3f}")
    print(f"  低于 0.15 的比例: {sum(1 for r in rates_3 if r < 0.15) / len(rates_3) * 100:.1f}%")

    # 公式 4: 归一化 cov_80
    cov80_min, cov80_max = min(cov80_vals), max(cov80_vals)
    rates_4 = []
    for f in all_features:
        norm = (f['cov_80'] - cov80_min) / (cov80_max - cov80_min + 1e-10)
        rate = min_rate + norm * (max_rate - min_rate)
        rates_4.append(rate)
    print(f"\n公式 4: 归一化 cov_80 映射到 [min_rate, max_rate]")
    print(f"  Rate 分布: mean={np.mean(rates_4):.3f}, min={min(rates_4):.3f}, max={max(rates_4):.3f}")
    print(f"  低于 0.15 的比例: {sum(1 for r in rates_4 if r < 0.15) / len(rates_4) * 100:.1f}%")


if __name__ == "__main__":
    main()
