#!/usr/bin/env python3
"""
分析 rate=0.05 vs rate=0.3 的结果（去重版本）
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


def load_and_compare_results():
    """加载并比较不同 rate 的结果"""
    files = {
        'rate_0.05': 'DraftModel_global_topk_10_rate_0.05_revert_rope.csv',
        'rate_0.3': 'DraftModel_global_topk_10_rate_0.3.csv',
        'rate_1': 'DraftModel_global_topk_10_rate_1.csv',
    }

    results = {}
    for key, filename in files.items():
        filepath = os.path.join(RESULTS_DIR, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            results[key] = df
            correct = df['Correct'].sum()
            total = len(df)
            print(f"{key}: {correct}/{total} correct ({correct/total*100:.1f}%)")

    return results


def categorize_questions(results):
    """将问题分类（去重）"""
    df_005 = results.get('rate_0.05')
    df_03 = results.get('rate_0.3')
    df_1 = results.get('rate_1')

    if df_005 is None or df_03 is None:
        return None

    min_rows = min(len(df_005), len(df_03))
    if df_1 is not None:
        min_rows = min(min_rows, len(df_1))

    categories = {'easy': [], 'medium': [], 'hard': []}
    seen_questions = set()  # 用于去重

    for i in range(min_rows):
        correct_005 = df_005.iloc[i]['Correct']
        correct_03 = df_03.iloc[i]['Correct']
        correct_1 = df_1.iloc[i]['Correct'] if df_1 is not None else None

        sub_q = df_005.iloc[i]['Sub Question']
        main_q = df_005.iloc[i]['Main Question']

        # 去重：使用子问题的前80字符作为key
        q_key = sub_q[:80] if isinstance(sub_q, str) else str(sub_q)[:80]
        if q_key in seen_questions:
            continue
        seen_questions.add(q_key)

        info = {
            'idx': i,
            'sub_q': sub_q,
            'main_q': main_q,
            'correct_005': correct_005,
            'correct_03': correct_03,
            'correct_1': correct_1,
        }

        if correct_005:
            categories['easy'].append(info)
        elif correct_03:
            categories['medium'].append(info)
        else:
            categories['hard'].append(info)

    total = sum(len(v) for v in categories.values())
    print(f"\n=== Question Categories (n={total}, deduplicated) ===")
    print(f"Easy (rate=0.05 correct):    {len(categories['easy'])} ({len(categories['easy'])/total*100:.1f}%)")
    print(f"Medium (need rate>0.05):     {len(categories['medium'])} ({len(categories['medium'])/total*100:.1f}%)")
    print(f"Hard (rate=0.3 still wrong): {len(categories['hard'])} ({len(categories['hard'])/total*100:.1f}%)")

    return categories


def compute_draft_attention_features(model, tokenizer, text, device="cuda:0"):
    """计算 draft model 的 attention 特征"""
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

    return layer_attentions, seq_len


def extract_attention_features(layer_attentions, system_len, doc_len, query_len):
    """从 attention 中提取特征"""
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

    features = {}

    sorted_idx = np.argsort(aggregated_attn)[::-1]
    sorted_attn = aggregated_attn[sorted_idx]
    cumsum = np.cumsum(sorted_attn) / (sorted_attn.sum() + 1e-10)

    for coverage in [0.5, 0.7, 0.8, 0.85, 0.9, 0.95]:
        count = np.searchsorted(cumsum, coverage) + 1
        features[f'cov_{int(coverage*100)}'] = count / doc_len

    for k in [5, 10, 20, 50]:
        if k <= len(aggregated_attn):
            features[f'top{k}_ratio'] = float(sorted_attn[:k].sum() / (sorted_attn.sum() + 1e-10))

    sorted_attn_asc = np.sort(aggregated_attn)
    n = len(sorted_attn_asc)
    index = np.arange(1, n + 1)
    gini = ((2 * index - n - 1) * sorted_attn_asc).sum() / (n * sorted_attn_asc.sum() + 1e-10)
    features['gini'] = float(gini)

    p = aggregated_attn / (aggregated_attn.sum() + 1e-10)
    p = np.clip(p, 1e-10, 1.0)
    entropy = -np.sum(p * np.log(p))
    max_entropy = np.log(doc_len) if doc_len > 0 else 1
    features['norm_entropy'] = float(entropy / max_entropy)

    features['max_attn'] = float(np.max(aggregated_attn))
    features['min_layer_entropy'] = float(np.min(layer_entropies))

    return features


def main():
    device = "cuda:0"

    print("Loading results...")
    results = load_and_compare_results()

    categories = categorize_questions(results)
    if categories is None:
        return

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
    print(f"Model loaded: {model.config.num_hidden_layers} layers")

    sub_q_to_data = {}
    for i, item in enumerate(data):
        intermediate = item.get('intermediate_context', [])
        for sub_idx, sub_q in enumerate(intermediate):
            query = sub_q.get('query', '')
            if query.startswith("Intermediate query"):
                colon_pos = query.find(":")
                if colon_pos != -1:
                    query = query[colon_pos + 1:].strip()
            sub_q_to_data[query[:80]] = (i, sub_idx, sub_q, query)

    category_features = {'easy': [], 'medium': []}
    max_samples = 40

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

            i, sub_idx, sub_q_data, full_query = sub_q_to_data[found_key]
            docs = sub_q_data.get('retrieve docs', [])[:10]
            if not docs:
                continue

            system_prompt = "You are a helpful assistant."
            docs_text = "\n".join(docs)
            full_text = f"{system_prompt}\n\n{docs_text}\n\nQuestion: {full_query}\nAnswer:"

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
                category_features[category].append(features)
                print(f"  [{len(category_features[category])}] gini={features['gini']:.3f}, "
                      f"cov85={features['cov_85']:.3f}, top10={features['top10_ratio']:.3f}")
            except Exception as e:
                print(f"  Error: {e}")
                continue

    # 统计分析
    print("\n" + "=" * 70)
    print("FEATURE COMPARISON (deduplicated)")
    print("=" * 70)

    feature_keys = ['gini', 'norm_entropy', 'top10_ratio', 'cov_85', 'cov_90']
    separation_scores = {}

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
                separation_scores[key] = sep
                direction = "↑" if easy_mean > medium_mean else "↓"
                print(f"  Separation: {sep:.3f} (Easy {direction})")

    # 核心发现
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    if category_features['easy'] and category_features['medium']:
        easy_gini = [f['gini'] for f in category_features['easy']]
        medium_gini = [f['gini'] for f in category_features['medium']]
        easy_cov85 = [f['cov_85'] for f in category_features['easy']]
        medium_cov85 = [f['cov_85'] for f in category_features['medium']]

        # 检查特征重叠
        gini_overlap = len([g for g in easy_gini if min(medium_gini) <= g <= max(medium_gini)]) / len(easy_gini)
        cov85_overlap = len([c for c in easy_cov85 if min(medium_cov85) <= c <= max(medium_cov85)]) / len(easy_cov85)

        print(f"""
1. Easy 问题 (n={len(category_features['easy'])}):
   - gini: {np.mean(easy_gini):.3f} ± {np.std(easy_gini):.3f} (range: {min(easy_gini):.3f} - {max(easy_gini):.3f})
   - cov_85: {np.mean(easy_cov85):.3f} ± {np.std(easy_cov85):.3f} (range: {min(easy_cov85):.3f} - {max(easy_cov85):.3f})

2. Medium 问题 (n={len(category_features['medium'])}):
   - gini: {np.mean(medium_gini):.3f} ± {np.std(medium_gini):.3f} (range: {min(medium_gini):.3f} - {max(medium_gini):.3f})
   - cov_85: {np.mean(medium_cov85):.3f} ± {np.std(medium_cov85):.3f} (range: {min(medium_cov85):.3f} - {max(medium_cov85):.3f})

3. 特征重叠度:
   - Gini 重叠: {gini_overlap*100:.1f}%
   - Cov_85 重叠: {cov85_overlap*100:.1f}%
""")

        if gini_overlap > 0.7 and cov85_overlap > 0.7:
            print("""
⚠️ 结论：Attention 特征的重叠度很高，无法可靠区分 Easy 和 Medium 问题。

这意味着：
- 问题是否需要重算，可能更多取决于问题本身的语义难度
- 而非 attention 分布的集中程度

建议的策略：
- 使用固定的保守 rate（如 0.15-0.20）作为默认值
- 只在 attention 极度集中（如 gini > 0.85）时降低到最小 rate
""")
        else:
            # 计算最优阈值
            all_gini = easy_gini + medium_gini
            all_labels = [0] * len(easy_gini) + [1] * len(medium_gini)

            best_acc = 0
            best_threshold = 0
            for threshold in np.linspace(min(all_gini), max(all_gini), 50):
                preds = [1 if g > threshold else 0 for g in all_gini]
                acc = sum(p == l for p, l in zip(preds, all_labels)) / len(all_labels)
                if acc > best_acc:
                    best_acc = acc
                    best_threshold = threshold

            print(f"""
✓ 可以尝试基于 attention 特征区分：
  最佳 Gini 阈值: {best_threshold:.3f}
  分类准确率: {best_acc*100:.1f}%
""")


if __name__ == "__main__":
    main()
