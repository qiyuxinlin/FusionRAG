#!/usr/bin/env python3
"""
分析 rate=0.05 vs rate=0.3 vs rate=1 的结果，
找出哪些问题不需要重算（rate=0.05 就对）vs 需要重算的问题。
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
    """将问题分类为 easy/medium/hard"""
    df_005 = results.get('rate_0.05')
    df_03 = results.get('rate_0.3')
    df_1 = results.get('rate_1')

    if df_005 is None or df_03 is None:
        print("Missing required result files!")
        return None

    # 确保行数一致
    min_rows = min(len(df_005), len(df_03))
    if df_1 is not None:
        min_rows = min(min_rows, len(df_1))

    categories = {
        'easy': [],      # rate=0.05 就能答对
        'medium': [],    # rate=0.05 错，rate=0.3 对
        'hard': [],      # rate=0.3 也错
    }

    for i in range(min_rows):
        correct_005 = df_005.iloc[i]['Correct']
        correct_03 = df_03.iloc[i]['Correct']
        correct_1 = df_1.iloc[i]['Correct'] if df_1 is not None else None

        sub_q = df_005.iloc[i]['Sub Question']
        main_q = df_005.iloc[i]['Main Question']

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

    print(f"\n=== Question Categories (n={min_rows}) ===")
    print(f"Easy (rate=0.05 correct):    {len(categories['easy'])} ({len(categories['easy'])/min_rows*100:.1f}%)")
    print(f"Medium (need rate>0.05):     {len(categories['medium'])} ({len(categories['medium'])/min_rows*100:.1f}%)")
    print(f"Hard (rate=0.3 still wrong): {len(categories['hard'])} ({len(categories['hard'])/min_rows*100:.1f}%)")

    # 分析 medium 类别中，有多少在 rate=1 时能答对
    if df_1 is not None:
        medium_recoverable = sum(1 for q in categories['medium'] if q['correct_1'])
        hard_recoverable = sum(1 for q in categories['hard'] if q['correct_1'])
        print(f"\nMedium questions correct at rate=1: {medium_recoverable}/{len(categories['medium'])}")
        print(f"Hard questions correct at rate=1: {hard_recoverable}/{len(categories['hard'])}")

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
    for k in [5, 10, 20, 50]:
        if k <= len(aggregated_attn):
            features[f'top{k}_ratio'] = float(sorted_attn[:k].sum() / (sorted_attn.sum() + 1e-10))

    # 3. Gini 系数
    sorted_attn_asc = np.sort(aggregated_attn)
    n = len(sorted_attn_asc)
    index = np.arange(1, n + 1)
    gini = ((2 * index - n - 1) * sorted_attn_asc).sum() / (n * sorted_attn_asc.sum() + 1e-10)
    features['gini'] = float(gini)

    # 4. 归一化熵
    p = aggregated_attn / (aggregated_attn.sum() + 1e-10)
    p = np.clip(p, 1e-10, 1.0)
    entropy = -np.sum(p * np.log(p))
    max_entropy = np.log(doc_len) if doc_len > 0 else 1
    features['norm_entropy'] = float(entropy / max_entropy)

    # 5. 最大 attention 值
    features['max_attn'] = float(np.max(aggregated_attn))

    # 6. 选择层的最低熵
    features['min_layer_entropy'] = float(np.min(layer_entropies))

    return features


def main():
    device = "cuda:0"

    # 加载并比较结果
    print("Loading results...")
    results = load_and_compare_results()

    categories = categorize_questions(results)
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
            sub_q_to_data[query[:80]] = (i, sub_idx, sub_q, query)

    # 为每个类别收集 attention 特征
    category_features = {
        'easy': [],
        'medium': [],
    }

    max_samples_per_category = 50

    for category in ['easy', 'medium']:
        print(f"\n=== Analyzing {category} cases ===")
        samples = categories[category][:max_samples_per_category * 2]  # 多取一些以防匹配失败

        for sample in samples:
            if len(category_features[category]) >= max_samples_per_category:
                break

            sub_q = sample['sub_q']

            # 查找对应的数据
            found_key = None
            for key in sub_q_to_data.keys():
                if key in sub_q or sub_q[:60] in key:
                    found_key = key
                    break

            if found_key is None:
                continue

            i, sub_idx, sub_q_data, full_query = sub_q_to_data[found_key]
            docs = sub_q_data.get('retrieve docs', [])[:10]
            if not docs:
                continue

            # 构建输入
            system_prompt = "You are a helpful assistant."
            docs_text = "\n".join(docs)
            full_text = f"{system_prompt}\n\n{docs_text}\n\nQuestion: {full_query}\nAnswer:"

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
                features['query'] = full_query[:50]
                category_features[category].append(features)
                print(f"  [{len(category_features[category])}] doc_len={doc_len}, "
                      f"gini={features['gini']:.3f}, top10={features['top10_ratio']:.3f}, "
                      f"cov85={features['cov_85']:.3f}")
            except Exception as e:
                print(f"  Error: {e}")
                continue

    # 统计分析
    print("\n" + "=" * 70)
    print("FEATURE COMPARISON: Easy (rate=0.05 OK) vs Medium (need higher rate)")
    print("=" * 70)

    feature_keys = ['gini', 'norm_entropy', 'top10_ratio', 'top20_ratio',
                    'cov_80', 'cov_85', 'cov_90', 'max_attn', 'min_layer_entropy']

    separation_scores = {}

    for key in feature_keys:
        easy_vals = [f[key] for f in category_features['easy'] if key in f]
        medium_vals = [f[key] for f in category_features['medium'] if key in f]

        if easy_vals and medium_vals:
            easy_mean = np.mean(easy_vals)
            medium_mean = np.mean(medium_vals)
            overall_std = np.std(easy_vals + medium_vals)

            print(f"\n{key}:")
            print(f"  Easy:   mean={easy_mean:.4f}, std={np.std(easy_vals):.4f}")
            print(f"  Medium: mean={medium_mean:.4f}, std={np.std(medium_vals):.4f}")

            if overall_std > 0:
                sep = abs(easy_mean - medium_mean) / overall_std
                separation_scores[key] = sep
                direction = "Easy > Medium" if easy_mean > medium_mean else "Easy < Medium"
                print(f"  Separation: {sep:.3f} ({direction})")

    # 找出最好的区分特征
    print("\n" + "=" * 70)
    print("BEST DISCRIMINATING FEATURES (sorted by separation)")
    print("=" * 70)
    sorted_features = sorted(separation_scores.items(), key=lambda x: x[1], reverse=True)
    for feat, sep in sorted_features:
        print(f"  {feat}: {sep:.3f}")

    # 计算建议的阈值
    print("\n" + "=" * 70)
    print("THRESHOLD RECOMMENDATIONS")
    print("=" * 70)

    if category_features['easy'] and category_features['medium']:
        # 使用最佳区分特征
        best_feature = sorted_features[0][0] if sorted_features else 'gini'

        easy_vals = [f[best_feature] for f in category_features['easy']]
        medium_vals = [f[best_feature] for f in category_features['medium']]

        # 计算阈值（使用两组的中间点）
        easy_mean = np.mean(easy_vals)
        medium_mean = np.mean(medium_vals)
        threshold = (easy_mean + medium_mean) / 2

        # 计算在该阈值下的分类准确率
        if easy_mean > medium_mean:
            # 高于阈值 = easy
            easy_correct = sum(1 for v in easy_vals if v > threshold)
            medium_correct = sum(1 for v in medium_vals if v <= threshold)
        else:
            # 低于阈值 = easy
            easy_correct = sum(1 for v in easy_vals if v < threshold)
            medium_correct = sum(1 for v in medium_vals if v >= threshold)

        total_correct = easy_correct + medium_correct
        total = len(easy_vals) + len(medium_vals)

        print(f"\nBest feature: {best_feature}")
        print(f"  Threshold: {threshold:.4f}")
        print(f"  Classification accuracy: {total_correct}/{total} ({total_correct/total*100:.1f}%)")
        print(f"  Easy correctly classified: {easy_correct}/{len(easy_vals)}")
        print(f"  Medium correctly classified: {medium_correct}/{len(medium_vals)}")

    # 设计公式
    print("\n" + "=" * 70)
    print("SUGGESTED FORMULA FOR DYNAMIC RATE")
    print("=" * 70)

    if category_features['easy'] and category_features['medium']:
        easy_cov85 = np.mean([f['cov_85'] for f in category_features['easy']])
        medium_cov85 = np.mean([f['cov_85'] for f in category_features['medium']])
        easy_gini = np.mean([f['gini'] for f in category_features['easy']])
        medium_gini = np.mean([f['gini'] for f in category_features['medium']])

        print(f"""
观察结果:
  - Easy 问题: cov_85 平均 = {easy_cov85:.3f}, gini 平均 = {easy_gini:.3f}
  - Medium 问题: cov_85 平均 = {medium_cov85:.3f}, gini 平均 = {medium_gini:.3f}

建议公式:
  concentration_score = gini * (1 - cov_85)

  if concentration_score > 0.5:  # attention 非常集中
      rate = 0.05  # 几乎不需要重算
  else:
      rate = cov_85 * 1.2  # 基于覆盖率计算
      rate = clip(rate, 0.05, 0.30)
""")

    # 保存结果
    output = {
        'easy': category_features['easy'],
        'medium': category_features['medium'],
        'separation_scores': separation_scores,
    }
    with open('rate_comparison_analysis.json', 'w') as f:
        json.dump(output, f, indent=2, default=float)

    print("\nResults saved to rate_comparison_analysis.json")


if __name__ == "__main__":
    main()
