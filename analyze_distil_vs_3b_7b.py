#!/usr/bin/env python3
"""
对比 3B、DistilQwen2.5-3B 和 7B 的 attention 分布

分析蒸馏模型的 attention 是更接近原始 3B 还是 7B
"""

import json
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoConfig

project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_dir)

from per_head_generation import find_connected_components, rotate_half


def compute_attention_for_analysis(model, input_ids, query_start, device="cuda:0"):
    """计算模型的 attention 分布"""
    seq_len = input_ids.shape[1]
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    num_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // num_heads

    layer_attention_scores = {}

    with torch.no_grad():
        inputs_embeds = model.model.embed_tokens(input_ids.to(device))
        hidden_states = inputs_embeds
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

        if hasattr(model.model, 'rotary_emb'):
            rotary_emb = model.model.rotary_emb
            cos, sin = rotary_emb(hidden_states, position_ids)
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
            use_global_rope = True
        else:
            use_global_rope = False
            cos, sin = None, None

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

            if not use_global_rope:
                cos, sin = layer.self_attn.rotary_emb(value_states, position_ids)
                cos = cos.unsqueeze(1)
                sin = sin.unsqueeze(1)
            query_states = (query_states * cos) + (rotate_half(query_states) * sin)
            key_states = (key_states * cos) + (rotate_half(key_states) * sin)

            n_rep = num_heads // num_kv_heads
            key_states_expanded = key_states.repeat_interleave(n_rep, dim=1)
            value_states_expanded = value_states.repeat_interleave(n_rep, dim=1)

            # 只对后 50% 的层计算 attention
            if layer_idx >= num_layers // 2:
                query_states_subset = query_states[:, :, query_start:, :]
                attn_weights_subset = torch.matmul(
                    query_states_subset.float(),
                    key_states_expanded.float().transpose(2, 3)
                ) / (head_dim ** 0.5)

                query_positions = torch.arange(query_start, seq_len, device=device)
                key_positions = torch.arange(seq_len, device=device)
                causal_mask = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
                attn_weights_subset = attn_weights_subset.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

                attn_weights_subset = F.softmax(attn_weights_subset, dim=-1)
                layer_attention_scores[layer_idx] = attn_weights_subset[0].cpu().float().numpy()

                attn_output_subset = torch.matmul(
                    attn_weights_subset.to(value_states_expanded.dtype),
                    value_states_expanded
                )

                if query_start > 0:
                    query_states_prefix = query_states[:, :, :query_start, :]
                    key_states_prefix = key_states_expanded[:, :, :query_start, :]
                    value_states_prefix = value_states_expanded[:, :, :query_start, :]
                    attn_output_prefix = F.scaled_dot_product_attention(
                        query_states_prefix, key_states_prefix, value_states_prefix, is_causal=True
                    )
                    attn_output = torch.cat([attn_output_prefix, attn_output_subset], dim=2)
                else:
                    attn_output = attn_output_subset
            else:
                attn_output = F.scaled_dot_product_attention(
                    query_states, key_states_expanded, value_states_expanded, is_causal=True
                )

            attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
            attn_output = layer.self_attn.o_proj(attn_output)
            hidden_states = residual + attn_output
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states

    return layer_attention_scores


def extract_doc_attention(attention_scores, system_len, doc_len, query_len):
    """提取 query->doc 的 attention"""
    total_len = system_len + doc_len + query_len
    doc_start = system_len
    doc_end = system_len + doc_len
    query_start_pos = system_len + doc_len

    layer_doc_attention = {}
    layer_entropy = {}

    for layer_idx in sorted(attention_scores.keys()):
        layer_attn = attention_scores[layer_idx]  # [num_heads, query_len, seq_len]
        query_to_doc = layer_attn[:, :, doc_start:doc_end]  # [num_heads, query_len, doc_len]
        doc_attention_avg = query_to_doc.mean(axis=(0, 1))  # [doc_len]
        layer_doc_attention[layer_idx] = doc_attention_avg

        # 计算熵
        p = doc_attention_avg / (doc_attention_avg.sum() + 1e-10)
        p = np.clip(p, 1e-10, 1.0)
        entropy = -np.sum(p * np.log(p))
        layer_entropy[layer_idx] = entropy

    return layer_doc_attention, layer_entropy


def select_tokens(layer_doc_attention, layer_entropy, doc_len, target_ratio=0.2):
    """根据 attention 选择 tokens"""
    # 选择熵最低的 4 层
    sorted_layers = sorted(layer_entropy.items(), key=lambda x: x[1])
    layers_to_use = [layer_idx for layer_idx, _ in sorted_layers[:4]]

    # 聚合 attention
    multi_layer_attn = np.stack([layer_doc_attention[l] for l in layers_to_use]).mean(axis=0)

    target_count = int(doc_len * target_ratio)

    # 找高 attention 位置
    mean_attn = np.mean(multi_layer_attn)
    std_attn = np.std(multi_layer_attn)
    threshold = mean_attn + 0.5 * std_attn
    high_attn_positions = list(np.where(multi_layer_attn > threshold)[0])

    # 连通分量
    components = find_connected_components(high_attn_positions, max_gap=2)
    component_scores = [(comp, sum(multi_layer_attn[p] for p in comp)) for comp in components]
    component_scores.sort(key=lambda x: x[1], reverse=True)

    # 贪心选择
    selected = set()
    for comp, _ in component_scores:
        extended_comp = set()
        for p in comp:
            for offset in range(-1, 2):
                new_p = p + offset
                if 0 <= new_p < doc_len:
                    extended_comp.add(new_p)
        if len(selected) + len(extended_comp - selected) <= target_count * 1.1:
            selected.update(extended_comp)

    # 补充
    if len(selected) < target_count:
        sorted_indices = np.argsort(multi_layer_attn)[::-1]
        for pos in sorted_indices:
            if pos not in selected:
                selected.add(int(pos))
                if len(selected) >= target_count:
                    break

    while len(selected) > target_count:
        min_pos = min(selected, key=lambda p: multi_layer_attn[p])
        selected.remove(min_pos)

    return sorted(list(selected)), layers_to_use, multi_layer_attn


def compare_attentions(attn_a, attn_b, name_a, name_b):
    """比较两个 attention 分布"""
    correlation = np.corrcoef(attn_a, attn_b)[0, 1]

    # KL divergence (symmetric)
    p = attn_a / (attn_a.sum() + 1e-10)
    q = attn_b / (attn_b.sum() + 1e-10)
    p = np.clip(p, 1e-10, 1.0)
    q = np.clip(q, 1e-10, 1.0)
    kl_pq = np.sum(p * np.log(p / q))
    kl_qp = np.sum(q * np.log(q / p))
    kl_symmetric = (kl_pq + kl_qp) / 2

    # Cosine similarity
    cosine_sim = np.dot(attn_a, attn_b) / (np.linalg.norm(attn_a) * np.linalg.norm(attn_b) + 1e-10)

    # L2 distance
    l2_dist = np.linalg.norm(attn_a - attn_b)

    return {
        'correlation': correlation,
        'kl_symmetric': kl_symmetric,
        'cosine_sim': cosine_sim,
        'l2_dist': l2_dist,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--max_cases', type=int, default=5)
    parser.add_argument('--rate', type=float, default=0.2)
    args = parser.parse_args()

    device = args.device

    # Load critical cases
    with open('/mnt/data/wjh/FusionRAG/critical_cases_3b_vs_7b.json', 'r', encoding='utf-8') as f:
        critical_cases = json.load(f)

    print(f"Loaded {len(critical_cases)} critical cases")
    print(f"Will analyze {min(args.max_cases, len(critical_cases))} cases\n")

    # Load models
    from test_fusionrag_reflect import load_model

    models_info = [
        ('3B', '/mnt/data/models/Qwen2.5-3B-Instruct'),
        ('Distil-3B', '/mnt/data/models/DistilQwen2.5-3B-Instruct-BF16'),
        ('7B', '/mnt/data/models/Qwen2.5-7B-Instruct'),
    ]

    models = {}
    for name, path in models_info:
        print(f"Loading {name} from {path}...")
        config = AutoConfig.from_pretrained(path, trust_remote_code=True)
        config._attn_implementation = "sdpa"
        model, _ = load_model('qwen', path, config, device, use_multi_gpu=False)
        model.eval()
        models[name] = model
        print(f"  Loaded: {model.config.num_hidden_layers} layers")

    tokenizer = AutoTokenizer.from_pretrained(models_info[0][1], trust_remote_code=True)

    # Load dataset
    print("\nLoading dataset...")
    from test_fusionrag_reflect import prepare_reflect_data

    questions_data, system_tensor, _, _ = prepare_reflect_data(
        './data/result_reflect.json', tokenizer, '/mnt/data/models/bge-m3-FP16', 'qwen', 10,
        max_main_questions=200, preprocess=False
    )

    system_len = system_tensor.shape[0]

    # 统计结果
    all_comparisons = {
        '3B_vs_7B': [],
        'Distil_vs_7B': [],
        'Distil_vs_3B': [],
    }

    all_iou = {
        '3B_vs_7B': [],
        'Distil_vs_7B': [],
        'Distil_vs_3B': [],
    }

    for case_idx, case in enumerate(critical_cases[:args.max_cases]):
        print(f"\n{'='*80}")
        print(f"Case {case_idx + 1}/{min(args.max_cases, len(critical_cases))}")
        print(f"{'='*80}")
        print(f"Question: {case['sub_question'][:80]}...")

        # 找到对应的问题数据
        found = False
        for q_data in questions_data:
            for sub_q_info in q_data['sub_questions']:
                if sub_q_info['query'] == case['sub_question']:
                    found = True
                    break
            if found:
                break

        if not found:
            print("  WARNING: Could not find question")
            continue

        # 构建输入
        doc_chunk_ids = sub_q_info['chunk_ids']
        doc_tensors = q_data['doc_tensors']
        sub_q_doc_tensors = [doc_tensors[chunk_id - 1] for chunk_id in doc_chunk_ids]

        question_text = f"<|im_end|>\n<|im_start|>user\nQuestion: {sub_q_info['query']}<|im_end|>\n<|im_start|>assistant\nAnswer: "
        question_tokens = tokenizer.encode(question_text, add_special_tokens=False)
        question_tensor = torch.tensor(question_tokens, dtype=torch.long)

        all_tokens = [system_tensor] + sub_q_doc_tensors + [question_tensor]
        full_input = torch.cat(all_tokens).unsqueeze(0).to(device)

        doc_len = sum(t.shape[0] for t in sub_q_doc_tensors)
        query_len = question_tensor.shape[0]
        query_start = system_len + doc_len

        print(f"  Input: system={system_len}, doc={doc_len}, query={query_len}")

        # 计算各模型的 attention
        attentions = {}
        selections = {}
        aggregated_attns = {}

        for name, model in models.items():
            print(f"  Computing {name} attention...")
            attn_scores = compute_attention_for_analysis(model, full_input, query_start, device)
            layer_doc_attn, layer_entropy = extract_doc_attention(attn_scores, system_len, doc_len, query_len)
            selected, layers_used, agg_attn = select_tokens(layer_doc_attn, layer_entropy, doc_len, args.rate)

            attentions[name] = layer_doc_attn
            selections[name] = set(selected)
            aggregated_attns[name] = agg_attn

            print(f"    Selected layers: {layers_used}")
            print(f"    Selected tokens: {len(selected)}")

        # 比较 attention 分布
        print(f"\n  === Attention Distribution Comparison ===")

        comparisons = [
            ('3B', '7B', '3B_vs_7B'),
            ('Distil-3B', '7B', 'Distil_vs_7B'),
            ('Distil-3B', '3B', 'Distil_vs_3B'),
        ]

        for name_a, name_b, key in comparisons:
            comp = compare_attentions(aggregated_attns[name_a], aggregated_attns[name_b], name_a, name_b)
            all_comparisons[key].append(comp)
            print(f"  {name_a} vs {name_b}:")
            print(f"    Correlation: {comp['correlation']:.4f}")
            print(f"    Cosine Sim:  {comp['cosine_sim']:.4f}")
            print(f"    KL Divergence: {comp['kl_symmetric']:.4f}")

        # 比较 token 选择
        print(f"\n  === Token Selection Comparison ===")

        for name_a, name_b, key in comparisons:
            set_a = selections[name_a]
            set_b = selections[name_b]
            overlap = len(set_a & set_b)
            union = len(set_a | set_b)
            iou = overlap / union if union > 0 else 0
            all_iou[key].append(iou)
            print(f"  {name_a} vs {name_b}: IoU = {iou:.4f} (overlap={overlap}, union={union})")

        torch.cuda.empty_cache()

    # 汇总统计
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")

    print("\n=== Attention Distribution ===")
    print(f"{'Comparison':<20} {'Correlation':<15} {'Cosine Sim':<15} {'KL Divergence':<15}")
    print("-" * 65)
    for key in ['3B_vs_7B', 'Distil_vs_7B', 'Distil_vs_3B']:
        if all_comparisons[key]:
            avg_corr = np.mean([c['correlation'] for c in all_comparisons[key]])
            avg_cos = np.mean([c['cosine_sim'] for c in all_comparisons[key]])
            avg_kl = np.mean([c['kl_symmetric'] for c in all_comparisons[key]])
            print(f"{key:<20} {avg_corr:<15.4f} {avg_cos:<15.4f} {avg_kl:<15.4f}")

    print("\n=== Token Selection IoU ===")
    print(f"{'Comparison':<20} {'Avg IoU':<15}")
    print("-" * 35)
    for key in ['3B_vs_7B', 'Distil_vs_7B', 'Distil_vs_3B']:
        if all_iou[key]:
            avg_iou = np.mean(all_iou[key])
            print(f"{key:<20} {avg_iou:<15.4f}")

    # 判断 Distil-3B 更接近谁
    print("\n=== Conclusion ===")
    if all_comparisons['Distil_vs_7B'] and all_comparisons['Distil_vs_3B']:
        distil_7b_corr = np.mean([c['correlation'] for c in all_comparisons['Distil_vs_7B']])
        distil_3b_corr = np.mean([c['correlation'] for c in all_comparisons['Distil_vs_3B']])
        distil_7b_iou = np.mean(all_iou['Distil_vs_7B'])
        distil_3b_iou = np.mean(all_iou['Distil_vs_3B'])

        print(f"Distil-3B vs 7B: correlation={distil_7b_corr:.4f}, IoU={distil_7b_iou:.4f}")
        print(f"Distil-3B vs 3B: correlation={distil_3b_corr:.4f}, IoU={distil_3b_iou:.4f}")

        if distil_3b_corr > distil_7b_corr:
            print(f"\n→ Distil-3B attention is MORE SIMILAR to 3B (correlation diff: {distil_3b_corr - distil_7b_corr:.4f})")
        else:
            print(f"\n→ Distil-3B attention is MORE SIMILAR to 7B (correlation diff: {distil_7b_corr - distil_3b_corr:.4f})")


if __name__ == '__main__':
    main()
