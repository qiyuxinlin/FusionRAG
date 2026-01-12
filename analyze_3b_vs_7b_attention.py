#!/usr/bin/env python3
"""
分析 3B 和 7B draft model 在关键失败案例中的 attention 分布差异

目标：找出 7B 能答对但 3B 答错的案例中，两者 token 选择的差异特征
"""

import json
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoConfig

# Add project directory to path
project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_dir)

from per_head_generation import (
    find_connected_components,
    rotate_half
)


def compute_draft_attention_for_analysis(
    draft_model,
    input_ids,
    query_start,
    device="cuda:0"
):
    """
    计算 draft model 的 attention，返回更详细的分析信息
    """
    seq_len = input_ids.shape[1]
    query_len = seq_len - query_start
    num_layers = draft_model.config.num_hidden_layers
    num_heads = draft_model.config.num_attention_heads
    num_kv_heads = draft_model.config.num_key_value_heads
    head_dim = draft_model.config.hidden_size // num_heads

    layer_attention_scores = {}

    with torch.no_grad():
        inputs_embeds = draft_model.model.embed_tokens(input_ids.to(device))
        hidden_states = inputs_embeds
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

        # 获取 rotary_emb
        if hasattr(draft_model.model, 'rotary_emb'):
            rotary_emb = draft_model.model.rotary_emb
            cos, sin = rotary_emb(hidden_states, position_ids)
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
            use_global_rope = True
        else:
            use_global_rope = False
            cos, sin = None, None

        for layer_idx in range(num_layers):
            layer = draft_model.model.layers[layer_idx]

            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)

            bsz, q_len, _ = hidden_states.size()

            # Q, K, V projections
            query_states = layer.self_attn.q_proj(hidden_states)
            key_states = layer.self_attn.k_proj(hidden_states)
            value_states = layer.self_attn.v_proj(hidden_states)

            # Reshape
            query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

            # Apply RoPE
            if not use_global_rope:
                cos, sin = layer.self_attn.rotary_emb(value_states, position_ids)
                cos = cos.unsqueeze(1)
                sin = sin.unsqueeze(1)
            query_states = (query_states * cos) + (rotate_half(query_states) * sin)
            key_states = (key_states * cos) + (rotate_half(key_states) * sin)

            # Expand K, V for GQA
            n_rep = num_heads // num_kv_heads
            key_states_expanded = key_states.repeat_interleave(n_rep, dim=1)
            value_states_expanded = value_states.repeat_interleave(n_rep, dim=1)

            # 对于后 50% 的层，计算 query→all 的 attention
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
                        query_states_prefix,
                        key_states_prefix,
                        value_states_prefix,
                        is_causal=True
                    )
                    attn_output = torch.cat([attn_output_prefix, attn_output_subset], dim=2)
                else:
                    attn_output = attn_output_subset
            else:
                attn_output = F.scaled_dot_product_attention(
                    query_states,
                    key_states_expanded,
                    value_states_expanded,
                    is_causal=True
                )

            attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
            attn_output = layer.self_attn.o_proj(attn_output)

            hidden_states = residual + attn_output

            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states

    return layer_attention_scores


def analyze_attention_features(attention_scores, system_len, doc_len, query_len):
    """
    从 attention 分布中提取特征
    """
    total_len = system_len + doc_len + query_len
    doc_start = system_len
    doc_end = system_len + doc_len
    query_start = system_len + doc_len

    # 计算每层的统计信息
    layer_stats = {}
    layer_attention = {}

    for layer_idx in sorted(attention_scores.keys()):
        layer_attn = attention_scores[layer_idx]  # [num_heads, query_len, seq_len]

        # 提取 query→doc attention
        query_to_doc = layer_attn[:, :, doc_start:doc_end]  # [num_heads, query_len, doc_len]

        # 对所有 heads 和 query positions 平均
        doc_attention_avg = query_to_doc.mean(axis=(0, 1))  # [doc_len]
        layer_attention[layer_idx] = doc_attention_avg

        # 计算熵
        p = doc_attention_avg / (doc_attention_avg.sum() + 1e-10)
        p = np.clip(p, 1e-10, 1.0)
        entropy = -np.sum(p * np.log(p))

        # 计算其他统计量
        layer_stats[layer_idx] = {
            'entropy': entropy,
            'mean': float(np.mean(doc_attention_avg)),
            'std': float(np.std(doc_attention_avg)),
            'max': float(np.max(doc_attention_avg)),
            'min': float(np.min(doc_attention_avg)),
            'gini': compute_gini(doc_attention_avg),
            # Top-k coverage
            'top10_coverage': compute_topk_coverage(doc_attention_avg, k=10),
            'top20_coverage': compute_topk_coverage(doc_attention_avg, k=20),
            'top50_coverage': compute_topk_coverage(doc_attention_avg, k=50),
        }

    return layer_stats, layer_attention


def compute_gini(values):
    """计算基尼系数（衡量分布不均匀程度）"""
    values = np.sort(values)
    n = len(values)
    cumsum = np.cumsum(values)
    return (2 * np.sum((np.arange(1, n+1) * values)) - (n + 1) * cumsum[-1]) / (n * cumsum[-1] + 1e-10)


def compute_topk_coverage(attention, k):
    """计算 top-k tokens 覆盖了多少 attention"""
    sorted_attn = np.sort(attention)[::-1]
    return float(sorted_attn[:k].sum() / (attention.sum() + 1e-10))


def select_tokens_and_analyze(attention_scores, system_len, doc_len, query_len, target_ratio=0.2):
    """
    选择 tokens 并返回详细分析
    """
    total_len = system_len + doc_len + query_len
    doc_start = system_len
    doc_end = system_len + doc_len
    query_start = system_len + doc_len

    # 计算每层熵
    layer_entropy = {}
    layer_attention = {}

    for layer_idx in sorted(attention_scores.keys()):
        layer_attn = attention_scores[layer_idx]
        query_to_doc = layer_attn[:, :, doc_start:doc_end]
        doc_attention_avg = query_to_doc.mean(axis=(0, 1))
        layer_attention[layer_idx] = doc_attention_avg

        p = doc_attention_avg / (doc_attention_avg.sum() + 1e-10)
        p = np.clip(p, 1e-10, 1.0)
        entropy = -np.sum(p * np.log(p))
        layer_entropy[layer_idx] = entropy

    # 选择熵最低的 top-4 层
    top_k = 4
    sorted_layers = sorted(layer_entropy.items(), key=lambda x: x[1])
    layers_to_use = [layer_idx for layer_idx, _ in sorted_layers[:top_k]]

    # 聚合选中层的 attention
    multi_layer_attn = np.stack([layer_attention[l] for l in layers_to_use]).mean(axis=0)

    target_count = int(doc_len * target_ratio)

    # 找高 attention 位置
    mean_attn = np.mean(multi_layer_attn)
    std_attn = np.std(multi_layer_attn)
    threshold = mean_attn + 0.5 * std_attn

    high_attn_positions = list(np.where(multi_layer_attn > threshold)[0])

    # 连通分量分析
    components = find_connected_components(high_attn_positions, max_gap=2)

    # 计算分量分数并排序
    component_scores = []
    for comp in components:
        total_score = sum(multi_layer_attn[p] for p in comp)
        component_scores.append((comp, total_score))
    component_scores.sort(key=lambda x: x[1], reverse=True)

    # 贪心选择
    selected = set()
    for comp, total_score in component_scores:
        extended_comp = set()
        for p in comp:
            for offset in range(-1, 2):
                new_p = p + offset
                if 0 <= new_p < doc_len:
                    extended_comp.add(new_p)

        new_positions = extended_comp - selected
        if len(selected) + len(new_positions) <= target_count * 1.1:
            selected.update(extended_comp)

    # 补充到目标数量
    if len(selected) < target_count:
        sorted_indices = np.argsort(multi_layer_attn)[::-1]
        for pos in sorted_indices:
            if pos not in selected:
                selected.add(int(pos))
                if len(selected) >= target_count:
                    break

    # 移除多余的
    while len(selected) > target_count:
        min_pos = min(selected, key=lambda p: multi_layer_attn[p])
        selected.remove(min_pos)

    selected_list = sorted(list(selected))

    return {
        'selected_positions': selected_list,
        'selected_layers': layers_to_use,
        'layer_entropy': layer_entropy,
        'attention_scores': multi_layer_attn,
        'num_components': len(components),
        'num_high_attn_positions': len(high_attn_positions),
        'threshold': threshold,
        'mean_attn': mean_attn,
        'std_attn': std_attn,
    }


def compare_3b_7b_selections(selection_3b, selection_7b, doc_len):
    """
    比较 3B 和 7B 的选择差异
    """
    set_3b = set(selection_3b['selected_positions'])
    set_7b = set(selection_7b['selected_positions'])

    overlap = set_3b & set_7b
    only_3b = set_3b - set_7b
    only_7b = set_7b - set_3b

    # IoU
    iou = len(overlap) / (len(set_3b | set_7b) + 1e-10)

    # Attention 分数差异
    attn_3b = selection_3b['attention_scores']
    attn_7b = selection_7b['attention_scores']

    # 计算相关性
    correlation = np.corrcoef(attn_3b, attn_7b)[0, 1]

    # 7B 独有位置的 attention 在 3B 中的排名
    only_7b_ranks_in_3b = []
    sorted_3b_indices = np.argsort(attn_3b)[::-1]
    rank_dict_3b = {idx: rank for rank, idx in enumerate(sorted_3b_indices)}
    for pos in only_7b:
        only_7b_ranks_in_3b.append(rank_dict_3b.get(pos, doc_len))

    return {
        'overlap': len(overlap),
        'only_3b': len(only_3b),
        'only_7b': len(only_7b),
        'iou': iou,
        'correlation': correlation,
        'only_7b_avg_rank_in_3b': np.mean(only_7b_ranks_in_3b) if only_7b_ranks_in_3b else 0,
        'layer_overlap': len(set(selection_3b['selected_layers']) & set(selection_7b['selected_layers'])),
        '7b_entropy_diff': np.mean(list(selection_7b['layer_entropy'].values())) - np.mean(list(selection_3b['layer_entropy'].values())),
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
    print(f"Will analyze {min(args.max_cases, len(critical_cases))} cases")

    # Load models
    print("\n[1] Loading 3B model...")
    model_3b_path = '/mnt/data/models/Qwen2.5-3B-Instruct'
    config_3b = AutoConfig.from_pretrained(model_3b_path, trust_remote_code=True)
    config_3b._attn_implementation = "sdpa"

    from test_fusionrag_reflect import load_model
    model_3b, _ = load_model('qwen', model_3b_path, config_3b, device, use_multi_gpu=False)
    model_3b.eval()

    print("\n[2] Loading 7B model...")
    model_7b_path = '/mnt/data/models/Qwen2.5-7B-Instruct'
    config_7b = AutoConfig.from_pretrained(model_7b_path, trust_remote_code=True)
    config_7b._attn_implementation = "sdpa"
    model_7b, _ = load_model('qwen', model_7b_path, config_7b, device, use_multi_gpu=False)
    model_7b.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_7b_path, trust_remote_code=True)

    # Load data to get document texts
    print("\n[3] Loading dataset...")
    from test_fusionrag_reflect import load_system_prompt, prepare_reflect_data

    # Load full dataset
    data_path = './data/result_reflect.json'
    bge_model_path = '/mnt/data/models/bge-m3-FP16'

    questions_data, system_tensor, _, _ = prepare_reflect_data(
        data_path, tokenizer, bge_model_path, 'qwen', 10,
        max_main_questions=200,
        preprocess=False
    )

    system_len = system_tensor.shape[0]

    # 分析结果
    analysis_results = []

    for case_idx, case in enumerate(critical_cases[:args.max_cases]):
        print(f"\n{'='*80}")
        print(f"Case {case_idx + 1}/{min(args.max_cases, len(critical_cases))}")
        print(f"{'='*80}")
        print(f"Question: {case['sub_question'][:100]}...")
        print(f"Ground truth: {case['ground_truth'][:100]}...")
        print(f"3B predicted: {case['pred_3b'][:100]}...")
        print(f"7B predicted: {case['pred_7b'][:100]}...")

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
            print("  WARNING: Could not find matching question in dataset")
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

        print(f"\n  Input: system={system_len}, doc={doc_len}, query={query_len}")

        # 计算 attention
        print("  Computing 3B attention...")
        attn_3b = compute_draft_attention_for_analysis(model_3b, full_input, query_start, device)

        print("  Computing 7B attention...")
        attn_7b = compute_draft_attention_for_analysis(model_7b, full_input, query_start, device)

        # 分析和选择
        print("  Selecting tokens...")
        selection_3b = select_tokens_and_analyze(attn_3b, system_len, doc_len, query_len, args.rate)
        selection_7b = select_tokens_and_analyze(attn_7b, system_len, doc_len, query_len, args.rate)

        # 比较
        comparison = compare_3b_7b_selections(selection_3b, selection_7b, doc_len)

        print(f"\n  === Comparison Results ===")
        print(f"  Selection overlap: {comparison['overlap']} tokens")
        print(f"  Only 3B selected: {comparison['only_3b']} tokens")
        print(f"  Only 7B selected: {comparison['only_7b']} tokens")
        print(f"  IoU: {comparison['iou']:.4f}")
        print(f"  Attention correlation: {comparison['correlation']:.4f}")
        print(f"  Avg rank of 7B-only positions in 3B: {comparison['only_7b_avg_rank_in_3b']:.1f}")
        print(f"  Layer overlap: {comparison['layer_overlap']}/4")
        print(f"  Entropy diff (7B - 3B): {comparison['7b_entropy_diff']:.4f}")

        print(f"\n  3B selected layers: {selection_3b['selected_layers']}")
        print(f"  7B selected layers: {selection_7b['selected_layers']}")

        # 看看 7B 独有选择的位置对应什么文本
        if comparison['only_7b'] > 0:
            all_doc_tokens = torch.cat(sub_q_doc_tensors)
            only_7b_positions = sorted(set(selection_7b['selected_positions']) - set(selection_3b['selected_positions']))

            print(f"\n  === 7B-only selected positions (first 10) ===")
            for i, pos in enumerate(only_7b_positions[:10]):
                # 获取上下文
                start = max(0, pos - 2)
                end = min(doc_len, pos + 3)
                context_tokens = all_doc_tokens[start:end].tolist()
                context_text = tokenizer.decode(context_tokens)
                attn_3b_val = selection_3b['attention_scores'][pos]
                attn_7b_val = selection_7b['attention_scores'][pos]
                print(f"    pos {pos}: '{context_text}' | 3B attn: {attn_3b_val:.6f}, 7B attn: {attn_7b_val:.6f}")

        analysis_results.append({
            'case': case,
            'comparison': comparison,
            'selection_3b_layers': selection_3b['selected_layers'],
            'selection_7b_layers': selection_7b['selected_layers'],
            'doc_len': doc_len,
        })

        torch.cuda.empty_cache()

    # 汇总统计
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")

    if analysis_results:
        avg_iou = np.mean([r['comparison']['iou'] for r in analysis_results])
        avg_corr = np.mean([r['comparison']['correlation'] for r in analysis_results])
        avg_only_7b = np.mean([r['comparison']['only_7b'] for r in analysis_results])
        avg_rank = np.mean([r['comparison']['only_7b_avg_rank_in_3b'] for r in analysis_results])
        avg_layer_overlap = np.mean([r['comparison']['layer_overlap'] for r in analysis_results])

        print(f"Average IoU: {avg_iou:.4f}")
        print(f"Average attention correlation: {avg_corr:.4f}")
        print(f"Average # of 7B-only positions: {avg_only_7b:.1f}")
        print(f"Average rank of 7B-only positions in 3B: {avg_rank:.1f}")
        print(f"Average layer overlap: {avg_layer_overlap:.2f}/4")

    # Save results
    output_path = '/mnt/data/wjh/FusionRAG/analysis_3b_vs_7b_attention.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
