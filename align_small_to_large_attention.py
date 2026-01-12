#!/usr/bin/env python3
"""
探索对齐小模型和大模型 attention 分布的方法

方法1: 层映射 (Layer Mapping)
  - 3B 有 36 层，7B 有 28 层
  - 找到语义上对应的层，而不是用熵选层

方法2: Attention 后处理校准
  - 学习一个简单的变换，将 3B 的 attention 校准到 7B 的分布

方法3: 多信号融合
  - 结合 attention + embedding similarity 来选择 tokens

方法4: 固定层选择
  - 不用熵选层，而是固定使用与 7B 对应的层
"""

import json
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoConfig
from scipy import stats

project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_dir)

from per_head_generation import find_connected_components, rotate_half


def compute_all_layer_attention(model, input_ids, query_start, device="cuda:0"):
    """计算所有层的 attention（用于分析层映射）"""
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

            # 计算所有层的 attention（不只是后 50%）
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

            # Continue forward pass with SDPA
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


def extract_doc_attention_per_layer(attention_scores, system_len, doc_len, query_len):
    """提取每层的 query->doc attention"""
    doc_start = system_len
    doc_end = system_len + doc_len

    layer_doc_attention = {}
    for layer_idx in sorted(attention_scores.keys()):
        layer_attn = attention_scores[layer_idx]
        query_to_doc = layer_attn[:, :, doc_start:doc_end]
        doc_attention_avg = query_to_doc.mean(axis=(0, 1))
        layer_doc_attention[layer_idx] = doc_attention_avg

    return layer_doc_attention


def find_layer_mapping(attn_3b, attn_7b, system_len, doc_len, query_len):
    """
    方法1: 找到 3B 和 7B 层之间的最佳映射

    对于 7B 的每一层，找到 3B 中 attention 分布最相似的层
    """
    doc_attn_3b = extract_doc_attention_per_layer(attn_3b, system_len, doc_len, query_len)
    doc_attn_7b = extract_doc_attention_per_layer(attn_7b, system_len, doc_len, query_len)

    # 计算所有层对之间的相关性
    correlation_matrix = np.zeros((len(doc_attn_7b), len(doc_attn_3b)))

    for i, layer_7b in enumerate(sorted(doc_attn_7b.keys())):
        for j, layer_3b in enumerate(sorted(doc_attn_3b.keys())):
            corr = np.corrcoef(doc_attn_7b[layer_7b], doc_attn_3b[layer_3b])[0, 1]
            correlation_matrix[i, j] = corr

    # 对于 7B 的每层，找到 3B 中最相似的层
    layer_mapping = {}
    layers_7b = sorted(doc_attn_7b.keys())
    layers_3b = sorted(doc_attn_3b.keys())

    for i, layer_7b in enumerate(layers_7b):
        best_3b_idx = np.argmax(correlation_matrix[i])
        best_3b_layer = layers_3b[best_3b_idx]
        best_corr = correlation_matrix[i, best_3b_idx]
        layer_mapping[layer_7b] = (best_3b_layer, best_corr)

    return layer_mapping, correlation_matrix, layers_7b, layers_3b


def select_tokens_with_mapped_layers(doc_attn_3b, mapped_layers, doc_len, target_ratio=0.2):
    """使用映射后的层来选择 tokens"""
    # 聚合映射层的 attention
    selected_attns = [doc_attn_3b[layer_3b] for layer_3b in mapped_layers]
    multi_layer_attn = np.stack(selected_attns).mean(axis=0)

    target_count = int(doc_len * target_ratio)

    # 标准选择流程
    mean_attn = np.mean(multi_layer_attn)
    std_attn = np.std(multi_layer_attn)
    threshold = mean_attn + 0.5 * std_attn
    high_attn_positions = list(np.where(multi_layer_attn > threshold)[0])

    components = find_connected_components(high_attn_positions, max_gap=2)
    component_scores = [(comp, sum(multi_layer_attn[p] for p in comp)) for comp in components]
    component_scores.sort(key=lambda x: x[1], reverse=True)

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

    return sorted(list(selected)), multi_layer_attn


def calibrate_attention_distribution(attn_3b, attn_7b):
    """
    方法2: 学习一个简单的变换来校准 attention 分布

    使用分位数匹配 (Quantile Matching)
    """
    # 对 3B 的 attention 进行分位数变换，使其分布匹配 7B
    sorted_3b = np.sort(attn_3b)
    sorted_7b = np.sort(attn_7b)

    # 创建映射函数
    ranks_3b = stats.rankdata(attn_3b, method='ordinal') - 1
    calibrated = sorted_7b[np.clip(ranks_3b, 0, len(sorted_7b) - 1)]

    return calibrated


def combine_attention_with_embedding_similarity(
    model, input_ids, doc_attn, system_len, doc_len, query_len, device="cuda:0", alpha=0.5
):
    """
    方法3: 结合 attention 和 embedding similarity
    """
    with torch.no_grad():
        # 获取中间层的 hidden states
        outputs = model(input_ids.to(device), output_hidden_states=True, use_cache=False)
        num_layers = len(outputs.hidden_states)
        mid_layer = num_layers // 2
        hidden_states = outputs.hidden_states[mid_layer][0]  # (seq_len, hidden_dim)

        # Query embedding (平均)
        query_start = system_len + doc_len
        query_emb = hidden_states[query_start:].mean(dim=0)  # (hidden_dim,)

        # Document embeddings
        doc_start = system_len
        doc_end = system_len + doc_len
        doc_emb = hidden_states[doc_start:doc_end]  # (doc_len, hidden_dim)

        # Cosine similarity
        query_emb = F.normalize(query_emb.float().unsqueeze(0), dim=-1)
        doc_emb = F.normalize(doc_emb.float(), dim=-1)
        similarity = torch.mm(doc_emb, query_emb.T).squeeze().cpu().numpy()

        # 归一化 similarity 到 [0, 1]
        similarity = (similarity - similarity.min()) / (similarity.max() - similarity.min() + 1e-10)

    # 归一化 attention 到 [0, 1]
    attn_norm = (doc_attn - doc_attn.min()) / (doc_attn.max() - doc_attn.min() + 1e-10)

    # 融合
    combined = alpha * attn_norm + (1 - alpha) * similarity

    return combined, similarity


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

    print("Loading 3B model...")
    config_3b = AutoConfig.from_pretrained('/mnt/data/models/Qwen2.5-3B-Instruct', trust_remote_code=True)
    config_3b._attn_implementation = "sdpa"
    model_3b, _ = load_model('qwen', '/mnt/data/models/Qwen2.5-3B-Instruct', config_3b, device, use_multi_gpu=False)
    model_3b.eval()

    print("Loading 7B model...")
    config_7b = AutoConfig.from_pretrained('/mnt/data/models/Qwen2.5-7B-Instruct', trust_remote_code=True)
    config_7b._attn_implementation = "sdpa"
    model_7b, _ = load_model('qwen', '/mnt/data/models/Qwen2.5-7B-Instruct', config_7b, device, use_multi_gpu=False)
    model_7b.eval()

    tokenizer = AutoTokenizer.from_pretrained('/mnt/data/models/Qwen2.5-3B-Instruct', trust_remote_code=True)

    # Load dataset
    print("\nLoading dataset...")
    from test_fusionrag_reflect import prepare_reflect_data

    questions_data, system_tensor, _, _ = prepare_reflect_data(
        './data/result_reflect.json', tokenizer, '/mnt/data/models/bge-m3-FP16', 'qwen', 10,
        max_main_questions=200, preprocess=False
    )

    system_len = system_tensor.shape[0]

    # 收集层映射统计
    all_layer_mappings = []

    # 收集各方法的 IoU
    results = {
        'baseline_3b': [],       # 原始 3B 熵选层
        'layer_mapping': [],     # 方法1: 层映射
        'calibration': [],       # 方法2: 分布校准
        'combined': [],          # 方法3: 融合 embedding
        'oracle_7b': [],         # Oracle 7B (上界)
    }

    for case_idx, case in enumerate(critical_cases[:args.max_cases]):
        print(f"\n{'='*80}")
        print(f"Case {case_idx + 1}/{min(args.max_cases, len(critical_cases))}")
        print(f"{'='*80}")
        print(f"Question: {case['sub_question'][:60]}...")

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

        # 计算所有层的 attention
        print("  Computing all-layer attention for 3B...")
        attn_3b = compute_all_layer_attention(model_3b, full_input, query_start, device)
        doc_attn_3b = extract_doc_attention_per_layer(attn_3b, system_len, doc_len, query_len)

        print("  Computing all-layer attention for 7B...")
        attn_7b = compute_all_layer_attention(model_7b, full_input, query_start, device)
        doc_attn_7b = extract_doc_attention_per_layer(attn_7b, system_len, doc_len, query_len)

        # =========================================================================
        # 方法1: 层映射分析
        # =========================================================================
        print("\n  [Method 1] Layer Mapping Analysis...")
        layer_mapping, corr_matrix, layers_7b, layers_3b = find_layer_mapping(
            attn_3b, attn_7b, system_len, doc_len, query_len
        )

        print("    7B Layer -> Best 3B Layer (correlation):")
        # 只显示 7B 后半部分层的映射（这些是我们关心的）
        for layer_7b in layers_7b[len(layers_7b)//2:]:
            best_3b, corr = layer_mapping[layer_7b]
            print(f"      Layer {layer_7b} -> Layer {best_3b} (corr={corr:.4f})")

        all_layer_mappings.append(layer_mapping)

        # 使用 7B 熵选层对应的 3B 层
        # 先找 7B 的熵选层
        layer_entropy_7b = {}
        for layer_idx, attn in doc_attn_7b.items():
            p = attn / (attn.sum() + 1e-10)
            p = np.clip(p, 1e-10, 1.0)
            entropy = -np.sum(p * np.log(p))
            layer_entropy_7b[layer_idx] = entropy

        sorted_layers_7b = sorted(layer_entropy_7b.items(), key=lambda x: x[1])
        top4_7b = [l for l, _ in sorted_layers_7b[:4]]

        # 映射到 3B 的层
        mapped_3b_layers = [layer_mapping[l][0] for l in top4_7b]
        print(f"    7B entropy-selected layers: {top4_7b}")
        print(f"    Mapped 3B layers: {mapped_3b_layers}")

        # 使用映射层选择 tokens
        selected_mapped, attn_mapped = select_tokens_with_mapped_layers(
            doc_attn_3b, mapped_3b_layers, doc_len, args.rate
        )

        # =========================================================================
        # Baseline: 3B 熵选层
        # =========================================================================
        layer_entropy_3b = {}
        for layer_idx, attn in doc_attn_3b.items():
            p = attn / (attn.sum() + 1e-10)
            p = np.clip(p, 1e-10, 1.0)
            entropy = -np.sum(p * np.log(p))
            layer_entropy_3b[layer_idx] = entropy

        sorted_layers_3b = sorted(layer_entropy_3b.items(), key=lambda x: x[1])
        top4_3b = [l for l, _ in sorted_layers_3b[:4]]
        print(f"    3B entropy-selected layers: {top4_3b}")

        selected_baseline, attn_baseline = select_tokens_with_mapped_layers(
            doc_attn_3b, top4_3b, doc_len, args.rate
        )

        # =========================================================================
        # Oracle: 7B 选择
        # =========================================================================
        selected_oracle, attn_oracle = select_tokens_with_mapped_layers(
            doc_attn_7b, top4_7b, doc_len, args.rate
        )

        # =========================================================================
        # 方法2: 分布校准
        # =========================================================================
        print("\n  [Method 2] Distribution Calibration...")
        # 使用 3B baseline 的聚合 attention，校准到 7B 分布
        calibrated_attn = calibrate_attention_distribution(attn_baseline, attn_oracle)

        # 用校准后的 attention 选择
        target_count = int(doc_len * args.rate)
        sorted_indices = np.argsort(calibrated_attn)[::-1]
        selected_calibrated = set(sorted_indices[:target_count].tolist())
        selected_calibrated = sorted(list(selected_calibrated))

        # =========================================================================
        # 方法3: 融合 embedding similarity
        # =========================================================================
        print("\n  [Method 3] Combined with Embedding Similarity...")
        combined_attn, emb_sim = combine_attention_with_embedding_similarity(
            model_3b, full_input, attn_baseline, system_len, doc_len, query_len, device, alpha=0.7
        )

        sorted_indices = np.argsort(combined_attn)[::-1]
        selected_combined = set(sorted_indices[:target_count].tolist())
        selected_combined = sorted(list(selected_combined))

        # =========================================================================
        # 计算 IoU
        # =========================================================================
        print("\n  === Results ===")
        set_oracle = set(selected_oracle)

        for name, selected in [
            ('baseline_3b', selected_baseline),
            ('layer_mapping', selected_mapped),
            ('calibration', selected_calibrated),
            ('combined', selected_combined),
        ]:
            set_sel = set(selected)
            overlap = len(set_sel & set_oracle)
            union = len(set_sel | set_oracle)
            iou = overlap / union if union > 0 else 0
            results[name].append(iou)
            print(f"    {name:15s}: IoU with 7B = {iou:.4f}")

        results['oracle_7b'].append(1.0)  # 自己和自己的 IoU 是 1

        torch.cuda.empty_cache()

    # =========================================================================
    # 汇总统计
    # =========================================================================
    print(f"\n{'='*80}")
    print("SUMMARY: Average IoU with 7B Oracle")
    print(f"{'='*80}")

    for name in ['baseline_3b', 'layer_mapping', 'calibration', 'combined']:
        if results[name]:
            avg_iou = np.mean(results[name])
            std_iou = np.std(results[name])
            print(f"  {name:15s}: {avg_iou:.4f} ± {std_iou:.4f}")

    # 分析层映射的统计规律
    print(f"\n{'='*80}")
    print("Layer Mapping Statistics (7B -> 3B)")
    print(f"{'='*80}")

    if all_layer_mappings:
        # 统计每个 7B 层最常映射到哪个 3B 层
        from collections import Counter
        layer_counts = {layer_7b: Counter() for layer_7b in all_layer_mappings[0].keys()}

        for mapping in all_layer_mappings:
            for layer_7b, (layer_3b, _) in mapping.items():
                layer_counts[layer_7b][layer_3b] += 1

        print("\n  Most common mappings (7B -> 3B):")
        for layer_7b in sorted(layer_counts.keys()):
            if layer_7b >= 14:  # 只显示后半部分
                most_common = layer_counts[layer_7b].most_common(3)
                mapping_str = ", ".join([f"L{l3b}({cnt})" for l3b, cnt in most_common])
                print(f"    7B Layer {layer_7b:2d} -> {mapping_str}")


if __name__ == '__main__':
    main()
