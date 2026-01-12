#!/usr/bin/env python3
"""
分析 DraftModel 失败案例中，重算 vs 不重算 chunk1 时 KV Cache 的差异
"""

import os
import json
import torch
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import re


def extract_sub_question(query_text):
    if ':' in query_text:
        return query_text.split(':', 1)[1].strip()
    return query_text.strip()


def load_failure_cases(csv_path, dataset_path):
    """加载失败案例并匹配数据集信息"""
    # 加载数据集
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    # 构建索引
    index = {}
    for item in data:
        main_q = item.get('question', '')
        for sub_idx, sub_q_info in enumerate(item.get('intermediate_context', [])):
            raw_query = sub_q_info.get('query', '')
            sub_q = extract_sub_question(raw_query)
            chunks = sub_q_info.get('retrieve docs', [])
            chunk_texts = [c if isinstance(c, str) else c.get('text', '') for c in chunks]
            index[(main_q, sub_q)] = {
                'chunks': chunk_texts,
                'sub_q_info': sub_q_info,
                'main_item': item,
            }

    # 读取 CSV
    df = pd.read_csv(csv_path)
    df['Correct_bool'] = df['Correct'].astype(str).str.lower().isin(['true', 'yes', '1'])
    df['Rate1_Correct_bool'] = df['Rate1_Correct'].astype(str).str.lower().isin(['true', 'yes', '1'])

    # 筛选失败案例
    failures = df[~df['Correct_bool'] & df['Rate1_Correct_bool']]

    cases = []
    for _, row in failures.iterrows():
        key = (row['Main Question'], row['Sub Question'])
        if key in index:
            cases.append({
                'main_q': row['Main Question'],
                'sub_q': row['Sub Question'],
                'gt': row['Ground Truth'],
                'draft_pred': row['Predicted'],
                'rate1_pred': row['Rate1_Predicted'],
                'chunks': index[key]['chunks'],
            })

    return cases


def compute_kvcache_with_recompute(model, tokenizer, system_prompt, chunks, question,
                                    recompute_chunk0=False, device="cuda:0"):
    """
    计算 KV Cache，可选是否重算 chunk0

    返回每一层的 KV Cache
    """
    # 构建完整的 prompt
    # passages = [system, chunk1, chunk2, ..., chunkn, question]
    passages = [system_prompt] + chunks + [question]

    # Tokenize 每个 passage
    passage_tokens = []
    passage_lengths = []
    for p in passages:
        tokens = tokenizer.encode(p, add_special_tokens=False)
        passage_tokens.append(tokens)
        passage_lengths.append(len(tokens))

    # 完整的 input_ids
    all_tokens = []
    for tokens in passage_tokens:
        all_tokens.extend(tokens)
    input_ids = torch.tensor([all_tokens], dtype=torch.long, device=device)

    # 计算 chunk0 的位置范围（chunk0 是 passages[1]，即第二个元素）
    system_len = passage_lengths[0]
    chunk0_start = system_len
    chunk0_end = system_len + passage_lengths[1]

    print(f"  Total tokens: {len(all_tokens)}")
    print(f"  System prompt: 0-{system_len}")
    print(f"  Chunk0: {chunk0_start}-{chunk0_end} (len={passage_lengths[1]})")
    print(f"  Recompute chunk0: {recompute_chunk0}")

    # 使用模型进行 forward pass
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            use_cache=True,
            return_dict=True,
            output_attentions=False,
        )

    # 提取每一层的 KV Cache
    past_key_values = outputs.past_key_values

    layer_kv_caches = []
    for layer_idx, (k, v) in enumerate(past_key_values):
        # k, v shape: [batch, num_kv_heads, seq_len, head_dim]
        layer_kv_caches.append({
            'key': k.clone(),
            'value': v.clone(),
        })

    return layer_kv_caches, {
        'chunk0_start': chunk0_start,
        'chunk0_end': chunk0_end,
        'passage_lengths': passage_lengths,
        'total_len': len(all_tokens),
    }


def compute_kvcache_with_preprocess(model, tokenizer, system_prompt, chunks, question,
                                     preprocess_cache_dir, doc_id, device="cuda:0"):
    """
    使用预处理的 KV Cache（模拟不重算 chunk0 的情况）

    这里我们模拟的是：
    1. system prompt + chunks 使用预处理的 KV Cache
    2. question 需要实时计算

    但为了对比，我们直接计算完整的 KV Cache，然后标记 chunk0 的位置
    """
    # 构建完整的 prompt
    passages = [system_prompt] + chunks + [question]

    # Tokenize
    passage_tokens = []
    passage_lengths = []
    for p in passages:
        tokens = tokenizer.encode(p, add_special_tokens=False)
        passage_tokens.append(tokens)
        passage_lengths.append(len(tokens))

    all_tokens = []
    for tokens in passage_tokens:
        all_tokens.extend(tokens)
    input_ids = torch.tensor([all_tokens], dtype=torch.long, device=device)

    system_len = passage_lengths[0]
    chunk0_start = system_len
    chunk0_end = system_len + passage_lengths[1]

    # 尝试加载预处理的 KV Cache
    preprocess_kv = {}
    num_layers = model.config.num_hidden_layers

    for layer_idx in range(num_layers):
        key_file = os.path.join(preprocess_cache_dir, f"{doc_id}_{layer_idx}_key.pt")
        value_file = os.path.join(preprocess_cache_dir, f"{doc_id}_{layer_idx}_value.pt")

        if os.path.exists(key_file) and os.path.exists(value_file):
            preprocess_kv[layer_idx] = {
                'key': torch.load(key_file, map_location=device),
                'value': torch.load(value_file, map_location=device),
            }

    return preprocess_kv, {
        'chunk0_start': chunk0_start,
        'chunk0_end': chunk0_end,
        'passage_lengths': passage_lengths,
        'total_len': len(all_tokens),
    }


def analyze_kvcache_difference(kv_full, kv_preprocess, chunk0_start, chunk0_end, num_layers):
    """
    分析两种 KV Cache 在 chunk0 范围内的差异
    """
    results = []

    for layer_idx in range(num_layers):
        if layer_idx not in kv_preprocess:
            continue

        full_k = kv_full[layer_idx]['key']  # [1, num_kv_heads, seq_len, head_dim]
        full_v = kv_full[layer_idx]['value']

        pre_k = kv_preprocess[layer_idx]['key']
        pre_v = kv_preprocess[layer_idx]['value']

        # 提取 chunk0 范围
        # 注意：预处理的 KV Cache 可能只包含文档部分，需要对齐
        pre_seq_len = pre_k.shape[2]

        # 计算差异（如果长度匹配）
        if pre_seq_len >= chunk0_end:
            chunk0_k_full = full_k[:, :, chunk0_start:chunk0_end, :]
            chunk0_v_full = full_v[:, :, chunk0_start:chunk0_end, :]
            chunk0_k_pre = pre_k[:, :, chunk0_start:chunk0_end, :]
            chunk0_v_pre = pre_v[:, :, chunk0_start:chunk0_end, :]

            # 计算差异指标
            k_diff = (chunk0_k_full - chunk0_k_pre).abs()
            v_diff = (chunk0_v_full - chunk0_v_pre).abs()

            k_diff_mean = k_diff.mean().item()
            k_diff_max = k_diff.max().item()
            v_diff_mean = v_diff.mean().item()
            v_diff_max = v_diff.max().item()

            # 相对差异
            k_rel_diff = (k_diff / (chunk0_k_full.abs() + 1e-8)).mean().item()
            v_rel_diff = (v_diff / (chunk0_v_full.abs() + 1e-8)).mean().item()
        else:
            k_diff_mean = k_diff_max = v_diff_mean = v_diff_max = float('nan')
            k_rel_diff = v_rel_diff = float('nan')

        results.append({
            'layer': layer_idx,
            'k_diff_mean': k_diff_mean,
            'k_diff_max': k_diff_max,
            'v_diff_mean': v_diff_mean,
            'v_diff_max': v_diff_max,
            'k_rel_diff': k_rel_diff,
            'v_rel_diff': v_rel_diff,
        })

    return results


def main():
    device = "cuda:0"

    # 路径配置
    model_path = "/mnt/data/models/Qwen2.5-7B-Instruct"
    csv_path = "/mnt/data/reflect/Qwen2.5-7B-Instruct/2wikimqa/results/DraftModel_global_topk_10_rate_0.3_draft_Qwen2.5-3B-Instruct_revert_rope.csv"
    dataset_path = "/mnt/data/wjh/FusionRAG/data/2wikimqa_reflect.json"
    preprocess_cache_dir = "/mnt/data/reflect/Qwen2.5-7B-Instruct/2wikimqa/preprocess_kv_cache_global"

    system_prompt = """You are a helpful assistant. Answer the question based on the given context.
If the context doesn't contain enough information, say so clearly."""

    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    print("Loading failure cases...")
    cases = load_failure_cases(csv_path, dataset_path)
    print(f"Found {len(cases)} failure cases")

    # 只分析前 3 个案例（完整分析太耗时）
    num_analyze = min(3, len(cases))

    all_layer_diffs = []

    for case_idx, case in enumerate(cases[:num_analyze]):
        print(f"\n{'='*80}")
        print(f"Case {case_idx + 1}: {case['sub_q'][:60]}...")
        print(f"GT: {case['gt']}")
        print(f"Draft: {case['draft_pred']}")
        print(f"Rate1: {case['rate1_pred']}")
        print(f"{'='*80}")

        # 方式1: 完整计算（相当于 rate=1，重算所有 token）
        print("\n[Computing full KV Cache (rate=1)...]")
        kv_full, info = compute_kvcache_with_recompute(
            model, tokenizer, system_prompt, case['chunks'], case['sub_q'],
            recompute_chunk0=True, device=device
        )

        # 方式2: 使用预处理的 KV Cache（不重算）
        # 这里我们模拟：加载预处理的 KV Cache，看看和完整计算的差异
        print("\n[Loading preprocess KV Cache...]")
        # 注意：预处理的 KV Cache 是按文档 ID 存储的，需要找到对应的 ID
        # 这里简化处理，我们直接比较完整计算和只计算 system+question 的差异

        # 我们换一种方式：计算不包含 chunk0 attention 影响的 KV
        # 即：让 chunk0 的 token 无法 attend 到其他 token（模拟 prefix cache）

        # 简化分析：直接看完整 KV Cache 中 chunk0 部分的统计特征
        print("\n[Analyzing chunk0 KV Cache statistics...]")

        chunk0_start = info['chunk0_start']
        chunk0_end = info['chunk0_end']
        num_layers = len(kv_full)

        print(f"\nChunk0 position: {chunk0_start} - {chunk0_end} (len={chunk0_end - chunk0_start})")
        print(f"\n{'Layer':<6} {'K_mean':<12} {'K_std':<12} {'V_mean':<12} {'V_std':<12} {'K_norm':<12} {'V_norm':<12}")
        print("-" * 78)

        layer_stats = []
        for layer_idx in range(num_layers):
            k = kv_full[layer_idx]['key'][:, :, chunk0_start:chunk0_end, :]
            v = kv_full[layer_idx]['value'][:, :, chunk0_start:chunk0_end, :]

            k_mean = k.mean().item()
            k_std = k.std().item()
            v_mean = v.mean().item()
            v_std = v.std().item()
            k_norm = k.norm().item()
            v_norm = v.norm().item()

            layer_stats.append({
                'layer': layer_idx,
                'k_mean': k_mean,
                'k_std': k_std,
                'v_mean': v_mean,
                'v_std': v_std,
                'k_norm': k_norm,
                'v_norm': v_norm,
            })

            if layer_idx % 4 == 0 or layer_idx == num_layers - 1:
                print(f"{layer_idx:<6} {k_mean:<12.6f} {k_std:<12.6f} {v_mean:<12.6f} {v_std:<12.6f} {k_norm:<12.2f} {v_norm:<12.2f}")

        all_layer_diffs.append({
            'case': case_idx,
            'sub_q': case['sub_q'],
            'layer_stats': layer_stats,
        })

        # 清理显存
        del kv_full
        torch.cuda.empty_cache()

    print(f"\n{'='*80}")
    print("Analysis Complete")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
