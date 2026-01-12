#!/usr/bin/env python3
"""
分析 DraftModel 失败案例中，预处理 KV Cache vs 实时计算 KV Cache 的差异

核心对比：
1. 预处理 KV Cache：每个 chunk 独立计算，position 从各自起点开始
2. 实时计算 KV Cache：整个序列一起计算，position 连续

差异主要来自 RoPE position embedding
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
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    index = {}
    for item_idx, item in enumerate(data):
        main_q = item.get('question', '')
        for sub_idx, sub_q_info in enumerate(item.get('intermediate_context', [])):
            raw_query = sub_q_info.get('query', '')
            sub_q = extract_sub_question(raw_query)
            chunks = sub_q_info.get('retrieve docs', [])
            chunk_texts = [c if isinstance(c, str) else c.get('text', '') for c in chunks]
            index[(main_q, sub_q)] = {
                'chunks': chunk_texts,
                'item_idx': item_idx,
                'sub_idx': sub_idx,
            }

    df = pd.read_csv(csv_path)
    df['Correct_bool'] = df['Correct'].astype(str).str.lower().isin(['true', 'yes', '1'])
    df['Rate1_Correct_bool'] = df['Rate1_Correct'].astype(str).str.lower().isin(['true', 'yes', '1'])
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
                'item_idx': index[key]['item_idx'],
                'sub_idx': index[key]['sub_idx'],
            })

    return cases


def compute_kv_independent(model, tokenizer, system_prompt, chunk, start_position, device="cuda:0"):
    """
    独立计算单个 chunk 的 KV Cache（模拟预处理）
    使用指定的 start_position 作为 RoPE 起点
    """
    tokens = tokenizer.encode(system_prompt + chunk, add_special_tokens=False)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

    # 使用指定的 position
    position_ids = torch.arange(start_position, start_position + len(tokens), device=device).unsqueeze(0)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            position_ids=position_ids,
            use_cache=True,
            return_dict=True,
        )

    return outputs.past_key_values, len(tokens)


def compute_kv_joint(model, tokenizer, passages, device="cuda:0"):
    """
    联合计算整个序列的 KV Cache（实时计算）
    """
    all_tokens = []
    passage_boundaries = [0]

    for p in passages:
        tokens = tokenizer.encode(p, add_special_tokens=False)
        all_tokens.extend(tokens)
        passage_boundaries.append(len(all_tokens))

    input_ids = torch.tensor([all_tokens], dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            use_cache=True,
            return_dict=True,
        )

    return outputs.past_key_values, passage_boundaries


def compare_kv_caches(kv1, kv2, start1, end1, start2, end2, layer_idx):
    """
    比较两个 KV Cache 在指定范围内的差异
    """
    k1 = kv1[layer_idx][0][:, :, start1:end1, :]  # [1, num_heads, seq_len, head_dim]
    v1 = kv1[layer_idx][1][:, :, start1:end1, :]
    k2 = kv2[layer_idx][0][:, :, start2:end2, :]
    v2 = kv2[layer_idx][1][:, :, start2:end2, :]

    # 确保长度一致
    min_len = min(k1.shape[2], k2.shape[2])
    k1, k2 = k1[:, :, :min_len, :], k2[:, :, :min_len, :]
    v1, v2 = v1[:, :, :min_len, :], v2[:, :, :min_len, :]

    k_diff = (k1 - k2).abs()
    v_diff = (v1 - v2).abs()

    # 按 head 统计
    num_heads = k1.shape[1]
    head_k_diffs = []
    head_v_diffs = []
    for h in range(num_heads):
        head_k_diffs.append(k_diff[:, h, :, :].mean().item())
        head_v_diffs.append(v_diff[:, h, :, :].mean().item())

    return {
        'k_diff_mean': k_diff.mean().item(),
        'k_diff_max': k_diff.max().item(),
        'v_diff_mean': v_diff.mean().item(),
        'v_diff_max': v_diff.max().item(),
        'k_norm1': k1.norm().item(),
        'k_norm2': k2.norm().item(),
        'head_k_diffs': head_k_diffs,
        'head_v_diffs': head_v_diffs,
    }


def main():
    device = "cuda:0"

    model_path = "/mnt/data/models/Qwen2.5-7B-Instruct"
    csv_path = "/mnt/data/reflect/Qwen2.5-7B-Instruct/2wikimqa/results/DraftModel_global_topk_10_rate_0.3_draft_Qwen2.5-3B-Instruct_revert_rope.csv"
    dataset_path = "/mnt/data/wjh/FusionRAG/data/2wikimqa_reflect.json"

    system_prompt = "You are a helpful assistant. Answer the question based on the given context.\n"

    print("Loading model...")
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

    num_layers = model.config.num_hidden_layers
    num_kv_heads = model.config.num_key_value_heads

    # 分析每个案例
    all_results = []

    for case_idx, case in enumerate(cases[:5]):  # 分析前 5 个
        print(f"\n{'='*80}")
        print(f"Case {case_idx + 1}: {case['sub_q'][:60]}...")
        print(f"GT: {case['gt']}")
        print(f"Draft: {case['draft_pred']}")
        print(f"{'='*80}")

        chunks = case['chunks']
        question = case['sub_q']

        # 构建完整 passages
        passages = [system_prompt] + chunks + [question]

        # 1. 联合计算（rate=1 方式）
        print("\n[Joint computation (rate=1)]...")
        kv_joint, boundaries = compute_kv_joint(model, tokenizer, passages, device)

        system_len = boundaries[1] - boundaries[0]
        chunk0_start = boundaries[1]
        chunk0_end = boundaries[2]
        chunk0_len = chunk0_end - chunk0_start

        print(f"  System: 0-{system_len}")
        print(f"  Chunk0: {chunk0_start}-{chunk0_end} (len={chunk0_len})")
        print(f"  Total: {boundaries[-1]}")

        # 2. 独立计算 chunk0（预处理方式）
        # 关键：预处理时 chunk0 的 position 从哪里开始？
        # 方式 A：从 0 开始（错误的 RoPE）
        # 方式 B：从 system_len 开始（正确的 RoPE，但没有完整上下文）

        print("\n[Independent computation (preprocess)]...")

        # 方式 A：错误的 position（从 0 开始）
        chunk0_text = chunks[0]
        chunk0_tokens = tokenizer.encode(chunk0_text, add_special_tokens=False)
        chunk0_input = torch.tensor([chunk0_tokens], dtype=torch.long, device=device)

        # position 从 system_len 开始（假设预处理知道 system 的长度）
        # 但预处理时只有 chunk 本身，没有其他 chunks 的影响
        position_ids_correct = torch.arange(chunk0_start, chunk0_start + len(chunk0_tokens), device=device).unsqueeze(0)

        with torch.no_grad():
            outputs_independent = model(
                input_ids=chunk0_input,
                position_ids=position_ids_correct,
                use_cache=True,
                return_dict=True,
            )
        kv_independent = outputs_independent.past_key_values

        # 3. 比较差异
        print(f"\n{'='*80}")
        print("KV Cache Difference Analysis (Chunk0)")
        print(f"{'='*80}")
        print(f"\n{'Layer':<6} {'K_diff_mean':<14} {'K_diff_max':<14} {'V_diff_mean':<14} {'V_diff_max':<14} {'K_rel_diff':<14}")
        print("-" * 86)

        layer_results = []
        for layer_idx in range(num_layers):
            # 从联合计算中提取 chunk0 部分
            k_joint = kv_joint[layer_idx][0][:, :, chunk0_start:chunk0_end, :]
            v_joint = kv_joint[layer_idx][1][:, :, chunk0_start:chunk0_end, :]

            # 独立计算的 chunk0
            k_indep = kv_independent[layer_idx][0]
            v_indep = kv_independent[layer_idx][1]

            # 对齐长度
            min_len = min(k_joint.shape[2], k_indep.shape[2])
            k_joint_aligned = k_joint[:, :, :min_len, :]
            v_joint_aligned = v_joint[:, :, :min_len, :]
            k_indep_aligned = k_indep[:, :, :min_len, :]
            v_indep_aligned = v_indep[:, :, :min_len, :]

            k_diff = (k_joint_aligned - k_indep_aligned).abs()
            v_diff = (v_joint_aligned - v_indep_aligned).abs()

            k_diff_mean = k_diff.mean().item()
            k_diff_max = k_diff.max().item()
            v_diff_mean = v_diff.mean().item()
            v_diff_max = v_diff.max().item()

            # 相对差异
            k_rel_diff = (k_diff / (k_joint_aligned.abs() + 1e-8)).mean().item()

            layer_results.append({
                'layer': layer_idx,
                'k_diff_mean': k_diff_mean,
                'k_diff_max': k_diff_max,
                'v_diff_mean': v_diff_mean,
                'v_diff_max': v_diff_max,
                'k_rel_diff': k_rel_diff,
            })

            if layer_idx % 4 == 0 or layer_idx == num_layers - 1:
                print(f"{layer_idx:<6} {k_diff_mean:<14.6f} {k_diff_max:<14.6f} {v_diff_mean:<14.6f} {v_diff_max:<14.6f} {k_rel_diff:<14.6f}")

        all_results.append({
            'case': case_idx,
            'sub_q': case['sub_q'],
            'layer_results': layer_results,
        })

        # 清理显存
        del kv_joint, kv_independent
        torch.cuda.empty_cache()

    # 汇总分析
    print(f"\n{'='*80}")
    print("Summary: Average difference across all cases")
    print(f"{'='*80}")

    avg_k_diff = np.zeros(num_layers)
    avg_v_diff = np.zeros(num_layers)

    for result in all_results:
        for lr in result['layer_results']:
            avg_k_diff[lr['layer']] += lr['k_diff_mean']
            avg_v_diff[lr['layer']] += lr['v_diff_mean']

    avg_k_diff /= len(all_results)
    avg_v_diff /= len(all_results)

    print(f"\n{'Layer':<6} {'Avg_K_diff':<14} {'Avg_V_diff':<14}")
    print("-" * 34)
    for layer_idx in range(num_layers):
        if layer_idx % 4 == 0 or layer_idx == num_layers - 1:
            print(f"{layer_idx:<6} {avg_k_diff[layer_idx]:<14.6f} {avg_v_diff[layer_idx]:<14.6f}")


if __name__ == '__main__':
    main()
