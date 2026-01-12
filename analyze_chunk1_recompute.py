#!/usr/bin/env python3
"""
分析在 FusionRAG 框架下，重算时包含/不包含 chunk1 的 KV Cache 差异

核心对比：
- 方式 A：选中重算的 token 包含 chunk1 中的 token
- 方式 B：选中重算的 token 不包含 chunk1 中的 token
"""

import os
import sys
import json
import torch
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# 添加项目路径
sys.path.insert(0, '/mnt/data/wjh/FusionRAG')
from ktransformers.util.utils import prepare_data


def load_failure_cases(csv_path, dataset_path):
    """加载失败案例"""
    def extract_sub_question(query_text):
        if ':' in query_text:
            return query_text.split(':', 1)[1].strip()
        return query_text.strip()

    with open(dataset_path, 'r') as f:
        data = json.load(f)

    index = {}
    for item in data:
        main_q = item.get('question', '')
        for sub_q_info in item.get('intermediate_context', []):
            raw_query = sub_q_info.get('query', '')
            sub_q = extract_sub_question(raw_query)
            chunks = sub_q_info.get('retrieve docs', [])
            chunk_texts = [c if isinstance(c, str) else c.get('text', '') for c in chunks]
            index[(main_q, sub_q)] = chunk_texts

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
                'chunks': index[key],
            })
    return cases


def simulate_fusionrag_recompute(model, tokenizer, system_prompt, chunks, question,
                                   rate, include_chunk1, device="cuda:0"):
    """
    模拟 FusionRAG 的重算过程

    Args:
        include_chunk1: 是否在重算时包含 chunk1 的 token
    """
    num_layers = model.config.num_hidden_layers
    num_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // model.config.num_attention_heads

    # 构建完整输入: [system, chunk1, chunk2, ..., question]
    passages = [system_prompt] + chunks + [question]

    # Tokenize
    all_tokens = []
    boundaries = [0]
    for p in passages:
        tokens = tokenizer.encode(p, add_special_tokens=False)
        all_tokens.extend(tokens)
        boundaries.append(len(all_tokens))

    input_ids = torch.tensor([all_tokens], dtype=torch.long, device=device)
    total_len = len(all_tokens)

    system_len = boundaries[1]
    chunk1_start = boundaries[1]
    chunk1_end = boundaries[2]
    doc_start = boundaries[1]
    doc_end = boundaries[-2]  # 不包含 question
    doc_len = doc_end - doc_start

    print(f"  Total: {total_len}, System: 0-{system_len}, Chunk1: {chunk1_start}-{chunk1_end}, Docs: {doc_start}-{doc_end}")

    # Step 1: 完整 prefill 获取所有 KV Cache（模拟预处理拼接后的结果）
    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=True, return_dict=True)
    kv_full = outputs.past_key_values

    # Step 2: 模拟重算
    # 选择要重算的 token（按 rate 比例）
    num_recompute = int(doc_len * rate)

    if include_chunk1:
        # 包含 chunk1：从整个 doc 范围内选择
        # 简化：选择前 num_recompute 个 token
        recompute_positions = list(range(doc_start, min(doc_start + num_recompute, doc_end)))
    else:
        # 不包含 chunk1：只从 chunk2 及之后选择
        chunk2_start = boundaries[2]
        remaining_doc_len = doc_end - chunk2_start
        num_from_chunk2 = min(num_recompute, remaining_doc_len)
        recompute_positions = list(range(chunk2_start, chunk2_start + num_from_chunk2))

    print(f"  Recompute positions: {recompute_positions[:5]}...{recompute_positions[-5:] if len(recompute_positions) > 5 else ''}")
    print(f"  Num recompute: {len(recompute_positions)}, Include chunk1: {include_chunk1}")

    # Step 3: 重算选中的 token
    # 提取选中位置的 token
    if len(recompute_positions) > 0:
        recompute_tokens = [all_tokens[pos] for pos in recompute_positions]
        recompute_input = torch.tensor([recompute_tokens], dtype=torch.long, device=device)

        # 使用正确的 position_ids
        position_ids = torch.tensor([recompute_positions], dtype=torch.long, device=device)

        # 重新计算这些位置的 KV
        with torch.no_grad():
            recompute_outputs = model(
                input_ids=recompute_input,
                position_ids=position_ids,
                use_cache=True,
                return_dict=True,
            )
        kv_recompute = recompute_outputs.past_key_values

        # Step 4: 用重算的 KV 更新原来的 KV Cache
        kv_updated = []
        for layer_idx in range(num_layers):
            k_full = kv_full[layer_idx][0].clone()
            v_full = kv_full[layer_idx][1].clone()

            # 更新选中位置的 KV
            for i, pos in enumerate(recompute_positions):
                k_full[:, :, pos, :] = kv_recompute[layer_idx][0][:, :, i, :]
                v_full[:, :, pos, :] = kv_recompute[layer_idx][1][:, :, i, :]

            kv_updated.append((k_full, v_full))
    else:
        kv_updated = [(kv_full[i][0].clone(), kv_full[i][1].clone()) for i in range(num_layers)]

    return kv_updated, {
        'boundaries': boundaries,
        'total_len': total_len,
        'chunk1_start': chunk1_start,
        'chunk1_end': chunk1_end,
        'recompute_positions': recompute_positions,
    }


def main():
    device = "cuda:0"

    model_path = "/mnt/data/models/Qwen2.5-7B-Instruct"
    csv_path = "/mnt/data/reflect/Qwen2.5-7B-Instruct/2wikimqa/results/DraftModel_global_topk_10_rate_0.3_draft_Qwen2.5-3B-Instruct_revert_rope.csv"
    dataset_path = "/mnt/data/wjh/FusionRAG/data/2wikimqa_reflect.json"

    # 使用和 FusionRAG 一致的 system prompt
    prompt_config = json.load(open('/mnt/data/wjh/FusionRAG/config/dataset2prompt_few-shot.json'))
    system_prompt = prompt_config['system_prompt']['Qwen2.5']['2wikimqa']

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
    rate = 0.3  # 和 DraftModel 实验一致

    # 分析前 3 个案例
    for case_idx, case in enumerate(cases[:3]):
        print(f"\n{'='*80}")
        print(f"Case {case_idx + 1}: {case['sub_q'][:60]}...")
        print(f"GT: {case['gt']}")
        print(f"Draft: {case['draft_pred']}")
        print(f"Rate1: {case['rate1_pred']}")
        print(f"{'='*80}")

        # 方式 A: 重算包含 chunk1
        print("\n[方式 A: 重算包含 chunk1]")
        kv_with_chunk1, info_a = simulate_fusionrag_recompute(
            model, tokenizer, system_prompt, case['chunks'], case['sub_q'],
            rate=rate, include_chunk1=True, device=device
        )

        # 方式 B: 重算不包含 chunk1
        print("\n[方式 B: 重算不包含 chunk1]")
        kv_without_chunk1, info_b = simulate_fusionrag_recompute(
            model, tokenizer, system_prompt, case['chunks'], case['sub_q'],
            rate=rate, include_chunk1=False, device=device
        )

        # 比较差异
        print(f"\n{'='*80}")
        print("KV Cache 差异分析 (方式A vs 方式B)")
        print(f"{'='*80}")

        chunk1_start = info_a['chunk1_start']
        chunk1_end = info_a['chunk1_end']

        print(f"\n--- Chunk1 范围内的差异 ({chunk1_start}-{chunk1_end}) ---")
        print(f"{'Layer':<6} {'K_diff_mean':<14} {'K_diff_max':<14} {'V_diff_mean':<14} {'V_diff_max':<14}")
        print("-" * 62)

        for layer_idx in range(num_layers):
            k_a = kv_with_chunk1[layer_idx][0][:, :, chunk1_start:chunk1_end, :]
            v_a = kv_with_chunk1[layer_idx][1][:, :, chunk1_start:chunk1_end, :]
            k_b = kv_without_chunk1[layer_idx][0][:, :, chunk1_start:chunk1_end, :]
            v_b = kv_without_chunk1[layer_idx][1][:, :, chunk1_start:chunk1_end, :]

            k_diff = (k_a - k_b).abs()
            v_diff = (v_a - v_b).abs()

            if layer_idx % 4 == 0 or layer_idx == num_layers - 1:
                print(f"{layer_idx:<6} {k_diff.mean().item():<14.6f} {k_diff.max().item():<14.6f} "
                      f"{v_diff.mean().item():<14.6f} {v_diff.max().item():<14.6f}")

        # 整体差异
        print(f"\n--- 整体差异 ---")
        total_k_diff = 0
        total_v_diff = 0
        for layer_idx in range(num_layers):
            k_a = kv_with_chunk1[layer_idx][0]
            v_a = kv_with_chunk1[layer_idx][1]
            k_b = kv_without_chunk1[layer_idx][0]
            v_b = kv_without_chunk1[layer_idx][1]
            total_k_diff += (k_a - k_b).abs().mean().item()
            total_v_diff += (v_a - v_b).abs().mean().item()
        print(f"平均 K 差异: {total_k_diff / num_layers:.6f}")
        print(f"平均 V 差异: {total_v_diff / num_layers:.6f}")

        # 清理显存
        del kv_with_chunk1, kv_without_chunk1
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
