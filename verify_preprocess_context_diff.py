#!/usr/bin/env python3
"""
验证 FusionRAG 预处理 KV Cache 与实时计算的根本差异

核心假设：预处理时 chunk1 attend 到了 similar_docs，
         而推理时 chunk1 应该只 attend 到 system

这个脚本会：
1. 模拟预处理流程：system + similar_docs + chunk1 一起 prefill
2. 模拟推理流程：system + chunk1 一起 prefill
3. 比较两者 chunk1 的 KV Cache 差异
"""

import os
import sys
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, '/mnt/data/wjh/FusionRAG')


def main():
    device = "cuda:0"
    model_path = "/mnt/data/models/Qwen2.5-7B-Instruct"

    # 加载 system prompt
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

    # 模拟文档
    chunk1 = "Document: Albert Einstein was a German-born theoretical physicist who developed the theory of relativity.\n"
    similar_doc1 = "Document: Isaac Newton was an English mathematician and physicist who developed the laws of motion.\n"
    similar_doc2 = "Document: Niels Bohr was a Danish physicist who made contributions to understanding atomic structure.\n"

    # Tokenize
    system_tokens = tokenizer.encode(system_prompt, add_special_tokens=False)
    chunk1_tokens = tokenizer.encode(chunk1, add_special_tokens=False)
    similar1_tokens = tokenizer.encode(similar_doc1, add_special_tokens=False)
    similar2_tokens = tokenizer.encode(similar_doc2, add_special_tokens=False)

    system_len = len(system_tokens)
    chunk1_len = len(chunk1_tokens)
    similar1_len = len(similar1_tokens)
    similar2_len = len(similar2_tokens)

    print(f"\nToken lengths:")
    print(f"  System: {system_len}")
    print(f"  Chunk1: {chunk1_len}")
    print(f"  Similar1: {similar1_len}")
    print(f"  Similar2: {similar2_len}")

    # ========================================
    # 方式 A: 模拟推理时的计算 (system + chunk1)
    # ========================================
    print("\n" + "="*80)
    print("[方式 A] 推理时: system + chunk1")
    print("="*80)

    input_a = torch.tensor([system_tokens + chunk1_tokens], dtype=torch.long, device=device)

    with torch.no_grad():
        outputs_a = model(input_ids=input_a, use_cache=True, return_dict=True)
    kv_a = outputs_a.past_key_values

    # 提取 chunk1 部分的 KV
    # chunk1 在位置 [system_len, system_len + chunk1_len)
    chunk1_kv_a = []
    for layer_idx in range(len(kv_a)):
        k = kv_a[layer_idx][0][:, :, system_len:system_len + chunk1_len, :]
        v = kv_a[layer_idx][1][:, :, system_len:system_len + chunk1_len, :]
        chunk1_kv_a.append((k.clone(), v.clone()))

    print(f"  Chunk1 KV shape: K={chunk1_kv_a[0][0].shape}, V={chunk1_kv_a[0][1].shape}")

    # ========================================
    # 方式 B: 模拟预处理时的计算 (system + similar_docs + chunk1)
    # ========================================
    print("\n" + "="*80)
    print("[方式 B] 预处理时: system + similar1 + similar2 + chunk1")
    print("="*80)

    input_b = torch.tensor(
        [system_tokens + similar1_tokens + similar2_tokens + chunk1_tokens],
        dtype=torch.long, device=device
    )

    with torch.no_grad():
        outputs_b = model(input_ids=input_b, use_cache=True, return_dict=True)
    kv_b = outputs_b.past_key_values

    # 提取 chunk1 部分的 KV
    # chunk1 在位置 [system_len + similar1_len + similar2_len, ...]
    chunk1_start_b = system_len + similar1_len + similar2_len
    chunk1_kv_b = []
    for layer_idx in range(len(kv_b)):
        k = kv_b[layer_idx][0][:, :, chunk1_start_b:chunk1_start_b + chunk1_len, :]
        v = kv_b[layer_idx][1][:, :, chunk1_start_b:chunk1_start_b + chunk1_len, :]
        chunk1_kv_b.append((k.clone(), v.clone()))

    print(f"  Chunk1 start position: {chunk1_start_b}")
    print(f"  Chunk1 KV shape: K={chunk1_kv_b[0][0].shape}, V={chunk1_kv_b[0][1].shape}")

    # ========================================
    # 比较差异
    # ========================================
    print("\n" + "="*80)
    print("KV Cache 差异分析 (方式A: 推理时 vs 方式B: 预处理时)")
    print("="*80)

    num_layers = len(kv_a)
    print(f"\n{'Layer':<6} {'K_diff_mean':<14} {'K_diff_max':<14} {'V_diff_mean':<14} {'V_diff_max':<14}")
    print("-" * 62)

    total_k_diff = 0
    total_v_diff = 0
    layer_diffs = []

    for layer_idx in range(num_layers):
        k_a = chunk1_kv_a[layer_idx][0]
        v_a = chunk1_kv_a[layer_idx][1]
        k_b = chunk1_kv_b[layer_idx][0]
        v_b = chunk1_kv_b[layer_idx][1]

        # 注意：位置不同导致 RoPE 不同，所以 Key 会有差异
        # 但更重要的是 Value 的差异，因为 Value 是 attention 聚合的结果
        k_diff = (k_a - k_b).abs()
        v_diff = (v_a - v_b).abs()

        k_diff_mean = k_diff.mean().item()
        k_diff_max = k_diff.max().item()
        v_diff_mean = v_diff.mean().item()
        v_diff_max = v_diff.max().item()

        total_k_diff += k_diff_mean
        total_v_diff += v_diff_mean
        layer_diffs.append({
            'layer': layer_idx,
            'k_diff_mean': k_diff_mean,
            'v_diff_mean': v_diff_mean,
        })

        if layer_idx % 4 == 0 or layer_idx == num_layers - 1:
            print(f"{layer_idx:<6} {k_diff_mean:<14.6f} {k_diff_max:<14.6f} {v_diff_mean:<14.6f} {v_diff_max:<14.6f}")

    print(f"\n整体平均差异:")
    print(f"  K 差异: {total_k_diff / num_layers:.6f}")
    print(f"  V 差异: {total_v_diff / num_layers:.6f}")

    # ========================================
    # 关键分析：Value 差异来源
    # ========================================
    print("\n" + "="*80)
    print("差异分析结论")
    print("="*80)

    if total_v_diff / num_layers > 0.01:
        print("\n[结论] Value 存在显著差异！")
        print("\n原因：")
        print("  1. 方式 A: chunk1 只 attend 到 system (正确的推理上下文)")
        print("  2. 方式 B: chunk1 attend 到 system + similar_docs (预处理上下文)")
        print("\n  由于 similar_docs 在推理时不存在，预处理的 KV Cache")
        print("  包含了不应该存在的 attention 信息，导致 Value 值不准确。")
        print("\n  这就是为什么重算 chunk1 能提升答案质量！")
    else:
        print("\n[结论] 差异较小，可能主要来自 RoPE 位置差异")

    # 比较 RoPE 的影响
    print("\n" + "="*80)
    print("附加分析：RoPE 位置影响 vs Attention 上下文影响")
    print("="*80)

    # 计算在相同位置但不同 attention 上下文下的 Value 差异
    print("\n由于 Key 包含 RoPE，其差异同时反映位置和上下文变化")
    print("而 Value 不包含 RoPE，其差异主要反映 attention 上下文变化")
    print(f"\nV/K 差异比值 (越高说明上下文影响越大):")

    for layer_idx in [0, 8, 16, 24, 27]:
        if layer_idx < num_layers:
            k_diff = layer_diffs[layer_idx]['k_diff_mean']
            v_diff = layer_diffs[layer_idx]['v_diff_mean']
            ratio = v_diff / (k_diff + 1e-10)
            print(f"  Layer {layer_idx}: V/K = {ratio:.4f}")


if __name__ == '__main__':
    main()
