#!/usr/bin/env python3
"""
验证 chunk1 使用 preprocess=False KV cache 时的 RoPE 双重应用问题

问题假设：
1. prefill_and_save_kv_cache 保存的 KV cache 已包含正确的 RoPE（位置 system_len 开始）
2. 加载时如果 revert_rope=True，会再次应用 RoPE 变换
3. 这导致 RoPE 被错误叠加

这个脚本验证：
- 方式 A: 正确的推理（直接 prefill，不做额外 RoPE 变换）
- 方式 B: 模拟当前代码（prefill 后再应用一次 RoPE）
"""

import os
import sys
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, '/mnt/data/wjh/FusionRAG')


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


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

    # Tokenize
    system_tokens = tokenizer.encode(system_prompt, add_special_tokens=False)
    chunk1_tokens = tokenizer.encode(chunk1, add_special_tokens=False)

    system_len = len(system_tokens)
    chunk1_len = len(chunk1_tokens)

    print(f"\nToken lengths:")
    print(f"  System: {system_len}")
    print(f"  Chunk1: {chunk1_len}")

    # ========================================
    # 方式 A: 正确的推理（直接 prefill system + chunk1）
    # ========================================
    print("\n" + "="*80)
    print("[方式 A] 正确的推理: system + chunk1 直接 prefill")
    print("="*80)

    input_a = torch.tensor([system_tokens + chunk1_tokens], dtype=torch.long, device=device)

    with torch.no_grad():
        outputs_a = model(input_ids=input_a, use_cache=True, return_dict=True)
    kv_a = outputs_a.past_key_values

    # 提取 chunk1 部分的 KV
    chunk1_key_a = kv_a[0][0][:, :, system_len:system_len + chunk1_len, :].clone()
    chunk1_value_a = kv_a[0][1][:, :, system_len:system_len + chunk1_len, :].clone()

    print(f"  Chunk1 Key shape: {chunk1_key_a.shape}")

    # ========================================
    # 方式 B: 模拟 preprocess=False + revert_rope=True 的问题
    # ========================================
    print("\n" + "="*80)
    print("[方式 B] 模拟问题: prefill 后再应用一次 RoPE (position=0)")
    print("="*80)

    # 假设这是从 preprocess=False 加载的 KV cache（已包含正确 RoPE）
    # 模拟加载时的 RoPE 变换
    chunk1_key_b = chunk1_key_a.clone()

    # 模拟 load_kv_and_generate 中的 RoPE 变换 (第 1012-1018 行)
    # position_ids = past_len - system_len = system_len - system_len = 0
    try:
        rotary_emb = model.model.layers[0].self_attn.rotary_emb
    except:
        rotary_emb = model.model.rotary_emb
    position_ids = torch.full((1, chunk1_len), 0, device=device)  # 对于 chunk1，past_len = system_len

    cos, sin = rotary_emb(chunk1_key_b[0].to(device), position_ids)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    # 应用 RoPE 变换
    chunk1_key_b_transformed = (chunk1_key_b * cos) + (rotate_half(chunk1_key_b) * sin)

    print(f"  After RoPE transform, Key shape: {chunk1_key_b_transformed.shape}")

    # ========================================
    # 比较差异
    # ========================================
    print("\n" + "="*80)
    print("Key 差异分析 (正确 vs 错误双重 RoPE)")
    print("="*80)

    key_diff = (chunk1_key_a - chunk1_key_b_transformed).abs()
    print(f"\n  Key diff mean: {key_diff.mean().item():.6f}")
    print(f"  Key diff max:  {key_diff.max().item():.6f}")

    # 检查是否接近 0（如果方式 B 的变换是恒等变换的话）
    if key_diff.mean().item() < 0.001:
        print("\n  [结论] 差异很小，position=0 的 RoPE 变换接近恒等变换")
    else:
        print("\n  [结论] 存在显著差异！RoPE 被错误叠加")
        print("  这解释了为什么重算 chunk1 能提升效果")

    # ========================================
    # 额外验证：对比所有层的差异
    # ========================================
    print("\n" + "="*80)
    print("所有层的 Key 差异")
    print("="*80)

    num_layers = len(kv_a)
    print(f"\n{'Layer':<6} {'K_diff_mean':<14} {'K_diff_max':<14}")
    print("-" * 34)

    total_diff = 0
    for layer_idx in range(num_layers):
        k_original = kv_a[layer_idx][0][:, :, system_len:system_len + chunk1_len, :]

        # 应用同样的 RoPE 变换
        k_transformed = (k_original * cos) + (rotate_half(k_original) * sin)

        k_diff = (k_original - k_transformed).abs()
        k_diff_mean = k_diff.mean().item()
        k_diff_max = k_diff.max().item()
        total_diff += k_diff_mean

        if layer_idx % 4 == 0 or layer_idx == num_layers - 1:
            print(f"{layer_idx:<6} {k_diff_mean:<14.6f} {k_diff_max:<14.6f}")

    print(f"\n平均 Key 差异: {total_diff / num_layers:.6f}")

    # ========================================
    # 关键洞察：RoPE position=0 不是恒等变换
    # ========================================
    print("\n" + "="*80)
    print("RoPE 分析")
    print("="*80)

    print("\nRoPE position=0 的 cos/sin 值 (前 8 维):")
    print(f"  cos: {cos[0, 0, 0, :8].tolist()}")
    print(f"  sin: {sin[0, 0, 0, :8].tolist()}")

    # cos=1, sin=0 时才是恒等变换
    is_identity = (cos.abs() - 1.0).abs().max() < 0.01 and sin.abs().max() < 0.01
    if is_identity:
        print("\n  RoPE(position=0) ≈ 恒等变换")
    else:
        print("\n  RoPE(position=0) ≠ 恒等变换")
        print("  这意味着应用 position=0 的 RoPE 会改变 Key 值！")


if __name__ == '__main__':
    main()
