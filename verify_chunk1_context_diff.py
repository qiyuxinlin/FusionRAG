#!/usr/bin/env python3
"""
验证 chunk1 的 KV cache 差异

关键洞察：
- preprocess=False 的 chunk1 KV cache 是用 system + chunk1 计算的
- chunk1 的 Value 只聚合了 system 的信息
- 但在实际推理时，上下文是 system + chunk1 + chunk2 + ... + question
- 如果重算 chunk1，它可以 attend 到所有其他 chunks

这个脚本验证：
- 方式 A: chunk1 只 attend 到 system（预处理方式）
- 方式 B: chunk1 attend 到 system + chunk2（更完整的上下文）
"""

import os
import sys
import json
import torch
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
    chunk2 = "Document: Isaac Newton was an English mathematician and physicist who developed the laws of motion.\n"

    # Tokenize
    system_tokens = tokenizer.encode(system_prompt, add_special_tokens=False)
    chunk1_tokens = tokenizer.encode(chunk1, add_special_tokens=False)
    chunk2_tokens = tokenizer.encode(chunk2, add_special_tokens=False)

    system_len = len(system_tokens)
    chunk1_len = len(chunk1_tokens)
    chunk2_len = len(chunk2_tokens)

    print(f"\nToken lengths:")
    print(f"  System: {system_len}")
    print(f"  Chunk1: {chunk1_len}")
    print(f"  Chunk2: {chunk2_len}")

    # ========================================
    # 方式 A: chunk1 只 attend 到 system（预处理方式）
    # ========================================
    print("\n" + "="*80)
    print("[方式 A] 预处理方式: system + chunk1 (chunk1 只看到 system)")
    print("="*80)

    input_a = torch.tensor([system_tokens + chunk1_tokens], dtype=torch.long, device=device)

    with torch.no_grad():
        outputs_a = model(input_ids=input_a, use_cache=True, return_dict=True)
    kv_a = outputs_a.past_key_values

    # ========================================
    # 方式 B: chunk1 attend 到 system + chunk2（更完整上下文）
    # 这模拟了在推理时重算 chunk1 的情况
    # ========================================
    print("\n" + "="*80)
    print("[方式 B] 重算时: system + chunk2 + chunk1 (chunk1 能看到 chunk2)")
    print("="*80)

    # 这里我们把 chunk2 放在 chunk1 前面，这样 chunk1 可以 attend 到 chunk2
    # 注意：这改变了 position，但我们关注的是 Value 的差异
    input_b = torch.tensor([system_tokens + chunk2_tokens + chunk1_tokens], dtype=torch.long, device=device)

    with torch.no_grad():
        outputs_b = model(input_ids=input_b, use_cache=True, return_dict=True)
    kv_b = outputs_b.past_key_values

    # 提取 chunk1 部分的 KV
    # 方式 A: chunk1 在位置 [system_len, system_len + chunk1_len)
    # 方式 B: chunk1 在位置 [system_len + chunk2_len, system_len + chunk2_len + chunk1_len)
    chunk1_start_a = system_len
    chunk1_start_b = system_len + chunk2_len

    # ========================================
    # 比较 Value 差异（Value 不包含 RoPE，纯粹反映 attention 聚合）
    # ========================================
    print("\n" + "="*80)
    print("Value 差异分析 (预处理 vs 重算)")
    print("注意：Value 反映了 attention 聚合的结果")
    print("="*80)

    num_layers = len(kv_a)
    print(f"\n{'Layer':<6} {'V_diff_mean':<14} {'V_diff_max':<14} {'V_norm_A':<14} {'V_norm_B':<14}")
    print("-" * 62)

    total_v_diff = 0
    for layer_idx in range(num_layers):
        v_a = kv_a[layer_idx][1][:, :, chunk1_start_a:chunk1_start_a + chunk1_len, :]
        v_b = kv_b[layer_idx][1][:, :, chunk1_start_b:chunk1_start_b + chunk1_len, :]

        v_diff = (v_a - v_b).abs()
        v_diff_mean = v_diff.mean().item()
        v_diff_max = v_diff.max().item()
        v_norm_a = v_a.norm().item()
        v_norm_b = v_b.norm().item()

        total_v_diff += v_diff_mean

        if layer_idx % 4 == 0 or layer_idx == num_layers - 1:
            print(f"{layer_idx:<6} {v_diff_mean:<14.6f} {v_diff_max:<14.6f} {v_norm_a:<14.2f} {v_norm_b:<14.2f}")

    print(f"\n平均 Value 差异: {total_v_diff / num_layers:.6f}")

    # ========================================
    # 结论
    # ========================================
    print("\n" + "="*80)
    print("结论")
    print("="*80)

    if total_v_diff / num_layers > 0.01:
        print("\n[发现] Value 存在显著差异！")
        print("\n原因：")
        print("  - 预处理时：chunk1 的 Value 只聚合了 system 的信息")
        print("  - 重算时：chunk1 的 Value 可以聚合 system + 其他 chunks 的信息")
        print("\n  即使 chunk1 的 RoPE 位置是正确的，")
        print("  其 Value 反映的是预处理时的有限上下文，")
        print("  而不是推理时的完整上下文。")
        print("\n  重算 chunk1 允许其 Value 整合完整上下文的信息，")
        print("  这可能提升答案质量，尤其当答案在 chunk1 中时。")
    else:
        print("\n差异较小")


if __name__ == '__main__':
    main()
