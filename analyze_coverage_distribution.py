#!/usr/bin/env python3
"""
分析不同覆盖率需要的token比例，用于设计动态rate公式。
"""

import os
import sys
import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

sys.path.insert(0, '/mnt/data/wjh/FusionRAG')
from ktransformers.util.utils import rotate_half

DRAFT_MODEL_PATH = "/mnt/data/models/Qwen2.5-3B-Instruct"


def compute_draft_attention_and_coverage(model, tokenizer, input_text, device="cuda:0"):
    """计算attention并返回各覆盖率需要的token比例"""
    config = model.config
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // num_heads

    input_ids = tokenizer(input_text, return_tensors="pt")["input_ids"].to(device)
    seq_len = input_ids.shape[1]

    # 只收集后半部分层的attention
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

            # rotary_emb可能在不同位置
            if hasattr(layer.self_attn, 'rotary_emb'):
                rotary_emb = layer.self_attn.rotary_emb
            elif hasattr(model.model, 'rotary_emb'):
                rotary_emb = model.model.rotary_emb
            else:
                # 尝试从第一层获取
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

            # 只保存后半部分层
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


def analyze_coverage(layer_attentions, system_len, doc_len, query_len, use_entropy_selection=True):
    """分析各覆盖率需要的token比例"""
    total_len = system_len + doc_len + query_len
    doc_start = system_len
    doc_end = system_len + doc_len
    query_start = system_len + doc_len

    # 收集各层的attention和熵
    layer_doc_attentions = []
    layer_entropies = []

    for layer_attn in layer_attentions:
        # 提取 query→doc attention
        query_to_doc = layer_attn[:, query_start:total_len, doc_start:doc_end]
        doc_attention_avg = query_to_doc.mean(axis=(0, 1))
        layer_doc_attentions.append(doc_attention_avg)

        # 计算该层的熵
        p = doc_attention_avg / (doc_attention_avg.sum() + 1e-10)
        p = np.clip(p, 1e-10, 1.0)
        entropy = -np.sum(p * np.log(p))
        layer_entropies.append(entropy)

    if use_entropy_selection:
        # 选择熵最低的4层（和实际DraftModelDynamic一致）
        sorted_indices = np.argsort(layer_entropies)[:4]
        selected_attentions = [layer_doc_attentions[i] for i in sorted_indices]
        aggregated_attn = np.mean(selected_attentions, axis=0)
    else:
        # 平均所有层
        aggregated_attn = np.mean(layer_doc_attentions, axis=0)

    # 计算各覆盖率
    sorted_indices = np.argsort(aggregated_attn)[::-1]
    sorted_attn = aggregated_attn[sorted_indices]
    cumsum = np.cumsum(sorted_attn) / (sorted_attn.sum() + 1e-10)

    coverage_ratios = {}
    for coverage in [0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]:
        count = np.searchsorted(cumsum, coverage) + 1
        coverage_ratios[f'coverage_{int(coverage*100)}'] = count / doc_len

    # Gini系数
    sorted_attn_asc = np.sort(aggregated_attn)
    n = len(sorted_attn_asc)
    index = np.arange(1, n + 1)
    gini = ((2 * index - n - 1) * sorted_attn_asc).sum() / (n * sorted_attn_asc.sum() + 1e-10)
    coverage_ratios['gini'] = gini

    # 归一化熵
    p = aggregated_attn / (aggregated_attn.sum() + 1e-10)
    p = np.clip(p, 1e-10, 1.0)
    entropy = -np.sum(p * np.log(p))
    max_entropy = np.log(doc_len) if doc_len > 0 else 1
    coverage_ratios['normalized_entropy'] = entropy / max_entropy

    return coverage_ratios


def main():
    device = "cuda:0"

    print("Loading draft model...")
    tokenizer = AutoTokenizer.from_pretrained(DRAFT_MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        DRAFT_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map=device
    )
    model.eval()
    print(f"Model loaded: {model.config.num_hidden_layers} layers")

    # 加载测试数据
    with open('/mnt/data/wjh/FusionRAG/result_reflect.json') as f:
        data = json.load(f)

    all_coverage_stats = []

    # 只测试前20个问题
    for i, item in enumerate(data[:20]):
        main_q = item['question']
        intermediate = item.get('intermediate_context', [])

        for sub_idx, sub_q in enumerate(intermediate):
            query = sub_q.get('query', '')
            if query.startswith("Intermediate query"):
                colon_pos = query.find(":")
                if colon_pos != -1:
                    query = query[colon_pos + 1:].strip()

            docs = sub_q.get('retrieve docs', [])[:10]
            if not docs:
                continue

            # 构建输入
            system_prompt = "You are a helpful assistant."
            docs_text = "\n".join(docs)
            full_text = f"{system_prompt}\n\n{docs_text}\n\nQuestion: {query}\nAnswer:"

            # 计算长度
            system_tokens = tokenizer(system_prompt, return_tensors="pt")["input_ids"][0]
            docs_tokens = tokenizer(docs_text, return_tensors="pt")["input_ids"][0]
            full_tokens = tokenizer(full_text, return_tensors="pt")["input_ids"][0]

            system_len = len(system_tokens)
            doc_len = len(docs_tokens)
            query_len = len(full_tokens) - system_len - doc_len

            if doc_len < 100:  # 跳过太短的文档
                continue

            try:
                layer_attentions, seq_len = compute_draft_attention_and_coverage(
                    model, tokenizer, full_text, device
                )
                coverage = analyze_coverage(layer_attentions, system_len, doc_len, query_len)
                coverage['doc_len'] = doc_len
                coverage['question'] = query[:50]
                all_coverage_stats.append(coverage)

                print(f"[{len(all_coverage_stats)}] doc_len={doc_len}, "
                      f"cov85={coverage['coverage_85']*100:.1f}%, "
                      f"cov95={coverage['coverage_95']*100:.1f}%, "
                      f"cov99={coverage['coverage_99']*100:.1f}%")

            except Exception as e:
                print(f"Error: {e}")
                continue

            if len(all_coverage_stats) >= 30:
                break

        if len(all_coverage_stats) >= 30:
            break

    # 统计分析
    print("\n" + "="*70)
    print("Coverage 分布统计")
    print("="*70)

    for key in ['coverage_50', 'coverage_70', 'coverage_80', 'coverage_85',
                'coverage_90', 'coverage_95', 'coverage_99', 'gini', 'normalized_entropy']:
        values = [s[key] for s in all_coverage_stats]
        print(f"\n{key}:")
        print(f"  min={min(values):.3f}, max={max(values):.3f}, "
              f"mean={np.mean(values):.3f}, median={np.median(values):.3f}")

    # 保存结果
    with open('coverage_stats.json', 'w') as f:
        json.dump(all_coverage_stats, f, indent=2)

    print("\n" + "="*70)
    print("公式建议")
    print("="*70)

    cov95_mean = np.mean([s['coverage_95'] for s in all_coverage_stats])
    cov99_mean = np.mean([s['coverage_99'] for s in all_coverage_stats])

    print(f"""
达到95%覆盖平均需要: {cov95_mean*100:.1f}% 的token
达到99%覆盖平均需要: {cov99_mean*100:.1f}% 的token

建议公式:
  rate = coverage_95_ratio  (直接使用95%覆盖率作为rate)

或者:
  rate = coverage_90_ratio * 1.2  (90%覆盖率 + 20%安全边际)
""")


if __name__ == "__main__":
    main()
