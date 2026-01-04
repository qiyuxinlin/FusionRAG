#!/usr/bin/env python3
"""
分析 Oracle vs DraftModel 选择的 token 差异

目的：理解为什么用主模型选择的 token 反而效果比小模型差
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoConfig
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

from test_fusionrag_reflect import load_model, prepare_reflect_data, load_system_prompt, PreprocessScope
from ktransformers.util.utils import (
    compute_draft_model_attention,
    smart_query_selection,
    entropy_layer_selection,
    find_connected_components
)


def analyze_attention_distribution(attention_dict, name="Model"):
    """分析 attention 分布特征"""
    print(f"\n{'='*60}")
    print(f"{name} Attention Distribution Analysis")
    print(f"{'='*60}")

    layer_stats = {}
    for layer_idx, attn in attention_dict.items():
        if isinstance(attn, torch.Tensor):
            attn = attn.cpu().numpy()

        # 归一化
        p = attn / (attn.sum() + 1e-10)
        p = np.clip(p, 1e-10, 1.0)

        # 计算熵
        entropy = -np.sum(p * np.log(p))
        max_entropy = np.log(len(attn))
        normalized_entropy = entropy / max_entropy

        # Gini 系数
        sorted_attn = np.sort(attn)
        n = len(sorted_attn)
        index = np.arange(1, n + 1)
        gini = ((2 * index - n - 1) * sorted_attn).sum() / (n * sorted_attn.sum() + 1e-10)

        # Top-k 集中度
        sorted_desc = np.sort(attn)[::-1]
        cumsum = np.cumsum(sorted_desc) / (sorted_desc.sum() + 1e-10)
        top_10_coverage = cumsum[min(int(len(attn)*0.1), len(attn)-1)]
        top_20_coverage = cumsum[min(int(len(attn)*0.2), len(attn)-1)]

        # 达到 85% 覆盖需要多少 token
        coverage_85_count = np.searchsorted(cumsum, 0.85) + 1
        coverage_85_ratio = coverage_85_count / len(attn)

        layer_stats[layer_idx] = {
            'entropy': entropy,
            'normalized_entropy': normalized_entropy,
            'gini': gini,
            'top_10_coverage': top_10_coverage,
            'top_20_coverage': top_20_coverage,
            'coverage_85_ratio': coverage_85_ratio,
            'max': np.max(attn),
            'mean': np.mean(attn),
            'std': np.std(attn)
        }

    # 打印统计
    print(f"\n{'Layer':<8} {'Entropy':<12} {'Norm_Ent':<12} {'Gini':<10} {'Top10%':<10} {'Top20%':<10} {'Cov85%':<10}")
    print("-" * 80)
    for layer_idx in sorted(layer_stats.keys()):
        s = layer_stats[layer_idx]
        print(f"{layer_idx:<8} {s['entropy']:<12.4f} {s['normalized_entropy']:<12.4f} "
              f"{s['gini']:<10.4f} {s['top_10_coverage']:<10.4f} {s['top_20_coverage']:<10.4f} "
              f"{s['coverage_85_ratio']:<10.4f}")

    return layer_stats


def compare_selections(oracle_indices, draft_indices, doc_len, oracle_attn, draft_attn):
    """对比两种方法选出的 token"""
    print(f"\n{'='*60}")
    print("Selection Comparison")
    print(f"{'='*60}")

    oracle_set = set(oracle_indices)
    draft_set = set(draft_indices)

    # 基本统计
    print(f"\nOracle selected: {len(oracle_indices)} tokens ({len(oracle_indices)/doc_len*100:.1f}%)")
    print(f"DraftModel selected: {len(draft_indices)} tokens ({len(draft_indices)/doc_len*100:.1f}%)")

    # 重叠分析
    overlap = oracle_set & draft_set
    only_oracle = oracle_set - draft_set
    only_draft = draft_set - oracle_set

    print(f"\nOverlap: {len(overlap)} tokens ({len(overlap)/len(oracle_set)*100:.1f}% of Oracle)")
    print(f"Only in Oracle: {len(only_oracle)} tokens")
    print(f"Only in DraftModel: {len(only_draft)} tokens")

    # Jaccard 相似度
    jaccard = len(overlap) / len(oracle_set | draft_set)
    print(f"Jaccard Similarity: {jaccard:.4f}")

    # 分析 Oracle 独有的 token 的 attention 特征
    if len(only_oracle) > 0:
        only_oracle_list = sorted(list(only_oracle))
        oracle_attn_np = oracle_attn.cpu().numpy() if isinstance(oracle_attn, torch.Tensor) else oracle_attn
        draft_attn_np = draft_attn.cpu().numpy() if isinstance(draft_attn, torch.Tensor) else draft_attn

        # Oracle 独有 token 在两个模型中的 attention 排名
        oracle_ranks = np.argsort(np.argsort(oracle_attn_np)[::-1])  # 排名 (0=最高)
        draft_ranks = np.argsort(np.argsort(draft_attn_np)[::-1])

        print(f"\n--- Oracle-only tokens analysis ---")
        print(f"These {len(only_oracle)} tokens are selected by Oracle but NOT by DraftModel:")

        oracle_only_oracle_ranks = [oracle_ranks[i] for i in only_oracle_list]
        oracle_only_draft_ranks = [draft_ranks[i] for i in only_oracle_list]

        print(f"  In Oracle attention: avg rank = {np.mean(oracle_only_oracle_ranks):.1f} (lower=more important)")
        print(f"  In Draft attention:  avg rank = {np.mean(oracle_only_draft_ranks):.1f}")
        print(f"  Rank difference: {np.mean(oracle_only_draft_ranks) - np.mean(oracle_only_oracle_ranks):.1f}")

    # 分析 DraftModel 独有的 token
    if len(only_draft) > 0:
        only_draft_list = sorted(list(only_draft))
        oracle_attn_np = oracle_attn.cpu().numpy() if isinstance(oracle_attn, torch.Tensor) else oracle_attn
        draft_attn_np = draft_attn.cpu().numpy() if isinstance(draft_attn, torch.Tensor) else draft_attn

        oracle_ranks = np.argsort(np.argsort(oracle_attn_np)[::-1])
        draft_ranks = np.argsort(np.argsort(draft_attn_np)[::-1])

        print(f"\n--- DraftModel-only tokens analysis ---")
        print(f"These {len(only_draft)} tokens are selected by DraftModel but NOT by Oracle:")

        draft_only_oracle_ranks = [oracle_ranks[i] for i in only_draft_list]
        draft_only_draft_ranks = [draft_ranks[i] for i in only_draft_list]

        print(f"  In Oracle attention: avg rank = {np.mean(draft_only_oracle_ranks):.1f}")
        print(f"  In Draft attention:  avg rank = {np.mean(draft_only_draft_ranks):.1f} (lower=more important)")
        print(f"  Rank difference: {np.mean(draft_only_oracle_ranks) - np.mean(draft_only_draft_ranks):.1f}")

    return {
        'overlap': len(overlap),
        'only_oracle': len(only_oracle),
        'only_draft': len(only_draft),
        'jaccard': jaccard
    }


def visualize_attention_comparison(oracle_attn, draft_attn, oracle_indices, draft_indices,
                                   doc_len, output_path='attention_oracle_vs_draft.png'):
    """可视化 attention 分布对比"""
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))

    oracle_attn_np = oracle_attn.cpu().numpy() if isinstance(oracle_attn, torch.Tensor) else oracle_attn
    draft_attn_np = draft_attn.cpu().numpy() if isinstance(draft_attn, torch.Tensor) else draft_attn

    # 归一化
    oracle_attn_norm = oracle_attn_np / (oracle_attn_np.max() + 1e-10)
    draft_attn_norm = draft_attn_np / (draft_attn_np.max() + 1e-10)

    x = np.arange(doc_len)

    # Plot 1: Attention distribution comparison
    ax1 = axes[0]
    ax1.plot(x, oracle_attn_norm, 'b-', alpha=0.7, label='Oracle (Main Model)', linewidth=0.8)
    ax1.plot(x, draft_attn_norm, 'r-', alpha=0.7, label='DraftModel (Small Model)', linewidth=0.8)
    ax1.set_xlabel('Token Position')
    ax1.set_ylabel('Normalized Attention')
    ax1.set_title('Attention Distribution Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Selection comparison
    ax2 = axes[1]
    oracle_mask = np.zeros(doc_len)
    draft_mask = np.zeros(doc_len)
    for idx in oracle_indices:
        if idx < doc_len:
            oracle_mask[idx] = 1
    for idx in draft_indices:
        if idx < doc_len:
            draft_mask[idx] = 1

    ax2.fill_between(x, 0, oracle_mask, alpha=0.5, color='blue', label='Oracle Selection')
    ax2.fill_between(x, 0, -draft_mask, alpha=0.5, color='red', label='DraftModel Selection')
    ax2.set_xlabel('Token Position')
    ax2.set_ylabel('Selected')
    ax2.set_title('Token Selection Comparison (Blue=Oracle, Red=DraftModel)')
    ax2.legend()
    ax2.set_ylim(-1.5, 1.5)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Attention difference
    ax3 = axes[2]
    diff = oracle_attn_norm - draft_attn_norm
    colors = ['blue' if d > 0 else 'red' for d in diff]
    ax3.bar(x, diff, color=colors, alpha=0.6, width=1.0)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_xlabel('Token Position')
    ax3.set_ylabel('Attention Difference (Oracle - Draft)')
    ax3.set_title('Attention Difference: Blue=Oracle higher, Red=Draft higher')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nVisualization saved to {output_path}")


def analyze_layer_selection_difference(oracle_layer_dict, draft_layer_dict, entropy_top_k=4):
    """分析熵选层的差异"""
    print(f"\n{'='*60}")
    print("Layer Selection Analysis (Entropy-based)")
    print(f"{'='*60}")

    # Oracle 熵选层
    oracle_layers, oracle_entropy = entropy_layer_selection(
        oracle_layer_dict, top_k=entropy_top_k, return_entropy=True
    )

    # DraftModel 熵选层
    draft_layers, draft_entropy = entropy_layer_selection(
        draft_layer_dict, top_k=entropy_top_k, return_entropy=True
    )

    print(f"\nOracle selected layers: {oracle_layers}")
    print(f"DraftModel selected layers: {draft_layers}")

    print(f"\nOracle layer entropy (lower = more concentrated):")
    for layer_idx in sorted(oracle_entropy.keys()):
        marker = " <-- selected" if layer_idx in oracle_layers else ""
        print(f"  Layer {layer_idx}: {oracle_entropy[layer_idx]:.4f}{marker}")

    print(f"\nDraftModel layer entropy:")
    for layer_idx in sorted(draft_entropy.keys()):
        marker = " <-- selected" if layer_idx in draft_layers else ""
        print(f"  Layer {layer_idx}: {draft_entropy[layer_idx]:.4f}{marker}")

    return oracle_layers, draft_layers, oracle_entropy, draft_entropy


def main():
    # 配置
    model_path = '/mnt/data/models/Qwen2.5-7B-Instruct'
    draft_model_path = '/mnt/data/models/Qwen2.5-3B-Instruct'
    data_path = './result_reflect.json'
    bge_model_path = '/mnt/data/models/bge-m3-FP16'
    device = "cuda:0"
    rate = 0.3
    topk = 10
    entropy_top_k = 4

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # 加载主模型 (单 GPU，避免多 GPU 问题)
    print("\nLoading main model (single GPU for fair comparison)...")
    main_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    main_config._attn_implementation = "sdpa"
    main_model, _ = load_model('qwen', model_path, main_config, device, use_multi_gpu=False)
    main_model.eval()
    print(f"Main model loaded: {main_model.config.num_hidden_layers} layers")

    # 加载 draft model
    print("\nLoading draft model...")
    draft_config = AutoConfig.from_pretrained(draft_model_path, trust_remote_code=True)
    draft_config._attn_implementation = "sdpa"
    draft_model, _ = load_model('qwen', draft_model_path, draft_config, device, use_multi_gpu=False)
    draft_model.eval()
    print(f"Draft model loaded: {draft_model.config.num_hidden_layers} layers")

    # 准备数据
    print("\nPreparing data...")
    questions_data, system_tensor, context_rank, corpus_lens = prepare_reflect_data(
        data_path, tokenizer, bge_model_path, 'qwen', topk, max_main_questions=1,
        preprocess=True, preprocess_scope=PreprocessScope.GLOBAL
    )

    # 取第一个问题进行分析
    q_data = questions_data[0]
    doc_tensors = q_data['doc_tensors']
    sub_q_info = q_data['sub_questions'][0]

    print(f"\nAnalyzing question: {sub_q_info['query'][:100]}...")

    # 构建输入
    system_len = system_tensor.shape[0]
    question_text = f"<|im_end|>\n<|im_start|>user\nQuestion: {sub_q_info['query']}<|im_end|>\n<|im_start|>assistant\nAnswer: "
    question_tokens = tokenizer.encode(question_text, add_special_tokens=False)
    question_tensor = torch.tensor(question_tokens, dtype=torch.long)

    doc_chunk_ids = sub_q_info['chunk_ids']
    sub_q_doc_tensors = [doc_tensors[chunk_id - 1] for chunk_id in doc_chunk_ids]

    passages = [system_tensor] + sub_q_doc_tensors + [question_tensor]
    passages_len = [p.shape[0] for p in passages]

    full_input = torch.cat(passages).unsqueeze(0).to(device)
    query_start = sum(passages_len[:-1])

    text_block1_len = passages_len[1]
    doc_len = sum(passages_len[2:-1])
    selection_start = system_len + text_block1_len

    print(f"\nInput structure:")
    print(f"  System: {system_len} tokens")
    print(f"  Doc block 1: {text_block1_len} tokens (cached, not selected)")
    print(f"  Doc blocks 2-n: {doc_len} tokens (selection area)")
    print(f"  Query: {passages_len[-1]} tokens")
    print(f"  Total: {full_input.shape[1]} tokens")

    # 计算 Oracle attention
    print("\n" + "="*60)
    print("Computing Oracle (Main Model) Attention...")
    print("="*60)
    oracle_attention = compute_draft_model_attention(main_model, full_input, query_start, device)
    torch.cuda.empty_cache()

    # 计算 DraftModel attention
    print("\n" + "="*60)
    print("Computing DraftModel (Small Model) Attention...")
    print("="*60)
    draft_attention = compute_draft_model_attention(draft_model, full_input, query_start, device)
    torch.cuda.empty_cache()

    # 收集各层的 query→doc attention
    oracle_layer_dict = {}
    for layer_idx, layer_attn in oracle_attention.items():
        query_to_doc = layer_attn[:, :, selection_start:selection_start + doc_len]
        doc_attention_avg = query_to_doc.mean(axis=(0, 1))
        oracle_layer_dict[layer_idx] = torch.tensor(doc_attention_avg, device=device)

    draft_layer_dict = {}
    for layer_idx, layer_attn in draft_attention.items():
        query_to_doc = layer_attn[:, :, selection_start:selection_start + doc_len]
        doc_attention_avg = query_to_doc.mean(axis=(0, 1))
        draft_layer_dict[layer_idx] = torch.tensor(doc_attention_avg, device=device)

    # 分析各层 attention 分布
    oracle_stats = analyze_attention_distribution(oracle_layer_dict, "Oracle (Main Model)")
    draft_stats = analyze_attention_distribution(draft_layer_dict, "DraftModel (Small Model)")

    # 分析熵选层差异
    oracle_layers, draft_layers, oracle_entropy, draft_entropy = analyze_layer_selection_difference(
        oracle_layer_dict, draft_layer_dict, entropy_top_k
    )

    # 聚合选中层的 attention
    oracle_multi_attn = torch.stack([oracle_layer_dict[idx] for idx in oracle_layers]).mean(dim=0)
    draft_multi_attn = torch.stack([draft_layer_dict[idx] for idx in draft_layers]).mean(dim=0)

    # 分析聚合后的 attention
    print(f"\n{'='*60}")
    print("Aggregated Attention Analysis")
    print(f"{'='*60}")

    oracle_agg_np = oracle_multi_attn.cpu().numpy()
    draft_agg_np = draft_multi_attn.cpu().numpy()

    # 计算相关性
    correlation = np.corrcoef(oracle_agg_np, draft_agg_np)[0, 1]
    print(f"\nCorrelation between Oracle and DraftModel attention: {correlation:.4f}")

    # Spearman rank correlation
    from scipy.stats import spearmanr
    spearman_corr, _ = spearmanr(oracle_agg_np, draft_agg_np)
    print(f"Spearman rank correlation: {spearman_corr:.4f}")

    # 执行 token 选择
    oracle_indices = smart_query_selection(
        attention_scores=oracle_multi_attn,
        doc_len=doc_len,
        target_ratio=rate,
        system_len=selection_start,
        device=device
    )
    # 转换为相对索引 (去掉 selection_start 偏移)
    oracle_indices_relative = [idx - selection_start for idx in oracle_indices]

    draft_indices = smart_query_selection(
        attention_scores=draft_multi_attn,
        doc_len=doc_len,
        target_ratio=rate,
        system_len=selection_start,
        device=device
    )
    draft_indices_relative = [idx - selection_start for idx in draft_indices]

    # 对比选择结果
    comparison = compare_selections(
        oracle_indices_relative, draft_indices_relative,
        doc_len, oracle_multi_attn, draft_multi_attn
    )

    # 可视化
    visualize_attention_comparison(
        oracle_multi_attn, draft_multi_attn,
        oracle_indices_relative, draft_indices_relative,
        doc_len, 'attention_oracle_vs_draft.png'
    )

    # 总结
    print(f"\n{'='*60}")
    print("SUMMARY: Why Oracle might perform worse than DraftModel")
    print(f"{'='*60}")

    # 计算关键指标
    oracle_gini_avg = np.mean([s['gini'] for s in oracle_stats.values()])
    draft_gini_avg = np.mean([s['gini'] for s in draft_stats.values()])

    oracle_entropy_avg = np.mean([s['normalized_entropy'] for s in oracle_stats.values()])
    draft_entropy_avg = np.mean([s['normalized_entropy'] for s in draft_stats.values()])

    print(f"\n1. Attention Concentration:")
    print(f"   Oracle avg Gini: {oracle_gini_avg:.4f} (higher = more concentrated)")
    print(f"   Draft avg Gini:  {draft_gini_avg:.4f}")
    if oracle_gini_avg < draft_gini_avg:
        print(f"   → DraftModel attention is MORE concentrated!")

    print(f"\n2. Attention Entropy:")
    print(f"   Oracle avg normalized entropy: {oracle_entropy_avg:.4f} (lower = more concentrated)")
    print(f"   Draft avg normalized entropy:  {draft_entropy_avg:.4f}")
    if oracle_entropy_avg > draft_entropy_avg:
        print(f"   → DraftModel attention has LOWER entropy (more focused)!")

    print(f"\n3. Selection Overlap:")
    print(f"   Jaccard similarity: {comparison['jaccard']:.4f}")
    print(f"   Only {comparison['overlap']} tokens selected by both methods")

    print(f"\n4. Layer Selection:")
    print(f"   Oracle uses layers: {oracle_layers} (from {main_model.config.num_hidden_layers} total)")
    print(f"   Draft uses layers:  {draft_layers} (from {draft_model.config.num_hidden_layers} total)")

    print(f"\n5. Possible Explanations:")
    print(f"   a) Small model has more focused attention (limited capacity forces prioritization)")
    print(f"   b) Large model's attention is more diffuse (attends to more context)")
    print(f"   c) The 'important' tokens for generation may differ from high-attention tokens")
    print(f"   d) Small model's attention pattern may better match the recomputation needs")


if __name__ == "__main__":
    main()
