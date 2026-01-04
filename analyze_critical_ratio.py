#!/usr/bin/env python3
"""
分析单个测试案例的临界重算比例

逐步增加重算比例，找到能答对的最小比例，分析临界点的特征
"""

import os
import sys
import torch
import numpy as np
from transformers import AutoTokenizer, AutoConfig
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_dir)

from test_fusionrag_reflect import load_model, prepare_reflect_data
from ktransformers.models.custom_cache import StaticCache
from ktransformers.util.utils import (
    compute_draft_model_attention,
    entropy_layer_selection,
    smart_query_selection,
    find_connected_components
)


def test_single_example_with_ratios(
    model,
    tokenizer,
    example_data,
    system_tensor,
    example_idx=4,
    sub_question_idx=1,
    test_ratios=None,
    cache_path='/mnt/data/reflect/',
    model_name='Qwen2.5-7B-Instruct',
    device='cuda:0'
):
    """
    测试单个样例在不同重算比例下的表现

    Args:
        test_ratios: 要测试的比例列表，如 [0.0, 0.05, 0.10, ..., 0.30]
    """
    if test_ratios is None:
        test_ratios = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    q_data = example_data
    sub_q_info = q_data['sub_questions'][sub_question_idx]

    print(f"\n{'='*100}")
    print(f"分析样例 {example_idx}, 子问题 {sub_question_idx}")
    print(f"{'='*100}")
    print(f"问题: {sub_q_info['query']}")
    print(f"标准答案: {sub_q_info['answer']}")
    print(f"{'='*100}\n")

    # 准备文档和 query
    doc_chunk_ids = sub_q_info['chunk_ids']
    doc_tensors = q_data['doc_tensors']
    sub_q_doc_tensors = [doc_tensors[chunk_id - 1] for chunk_id in doc_chunk_ids]

    question_text = f"<|im_end|>\n<|im_start|>user\nQuestion: {sub_q_info['query']}<|im_end|>\n<|im_start|>assistant\nAnswer: "
    question_tokens = tokenizer.encode(question_text, add_special_tokens=False)
    question_tensor = torch.tensor(question_tokens, dtype=torch.long)

    system_len = system_tensor.shape[0]
    doc_len = sum(t.shape[0] for t in sub_q_doc_tensors)
    query_len = question_tensor.shape[0]

    print(f"长度统计:")
    print(f"  System: {system_len} tokens")
    print(f"  Document: {doc_len} tokens")
    print(f"  Query: {query_len} tokens")
    print(f"  Total: {system_len + doc_len + query_len} tokens\n")

    # 计算 attention 分布（只需要计算一次）
    print("计算 Oracle attention 分布...")
    all_tokens = [system_tensor] + sub_q_doc_tensors + [question_tensor]
    full_input = torch.cat(all_tokens).unsqueeze(0).to(device)

    query_start = system_len + doc_len
    oracle_attention = compute_draft_model_attention(model, full_input, query_start, device)

    # 提取 query→doc attention（跳过第一个文档块，它被 prefix cache 命中）
    text_block1_len = sub_q_doc_tensors[0].shape[0]
    effective_doc_len = doc_len - text_block1_len
    selection_start = system_len + text_block1_len

    layer_attention_dict = {}
    for layer_idx, layer_attn in oracle_attention.items():
        query_to_doc = layer_attn[:, :, selection_start:selection_start + effective_doc_len]
        doc_attention_avg = query_to_doc.mean(axis=(0, 1))
        layer_attention_dict[layer_idx] = torch.tensor(doc_attention_avg, device=device)

    # 选择低熵层
    active_layers, layer_entropy = entropy_layer_selection(
        layer_attention_dict, top_k=4, return_entropy=True
    )
    print(f"选择的层: {active_layers}")

    # 聚合 attention
    layer_attentions = [layer_attention_dict[idx] for idx in active_layers]
    multi_layer_attn = torch.stack(layer_attentions).mean(dim=0).cpu().numpy()

    # 分析 attention 分布
    print(f"\nAttention 分布统计:")
    print(f"  Mean: {multi_layer_attn.mean():.6f}")
    print(f"  Std: {multi_layer_attn.std():.6f}")
    print(f"  Max: {multi_layer_attn.max():.6f}")
    print(f"  Min: {multi_layer_attn.min():.6f}")

    # Top-k concentration
    sorted_attn = np.sort(multi_layer_attn)[::-1]
    top1_ratio = sorted_attn[0]
    top3_ratio = sorted_attn[:3].sum()
    top5_ratio = sorted_attn[:5].sum()
    top10_ratio = sorted_attn[:10].sum()

    print(f"\nTop-k Concentration:")
    print(f"  Top-1: {top1_ratio:.2%}")
    print(f"  Top-3: {top3_ratio:.2%}")
    print(f"  Top-5: {top5_ratio:.2%}")
    print(f"  Top-10: {top10_ratio:.2%}")

    # Coverage analysis
    cumsum_attn = np.cumsum(sorted_attn)
    for threshold in [0.80, 0.85, 0.90, 0.95]:
        tokens_needed = np.searchsorted(cumsum_attn, threshold) + 1
        ratio = tokens_needed / effective_doc_len
        print(f"  {threshold*100:.0f}% coverage: {tokens_needed} tokens ({ratio:.2%})")

    # Connected components
    mean_attn = multi_layer_attn.mean()
    std_attn = multi_layer_attn.std()
    threshold = mean_attn + 0.5 * std_attn
    high_attn_positions = list(np.where(multi_layer_attn > threshold)[0])
    components = find_connected_components(high_attn_positions, max_gap=2)

    print(f"\nConnected Components:")
    print(f"  Number: {len(components)}")
    print(f"  Largest: {max([len(c) for c in components]) if components else 0} tokens")
    print(f"  High attention positions: {len(high_attn_positions)}")

    # Gini coefficient
    sorted_attn_gini = np.sort(multi_layer_attn)
    n = len(sorted_attn_gini)
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * sorted_attn_gini)) / (n * np.sum(sorted_attn_gini)) - (n + 1) / n
    print(f"\nGini Coefficient: {gini:.4f}")

    # 为每个比例测试生成
    print(f"\n{'='*100}")
    print("测试不同重算比例")
    print(f"{'='*100}\n")

    results = []

    for ratio in test_ratios:
        print(f"\n--- 测试比例: {ratio:.0%} ---")

        # 使用 smart_query_selection 选择 tokens
        selected_indices = smart_query_selection(
            attention_scores=torch.tensor(multi_layer_attn, device=device),
            doc_len=effective_doc_len,
            target_ratio=ratio,
            system_len=selection_start,
            device=device
        )

        num_selected = len(selected_indices)
        actual_ratio = num_selected / effective_doc_len

        print(f"  选中 {num_selected} tokens ({actual_ratio:.2%})")

        if num_selected > 0:
            # 分析选中的 tokens 的 attention 覆盖
            selected_attention_sum = sum(multi_layer_attn[idx - selection_start]
                                        for idx in selected_indices
                                        if selection_start <= idx < selection_start + effective_doc_len)
            coverage = selected_attention_sum / multi_layer_attn.sum()
            print(f"  Attention 覆盖: {coverage:.2%}")

        # 加载 KV cache 并生成
        # (简化版：这里只做选择分析，不实际生成)
        # 实际生成需要完整的 KV cache 加载和解码过程

        results.append({
            'ratio': ratio,
            'num_selected': num_selected,
            'actual_ratio': actual_ratio,
            'selected_indices': selected_indices,
        })

    # 可视化
    visualize_ratio_analysis(
        multi_layer_attn,
        results,
        effective_doc_len,
        example_idx,
        sub_question_idx
    )

    return results, multi_layer_attn


def visualize_ratio_analysis(multi_layer_attn, results, doc_len, example_idx, sub_q_idx):
    """可视化不同比例下的选择结果"""

    fig, axes = plt.subplots(len(results) + 1, 1, figsize=(15, 3 * (len(results) + 1)))

    # 第一个子图：原始 attention 分布
    ax = axes[0]
    x = np.arange(doc_len)
    ax.bar(x, multi_layer_attn, alpha=0.7, color='steelblue')
    ax.set_title('Original Attention Distribution')
    ax.set_xlabel('Token Position')
    ax.set_ylabel('Attention Score')
    ax.axhline(y=multi_layer_attn.mean(), color='red', linestyle='--', alpha=0.5, label='Mean')
    ax.legend()

    # 后续子图：每个比例的选择结果
    for i, result in enumerate(results):
        ax = axes[i + 1]

        # 创建颜色数组
        colors = ['green' if (idx in result['selected_indices']) else 'lightgray'
                  for idx in range(doc_len)]

        ax.bar(x, multi_layer_attn, color=colors, alpha=0.7)
        ax.set_title(f"Ratio {result['ratio']:.0%}: Selected {result['num_selected']} tokens ({result['actual_ratio']:.2%})")
        ax.set_xlabel('Token Position')
        ax.set_ylabel('Attention Score')

    plt.tight_layout()
    output_path = f'./critical_ratio_analysis_ex{example_idx}_sub{sub_q_idx}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n可视化已保存到: {output_path}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--example_idx', type=int, default=4)
    parser.add_argument('--sub_question_idx', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--model_path', type=str, default='/mnt/data/models/Qwen2.5-7B-Instruct')
    parser.add_argument('--data_path', type=str, default='./result_reflect.json')
    parser.add_argument('--bge_model_path', type=str, default='/mnt/data/models/bge-m3-FP16')

    args = parser.parse_args()

    # 设置 GPU
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device.split(':')[-1]

    print("="*100)
    print("临界重算比例分析工具")
    print("="*100)
    print(f"样例: {args.example_idx}, 子问题: {args.sub_question_idx}")
    print(f"设备: {args.device}")
    print("="*100)

    # 加载模型
    print("\n加载模型...")
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    model, device_map = load_model('qwen', args.model_path, config, args.device, use_multi_gpu=True)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # 加载数据
    print("加载数据...")
    questions_data, system_tensor, context_rank, corpus_lens = prepare_reflect_data(
        args.data_path, tokenizer, args.bge_model_path, 'qwen', 10,
        max_main_questions=args.example_idx + 1,
        preprocess=False
    )

    example_data = questions_data[args.example_idx]

    # 测试不同比例
    test_ratios = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]

    results, attention = test_single_example_with_ratios(
        model=model,
        tokenizer=tokenizer,
        example_data=example_data,
        system_tensor=system_tensor,
        example_idx=args.example_idx,
        sub_question_idx=args.sub_question_idx,
        test_ratios=test_ratios,
        device=args.device
    )

    # 总结分析
    print(f"\n{'='*100}")
    print("分析总结")
    print(f"{'='*100}")

    print("\n基于以上分析，建议的动态比例策略:")

    # 计算 attention 特征
    sorted_attn = np.sort(attention)[::-1]
    top1_ratio = sorted_attn[0]
    top3_ratio = sorted_attn[:3].sum()

    if top1_ratio > 0.5:
        print(f"  - Top-1 占 {top1_ratio:.0%} → 极度集中 → 建议 5-10%")
    elif top3_ratio > 0.7:
        print(f"  - Top-3 占 {top3_ratio:.0%} → 高度集中 → 建议 10-15%")
    else:
        print(f"  - 分布较均匀 → 建议 20-30%")

    print("\n注意: 最终需要实际测试生成质量来验证！")


if __name__ == '__main__':
    main()
