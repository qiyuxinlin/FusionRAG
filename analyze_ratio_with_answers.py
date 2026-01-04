#!/usr/bin/env python3
"""
分析不同重算比例下的答案变化

对每个测试样例：
1. 测试 0%, 5%, 10%, 15%, 20%, 25%, 30% 的重算比例
2. 记录每个比例下模型生成的答案
3. 分析 attention 分布特征
4. 找出规律：什么样的 attention 特征需要多高的重算比例
"""

import os
import sys
import torch
import numpy as np
import json
from transformers import AutoTokenizer, AutoConfig

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


def load_kv_cache(cache_path, model_name, example_idx, sub_q_idx, device):
    """加载预先保存的 KV cache"""
    cache_file = os.path.join(
        cache_path,
        f"{model_name}_ex{example_idx}_sub{sub_q_idx}_cache.pt"
    )

    if not os.path.exists(cache_file):
        raise FileNotFoundError(f"KV cache not found: {cache_file}")

    cache_data = torch.load(cache_file, map_location='cpu')
    return cache_data


def generate_with_cache(
    model,
    tokenizer,
    kv_cache,
    question_tensor,
    max_new_tokens=50,
    device='cuda:0'
):
    """使用 KV cache 生成答案"""
    model.eval()

    # 将 question_tensor 移到设备
    input_ids = question_tensor.unsqueeze(0).to(device)

    # 生成
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            past_key_values=kv_cache,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # 解码答案
    answer_ids = outputs[0][input_ids.shape[1]:]
    answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True)

    return answer_text.strip()


def test_ratio_with_answer_generation(
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
    测试不同重算比例下的答案生成
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

    # 分析 attention 分布特征
    print(f"\n{'='*100}")
    print("Attention 分布特征分析")
    print(f"{'='*100}")

    print(f"\n基本统计:")
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
    coverage_info = {}
    for threshold in [0.80, 0.85, 0.90, 0.95]:
        tokens_needed = np.searchsorted(cumsum_attn, threshold) + 1
        ratio = tokens_needed / effective_doc_len
        coverage_info[threshold] = {'tokens': tokens_needed, 'ratio': ratio}
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

    # 测试不同比例下的答案生成
    print(f"\n{'='*100}")
    print("测试不同重算比例下的答案生成")
    print(f"{'='*100}\n")

    results = []

    # 注意：这里简化了实现，实际需要：
    # 1. 加载完整的 prefix KV cache
    # 2. 根据选择的 tokens 重新计算部分 KV cache
    # 3. 使用组合后的 KV cache 生成答案
    #
    # 由于完整实现较复杂，这里先做选择分析和覆盖率计算
    # 需要用户提供如何加载和使用 KV cache 的接口

    for ratio in test_ratios:
        print(f"\n{'─'*100}")
        print(f"测试比例: {ratio:.0%}")
        print(f"{'─'*100}")

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

        print(f"选中 {num_selected} tokens ({actual_ratio:.2%})")

        if num_selected > 0:
            # 分析选中的 tokens 的 attention 覆盖
            selected_attention_sum = sum(multi_layer_attn[idx - selection_start]
                                        for idx in selected_indices
                                        if selection_start <= idx < selection_start + effective_doc_len)
            coverage = selected_attention_sum / multi_layer_attn.sum()
            print(f"Attention 覆盖: {coverage:.2%}")

            # 分析选中的 tokens 的分布
            doc_relative_positions = [idx - selection_start for idx in selected_indices
                                     if selection_start <= idx < selection_start + effective_doc_len]
            if doc_relative_positions:
                print(f"选中位置范围: [{min(doc_relative_positions)}, {max(doc_relative_positions)}]")
                print(f"位置跨度: {max(doc_relative_positions) - min(doc_relative_positions)}")

        # TODO: 这里需要实际生成答案
        # answer = generate_with_recomputed_cache(...)
        # 目前先标记为需要实现
        answer = "[需要实现答案生成]"

        results.append({
            'ratio': ratio,
            'num_selected': num_selected,
            'actual_ratio': actual_ratio,
            'coverage': coverage if num_selected > 0 else 0.0,
            'answer': answer,
        })

        print(f"生成答案: {answer}")

    # 汇总结果
    print(f"\n{'='*100}")
    print("结果汇总")
    print(f"{'='*100}\n")

    print(f"标准答案: {sub_q_info['answer']}\n")

    print("不同比例下的答案变化:")
    for r in results:
        print(f"  {r['ratio']:>5.0%}: {r['answer']} (coverage: {r['coverage']:.1%})")

    # 保存详细分析结果
    analysis_result = {
        'example_idx': example_idx,
        'sub_question_idx': sub_question_idx,
        'question': sub_q_info['query'],
        'ground_truth': sub_q_info['answer'],
        'doc_len': effective_doc_len,
        'attention_features': {
            'top1_ratio': float(top1_ratio),
            'top3_ratio': float(top3_ratio),
            'top5_ratio': float(top5_ratio),
            'top10_ratio': float(top10_ratio),
            'gini': float(gini),
            'num_components': len(components),
            'coverage_80': coverage_info[0.80]['ratio'],
            'coverage_85': coverage_info[0.85]['ratio'],
            'coverage_90': coverage_info[0.90]['ratio'],
        },
        'ratio_results': results
    }

    output_file = f'./ratio_answer_analysis_ex{example_idx}_sub{sub_question_idx}.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_result, f, indent=2, ensure_ascii=False)

    print(f"\n详细分析已保存到: {output_file}")

    return analysis_result


def analyze_multiple_examples(
    model,
    tokenizer,
    questions_data,
    system_tensor,
    example_indices=None,
    cache_path='/mnt/data/reflect/',
    model_name='Qwen2.5-7B-Instruct',
    device='cuda:0'
):
    """分析多个测试样例"""

    if example_indices is None:
        example_indices = [4, 5, 6, 7, 8]  # 默认测试几个样例

    all_results = []

    for idx in example_indices:
        if idx >= len(questions_data):
            continue

        example_data = questions_data[idx]

        # 选择第一个子问题
        if len(example_data['sub_questions']) == 0:
            continue

        result = test_ratio_with_answer_generation(
            model=model,
            tokenizer=tokenizer,
            example_data=example_data,
            system_tensor=system_tensor,
            example_idx=idx,
            sub_question_idx=0,
            cache_path=cache_path,
            model_name=model_name,
            device=device
        )

        all_results.append(result)

    # 综合分析
    print(f"\n{'='*100}")
    print("综合分析：Attention 特征 vs 需要的重算比例")
    print(f"{'='*100}\n")

    print("基于以上分析，总结规律：")
    print("\n1. Attention 集中度分析:")
    for result in all_results:
        features = result['attention_features']
        print(f"\n样例 {result['example_idx']}:")
        print(f"  Top-1: {features['top1_ratio']:.1%}, Top-3: {features['top3_ratio']:.1%}")
        print(f"  Gini: {features['gini']:.3f}")
        print(f"  Components: {features['num_components']}")
        print(f"  Coverage@85%: {features['coverage_85']:.1%}")
        # TODO: 根据实际答案变化，标注临界比例

    print("\n2. 建议的动态比例策略:")
    print("  [需要基于实际答案生成结果来设计]")

    # 保存综合分析
    summary_file = './ratio_answer_analysis_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n综合分析已保存到: {summary_file}")

    return all_results


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--example_indices', type=int, nargs='+', default=[4, 5, 6])
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--model_path', type=str, default='/mnt/data/models/Qwen2.5-7B-Instruct')
    parser.add_argument('--data_path', type=str, default='./result_reflect.json')
    parser.add_argument('--bge_model_path', type=str, default='/mnt/data/models/bge-m3-FP16')
    parser.add_argument('--cache_path', type=str, default='/mnt/data/reflect/')

    args = parser.parse_args()

    # 设置 GPU
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device.split(':')[-1]

    print("="*100)
    print("重算比例 vs 答案变化分析工具")
    print("="*100)
    print(f"测试样例: {args.example_indices}")
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
    max_idx = max(args.example_indices) + 1
    questions_data, system_tensor, context_rank, corpus_lens = prepare_reflect_data(
        args.data_path, tokenizer, args.bge_model_path, 'qwen', 10,
        max_main_questions=max_idx,
        preprocess=False
    )

    # 分析多个样例
    results = analyze_multiple_examples(
        model=model,
        tokenizer=tokenizer,
        questions_data=questions_data,
        system_tensor=system_tensor,
        example_indices=args.example_indices,
        cache_path=args.cache_path,
        model_name='Qwen2.5-7B-Instruct',
        device=args.device
    )

    print("\n注意：当前版本需要集成答案生成部分才能完整分析！")
    print("请提供如何加载和使用 KV cache 的接口。")


if __name__ == '__main__':
    main()
