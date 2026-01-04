#!/usr/bin/env python3
"""
测试单个样例在多个重算比例下的答案变化
只加载一次模型，然后测试所有比例，更高效
"""

import os
import sys
import torch
import numpy as np
import json
import csv
from transformers import AutoTokenizer, AutoConfig

project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

from test_fusionrag_reflect import (
    load_model, prepare_reflect_data, PreprocessScope
)
from ktransformers.util.utils import (
    compute_draft_model_attention,
    entropy_layer_selection,
    smart_query_selection,
    prefill_and_generate,
    find_connected_components
)
from ktransformers.models.custom_cache import StaticCache


def test_single_example_multiple_ratios(
    model,
    tokenizer,
    example_data,
    system_tensor,
    example_idx=4,
    sub_question_idx=0,
    test_ratios=None,
    cache_path='/mnt/data/reflect/',
    model_name='Qwen2.5-7B-Instruct',
    device='cuda:0',
    device_map=None
):
    """
    测试单个样例的一个子问题在不同重算比例下的表现
    """
    if test_ratios is None:
        test_ratios = [0.0, 0.10, 0.15, 0.20, 0.25, 0.30]

    q_data = example_data
    sub_q_info = q_data['sub_questions'][sub_question_idx]

    print(f"\n{'='*100}")
    print(f"测试样例 {example_idx}, 子问题 {sub_question_idx}")
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
    active_layers, _ = entropy_layer_selection(
        layer_attention_dict, top_k=4, return_entropy=True
    )
    print(f"选择的层: {active_layers}")

    # 聚合 attention
    layer_attentions = [layer_attention_dict[idx] for idx in active_layers]
    multi_layer_attn = torch.stack(layer_attentions).mean(dim=0).cpu().numpy()

    # 分析 attention 分布特征
    print(f"\n{'='*100}")
    print("Attention 分布特征")
    print(f"{'='*100}\n")

    sorted_attn = np.sort(multi_layer_attn)[::-1]
    top1_ratio = sorted_attn[0]
    top3_ratio = sorted_attn[:3].sum()
    top5_ratio = sorted_attn[:5].sum()

    print(f"Top-k Concentration:")
    print(f"  Top-1: {top1_ratio:.2%}")
    print(f"  Top-3: {top3_ratio:.2%}")
    print(f"  Top-5: {top5_ratio:.2%}")

    # Connected components
    mean_attn = multi_layer_attn.mean()
    std_attn = multi_layer_attn.std()
    threshold = mean_attn + 0.5 * std_attn
    high_attn_positions = list(np.where(multi_layer_attn > threshold)[0])
    components = find_connected_components(high_attn_positions, max_gap=2)

    print(f"\nConnected Components: {len(components)}")

    # Gini coefficient
    sorted_attn_gini = np.sort(multi_layer_attn)
    n = len(sorted_attn_gini)
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * sorted_attn_gini)) / (n * np.sum(sorted_attn_gini)) - (n + 1) / n
    print(f"Gini Coefficient: {gini:.4f}")

    # Coverage analysis
    cumsum_attn = np.cumsum(sorted_attn)
    for threshold in [0.80, 0.85, 0.90]:
        tokens_needed = np.searchsorted(cumsum_attn, threshold) + 1
        ratio = tokens_needed / effective_doc_len
        print(f"{threshold*100:.0f}% coverage: {tokens_needed} tokens ({ratio:.2%})")

    # 准备 KV cache 路径
    input_device = "cuda:0" if device_map is not None else device
    save_path = os.path.join(cache_path, model_name, 'kv_cache')

    # 为每个比例测试生成
    print(f"\n{'='*100}")
    print("测试不同重算比例下的答案生成")
    print(f"{'='*100}\n")

    results = []

    for ratio in test_ratios:
        print(f"\n{'─'*80}")
        print(f"测试比例: {ratio:.0%}")
        print(f"{'─'*80}")

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

        # 使用 prefill_and_generate 生成答案
        try:
            generated_tokens, extra_info = prefill_and_generate(
                model=model,
                tokenizer=tokenizer,
                passages=all_tokens,
                max_new_tokens=100,
                reprocess_method='Oracle',
                rate=ratio,
                system_len=system_len,
                use_entropy_selection=True,
                entropy_top_k=4,
                draft_layer_selection='entropy',
                preprocess=True,
                device=input_device,
                chunk_ids=doc_chunk_ids,
                device_map=device_map,
                draft_attention=oracle_attention,  # 复用已计算的 attention
            )

            # 解码答案
            answer = tokenizer.decode(torch.tensor(generated_tokens[:-1]), skip_special_tokens=True)
            print(f"生成答案: {answer}")

            # 计算 F1 和 EM
            from test_fusionrag_reflect import compute_f1, _exact_match_score
            f1_score = compute_f1(answer, sub_q_info['answer'], tokenizer)
            em_score = 1.0 if _exact_match_score(answer, sub_q_info['answer']) else 0.0

            print(f"F1: {f1_score:.4f}, EM: {em_score:.4f}")

            results.append({
                'ratio': ratio,
                'num_selected': num_selected,
                'actual_ratio': actual_ratio,
                'answer': answer,
                'f1': f1_score,
                'em': em_score,
                'correct': em_score > 0.5
            })

        except Exception as e:
            print(f"❌ 生成失败: {e}")
            results.append({
                'ratio': ratio,
                'num_selected': num_selected,
                'actual_ratio': actual_ratio,
                'answer': f"ERROR: {str(e)[:50]}",
                'f1': 0.0,
                'em': 0.0,
                'correct': False
            })

        torch.cuda.empty_cache()

    # 汇总结果
    print(f"\n{'='*100}")
    print("结果汇总")
    print(f"{'='*100}\n")

    print(f"标准答案: {sub_q_info['answer']}\n")
    print(f"{'Ratio':<8} {'Tokens':<10} {'F1':<10} {'EM':<8} {'Correct':<10} Answer")
    print("─" * 100)

    for r in results:
        correct_mark = "✓" if r['correct'] else "✗"
        print(f"{r['ratio']:<8.0%} {r['num_selected']:<10} {r['f1']:<10.4f} {r['em']:<8.2f} {correct_mark:<10} {r['answer'][:60]}")

    # 找出临界比例
    critical_ratio = None
    for i in range(len(results) - 1):
        if not results[i]['correct'] and results[i+1]['correct']:
            critical_ratio = results[i+1]['ratio']
            break

    if critical_ratio is not None:
        print(f"\n⚠️  临界比例: {critical_ratio:.0%} (从这个比例开始答对)")
    else:
        all_correct = all(r['correct'] for r in results)
        all_wrong = all(not r['correct'] for r in results)
        if all_correct:
            print(f"\n✓ 所有比例都能答对（包括 0%）")
        elif all_wrong:
            print(f"\n✗ 所有比例都答错（包括 30%）")

    # 保存结果
    output_data = {
        'example_idx': example_idx,
        'sub_question_idx': sub_question_idx,
        'question': sub_q_info['query'],
        'ground_truth': sub_q_info['answer'],
        'attention_features': {
            'top1_ratio': float(top1_ratio),
            'top3_ratio': float(top3_ratio),
            'top5_ratio': float(top5_ratio),
            'num_components': len(components),
            'gini': float(gini),
        },
        'critical_ratio': critical_ratio,
        'results': results
    }

    output_file = f'./ratio_test_ex{example_idx}_sub{sub_question_idx}.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n详细结果已保存到: {output_file}")

    return output_data


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--example_idx', type=int, default=4)
    parser.add_argument('--sub_question_idx', type=int, default=0)
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
    print("单样例多比例测试工具")
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
        preprocess=True,
        preprocess_scope=PreprocessScope.GLOBAL
    )

    example_data = questions_data[args.example_idx]

    # 测试
    result = test_single_example_multiple_ratios(
        model=model,
        tokenizer=tokenizer,
        example_data=example_data,
        system_tensor=system_tensor,
        example_idx=args.example_idx,
        sub_question_idx=args.sub_question_idx,
        cache_path=args.cache_path,
        model_name='Qwen2.5-7B-Instruct',
        device=args.device,
        device_map=device_map
    )

    print("\n✓ 测试完成！")


if __name__ == '__main__':
    main()
