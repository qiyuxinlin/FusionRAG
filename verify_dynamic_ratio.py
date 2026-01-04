#!/usr/bin/env python3
"""
快速验证 V2 动态比例算法是否真的产生不同的比例
只测试少量样例，输出动态比例分布
"""

import os
import sys
import torch
import numpy as np
from transformers import AutoTokenizer, AutoConfig

project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_dir)

from test_fusionrag_reflect import load_model, prepare_reflect_data
from ktransformers.models.custom_cache import StaticCache
from ktransformers.util.utils import (
    compute_draft_model_attention,
    entropy_layer_selection,
    compute_dynamic_ratio_comprehensive
)


def quick_test_dynamic_ratios(
    model,
    tokenizer,
    questions_data,
    system_tensor,
    num_samples=10,
    device='cuda:0'
):
    """
    快速测试多个样例的动态比例
    """
    print("="*100)
    print("V2 算法动态比例验证")
    print("="*100)
    print(f"测试样例数: {num_samples}")
    print()

    dynamic_ratios = []
    sample_details = []

    for idx in range(min(num_samples, len(questions_data))):
        q_data = questions_data[idx]

        # 随机选一个子问题
        if len(q_data['sub_questions']) == 0:
            continue

        sub_q_idx = 0
        sub_q_info = q_data['sub_questions'][sub_q_idx]

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

        # 计算 attention 分布
        all_tokens = [system_tensor] + sub_q_doc_tensors + [question_tensor]
        full_input = torch.cat(all_tokens).unsqueeze(0).to(device)

        query_start = system_len + doc_len
        oracle_attention = compute_draft_model_attention(model, full_input, query_start, device)

        # 提取 query→doc attention
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

        # 聚合 attention
        layer_attentions = [layer_attention_dict[idx] for idx in active_layers]
        multi_layer_attn = torch.stack(layer_attentions).mean(dim=0)

        # 计算动态比例
        budget_info = compute_dynamic_ratio_comprehensive(
            multi_layer_attn,
            system_len=selection_start,
            doc_len=effective_doc_len,
            query_len=query_len,
            base_ratio=0.3,
            min_ratio=0.05,
            max_ratio=0.30,
            device=device
        )

        dynamic_ratio = budget_info['dynamic_ratio']
        dynamic_ratios.append(dynamic_ratio)

        sample_details.append({
            'idx': idx,
            'sub_q_idx': sub_q_idx,
            'doc_len': effective_doc_len,
            'dynamic_ratio': dynamic_ratio,
            'base_coverage': budget_info['base_coverage_ratio'],
            'num_components': budget_info['num_components'],
            'gini': budget_info['gini_coefficient'],
            'concentration_score': budget_info['concentration']['concentration_score'],
            'top1_ratio': budget_info['concentration']['top1_ratio'],
        })

        print(f"样例 {idx}, 子问题 {sub_q_idx}:")
        print(f"  doc_len: {effective_doc_len}")
        print(f"  动态比例: {dynamic_ratio:.1%}")
        print(f"  Top-1: {budget_info['concentration']['top1_ratio']:.2%}")
        print(f"  Concentration score: {budget_info['concentration']['concentration_score']:.3f}")
        print(f"  Components: {budget_info['num_components']}")
        print(f"  Gini: {budget_info['gini_coefficient']:.3f}")
        print()

    # 统计分析
    print("="*100)
    print("动态比例分布统计")
    print("="*100)

    ratios_array = np.array(dynamic_ratios)
    print(f"样例数量: {len(dynamic_ratios)}")
    print(f"\n动态比例分布:")
    print(f"  Min:  {ratios_array.min():.1%}")
    print(f"  25%:  {np.percentile(ratios_array, 25):.1%}")
    print(f"  50%:  {np.percentile(ratios_array, 50):.1%}")
    print(f"  75%:  {np.percentile(ratios_array, 75):.1%}")
    print(f"  Max:  {ratios_array.max():.1%}")
    print(f"  Mean: {ratios_array.mean():.1%}")
    print(f"  Std:  {ratios_array.std():.1%}")

    # 检查是否真的动态
    unique_ratios = len(set(dynamic_ratios))
    print(f"\n唯一比例数: {unique_ratios}/{len(dynamic_ratios)}")

    if ratios_array.std() < 0.01:
        print("\n⚠️  警告: 标准差太小，算法仍然不够动态！")
    else:
        print("\n✓ 算法产生了不同的比例，具有动态性")

    # 显示详细分布
    print("\n详细分布:")
    ratio_bins = {}
    for r in dynamic_ratios:
        bin_key = f"{int(r * 100)}%"
        ratio_bins[bin_key] = ratio_bins.get(bin_key, 0) + 1

    for bin_key in sorted(ratio_bins.keys()):
        count = ratio_bins[bin_key]
        bar = '█' * count
        print(f"  {bin_key:>4}: {bar} ({count})")

    return sample_details


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--model_path', type=str, default='/mnt/data/models/Qwen2.5-7B-Instruct')
    parser.add_argument('--data_path', type=str, default='./result_reflect.json')
    parser.add_argument('--bge_model_path', type=str, default='/mnt/data/models/bge-m3-FP16')

    args = parser.parse_args()

    # 设置 GPU
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device.split(':')[-1]

    print("加载模型...")
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    model, device_map = load_model('qwen', args.model_path, config, args.device, use_multi_gpu=True)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    print("加载数据...")
    questions_data, system_tensor, context_rank, corpus_lens = prepare_reflect_data(
        args.data_path, tokenizer, args.bge_model_path, 'qwen', 10,
        max_main_questions=args.num_samples,
        preprocess=False
    )

    # 快速测试
    sample_details = quick_test_dynamic_ratios(
        model=model,
        tokenizer=tokenizer,
        questions_data=questions_data,
        system_tensor=system_tensor,
        num_samples=args.num_samples,
        device=args.device
    )


if __name__ == '__main__':
    main()
