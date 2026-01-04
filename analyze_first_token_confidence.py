#!/usr/bin/env python3
"""
分析 Draft Model 首 token 置信度与 rate=0.05 能否答对的关系

Draft model 做 Full Attention，获取首 token 的置信度（top-1 probability）
看看这个信号能否区分 stayed_correct 和 became_wrong
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json

sys.path.insert(0, '/mnt/data/wjh/FusionRAG')

from transformers import AutoTokenizer, AutoConfig
from test_fusionrag_reflect import load_model, prepare_reflect_data


def compute_first_token_confidence(model, input_ids, device='cuda:0'):
    """
    计算模型在给定输入下，首个生成 token 的置信度

    返回:
    - top1_prob: top-1 token 的概率
    - top5_prob: top-5 token 的累积概率
    - entropy: logits 的熵（越低越确定）
    - top1_token_id: top-1 token 的 id
    """
    model.eval()

    with torch.no_grad():
        outputs = model(input_ids, use_cache=False)
        # 获取最后一个位置的 logits
        last_logits = outputs.logits[0, -1, :]  # [vocab_size]

        # 计算概率分布
        probs = F.softmax(last_logits, dim=-1)

        # Top-1 概率
        top1_prob, top1_idx = probs.max(dim=-1)

        # Top-5 累积概率
        top5_probs, top5_indices = probs.topk(5)
        top5_prob = top5_probs.sum()

        # Top-10 累积概率
        top10_probs, _ = probs.topk(10)
        top10_prob = top10_probs.sum()

        # 熵
        log_probs = F.log_softmax(last_logits, dim=-1)
        entropy = -(probs * log_probs).sum()

        # 归一化熵 (除以 log(vocab_size))
        vocab_size = last_logits.shape[0]
        norm_entropy = entropy / np.log(vocab_size)

    return {
        'top1_prob': top1_prob.item(),
        'top5_prob': top5_prob.item(),
        'top10_prob': top10_prob.item(),
        'entropy': entropy.item(),
        'norm_entropy': norm_entropy.item(),
        'top1_token_id': top1_idx.item(),
    }


def analyze_questions(
    model,
    tokenizer,
    questions_data,
    system_tensor,
    results_df,
    device='cuda:0'
):
    """
    分析每个问题的首 token 置信度
    """
    all_results = []

    # 创建问题到结果的映射
    question_results = {}
    for _, row in results_df.iterrows():
        sub_q = row['Sub Question']
        question_results[sub_q] = {
            'correct_005': row['Correct'],
            'correct_1': row['Rate1_Correct'],
            'predicted_005': row['Predicted'],
            'predicted_1': row['Rate1_Predicted'],
            'ground_truth': row['Ground Truth'],
        }

    for q_idx, q_data in enumerate(questions_data):
        main_question = q_data['main_question']
        doc_tensors = q_data['doc_tensors']

        for sub_idx, sub_q_info in enumerate(q_data['sub_questions']):
            sub_question = sub_q_info['query']

            # 检查是否在结果中
            result_info = question_results.get(sub_question)
            if result_info is None:
                continue

            print(f"\nProcessing: {sub_question[:50]}...")

            # 准备输入
            doc_chunk_ids = sub_q_info['chunk_ids']
            sub_q_doc_tensors = [doc_tensors[chunk_id - 1] for chunk_id in doc_chunk_ids]

            question_text = f"<|im_end|>\n<|im_start|>user\nQuestion: {sub_q_info['query']}<|im_end|>\n<|im_start|>assistant\nAnswer: "
            question_tokens = tokenizer.encode(question_text, add_special_tokens=False)
            question_tensor = torch.tensor(question_tokens, dtype=torch.long)

            # 拼接完整输入
            all_tokens = [system_tensor] + sub_q_doc_tensors + [question_tensor]
            full_input = torch.cat(all_tokens).unsqueeze(0).to(device)

            # 计算首 token 置信度
            try:
                confidence_info = compute_first_token_confidence(model, full_input, device)
            except Exception as e:
                print(f"  Error: {e}")
                continue

            # 解码 top1 token
            top1_token = tokenizer.decode([confidence_info['top1_token_id']])

            # 分类
            if result_info['correct_1'] and result_info['correct_005']:
                category = 'stayed_correct'
            elif result_info['correct_1'] and not result_info['correct_005']:
                category = 'became_wrong'
            elif not result_info['correct_1'] and not result_info['correct_005']:
                category = 'stayed_wrong'
            else:
                category = 'became_correct'

            result = {
                'query': sub_question,
                'category': category,
                'top1_prob': confidence_info['top1_prob'],
                'top5_prob': confidence_info['top5_prob'],
                'top10_prob': confidence_info['top10_prob'],
                'entropy': confidence_info['entropy'],
                'norm_entropy': confidence_info['norm_entropy'],
                'top1_token': top1_token,
                'ground_truth': result_info['ground_truth'][:50],
                'predicted_005': result_info['predicted_005'][:50] if result_info['predicted_005'] else '',
            }

            all_results.append(result)
            print(f"  Category: {category}, top1_prob: {confidence_info['top1_prob']:.4f}, top1_token: '{top1_token}'")

    return all_results


def compare_groups(results):
    """比较不同组的置信度差异"""
    df = pd.DataFrame(results)

    print("\n" + "="*100)
    print("GROUP COMPARISON: First Token Confidence")
    print("="*100)

    stayed_correct = df[df['category'] == 'stayed_correct']
    became_wrong = df[df['category'] == 'became_wrong']

    print(f"\nSample sizes:")
    print(f"  stayed_correct: {len(stayed_correct)}")
    print(f"  became_wrong: {len(became_wrong)}")

    if len(became_wrong) == 0:
        print("No became_wrong samples found!")
        return df, {}

    # 比较各个特征
    feature_cols = ['top1_prob', 'top5_prob', 'top10_prob', 'entropy', 'norm_entropy']

    print(f"\n{'Feature':<15} {'Stayed Correct':<25} {'Became Wrong':<25} {'Diff':<12} {'Separation':<12}")
    print("-" * 90)

    separation_scores = {}

    for col in feature_cols:
        sc_mean = stayed_correct[col].mean()
        sc_std = stayed_correct[col].std()
        bw_mean = became_wrong[col].mean()
        bw_std = became_wrong[col].std()

        diff = bw_mean - sc_mean

        pooled_std = np.sqrt((sc_std**2 + bw_std**2) / 2)
        separation = abs(diff) / pooled_std if pooled_std > 0 else 0
        separation_scores[col] = separation

        print(f"{col:<15} {sc_mean:>8.4f} ± {sc_std:<8.4f}   {bw_mean:>8.4f} ± {bw_std:<8.4f}   {diff:>+8.4f}   {separation:>8.4f}")

    # 显示一些具体例子
    print("\n" + "="*100)
    print("EXAMPLES: became_wrong (低置信度?)")
    print("="*100)
    for _, row in became_wrong.head(10).iterrows():
        print(f"\nQ: {row['query'][:60]}...")
        print(f"  top1_prob: {row['top1_prob']:.4f}, top1_token: '{row['top1_token']}'")
        print(f"  Ground truth: {row['ground_truth']}")
        print(f"  Predicted@0.05: {row['predicted_005']}")

    print("\n" + "="*100)
    print("EXAMPLES: stayed_correct (高置信度?)")
    print("="*100)
    for _, row in stayed_correct.head(10).iterrows():
        print(f"\nQ: {row['query'][:60]}...")
        print(f"  top1_prob: {row['top1_prob']:.4f}, top1_token: '{row['top1_token']}'")

    return df, separation_scores


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--max_examples', type=int, default=None)
    parser.add_argument('--use_draft_model', action='store_true', help='Use draft model (3B) instead of main model (7B)')
    args = parser.parse_args()

    print("="*100)
    print("Analyzing First Token Confidence")
    print("="*100)

    # 读取 rate=0.05 结果
    results_df = pd.read_csv('/mnt/data/reflect/Qwen2.5-7B-Instruct/results/DraftModel_global_topk_10_rate_0.05_revert_rope.csv')
    print(f"Loaded {len(results_df)} results")

    # 选择模型
    if args.use_draft_model:
        model_path = '/mnt/data/models/Qwen2.5-3B-Instruct'
        print(f"Using Draft Model: {model_path}")
    else:
        model_path = '/mnt/data/models/Qwen2.5-7B-Instruct'
        print(f"Using Main Model: {model_path}")

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model, device_map = load_model('qwen', model_path, config, args.device, use_multi_gpu=True)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # 加载数据
    max_questions = args.max_examples if args.max_examples else None
    questions_data, system_tensor, _, _ = prepare_reflect_data(
        '/mnt/data/wjh/FusionRAG/result_reflect.json',
        tokenizer,
        '/mnt/data/models/bge-m3',
        'qwen',
        10,
        max_main_questions=max_questions,
        preprocess=False
    )

    # 分析问题
    results = analyze_questions(
        model, tokenizer, questions_data, system_tensor, results_df, args.device
    )

    # 保存结果
    output_file = './first_token_confidence_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 比较各组
    df, separation_scores = compare_groups(results)

    print(f"\nResults saved to {output_file}")


if __name__ == '__main__':
    main()
