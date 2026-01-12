#!/usr/bin/env python3
"""
收集每个问题的 attention 分布和最优重算比例

两阶段设计：
1. 第一次运行：计算 attention 分布 + 搜索每个 rate 的结果，保存到文件
2. 后续分析：直接加载保存的数据进行分析和建模

重算比例：0, 0.05, 0.10, ..., 1.0 (共 21 个点)
"""

import os
import sys
import json
import torch
import numpy as np
import pickle
from tqdm import tqdm
from datetime import datetime

sys.path.insert(0, '/mnt/data/wjh/FusionRAG')

from transformers import AutoTokenizer, AutoConfig
from openai import OpenAI
from ktransformers.util.utils import (
    load_kv_and_generate,
    prefill_and_generate,
    compute_draft_model_attention,
    entropy_layer_selection,
    compute_f1,
    _exact_match_score,
)
from ktransformers.models.custom_cache import StaticCache
from test_fusionrag_reflect import (
    prepare_reflect_data, load_system_prompt, PreprocessScope, load_model,
    judge_answer_with_openai  # 使用 DeepSeek API 判断答案正确性
)


def extract_attention_features(draft_attention, query_start, system_len, passages_len,
                                 entropy_top_k=4, device="cuda:0"):
    """
    从 draft model attention 中提取特征和原始分布
    """
    text_block1_len = passages_len[1] if len(passages_len) > 1 else 0
    selection_start = system_len + text_block1_len
    doc_len = sum(passages_len[2:-1]) if len(passages_len) > 2 else 0

    if doc_len == 0:
        return None

    # 收集各层的 query→doc attention
    layer_attention_dict = {}
    for layer_idx, layer_attn in draft_attention.items():
        query_to_doc = layer_attn[:, :, selection_start:selection_start + doc_len]
        doc_attention_avg = query_to_doc.mean(axis=(0, 1))
        layer_attention_dict[layer_idx] = torch.tensor(doc_attention_avg, device=device)

    # 熵选层
    active_layers, layer_entropy = entropy_layer_selection(
        layer_attention_dict, top_k=entropy_top_k, return_entropy=True
    )
    layer_attentions = [layer_attention_dict[idx] for idx in active_layers]
    aggregated_attn = torch.stack(layer_attentions).mean(dim=0).cpu().numpy()

    # 计算特征
    features = {}

    # 基本统计
    features['peak_strength'] = float(aggregated_attn.max())
    features['attention_mean'] = float(aggregated_attn.mean())
    features['attention_std'] = float(aggregated_attn.std())
    features['attention_var'] = float(aggregated_attn.var())

    # Top-k concentration
    sorted_attn = np.sort(aggregated_attn)[::-1]
    total = sorted_attn.sum()

    for k in [5, 10, 20, 50]:
        features[f'top{k}_concentration'] = float(sorted_attn[:k].sum() / total) if total > 0 else 0

    # Coverage ratios
    cumsum = np.cumsum(sorted_attn)
    for coverage in [0.5, 0.7, 0.8, 0.9, 0.95]:
        coverage_idx = np.where(cumsum >= coverage * total)[0]
        if len(coverage_idx) > 0:
            tokens_needed = coverage_idx[0] + 1
        else:
            tokens_needed = len(aggregated_attn)
        features[f'coverage_{int(coverage*100)}_ratio'] = float(tokens_needed / len(aggregated_attn))

    # Normalized entropy
    p = aggregated_attn / (aggregated_attn.sum() + 1e-10)
    p = np.clip(p, 1e-10, 1.0)
    entropy = -(p * np.log(p)).sum()
    max_entropy = np.log(len(aggregated_attn))
    features['normalized_entropy'] = float(entropy / max_entropy) if max_entropy > 0 else 0

    # Gini coefficient
    sorted_p = np.sort(p)
    n = len(sorted_p)
    cumsum_p = np.cumsum(sorted_p)
    gini = (2 * np.sum(np.arange(1, n+1) * sorted_p) / (n * sorted_p.sum()) - (n + 1) / n)
    features['gini'] = float(gini)

    # Layer consistency
    top_k_per_layer = min(50, doc_len)
    layer_top_tokens = []
    for idx in active_layers:
        top_indices = torch.topk(layer_attention_dict[idx], top_k_per_layer).indices
        layer_top_tokens.append(set(top_indices.tolist()))

    consistency_scores = []
    for i in range(len(layer_top_tokens) - 1):
        intersection = len(layer_top_tokens[i] & layer_top_tokens[i+1])
        union = len(layer_top_tokens[i] | layer_top_tokens[i+1])
        if union > 0:
            consistency_scores.append(intersection / union)
    features['layer_consistency'] = float(np.mean(consistency_scores)) if consistency_scores else 0.5

    # 文档长度
    features['doc_len'] = doc_len
    features['log_doc_len'] = float(np.log(doc_len + 1))

    # 活跃层
    features['active_layers'] = active_layers

    return {
        'features': features,
        'attention_distribution': aggregated_attn.tolist(),  # 保存原始分布
        'layer_entropy': {k: float(v) for k, v in layer_entropy.items()},
    }


def test_single_rate(model, tokenizer, past_key_values, passages, passages_len,
                     load_path, example_id, chunk_ids, rate,
                     draft_model=None, revert_rope=True, device="cuda:0", device_map=None,
                     original_kv_path=None, preprocess=True):
    """
    测试单个 rate 的效果 (使用 DraftModel 方法)
    返回生成的答案，判断交给调用方用 DeepSeek API 判断
    """
    # 重置 KV cache
    for layer_idx in range(len(past_key_values.key_cache)):
        past_key_values.past_tokens[layer_idx] = 0

    try:
        if rate == 1.0:
            inputs = torch.cat(passages).to(device).unsqueeze(0)
            generated_tokens, _, _ = prefill_and_generate(
                model, tokenizer, inputs, max_new_tokens=100, device=device, device_map=device_map
            )
        elif rate == 0:
            # rate=0: 完全不重算，只用 KV cache
            generated_tokens, _, _ = load_kv_and_generate(
                model, tokenizer, past_key_values, passages, load_path, example_id,
                max_new_tokens=100, revert_rope=revert_rope,
                reprocess_method='DraftModel', rate=0.001,  # 用很小的 rate 模拟不重算
                draft_model=draft_model,
                draft_layer_selection='entropy',
                preprocess=preprocess,
                chunk_ids=chunk_ids, device=device, device_map=device_map,
                original_kv_path=original_kv_path
            )
        else:
            generated_tokens, _, _ = load_kv_and_generate(
                model, tokenizer, past_key_values, passages, load_path, example_id,
                max_new_tokens=100, revert_rope=revert_rope,
                reprocess_method='DraftModel', rate=rate,
                draft_model=draft_model,
                draft_layer_selection='entropy',
                preprocess=preprocess,
                chunk_ids=chunk_ids, device=device, device_map=device_map,
                original_kv_path=original_kv_path
            )

        # generated_tokens 是 list，需要转成 tensor，并去掉最后一个 EOS token
        answer = tokenizer.decode(torch.tensor(generated_tokens[:-1]), skip_special_tokens=True).strip()
        if not answer:
            answer = "[EMPTY]"

        return {
            'rate': rate,
            'answer': answer,
            'error': None
        }
    except Exception as e:
        import traceback
        return {
            'rate': rate,
            'answer': None,
            'error': str(e) + '\n' + traceback.format_exc()
        }


def main():
    device = "cuda:0"  # 会被 CUDA_VISIBLE_DEVICES 映射

    # 路径配置
    model_path = "/mnt/data/models/Qwen2.5-7B-Instruct"
    draft_model_path = "/mnt/data/models/Qwen2.5-3B-Instruct"
    data_path = "/mnt/data/wjh/FusionRAG/data/result_reflect.json"  # musique 数据集
    bge_model_path = "/mnt/data/models/bge-m3-FP16"

    # 缓存路径 (musique)
    model_cache_root = "/mnt/data/reflect/Qwen2.5-7B-Instruct/musique"
    save_path = os.path.join(model_cache_root, 'kv_cache')
    preprocess_save_path = os.path.join(model_cache_root, 'preprocess_kv_cache_global')

    # 输出路径
    output_dir = "/mnt/data/wjh/FusionRAG/optimal_rate_search"
    os.makedirs(output_dir, exist_ok=True)

    # 要尝试的 rates: 0, 0.05, 0.10, ..., 1.0
    rates_to_try = [round(r * 0.05, 2) for r in range(21)]  # [0.0, 0.05, 0.10, ..., 1.0]

    # DeepSeek API 配置 (用于判断答案正确性)
    openai_client = OpenAI(
        api_key="sk-519d391217894b6e91e7c2ebf2a9f4df",
        base_url="https://api.deepseek.com/v1"
    )
    openai_model = "deepseek-chat"

    # 打印配置信息
    print("="*80)
    print("Configuration")
    print("="*80)
    print(f"Main Model: {model_path}")
    print(f"Draft Model: {draft_model_path}")
    print(f"Data Path: {data_path}")
    print(f"Output Dir: {output_dir}")
    print(f"Reprocess Method: DraftModel")
    print(f"Draft Layer Selection: entropy")
    print(f"Preprocess: True")
    print(f"Revert RoPE: True")
    print(f"Rates to try: {rates_to_try}")
    print(f"Judgment: DeepSeek API ({openai_model})")
    print("="*80)
    print()

    # 限制样本数 (设为 None 测试全部)
    max_main_questions = None  # 设为 None 测试全部问题

    print("="*80)
    print("Step 1: 加载模型")
    print("="*80)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config._attn_implementation = "sdpa"

    print("加载主模型 (7B)...")
    model, device_map = load_model('qwen', model_path, config, device, use_multi_gpu=False)
    model.eval()

    print("加载 Draft 模型 (3B)...")
    draft_config = AutoConfig.from_pretrained(draft_model_path, trust_remote_code=True)
    draft_config._attn_implementation = "sdpa"
    draft_model, _ = load_model('qwen', draft_model_path, draft_config, device, use_multi_gpu=False)
    draft_model.eval()

    print("="*80)
    print("Step 2: 准备数据")
    print("="*80)

    questions_data, system_tensor, context_rank, corpus_lens = prepare_reflect_data(
        data_path, tokenizer, bge_model_path, 'qwen2', topk=10, max_main_questions=max_main_questions,
        preprocess=True, preprocess_scope=PreprocessScope.GLOBAL
    )

    system_len = system_tensor.shape[0]

    # 初始化 KV cache (需要 passage_len 参数来初始化 importance_cache)
    max_cache_len = 32768
    past_key_values = StaticCache(
        config=config,
        max_batch_size=1,
        max_cache_len=max_cache_len,
        device=device,
        dtype=config.torch_dtype,
        passage_len=max_cache_len
    )

    print("="*80)
    print("Step 3: 收集数据")
    print("="*80)

    all_results = []
    total_questions = sum(len(q['sub_questions']) for q in questions_data)

    with tqdm(total=total_questions, desc="Processing") as pbar:
        for example_id, q_data in enumerate(questions_data):
            doc_tensors = q_data['doc_tensors']

            for sub_q_idx, sub_q_info in enumerate(q_data['sub_questions']):
                ground_truth = sub_q_info['answer']
                query = sub_q_info['query']
                chunk_ids = sub_q_info['chunk_ids']

                # 构建输入
                sub_q_doc_tensors = [doc_tensors[cid - 1] for cid in chunk_ids]

                # 构建 question tensor
                question_text = f"<|im_end|>\n<|im_start|>user\nQuestion: {query}<|im_end|>\n<|im_start|>assistant\nAnswer: "
                question_tokens = tokenizer.encode(question_text, add_special_tokens=False)
                question_tensor = torch.tensor(question_tokens, dtype=torch.long)

                passages = [system_tensor] + sub_q_doc_tensors + [question_tensor]
                passages_len = [p.shape[0] for p in passages]
                kv_chunk_ids = [0] + chunk_ids

                question_id = f"Q{example_id+1}_Sub{sub_q_idx+1}"

                # Step 3.1: 计算 Draft Model Attention
                query_start = sum(passages_len[:-1])
                full_input = torch.cat(passages).unsqueeze(0).to(device)

                try:
                    draft_attention = compute_draft_model_attention(
                        draft_model, full_input, query_start, device
                    )
                except Exception as e:
                    print(f"\n  跳过 {question_id}: 计算 attention 失败 - {e}")
                    pbar.update(1)
                    continue

                # Step 3.2: 提取 Attention 特征
                attention_data = extract_attention_features(
                    draft_attention, query_start, system_len, passages_len,
                    entropy_top_k=4, device=device
                )

                if attention_data is None:
                    pbar.update(1)
                    continue

                # Step 3.3: 从低到高测试 rates，找到第一个正确的就停止 (早停优化)
                print(f"\n{'='*60}")
                print(f"[{question_id}] Query: {query}")
                print(f"[{question_id}] Ground Truth: {ground_truth}")
                print(f"{'='*60}")

                rate_results = []
                min_correct_rate = None

                for rate in rates_to_try:
                    result = test_single_rate(
                        model, tokenizer, past_key_values, passages, passages_len,
                        preprocess_save_path, example_id, kv_chunk_ids, rate,
                        draft_model=draft_model, revert_rope=True,
                        device=device, device_map=device_map,
                        original_kv_path=save_path,  # chunk1 用原始 KV cache
                        preprocess=True
                    )

                    if result['error']:
                        print(f"  rate={rate:.2f}: ERROR - {result['error'][:100]}")
                        result['correct'] = False
                        rate_results.append(result)
                        continue

                    answer = result['answer']

                    # 使用 DeepSeek API 判断答案正确性
                    is_correct, reason = judge_answer_with_openai(
                        openai_client, openai_model,
                        query, answer, ground_truth
                    )
                    result['correct'] = is_correct
                    result['judgment_reason'] = reason
                    rate_results.append(result)

                    # 输出当前 rate 的结果 (完整答案，不截断)
                    status = "✓ CORRECT" if is_correct else "✗ wrong"
                    print(f"  rate={rate:.2f}: {status}")
                    print(f"    Answer: {answer}")

                    # 早停：如果答对了，记录 min_correct_rate 并停止
                    if is_correct:
                        min_correct_rate = rate
                        print(f"  >>> 找到 min_correct_rate = {rate:.2f}, 停止搜索")
                        break

                # 如果所有 rate 都试完还没答对
                if min_correct_rate is None:
                    print(f"  >>> 所有 rate 都答错，标记为 all_wrong")

                # 保存结果 (早停后只有部分 rate_results)
                question_result = {
                    'question_id': question_id,
                    'example_id': example_id,
                    'sub_q_idx': sub_q_idx,
                    'query': query,
                    'ground_truth': ground_truth,
                    'attention_features': attention_data['features'],
                    'attention_distribution': attention_data['attention_distribution'],
                    'layer_entropy': attention_data['layer_entropy'],
                    'rate_results': rate_results,
                    'min_correct_rate': min_correct_rate,
                    # 早停后只知道第一个正确的 rate，不知道更高的 rate 是否也正确
                    'all_correct_rates': [min_correct_rate] if min_correct_rate is not None else [],
                }
                all_results.append(question_result)

                # 更新进度条
                status = f"min_rate={min_correct_rate}" if min_correct_rate else "all_wrong"
                pbar.set_postfix_str(f"{question_id}: {status}")
                pbar.update(1)

                # 定期保存 checkpoint
                if len(all_results) % 20 == 0:
                    checkpoint_path = os.path.join(output_dir, 'checkpoint.pkl')
                    with open(checkpoint_path, 'wb') as f:
                        pickle.dump(all_results, f)

                torch.cuda.empty_cache()

    # 保存最终结果
    print("\n" + "="*80)
    print("Step 4: 保存结果")
    print("="*80)

    # 保存完整数据 (pickle)
    output_path = os.path.join(output_dir, 'full_results.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"完整数据已保存到: {output_path}")

    # 保存摘要 (JSON)
    summary = []
    for r in all_results:
        summary.append({
            'question_id': r['question_id'],
            'query': r['query'][:100],
            'ground_truth': r['ground_truth'],
            'min_correct_rate': r['min_correct_rate'],
            'features': r['attention_features'],
        })

    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"摘要已保存到: {summary_path}")

    # 打印统计
    print("\n" + "="*80)
    print("统计信息")
    print("="*80)

    min_rates = [r['min_correct_rate'] for r in all_results if r['min_correct_rate'] is not None]
    failed = [r for r in all_results if r['min_correct_rate'] is None]

    print(f"总问题数: {len(all_results)}")
    print(f"有正确答案的问题数: {len(min_rates)}")
    print(f"所有 rate 都答错的问题数: {len(failed)}")

    if min_rates:
        print(f"\n最小正确 rate 分布:")
        print(f"  平均值: {np.mean(min_rates):.3f}")
        print(f"  中位数: {np.median(min_rates):.3f}")
        print(f"  最小值: {np.min(min_rates):.3f}")
        print(f"  最大值: {np.max(min_rates):.3f}")
        print(f"  标准差: {np.std(min_rates):.3f}")

        # 分布直方图
        print(f"\n最小正确 rate 分布直方图:")
        bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
        hist, _ = np.histogram(min_rates, bins=bins)
        for i in range(len(bins)-1):
            bar = '█' * (hist[i] * 50 // max(hist) if max(hist) > 0 else 0)
            print(f"  [{bins[i]:.1f}, {bins[i+1]:.1f}): {hist[i]:3d} {bar}")

    return all_results


if __name__ == '__main__':
    main()
