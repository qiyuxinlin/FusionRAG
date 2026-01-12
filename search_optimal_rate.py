#!/usr/bin/env python3
"""
暴力搜索每个问题的最优重算比例，并收集 attention 特征用于建模

思路：
1. 对每个问题，尝试不同的 rate (0.1, 0.2, ..., 1.0)
2. 找到能答对的最小 rate (optimal_rate)
3. 收集该问题的 3B 模型 attention 特征
4. 最后用线性回归拟合: optimal_rate = f(attention_features)
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

sys.path.insert(0, '/mnt/data/wjh/FusionRAG')

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from ktransformers.util.utils import (
    load_kv_and_generate,
    prefill_and_generate,
    compute_draft_model_attention,
    entropy_layer_selection,
    smart_query_selection,
)
from ktransformers.models.custom_cache import StaticCache
from test_fusionrag_reflect import prepare_reflect_data, load_system_prompt, PreprocessScope


def compute_attention_features(draft_attention, query_start, system_len, passages_len,
                                entropy_top_k=4, device="cuda:0"):
    """
    从 draft model attention 中提取特征

    Returns:
        dict: 包含各种 attention 特征
    """
    # 文本块1长度
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
    aggregated_attn = torch.stack(layer_attentions).mean(dim=0).cpu()

    features = {}

    # Feature 1: Peak strength
    features['peak_strength'] = aggregated_attn.max().item()

    # Feature 2: Top-k concentration
    sorted_attn, _ = torch.sort(aggregated_attn, descending=True)
    total = sorted_attn.sum()

    features['top5_concentration'] = sorted_attn[:5].sum().item() / total.item() if total > 0 else 0
    features['top10_concentration'] = sorted_attn[:10].sum().item() / total.item() if total > 0 else 0
    features['top20_concentration'] = sorted_attn[:20].sum().item() / total.item() if total > 0 else 0

    # Feature 3: Coverage ratios
    cumsum = torch.cumsum(sorted_attn, dim=0)
    for coverage in [0.5, 0.7, 0.9]:
        coverage_idx = (cumsum >= coverage * total).nonzero(as_tuple=True)[0]
        if len(coverage_idx) > 0:
            tokens_needed = coverage_idx[0].item() + 1
        else:
            tokens_needed = len(aggregated_attn)
        features[f'coverage_{int(coverage*100)}_ratio'] = tokens_needed / len(aggregated_attn)

    # Feature 4: Normalized entropy
    p = aggregated_attn / (aggregated_attn.sum() + 1e-10)
    p = torch.clamp(p, min=1e-10)
    entropy = -(p * torch.log(p)).sum()
    max_entropy = np.log(len(aggregated_attn))
    features['normalized_entropy'] = (entropy / max_entropy).item()

    # Feature 5: Gini coefficient
    sorted_p = torch.sort(p)[0]
    n = len(sorted_p)
    cumsum_p = torch.cumsum(sorted_p, dim=0)
    gini = (2 * torch.sum((torch.arange(1, n+1).float() * sorted_p)) / (n * sorted_p.sum()) - (n + 1) / n).item()
    features['gini'] = gini

    # Feature 6: Variance and std
    features['attention_std'] = aggregated_attn.std().item()
    features['attention_var'] = aggregated_attn.var().item()

    # Feature 7: Layer consistency
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
    features['layer_consistency'] = np.mean(consistency_scores) if consistency_scores else 0.5

    # Feature 8: Document length (可能影响最优 rate)
    features['doc_len'] = doc_len
    features['log_doc_len'] = np.log(doc_len + 1)

    return features


def search_optimal_rate(model, tokenizer, past_key_values, passages, passages_len,
                        load_path, example_id, chunk_ids, ground_truth,
                        rates_to_try, revert_rope=True, device="cuda:0", device_map=None):
    """
    搜索能答对问题的最小 rate

    Returns:
        optimal_rate: 能答对的最小 rate (如果都答不对则返回 1.0)
        results: 每个 rate 的结果
    """
    from ktransformers.util.utils import compute_f1, _exact_match_score

    results = []
    optimal_rate = None

    for rate in rates_to_try:
        if rate == 1.0:
            # 完全重算
            inputs = torch.cat(passages).to(device).unsqueeze(0)
            generated_tokens, _, _ = prefill_and_generate(
                model, tokenizer, inputs, max_new_tokens=50, device=device, device_map=device_map
            )
        else:
            # 使用 KV cache + 部分重算
            generated_tokens, _, _ = load_kv_and_generate(
                model, tokenizer, past_key_values, passages, load_path, example_id,
                max_new_tokens=50, revert_rope=revert_rope,
                reprocess_method='FusionRAG', rate=rate,
                chunk_ids=chunk_ids, device=device, device_map=device_map
            )

        answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # 判断是否正确 (使用 F1 和 EM)
        f1 = compute_f1(answer, ground_truth)
        em = _exact_match_score(answer, ground_truth)

        # 简单判断: F1 > 0.5 或 EM = 1 认为正确
        is_correct = f1 > 0.5 or em == 1

        results.append({
            'rate': rate,
            'answer': answer,
            'f1': f1,
            'em': em,
            'correct': is_correct
        })

        if is_correct and optimal_rate is None:
            optimal_rate = rate

    # 如果都答不对，返回 1.0
    if optimal_rate is None:
        optimal_rate = 1.0

    return optimal_rate, results


def main():
    device = "cuda:0"

    # 路径配置
    model_path = "/mnt/data/models/Qwen2.5-7B-Instruct"
    draft_model_path = "/mnt/data/models/Qwen2.5-3B-Instruct"
    data_path = "/mnt/data/wjh/FusionRAG/data/2wikimqa_reflect.json"
    bge_model_path = "/mnt/data/models/bge-m3-FP16"

    # 缓存路径
    model_cache_root = "/mnt/data/reflect/Qwen2.5-7B-Instruct/2wikimqa"
    save_path = os.path.join(model_cache_root, 'kv_cache')
    preprocess_save_path = os.path.join(model_cache_root, 'preprocess_kv_cache_global')

    # 要尝试的 rates
    rates_to_try = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8, 1.0]

    # 限制样本数 (完整搜索太慢)
    max_samples = 50

    print("="*80)
    print("Step 1: 加载模型")
    print("="*80)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config._attn_implementation = "sdpa"

    print("加载主模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    print("加载 Draft 模型...")
    draft_config = AutoConfig.from_pretrained(draft_model_path, trust_remote_code=True)
    draft_config._attn_implementation = "sdpa"
    draft_model = AutoModelForCausalLM.from_pretrained(
        draft_model_path,
        config=draft_config,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    draft_model.eval()

    print("="*80)
    print("Step 2: 准备数据")
    print("="*80)

    questions_data, system_tensor, context_rank, corpus_lens = prepare_reflect_data(
        data_path, tokenizer, bge_model_path, 'qwen2', topk=10, max_samples=max_samples,
        preprocess=True, preprocess_scope=PreprocessScope.GLOBAL
    )

    system_len = system_tensor.shape[0]

    # 初始化 KV cache
    past_key_values = StaticCache(
        config=config,
        max_batch_size=1,
        max_cache_len=config.max_position_embeddings,
        device=device,
        dtype=torch.bfloat16
    )

    print("="*80)
    print("Step 3: 搜索最优 rate 并收集特征")
    print("="*80)

    all_data = []

    for example_id, q_data in enumerate(tqdm(questions_data, desc="Processing questions")):
        doc_tensors = q_data['doc_tensors']

        for sub_q_idx, sub_q_info in enumerate(q_data['sub_questions']):
            ground_truth = sub_q_info['answer']
            query = sub_q_info['query']
            chunk_ids = sub_q_info['chunk_ids']

            # 构建输入
            sub_q_doc_tensors = [doc_tensors[cid - 1] for cid in chunk_ids]
            question_tensor = sub_q_info['question_tensor']

            passages = [system_tensor] + sub_q_doc_tensors + [question_tensor]
            passages_len = [p.shape[0] for p in passages]
            kv_chunk_ids = [0] + chunk_ids

            # Step 3.1: 计算 Draft Model Attention
            query_start = sum(passages_len[:-1])
            full_input = torch.cat(passages).unsqueeze(0).to(device)

            try:
                draft_attention = compute_draft_model_attention(
                    draft_model, full_input, query_start, device
                )
            except Exception as e:
                print(f"  跳过 Q{example_id+1}-Sub{sub_q_idx+1}: {e}")
                continue

            # Step 3.2: 提取 Attention 特征
            features = compute_attention_features(
                draft_attention, query_start, system_len, passages_len,
                entropy_top_k=4, device=device
            )

            if features is None:
                continue

            # Step 3.3: 搜索最优 rate
            # 重置 KV cache
            for layer_idx in range(len(past_key_values.key_cache)):
                past_key_values.past_tokens[layer_idx] = 0

            optimal_rate, rate_results = search_optimal_rate(
                model, tokenizer, past_key_values, passages, passages_len,
                preprocess_save_path, example_id, kv_chunk_ids, ground_truth,
                rates_to_try, revert_rope=True, device=device
            )

            # 收集数据
            data_point = {
                'example_id': example_id,
                'sub_q_idx': sub_q_idx,
                'query': query[:50],
                'ground_truth': ground_truth,
                'optimal_rate': optimal_rate,
                **features
            }
            all_data.append(data_point)

            # 打印进度
            if len(all_data) % 10 == 0:
                print(f"\n  已处理 {len(all_data)} 个样本, 当前最优 rate: {optimal_rate:.2f}")

            torch.cuda.empty_cache()

    print("="*80)
    print("Step 4: 拟合线性模型")
    print("="*80)

    df = pd.DataFrame(all_data)
    print(f"\n总样本数: {len(df)}")
    print(f"最优 rate 分布:\n{df['optimal_rate'].describe()}")

    # 保存原始数据
    df.to_csv('/mnt/data/wjh/FusionRAG/optimal_rate_data.csv', index=False)
    print(f"\n数据已保存到 optimal_rate_data.csv")

    # 准备特征矩阵
    feature_cols = [
        'peak_strength', 'top5_concentration', 'top10_concentration', 'top20_concentration',
        'coverage_50_ratio', 'coverage_70_ratio', 'coverage_90_ratio',
        'normalized_entropy', 'gini', 'attention_std', 'layer_consistency', 'log_doc_len'
    ]

    X = df[feature_cols].values
    y = df['optimal_rate'].values

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 拟合多种模型
    models = {
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.01),
    }

    print("\n模型拟合结果:")
    print("-" * 60)

    best_model = None
    best_score = -np.inf

    for name, reg_model in models.items():
        # 交叉验证
        scores = cross_val_score(reg_model, X_scaled, y, cv=5, scoring='r2')
        mean_score = scores.mean()

        print(f"{name}: R² = {mean_score:.4f} (+/- {scores.std():.4f})")

        if mean_score > best_score:
            best_score = mean_score
            best_model = reg_model

    # 用最佳模型拟合全部数据
    best_model.fit(X_scaled, y)

    print("\n" + "="*80)
    print("最佳模型系数 (标准化后):")
    print("="*80)

    if hasattr(best_model, 'coef_'):
        coef_df = pd.DataFrame({
            'feature': feature_cols,
            'coefficient': best_model.coef_
        }).sort_values('coefficient', key=abs, ascending=False)

        print(coef_df.to_string(index=False))
        print(f"\n截距 (intercept): {best_model.intercept_:.4f}")

    # 生成公式
    print("\n" + "="*80)
    print("拟合公式 (需要先标准化特征):")
    print("="*80)

    formula_parts = []
    for feat, coef in zip(feature_cols, best_model.coef_):
        if abs(coef) > 0.001:
            sign = "+" if coef > 0 else ""
            formula_parts.append(f"{sign}{coef:.4f} * {feat}")

    formula = f"optimal_rate = {best_model.intercept_:.4f} " + " ".join(formula_parts)
    print(formula)

    # 保存模型参数
    model_params = {
        'feature_cols': feature_cols,
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_std': scaler.scale_.tolist(),
        'coefficients': best_model.coef_.tolist(),
        'intercept': float(best_model.intercept_),
        'r2_score': float(best_score),
    }

    with open('/mnt/data/wjh/FusionRAG/optimal_rate_model.json', 'w') as f:
        json.dump(model_params, f, indent=2)

    print(f"\n模型参数已保存到 optimal_rate_model.json")

    return df, best_model, scaler


if __name__ == '__main__':
    main()
