#!/usr/bin/env python3
"""
改进的 DraftModel token 选择算法 V2
核心思路：用 Query-Token 语义相似度来修正 3B 的 attention 偏差

问题根源：3B 模型的 attention 不够准确，无法正确识别答案相关 token
解决方案：
1. 使用 query 的 embedding 与 document token embedding 的相似度作为先验
2. 结合 3B attention 和语义相似度进行选择
3. 对不同层的 attention 进行加权组合
"""
import sys
sys.path.insert(0, '/mnt/data/wjh/FusionRAG')

import torch
import torch.nn.functional as F
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained('/mnt/data/models/Qwen2.5-7B-Instruct', trust_remote_code=True)

def compute_query_doc_similarity(model, input_ids, query_start, doc_start, doc_end, device):
    """
    计算 query tokens 和 document tokens 的语义相似度

    使用模型的 hidden states 来获取 token embeddings
    """
    model.eval()

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            output_hidden_states=True,
            use_cache=False
        )

        # 使用中间层的 hidden states (比最后一层更通用)
        num_layers = len(outputs.hidden_states)
        mid_layer = num_layers // 2
        hidden_states = outputs.hidden_states[mid_layer][0]  # (seq_len, hidden_dim)

        # Query tokens embedding (平均)
        query_emb = hidden_states[query_start:].mean(dim=0)  # (hidden_dim,)

        # Document tokens embedding
        doc_emb = hidden_states[doc_start:doc_end]  # (doc_len, hidden_dim)

        # 计算余弦相似度
        query_emb = F.normalize(query_emb.unsqueeze(0), dim=-1)  # (1, hidden_dim)
        doc_emb = F.normalize(doc_emb, dim=-1)  # (doc_len, hidden_dim)
        similarity = torch.mm(doc_emb, query_emb.T).squeeze()  # (doc_len,)

        # 转换为正值 [0, 1]
        similarity = (similarity + 1) / 2

    return similarity.cpu()


def compute_attention_with_layers(model, input_ids, query_start, doc_start, doc_end, device,
                                   layer_strategy='last4'):
    """
    计算 attention 分数，支持不同的层选择策略

    layer_strategy:
        - 'last4': 最后 4 层 (原始方法)
        - 'all_weighted': 所有层加权 (后面层权重更大)
        - 'entropy_select': 熵选层
        - 'mid4': 中间 4 层
    """
    model.eval()
    num_layers = model.config.num_hidden_layers

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            output_attentions=True,
            use_cache=False
        )

        attentions = outputs.attentions

        if layer_strategy == 'last4':
            layers = list(range(num_layers - 4, num_layers))
        elif layer_strategy == 'mid4':
            mid = num_layers // 2
            layers = list(range(mid - 2, mid + 2))
        elif layer_strategy == 'all_weighted':
            layers = list(range(num_layers))
        else:
            layers = list(range(num_layers - 4, num_layers))

        layer_attentions = {}
        for layer_idx in layers:
            attn = attentions[layer_idx][0, :, query_start:, doc_start:doc_end]
            attn_avg = attn.mean(dim=(0, 1))  # (doc_len,)
            layer_attentions[layer_idx] = attn_avg.cpu()

        # 聚合
        if layer_strategy == 'all_weighted':
            # 后面层权重更大
            weights = torch.tensor([1.0 + i * 0.5 for i in range(num_layers)])
            weights = weights / weights.sum()
            combined = torch.zeros(doc_end - doc_start)
            for i, layer_idx in enumerate(layers):
                combined += weights[i] * layer_attentions[layer_idx]
        elif layer_strategy == 'entropy_select':
            # 熵选层
            layer_entropies = {}
            for layer_idx, attn in layer_attentions.items():
                attn_norm = attn / (attn.sum() + 1e-10)
                entropy = -torch.sum(attn_norm * torch.log(attn_norm + 1e-10))
                layer_entropies[layer_idx] = entropy.item()
            sorted_layers = sorted(layer_entropies.items(), key=lambda x: x[1])
            selected = [l[0] for l in sorted_layers[:4]]
            combined = torch.stack([layer_attentions[l] for l in selected]).mean(dim=0)
        else:
            combined = torch.stack([layer_attentions[l] for l in layers]).mean(dim=0)

    return combined


def hybrid_token_selection(attention_scores, similarity_scores, rate, doc_len,
                           attention_weight=0.5, use_rerank=True):
    """
    混合选择策略：结合 attention 和语义相似度

    Args:
        attention_scores: (doc_len,) 来自 draft model
        similarity_scores: (doc_len,) query-token 相似度
        rate: 选择比例
        attention_weight: attention 的权重 (0-1)
        use_rerank: 是否对 attention top-k 用相似度重排序

    Returns:
        selected_positions: 选中的位置列表
    """
    num_select = max(1, int(doc_len * rate))

    # 归一化
    attn = attention_scores / (attention_scores.sum() + 1e-10)
    sim = similarity_scores / (similarity_scores.sum() + 1e-10)

    if use_rerank:
        # 策略1: 先用 attention 选 2x 候选，再用相似度重排序
        candidate_count = min(num_select * 2, doc_len)
        _, attn_top = torch.topk(attn, candidate_count)

        # 在候选中按相似度排序
        candidate_sim = sim[attn_top]
        _, rerank_idx = torch.topk(candidate_sim, num_select)
        selected = attn_top[rerank_idx]
    else:
        # 策略2: 加权组合
        combined = attention_weight * attn + (1 - attention_weight) * sim
        _, selected = torch.topk(combined, num_select)

    return selected.sort()[0].tolist()


def adaptive_layer_selection(model, input_ids, query_start, doc_start, doc_end, device):
    """
    自适应层选择：根据 query 在不同层的 attention 分布特征选择最佳层

    思路：好的层应该有明确的 attention 峰值，而不是均匀分布
    """
    model.eval()
    num_layers = model.config.num_hidden_layers

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            output_attentions=True,
            use_cache=False
        )

        layer_quality = {}
        layer_attentions = {}

        for layer_idx in range(num_layers):
            attn = outputs.attentions[layer_idx][0, :, query_start:, doc_start:doc_end]
            attn_avg = attn.mean(dim=(0, 1))
            layer_attentions[layer_idx] = attn_avg.cpu()

            # 计算层质量指标：峰值显著性
            attn_norm = attn_avg / (attn_avg.sum() + 1e-10)
            # Gini 系数 (越高越集中)
            sorted_attn = torch.sort(attn_norm)[0]
            n = len(sorted_attn)
            cumsum = torch.cumsum(sorted_attn, dim=0)
            gini = (2 * torch.sum((torch.arange(1, n+1, device=sorted_attn.device).float() * sorted_attn)) / (n * sorted_attn.sum())) - (n + 1) / n
            # Top-10% 占比
            top_k = max(1, int(n * 0.1))
            top_ratio = attn_norm.topk(top_k)[0].sum().item()

            layer_quality[layer_idx] = {
                'gini': gini.item() if not torch.isnan(gini) else 0,
                'top_ratio': top_ratio,
                'combined': gini.item() * top_ratio if not torch.isnan(gini) else 0
            }

    # 选择 combined 最高的 4 层
    sorted_layers = sorted(layer_quality.items(), key=lambda x: x[1]['combined'], reverse=True)
    selected_layers = [l[0] for l in sorted_layers[:4]]

    return selected_layers, layer_quality, layer_attentions


def test_methods(case_data, model, tokenizer, device, rate=0.2):
    """测试各种改进方法"""
    input_ids = case_data['input_ids'].to(device)
    system_len = case_data['system_len']
    doc_len = case_data['doc_len']
    query_start = case_data['query_start']
    doc_start = system_len
    doc_end = system_len + doc_len

    results = {}

    # 1. 原始方法 (最后4层 + 熵选)
    attn_original = compute_attention_with_layers(
        model, input_ids, query_start, doc_start, doc_end, device,
        layer_strategy='entropy_select'
    )
    num_select = max(1, int(doc_len * rate))
    _, indices = torch.topk(attn_original, num_select)
    results['1_original'] = (indices + doc_start).sort()[0].tolist()

    # 2. 中间层策略
    attn_mid = compute_attention_with_layers(
        model, input_ids, query_start, doc_start, doc_end, device,
        layer_strategy='mid4'
    )
    _, indices = torch.topk(attn_mid, num_select)
    results['2_mid_layers'] = (indices + doc_start).sort()[0].tolist()

    # 3. 自适应层选择
    selected_layers, layer_quality, layer_attns = adaptive_layer_selection(
        model, input_ids, query_start, doc_start, doc_end, device
    )
    combined_attn = torch.stack([layer_attns[l] for l in selected_layers]).mean(dim=0)
    _, indices = torch.topk(combined_attn, num_select)
    results['3_adaptive_layers'] = (indices + doc_start).sort()[0].tolist()
    results['3_adaptive_layers_info'] = selected_layers

    # 4. Query-Token 相似度辅助
    similarity = compute_query_doc_similarity(
        model, input_ids, query_start, doc_start, doc_end, device
    )
    # 4a. 相似度重排序
    selected = hybrid_token_selection(attn_original, similarity, rate, doc_len,
                                      use_rerank=True)
    results['4a_sim_rerank'] = [s + doc_start for s in selected]

    # 4b. 加权组合 (0.7 attention + 0.3 similarity)
    selected = hybrid_token_selection(attn_original, similarity, rate, doc_len,
                                      attention_weight=0.7, use_rerank=False)
    results['4b_hybrid_0.7'] = [s + doc_start for s in selected]

    # 4c. 加权组合 (0.5 attention + 0.5 similarity)
    selected = hybrid_token_selection(attn_original, similarity, rate, doc_len,
                                      attention_weight=0.5, use_rerank=False)
    results['4c_hybrid_0.5'] = [s + doc_start for s in selected]

    return results, input_ids[0].cpu(), attn_original, similarity


def analyze_selection_quality(selected_positions, input_ids, answer_text, tokenizer, doc_start, doc_end):
    """分析选择质量"""
    answer_keywords = [w.lower() for w in answer_text.split() if len(w) > 3]
    answer_positions = []
    for i in range(doc_start, min(doc_end, len(input_ids))):
        tok_text = tokenizer.decode([input_ids[i]]).lower().strip()
        for word in answer_keywords:
            if word in tok_text or tok_text in word:
                answer_positions.append(i)
                break

    selected_set = set(selected_positions)
    answer_covered = len(set(answer_positions) & selected_set)

    return {
        'answer_coverage': answer_covered / len(answer_positions) if answer_positions else 0,
        'total_selected': len(selected_positions),
        'answer_positions': len(answer_positions),
        'answer_covered': answer_covered
    }


def main():
    device = "cuda:0"

    # 加载数据
    with open('result_reflect.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open('diff_cases_7b_vs_3b.json', 'r', encoding='utf-8') as f:
        diff_cases = json.load(f)

    config_path = "./config/dataset2prompt_few-shot.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    system_prompt = config["system_prompt"]["Qwen3"]["2wikimqa"]

    print("加载 3B 模型...")
    model = AutoModelForCausalLM.from_pretrained(
        '/mnt/data/models/Qwen2.5-3B-Instruct',
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="eager"
    )

    from analyze_token_selection_diff import prepare_case_input

    all_results = {}
    method_names = ['1_original', '2_mid_layers', '3_adaptive_layers',
                    '4a_sim_rerank', '4b_hybrid_0.7', '4c_hybrid_0.5']
    for m in method_names:
        all_results[m] = {'coverage': []}

    for i, case in enumerate(diff_cases[:5]):
        print(f"\n{'='*60}")
        print(f"案例 {i+1}: {case['sub_question'][:50]}...")
        print(f"{'='*60}")

        case_data = prepare_case_input(case, data, tokenizer, system_prompt)
        if case_data is None:
            continue

        results, input_ids, attn, sim = test_methods(case_data, model, tokenizer, device, rate=0.2)

        print(f"文档长度: {case_data['doc_len']} tokens")

        # 显示自适应层选择结果
        if '3_adaptive_layers_info' in results:
            print(f"自适应选择的层: {results['3_adaptive_layers_info']}")

        for method in method_names:
            if method in results:
                positions = results[method]
                quality = analyze_selection_quality(
                    positions, input_ids, case['ground_truth'], tokenizer,
                    case_data['system_len'], case_data['system_len'] + case_data['doc_len']
                )
                all_results[method]['coverage'].append(quality['answer_coverage'])
                print(f"  {method:20s}: 答案覆盖 {quality['answer_covered']:2d}/{quality['answer_positions']} ({quality['answer_coverage']*100:5.1f}%)")

    # 汇总
    print("\n" + "="*70)
    print("汇总统计 (rate=0.2)")
    print("="*70)
    print(f"{'方法':25s} | {'平均答案覆盖率':15s}")
    print("-"*45)
    for method in method_names:
        avg_coverage = np.mean(all_results[method]['coverage'])
        improvement = avg_coverage - np.mean(all_results['1_original']['coverage'])
        sign = '+' if improvement > 0 else ''
        print(f"{method:25s} | {avg_coverage*100:14.1f}% ({sign}{improvement*100:.1f}%)")

if __name__ == '__main__':
    main()
