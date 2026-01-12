#!/usr/bin/env python3
"""
实验：测试不同的 entropy_top_k 值对 3B 与 7B 选择重叠率的影响
配合 lower_threshold=0.3
"""
import sys
sys.path.insert(0, '/mnt/data/wjh/FusionRAG')

import torch
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from ktransformers.util.utils import (
    compute_draft_model_attention,
    entropy_layer_selection,
    find_connected_components,
    smart_query_selection
)


def get_selection_with_params(model, input_ids, system_len, doc_len, query_start, rate, device,
                               entropy_top_k=4, threshold_factor=0.5):
    """使用指定参数获取 token 选择"""
    draft_attention = compute_draft_model_attention(model, input_ids, query_start, device)

    doc_start = system_len
    doc_end = system_len + doc_len

    layer_attention_dict = {}
    for layer_idx, layer_attn in draft_attention.items():
        query_to_doc = layer_attn[:, :, doc_start:doc_end]
        doc_attention_avg = query_to_doc.mean(axis=(0, 1))
        layer_attention_dict[layer_idx] = torch.tensor(doc_attention_avg, device=device)

    # 熵选层
    active_layers, layer_entropy = entropy_layer_selection(
        layer_attention_dict, top_k=entropy_top_k, return_entropy=True
    )

    layer_attentions = [layer_attention_dict[idx] for idx in active_layers]
    multi_layer_attn = torch.stack(layer_attentions).mean(dim=0)

    # smart_query_selection
    selected = smart_query_selection(
        attention_scores=multi_layer_attn,
        doc_len=doc_len,
        target_ratio=rate,
        system_len=system_len,
        device=device,
        threshold_factor=threshold_factor
    )

    return set(selected), active_layers


def compute_overlap(sel_a, sel_b):
    """计算 Jaccard 重叠率"""
    common = sel_a & sel_b
    union = sel_a | sel_b
    return len(common) / len(union) if union else 0


def find_case_data(main_q, sub_q, data):
    for item in data:
        if item.get('question', '') == main_q:
            for sub in item.get('intermediate_context', []):
                query = sub.get('query', '')
                if query.startswith("Intermediate query"):
                    colon_pos = query.find(":")
                    if colon_pos != -1:
                        query = query[colon_pos + 1:].strip()
                if query == sub_q:
                    return item, sub
    return None, None


def prepare_case_input(case, data, tokenizer, system_prompt):
    main_item, sub_item = find_case_data(case['main_question'], case['sub_question'], data)
    if main_item is None or sub_item is None:
        return None

    docs = sub_item.get('retrieve docs', [])
    if not docs:
        return None

    doc_text = "\n\n".join(docs[:10])
    sub_q = case['sub_question']

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{doc_text}\n\nQuestion: {sub_q}"}
    ]

    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    system_text = tokenizer.apply_chat_template([{"role": "system", "content": system_prompt}],
                                                 tokenize=False, add_generation_prompt=False)
    system_len = len(tokenizer.encode(system_text))

    total_len = input_ids.shape[1]
    query_start = total_len - 30
    doc_len = query_start - system_len

    return {
        'input_ids': input_ids,
        'system_len': system_len,
        'doc_len': doc_len,
        'query_start': query_start
    }


def main():
    device = "cuda:0"
    rate = 0.2
    threshold_factor = 0.3  # 固定使用较低阈值

    print("加载数据和模型...")
    tokenizer = AutoTokenizer.from_pretrained('/mnt/data/models/Qwen2.5-7B-Instruct', trust_remote_code=True)

    with open('result_reflect.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open('diff_cases_7b_vs_3b.json', 'r', encoding='utf-8') as f:
        diff_cases = json.load(f)

    config_path = "./config/dataset2prompt_few-shot.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    system_prompt = config["system_prompt"]["Qwen3"]["2wikimqa"]

    print("加载 3B 模型...")
    model_3b = AutoModelForCausalLM.from_pretrained(
        '/mnt/data/models/Qwen2.5-3B-Instruct',
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="eager"
    )
    model_3b.eval()

    print("加载 7B 模型...")
    model_7b = AutoModelForCausalLM.from_pretrained(
        '/mnt/data/models/Qwen2.5-7B-Instruct',
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="eager"
    )
    model_7b.eval()

    # 测试不同的 entropy_top_k 值
    topk_values = [1, 2, 3, 4, 5, 6, 8, 10, 12]

    results = {k: [] for k in topk_values}
    layer_info = {k: [] for k in topk_values}

    print(f"\n测试不同 entropy_top_k 值 (threshold_factor={threshold_factor})")
    print("="*80)

    num_cases = min(10, len(diff_cases))  # 测试更多案例

    for i, case in enumerate(diff_cases[:num_cases]):
        print(f"\n案例 {i+1}/{num_cases}: {case['sub_question'][:50]}...")

        case_data = prepare_case_input(case, data, tokenizer, system_prompt)
        if case_data is None:
            print("  跳过")
            continue

        input_ids = case_data['input_ids'].to(device)
        system_len = case_data['system_len']
        doc_len = case_data['doc_len']
        query_start = case_data['query_start']

        # 获取 7B 参考选择 (使用默认参数 top_k=4, threshold=0.5)
        sel_7b, layers_7b = get_selection_with_params(
            model_7b, input_ids, system_len, doc_len, query_start, rate, device,
            entropy_top_k=4, threshold_factor=0.5
        )
        print(f"  7B: {len(sel_7b)} tokens, layers: {layers_7b}")

        torch.cuda.empty_cache()

        # 测试各种 entropy_top_k
        for topk in topk_values:
            sel_3b, layers_3b = get_selection_with_params(
                model_3b, input_ids, system_len, doc_len, query_start, rate, device,
                entropy_top_k=topk, threshold_factor=threshold_factor
            )

            overlap = compute_overlap(sel_3b, sel_7b)
            results[topk].append(overlap)
            layer_info[topk].append(layers_3b)

            torch.cuda.empty_cache()

        # 显示本案例结果
        print(f"  3B entropy_top_k 重叠率:")
        for topk in topk_values:
            print(f"    top_k={topk:2d}: {results[topk][-1]*100:5.1f}% (layers: {layer_info[topk][-1]})")

    # 汇总
    print("\n" + "="*80)
    print("汇总统计：不同 entropy_top_k 与 7B 的平均重叠率")
    print("="*80)

    summary = []
    for topk in topk_values:
        if results[topk]:
            avg_overlap = np.mean(results[topk])
            summary.append((topk, avg_overlap, results[topk]))

    summary.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'entropy_top_k':>15s} | {'平均重叠率':>10s} | {'各案例重叠率'}")
    print("-"*80)
    for topk, avg_overlap, case_overlaps in summary:
        overlaps_str = ', '.join([f'{o*100:.1f}%' for o in case_overlaps[:5]])
        print(f"{topk:>15d} | {avg_overlap*100:9.1f}% | {overlaps_str}...")

    best_topk, best_overlap, _ = summary[0]
    print(f"\n最佳 entropy_top_k: {best_topk} (重叠率 {best_overlap*100:.1f}%)")

    # 保存结果
    output = {
        'threshold_factor': threshold_factor,
        'topk_values': topk_values,
        'results': {str(k): v for k, v in results.items()},
        'summary': [(topk, avg) for topk, avg, _ in summary]
    }
    with open('entropy_topk_experiment.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n结果已保存到 entropy_topk_experiment.json")


if __name__ == '__main__':
    main()
