#!/usr/bin/env python3
"""
实验：让 3B 的 token 选择尽可能接近 7B

测试不同的改进策略：
1. 调整层选择（不用熵选层，用相对位置匹配）
2. 调整 attention 阈值
3. 调整连通分量 max_gap
4. 组合策略
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
    find_connected_components
)


def smart_query_selection_custom(attention_scores, doc_len, target_ratio, system_len,
                                  threshold_factor=0.5, max_gap=2, boundary_extend=1):
    """
    自定义参数的 smart_query_selection
    """
    if isinstance(attention_scores, torch.Tensor):
        attention_scores = attention_scores.float().cpu().numpy()

    target_count = int(doc_len * target_ratio)

    # Step 1: 找到高 attention 位置（可调阈值）
    mean_attn = np.mean(attention_scores)
    std_attn = np.std(attention_scores)
    threshold = mean_attn + threshold_factor * std_attn

    high_attn_positions = list(np.where(attention_scores > threshold)[0])

    # Step 2: 连通分量分析（可调 max_gap）
    components = find_connected_components(high_attn_positions, max_gap=max_gap)

    # Step 3: 计算每个分量的总 attention
    component_scores = []
    for comp in components:
        total_score = sum(attention_scores[p] for p in comp)
        component_scores.append((comp, total_score))

    component_scores.sort(key=lambda x: x[1], reverse=True)

    # Step 4: 贪心选择 + 边界扩展（可调扩展范围）
    selected = set()
    for comp, _ in component_scores:
        extended_comp = set()
        for p in comp:
            for offset in range(-boundary_extend, boundary_extend + 1):
                new_p = p + offset
                if 0 <= new_p < doc_len:
                    extended_comp.add(new_p)

        new_positions = extended_comp - selected
        if len(selected) + len(new_positions) <= target_count * 1.1:
            selected.update(extended_comp)

    # Step 5: 补充到目标数量
    if len(selected) < target_count:
        sorted_indices = np.argsort(attention_scores)[::-1]
        for pos in sorted_indices:
            if pos not in selected:
                selected.add(int(pos))
                if len(selected) >= target_count:
                    break

    while len(selected) > target_count:
        min_pos = min(selected, key=lambda p: attention_scores[p])
        selected.remove(min_pos)

    selected_global = [p + system_len for p in sorted(selected)]
    return selected_global


def get_7b_reference_selection(model_7b, input_ids, system_len, doc_len, query_start, rate, device):
    """获取 7B 的参考选择（作为 ground truth）"""
    draft_attention = compute_draft_model_attention(model_7b, input_ids, query_start, device)

    doc_start = system_len
    doc_end = system_len + doc_len

    layer_attention_dict = {}
    for layer_idx, layer_attn in draft_attention.items():
        query_to_doc = layer_attn[:, :, doc_start:doc_end]
        doc_attention_avg = query_to_doc.mean(axis=(0, 1))
        layer_attention_dict[layer_idx] = torch.tensor(doc_attention_avg, device=device)

    active_layers, _ = entropy_layer_selection(layer_attention_dict, top_k=4, return_entropy=True)
    layer_attentions = [layer_attention_dict[idx] for idx in active_layers]
    multi_layer_attn = torch.stack(layer_attentions).mean(dim=0)

    selected = smart_query_selection_custom(
        multi_layer_attn, doc_len, rate, system_len,
        threshold_factor=0.5, max_gap=2, boundary_extend=1
    )

    return set(selected), multi_layer_attn.cpu(), active_layers


def test_3b_strategy(model_3b, input_ids, system_len, doc_len, query_start, rate, device,
                     layer_strategy='entropy', threshold_factor=0.5, max_gap=2, boundary_extend=1,
                     fixed_layers=None):
    """
    测试不同的 3B 策略

    layer_strategy:
        - 'entropy': 熵选层（原始方法）
        - 'relative': 相对位置匹配 7B（选 7B 选的相对位置对应的 3B 层）
        - 'fixed': 使用固定层
        - 'middle': 使用中间层
    """
    draft_attention = compute_draft_model_attention(model_3b, input_ids, query_start, device)

    doc_start = system_len
    doc_end = system_len + doc_len

    layer_attention_dict = {}
    for layer_idx, layer_attn in draft_attention.items():
        query_to_doc = layer_attn[:, :, doc_start:doc_end]
        doc_attention_avg = query_to_doc.mean(axis=(0, 1))
        layer_attention_dict[layer_idx] = torch.tensor(doc_attention_avg, device=device)

    num_layers_3b = 36
    available_layers = sorted(layer_attention_dict.keys())

    # 选择层策略
    if layer_strategy == 'entropy':
        active_layers, _ = entropy_layer_selection(layer_attention_dict, top_k=4, return_entropy=True)
    elif layer_strategy == 'relative':
        # 7B 是 28 层，选的是 L14-22 (相对位置 50%-80%)
        # 3B 是 36 层，对应相对位置应该是 L18-29
        relative_positions = [0.50, 0.57, 0.64, 0.71]  # 7B 的 L14,16,18,20 相对位置
        active_layers = [int(p * num_layers_3b) for p in relative_positions]
        active_layers = [l for l in active_layers if l in available_layers]
        if len(active_layers) < 4:
            active_layers = available_layers[-4:]  # fallback
    elif layer_strategy == 'fixed' and fixed_layers:
        active_layers = [l for l in fixed_layers if l in available_layers]
    elif layer_strategy == 'middle':
        # 选择中间层 (40%-60%)
        mid_start = int(0.4 * num_layers_3b)
        mid_end = int(0.6 * num_layers_3b)
        active_layers = [l for l in range(mid_start, mid_end + 1) if l in available_layers][:4]
    else:
        active_layers, _ = entropy_layer_selection(layer_attention_dict, top_k=4, return_entropy=True)

    layer_attentions = [layer_attention_dict[idx] for idx in active_layers]
    multi_layer_attn = torch.stack(layer_attentions).mean(dim=0)

    selected = smart_query_selection_custom(
        multi_layer_attn, doc_len, rate, system_len,
        threshold_factor=threshold_factor,
        max_gap=max_gap,
        boundary_extend=boundary_extend
    )

    return set(selected), active_layers


def compute_overlap(sel_3b, sel_7b):
    """计算重叠率"""
    common = sel_3b & sel_7b
    union = sel_3b | sel_7b
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

    # 定义要测试的策略
    strategies = [
        # (名称, layer_strategy, threshold_factor, max_gap, boundary_extend, fixed_layers)
        ("baseline (entropy)", "entropy", 0.5, 2, 1, None),
        ("relative_layers", "relative", 0.5, 2, 1, None),
        ("middle_layers", "middle", 0.5, 2, 1, None),
        ("lower_threshold_0.3", "entropy", 0.3, 2, 1, None),
        ("lower_threshold_0.25", "entropy", 0.25, 2, 1, None),
        ("larger_gap_3", "entropy", 0.5, 3, 1, None),
        ("larger_gap_4", "entropy", 0.5, 4, 1, None),
        ("larger_boundary_2", "entropy", 0.5, 2, 2, None),
        ("combo: relative + lower_thresh", "relative", 0.3, 2, 1, None),
        ("combo: relative + larger_gap", "relative", 0.5, 3, 1, None),
        ("combo: middle + lower_thresh", "middle", 0.3, 2, 1, None),
        ("combo: all adjusted", "relative", 0.3, 3, 2, None),
    ]

    results = {name: [] for name, *_ in strategies}

    print(f"\n测试 {len(strategies)} 种策略，目标：最大化与 7B 的重叠率")
    print("="*80)

    for i, case in enumerate(diff_cases[:5]):  # 测试前 5 个案例
        print(f"\n案例 {i+1}: {case['sub_question'][:50]}...")

        case_data = prepare_case_input(case, data, tokenizer, system_prompt)
        if case_data is None:
            print("  跳过")
            continue

        input_ids = case_data['input_ids'].to(device)
        system_len = case_data['system_len']
        doc_len = case_data['doc_len']
        query_start = case_data['query_start']

        # 获取 7B 参考选择
        print("  获取 7B 参考选择...")
        sel_7b, _, layers_7b = get_7b_reference_selection(
            model_7b, input_ids, system_len, doc_len, query_start, rate, device
        )
        print(f"    7B 选择: {len(sel_7b)} tokens, 使用层: {layers_7b}")

        torch.cuda.empty_cache()

        # 测试各种 3B 策略
        print("  测试 3B 策略...")
        for name, layer_strategy, threshold_factor, max_gap, boundary_extend, fixed_layers in strategies:
            sel_3b, layers_3b = test_3b_strategy(
                model_3b, input_ids, system_len, doc_len, query_start, rate, device,
                layer_strategy=layer_strategy,
                threshold_factor=threshold_factor,
                max_gap=max_gap,
                boundary_extend=boundary_extend,
                fixed_layers=fixed_layers
            )

            overlap = compute_overlap(sel_3b, sel_7b)
            results[name].append(overlap)

            torch.cuda.empty_cache()

        # 显示本案例结果
        print(f"\n  本案例重叠率:")
        case_results = [(name, results[name][-1]) for name, *_ in strategies]
        case_results.sort(key=lambda x: x[1], reverse=True)
        for name, overlap in case_results[:5]:
            print(f"    {name:35s}: {overlap*100:5.1f}%")

    # 汇总
    print("\n" + "="*80)
    print("汇总统计：各策略与 7B 的平均重叠率")
    print("="*80)

    summary = []
    for name, *_ in strategies:
        if results[name]:
            avg_overlap = np.mean(results[name])
            summary.append((name, avg_overlap, results[name]))

    summary.sort(key=lambda x: x[1], reverse=True)

    baseline_overlap = summary[0][1] if summary[0][0] == "baseline (entropy)" else None
    for name, avg_overlap in [(n, a) for n, a, _ in summary if n == "baseline (entropy)"]:
        baseline_overlap = avg_overlap

    print(f"\n{'策略':40s} | {'平均重叠率':>10s} | {'vs baseline':>12s}")
    print("-"*70)
    for name, avg_overlap, case_overlaps in summary:
        delta = avg_overlap - baseline_overlap if baseline_overlap else 0
        sign = '+' if delta >= 0 else ''
        print(f"{name:40s} | {avg_overlap*100:9.1f}% | {sign}{delta*100:11.1f}%")

    # 找最佳策略
    best_name, best_overlap, _ = summary[0]
    print(f"\n最佳策略: {best_name} (重叠率 {best_overlap*100:.1f}%)")

    # 保存结果
    output = {
        'strategies': [name for name, *_ in strategies],
        'results': results,
        'summary': [(name, avg) for name, avg, _ in summary]
    }
    with open('3b_match_7b_experiment.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n结果已保存到 3b_match_7b_experiment.json")


if __name__ == '__main__':
    main()
