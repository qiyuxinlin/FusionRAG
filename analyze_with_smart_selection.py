#!/usr/bin/env python3
"""
对比简单 top-k vs smart_query_selection（连通分量+边界扩展）的答案覆盖率
"""
import sys
sys.path.insert(0, '/mnt/data/wjh/FusionRAG')

import torch
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM


def compute_attention_all_layers(model, input_ids, query_start, doc_start, doc_end):
    """计算所有层的 attention 分布"""
    model.eval()
    num_layers = model.config.num_hidden_layers

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            output_attentions=True,
            use_cache=False
        )

        layer_attentions = {}
        for layer_idx in range(num_layers):
            attn = outputs.attentions[layer_idx][0, :, query_start:, doc_start:doc_end]
            attn_avg = attn.mean(dim=(0, 1))
            layer_attentions[layer_idx] = attn_avg.cpu()

    return layer_attentions


def entropy_layer_selection(layer_attentions, top_k=4):
    """选择熵最低的层"""
    layer_entropies = {}
    for layer_idx, attn in layer_attentions.items():
        attn_norm = attn / (attn.sum() + 1e-10)
        entropy = -torch.sum(attn_norm * torch.log(attn_norm + 1e-10))
        layer_entropies[layer_idx] = entropy.item()

    sorted_layers = sorted(layer_entropies.items(), key=lambda x: x[1])
    selected = [l[0] for l in sorted_layers[:top_k]]
    return selected


def simple_topk_selection(attn, rate, doc_len, doc_start):
    """简单 top-k 选择"""
    num_select = max(1, int(doc_len * rate))
    _, indices = torch.topk(attn, num_select)
    return (indices + doc_start).sort()[0].tolist()


def smart_selection_simple(attn, rate, doc_len, doc_start):
    """
    简化版 smart selection：连通分量 + 边界扩展
    直接对单个 attention 向量操作
    """
    attn_np = attn.numpy()
    target_count = int(doc_len * rate)

    # Step 1: 找到高 attention 位置
    mean_attn = np.mean(attn_np)
    std_attn = np.std(attn_np)
    threshold = mean_attn + 0.5 * std_attn
    high_attn_positions = list(np.where(attn_np > threshold)[0])

    # Step 2: 找到连通群组 (max_gap=2)
    components = []
    if high_attn_positions:
        high_attn_positions.sort()
        current_comp = [high_attn_positions[0]]
        for pos in high_attn_positions[1:]:
            if pos - current_comp[-1] <= 2:
                current_comp.append(pos)
            else:
                components.append(current_comp)
                current_comp = [pos]
        components.append(current_comp)

    # Step 3: 计算每个群组的总 attention，排序
    component_scores = [(comp, sum(attn_np[p] for p in comp)) for comp in components]
    component_scores.sort(key=lambda x: x[1], reverse=True)

    # Step 4: 贪心选择 + 边界扩展 (±1)
    selected = set()
    for comp, _ in component_scores:
        extended = set()
        for p in comp:
            for offset in range(-1, 2):
                new_p = p + offset
                if 0 <= new_p < doc_len:
                    extended.add(new_p)
        if len(selected) + len(extended - selected) <= target_count * 1.1:
            selected.update(extended)

    # Step 5: 补充到目标数量
    if len(selected) < target_count:
        sorted_indices = np.argsort(attn_np)[::-1]
        for pos in sorted_indices:
            if pos not in selected:
                selected.add(pos)
                if len(selected) >= target_count:
                    break

    # Step 6: 如果超过目标，移除最低分的
    while len(selected) > target_count:
        min_pos = min(selected, key=lambda p: attn_np[p])
        selected.remove(min_pos)

    return [p + doc_start for p in sorted(selected)]


def compute_answer_coverage(selected_positions, input_ids, answer_text, tokenizer, doc_start, doc_end):
    """计算答案覆盖率"""
    answer_keywords = [w.lower() for w in answer_text.split() if len(w) > 2]

    answer_positions = []
    for i in range(doc_start, min(doc_end, len(input_ids))):
        tok_text = tokenizer.decode([input_ids[i]]).lower().strip()
        for word in answer_keywords:
            if word in tok_text or tok_text in word:
                answer_positions.append(i)
                break

    if not answer_positions:
        return 0, 0, []

    selected_set = set(selected_positions)
    covered = len([p for p in answer_positions if p in selected_set])
    coverage = covered / len(answer_positions)

    return coverage, len(answer_positions), covered


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

    query_text = f"Question: {sub_q}"
    query_ids = tokenizer.encode(query_text, add_special_tokens=False)

    total_len = input_ids.shape[1]
    query_start = total_len - len(query_ids) - 5
    doc_len = query_start - system_len

    return {
        'input_ids': input_ids,
        'system_len': system_len,
        'doc_len': doc_len,
        'query_start': query_start,
        'ground_truth': case['ground_truth']
    }


def main():
    device = "cuda:0"
    rate = 0.2

    print("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('/mnt/data/models/Qwen2.5-7B-Instruct', trust_remote_code=True)

    print("加载数据...")
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

    results = {
        'topk': [],
        'smart': [],
        'topk_max': [],
        'smart_max': []
    }

    print(f"\n对比 top-k vs smart_query_selection (rate={rate})")
    print("="*90)

    for i, case in enumerate(diff_cases):
        print(f"\n案例 {i+1}: {case['sub_question'][:50]}...")

        case_data = prepare_case_input(case, data, tokenizer, system_prompt)
        if case_data is None:
            print("  跳过")
            continue

        input_ids = case_data['input_ids'].to(device)
        system_len = case_data['system_len']
        doc_len = case_data['doc_len']
        query_start = case_data['query_start']
        doc_start = system_len
        doc_end = system_len + doc_len

        layer_attentions = compute_attention_all_layers(
            model, input_ids, query_start, doc_start, doc_end
        )

        input_ids_cpu = input_ids[0].cpu()

        # 熵选层聚合
        selected_layers = entropy_layer_selection(layer_attentions, top_k=4)
        attn_entropy = torch.stack([layer_attentions[l] for l in selected_layers]).mean(dim=0)

        # 所有层取 max
        attn_max = torch.stack(list(layer_attentions.values())).max(dim=0)[0]

        # 方法 1: 熵选层 + 简单 top-k
        sel_topk = simple_topk_selection(attn_entropy, rate, doc_len, doc_start)
        cov_topk, ans_cnt, _ = compute_answer_coverage(
            sel_topk, input_ids_cpu, case['ground_truth'], tokenizer, doc_start, doc_end
        )
        results['topk'].append(cov_topk)

        # 方法 2: 熵选层 + smart_query_selection
        sel_smart = smart_selection_simple(attn_entropy, rate, doc_len, doc_start)
        cov_smart, _, _ = compute_answer_coverage(
            sel_smart, input_ids_cpu, case['ground_truth'], tokenizer, doc_start, doc_end
        )
        results['smart'].append(cov_smart)

        # 方法 3: 所有层 max + 简单 top-k
        sel_topk_max = simple_topk_selection(attn_max, rate, doc_len, doc_start)
        cov_topk_max, _, _ = compute_answer_coverage(
            sel_topk_max, input_ids_cpu, case['ground_truth'], tokenizer, doc_start, doc_end
        )
        results['topk_max'].append(cov_topk_max)

        # 方法 4: 所有层 max + smart_query_selection
        sel_smart_max = smart_selection_simple(attn_max, rate, doc_len, doc_start)
        cov_smart_max, _, _ = compute_answer_coverage(
            sel_smart_max, input_ids_cpu, case['ground_truth'], tokenizer, doc_start, doc_end
        )
        results['smart_max'].append(cov_smart_max)

        print(f"  答案位置数: {ans_cnt}, 文档长度: {doc_len}")
        print(f"  熵选层 + top-k:    {cov_topk*100:5.1f}%  (选了 {len(sel_topk)} tokens)")
        print(f"  熵选层 + smart:    {cov_smart*100:5.1f}%  (选了 {len(sel_smart)} tokens, +{len(sel_smart)-len(sel_topk)})")
        print(f"  Max层 + top-k:     {cov_topk_max*100:5.1f}%")
        print(f"  Max层 + smart:     {cov_smart_max*100:5.1f}%  (选了 {len(sel_smart_max)} tokens)")

    # 汇总
    print("\n" + "="*90)
    print("汇总统计")
    print("="*90)
    print(f"{'方法':25s} | {'平均覆盖率':>12s} | {'提升':>10s}")
    print("-"*55)

    baseline = np.mean(results['topk'])
    for method, name in [('topk', '熵选层 + top-k'),
                         ('smart', '熵选层 + smart'),
                         ('topk_max', 'Max层 + top-k'),
                         ('smart_max', 'Max层 + smart')]:
        avg = np.mean(results[method])
        delta = avg - baseline
        sign = '+' if delta >= 0 else ''
        print(f"{name:25s} | {avg*100:11.1f}% | {sign}{delta*100:9.1f}%")


if __name__ == '__main__':
    main()
