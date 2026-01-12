#!/usr/bin/env python3
"""
分析"所有层取最大值"方法的答案覆盖率
对比：
1. 熵选层（当前方法）
2. 所有层取最大值（新方法）
3. 所有层取平均值
4. 单独最佳层
"""
import sys
sys.path.insert(0, '/mnt/data/wjh/FusionRAG')

import torch
import torch.nn.functional as F
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
            attn_avg = attn.mean(dim=(0, 1))  # (doc_len,)
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


def aggregate_attention(layer_attentions, method='max'):
    """聚合所有层的 attention

    method:
        - 'max': 每个位置取所有层的最大值
        - 'mean': 每个位置取所有层的平均值
        - 'entropy': 熵选层后平均
    """
    all_attn = torch.stack(list(layer_attentions.values()))  # [num_layers, doc_len]

    if method == 'max':
        combined, _ = all_attn.max(dim=0)  # [doc_len]
    elif method == 'mean':
        combined = all_attn.mean(dim=0)  # [doc_len]
    elif method == 'entropy':
        selected_layers = entropy_layer_selection(layer_attentions, top_k=4)
        selected_attn = torch.stack([layer_attentions[l] for l in selected_layers])
        combined = selected_attn.mean(dim=0)
    else:
        raise ValueError(f"Unknown method: {method}")

    return combined


def select_tokens(attn, rate, doc_len):
    """从 attention 中选择 tokens"""
    num_select = max(1, int(doc_len * rate))
    _, indices = torch.topk(attn, num_select)
    return indices.sort()[0].tolist()


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
        return 0, [], []

    selected_set = set(selected_positions)
    covered = [p for p in answer_positions if p in selected_set]
    coverage = len(covered) / len(answer_positions)

    return coverage, answer_positions, covered


def find_case_data(main_q, sub_q, data):
    """根据问题找到对应的数据"""
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
    """准备测试案例的输入"""
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

    # 统计结果
    methods = ['max', 'mean', 'entropy', 'best_single']
    results = {m: [] for m in methods}

    print(f"\n分析 {len(diff_cases)} 个差异案例...")
    print("="*80)

    for i, case in enumerate(diff_cases):
        print(f"\n案例 {i+1}: {case['sub_question'][:60]}...")

        case_data = prepare_case_input(case, data, tokenizer, system_prompt)
        if case_data is None:
            print("  跳过: 无法准备数据")
            continue

        input_ids = case_data['input_ids'].to(device)
        system_len = case_data['system_len']
        doc_len = case_data['doc_len']
        query_start = case_data['query_start']
        doc_start = system_len
        doc_end = system_len + doc_len

        # 计算所有层的 attention
        layer_attentions = compute_attention_all_layers(
            model, input_ids, query_start, doc_start, doc_end
        )

        input_ids_cpu = input_ids[0].cpu()

        # 方法 1: 所有层取最大值
        attn_max = aggregate_attention(layer_attentions, method='max')
        selected_max = select_tokens(attn_max, rate, doc_len)
        selected_max_global = [s + doc_start for s in selected_max]
        cov_max, _, _ = compute_answer_coverage(
            selected_max_global, input_ids_cpu, case['ground_truth'], tokenizer, doc_start, doc_end
        )
        results['max'].append(cov_max)

        # 方法 2: 所有层取平均值
        attn_mean = aggregate_attention(layer_attentions, method='mean')
        selected_mean = select_tokens(attn_mean, rate, doc_len)
        selected_mean_global = [s + doc_start for s in selected_mean]
        cov_mean, _, _ = compute_answer_coverage(
            selected_mean_global, input_ids_cpu, case['ground_truth'], tokenizer, doc_start, doc_end
        )
        results['mean'].append(cov_mean)

        # 方法 3: 熵选层
        attn_entropy = aggregate_attention(layer_attentions, method='entropy')
        selected_entropy = select_tokens(attn_entropy, rate, doc_len)
        selected_entropy_global = [s + doc_start for s in selected_entropy]
        cov_entropy, answer_pos, _ = compute_answer_coverage(
            selected_entropy_global, input_ids_cpu, case['ground_truth'], tokenizer, doc_start, doc_end
        )
        results['entropy'].append(cov_entropy)

        # 方法 4: 找最佳单层
        best_cov = 0
        best_layer = 0
        for layer_idx, attn in layer_attentions.items():
            selected = select_tokens(attn, rate, doc_len)
            selected_global = [s + doc_start for s in selected]
            cov, _, _ = compute_answer_coverage(
                selected_global, input_ids_cpu, case['ground_truth'], tokenizer, doc_start, doc_end
            )
            if cov > best_cov:
                best_cov = cov
                best_layer = layer_idx
        results['best_single'].append(best_cov)

        print(f"  文档长度: {doc_len}, 答案相关位置: {len(answer_pos)}")
        print(f"  Max:     {cov_max*100:5.1f}%")
        print(f"  Mean:    {cov_mean*100:5.1f}%")
        print(f"  Entropy: {cov_entropy*100:5.1f}%")
        print(f"  Best(L{best_layer}): {best_cov*100:5.1f}%")

    # 汇总
    print("\n" + "="*80)
    print("汇总统计 (rate=0.2)")
    print("="*80)
    print(f"{'方法':20s} | {'平均覆盖率':>12s} | {'标准差':>10s} | {'最小':>8s} | {'最大':>8s}")
    print("-"*70)

    for method in methods:
        if results[method]:
            arr = np.array(results[method])
            print(f"{method:20s} | {arr.mean()*100:11.1f}% | {arr.std()*100:9.1f}% | {arr.min()*100:7.1f}% | {arr.max()*100:7.1f}%")

    # 保存详细结果
    output = {
        'rate': rate,
        'num_cases': len(diff_cases),
        'results': {m: results[m] for m in methods},
        'summary': {m: {
            'mean': float(np.mean(results[m])) if results[m] else 0,
            'std': float(np.std(results[m])) if results[m] else 0,
            'min': float(np.min(results[m])) if results[m] else 0,
            'max': float(np.max(results[m])) if results[m] else 0
        } for m in methods}
    }

    with open('max_across_layers_analysis.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n结果已保存到 max_across_layers_analysis.json")


if __name__ == '__main__':
    main()
