#!/usr/bin/env python3
"""
深入分析 7B vs 3B draft model 的 token 选择差异
针对 7B 答对但 3B 答错的案例，分析两个模型选择了哪些不同的 tokens
"""
import sys
sys.path.insert(0, '/mnt/data/wjh/FusionRAG')

import torch
import json
import os
import gc
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载数据
with open('result_reflect.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

with open('diff_cases_7b_vs_3b.json', 'r', encoding='utf-8') as f:
    diff_cases = json.load(f)

print(f"差异案例数: {len(diff_cases)}")

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained('/mnt/data/models/Qwen2.5-7B-Instruct', trust_remote_code=True)

# 加载系统提示
config_path = "./config/dataset2prompt_few-shot.json"
with open(config_path, 'r', encoding='utf-8') as f:
    config = json.load(f)
system_prompt = config["system_prompt"]["Qwen3"]["2wikimqa"]

def compute_draft_attention_simple(model, input_ids, query_start, device="cuda:0"):
    """计算 draft model 的 attention (简化版，只取最后几层，使用 hook 逐层获取)"""
    model.eval()
    seq_len = input_ids.shape[1]
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads

    # 只计算最后 4 层 (足够用于熵选层)
    start_layer = num_layers - 4

    layer_attentions = {}
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, inputs, outputs):
            # outputs 是 tuple: (hidden_states, attn_weights, ...)
            if len(outputs) > 1 and outputs[1] is not None:
                attn = outputs[1]  # (batch, heads, seq, seq)
                attn_slice = attn[0, :, query_start:, :query_start]  # (heads, query_len, key_len)
                attn_avg = attn_slice.mean(dim=1)  # (heads, key_len)
                layer_attentions[layer_idx] = attn_avg.cpu()
        return hook_fn

    # 注册 hooks
    for layer_idx in range(start_layer, num_layers):
        layer = model.model.layers[layer_idx].self_attn
        hook = layer.register_forward_hook(make_hook(layer_idx))
        hooks.append(hook)

    try:
        with torch.no_grad():
            # 只对最后几层输出 attention
            outputs = model(
                input_ids=input_ids,
                output_attentions=True,
                use_cache=False
            )
    finally:
        # 移除 hooks
        for hook in hooks:
            hook.remove()

    return layer_attentions

def entropy_layer_selection(layer_attentions, top_k=4):
    """选择熵最低的层"""
    layer_entropies = {}
    for layer_idx, attn in layer_attentions.items():
        attn_flat = attn.mean(dim=0)
        attn_flat = attn_flat / (attn_flat.sum() + 1e-10)
        entropy = -torch.sum(attn_flat * torch.log(attn_flat + 1e-10))
        layer_entropies[layer_idx] = entropy.item()

    sorted_layers = sorted(layer_entropies.items(), key=lambda x: x[1])
    selected = [l[0] for l in sorted_layers[:top_k]]
    return selected, layer_entropies

def select_tokens_from_attention(layer_attentions, selected_layers, doc_len, rate, system_len):
    """从选定层的 attention 中选择 tokens"""
    all_attn = []
    for layer_idx in selected_layers:
        attn = layer_attentions[layer_idx].mean(dim=0)
        all_attn.append(attn)

    combined_attn = torch.stack(all_attn).mean(dim=0)
    doc_attn = combined_attn[system_len:system_len + doc_len]

    num_select = max(1, int(doc_len * rate))
    _, indices = torch.topk(doc_attn, num_select)
    selected_positions = (indices + system_len).sort()[0].tolist()

    return selected_positions, doc_attn

def find_case_data(main_q, sub_q, data):
    """根据问题找到对应的数据"""
    for item in data:
        # main_question 在 data 中是 'question'
        if item.get('question', '') == main_q:
            # 在 intermediate_context 中找子问题
            for sub in item.get('intermediate_context', []):
                query = sub.get('query', '')
                # 移除 "Intermediate queryXXX:" 前缀
                if query.startswith("Intermediate query"):
                    colon_pos = query.find(":")
                    if colon_pos != -1:
                        query = query[colon_pos + 1:].strip()

                if query == sub_q:
                    return item, sub
    return None, None

def compute_gini(values):
    """计算 Gini 系数"""
    values = values.flatten().numpy()
    values = np.sort(values)
    n = len(values)
    if np.sum(values) == 0:
        return 0
    return (2 * np.sum((np.arange(1, n+1) * values)) / (n * np.sum(values))) - (n + 1) / n

def prepare_case_input(case, data, tokenizer, system_prompt):
    """准备案例输入"""
    main_item, sub_item = find_case_data(case['main_question'], case['sub_question'], data)
    if main_item is None:
        print(f"  警告: 未找到 main_question: {case['main_question'][:50]}...")
        return None
    if sub_item is None:
        print(f"  警告: 未找到 sub_question: {case['sub_question'][:50]}...")
        return None

    # 获取文档
    docs = sub_item.get('retrieve docs', [])
    if not docs:
        print(f"  警告: 没有找到文档")
        return None

    doc_text = "\n\n".join(docs[:10])
    question = case['sub_question']

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Documents:\n{doc_text}\n\nQuestion: {question}"}
    ]

    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    # 找到 system prompt 边界
    system_text = tokenizer.apply_chat_template([{"role": "system", "content": system_prompt}], tokenize=False)
    system_ids = tokenizer.encode(system_text, return_tensors='pt')
    system_len = system_ids.shape[1]

    # 找到 query 开始位置
    full_text = tokenizer.decode(input_ids[0])
    query_marker = f"Question: {question}"
    query_start_char = full_text.find(query_marker)
    if query_start_char == -1:
        query_start_char = full_text.find("Question:")

    prefix_text = full_text[:query_start_char]
    prefix_ids = tokenizer.encode(prefix_text, return_tensors='pt')
    query_start = prefix_ids.shape[1]

    doc_len = query_start - system_len

    return {
        'input_ids': input_ids,
        'system_len': system_len,
        'doc_len': doc_len,
        'query_start': query_start,
        'doc_text': doc_text,
        'question': question
    }

def analyze_with_model(model_path, model_name, cases_data, device="cuda:0"):
    """用单个模型分析所有案例"""
    print(f"\n加载 {model_name} 模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="eager"  # 需要 eager 才能输出 attention
    )

    results = {}
    rate = 0.3

    for case_idx, (case, case_data) in enumerate(cases_data):
        if case_data is None:
            continue

        print(f"  分析案例 {case_idx + 1}...")
        input_ids = case_data['input_ids'].to(device)
        query_start = case_data['query_start']
        system_len = case_data['system_len']
        doc_len = case_data['doc_len']

        attn = compute_draft_attention_simple(model, input_ids, query_start, device)
        layers, layer_entropies = entropy_layer_selection(attn, top_k=4)
        selected, scores = select_tokens_from_attention(attn, layers, doc_len, rate, system_len)

        results[case_idx] = {
            'layers': layers,
            'layer_entropies': layer_entropies,
            'selected': selected,
            'scores': scores,
            'gini': compute_gini(scores),
            'input_ids': input_ids.cpu()
        }

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return results

def main():
    device = "cuda:0"  # CUDA_VISIBLE_DEVICES 会重映射设备

    # 准备所有案例的输入数据
    print("准备案例输入数据...")
    cases_data = []
    for i, case in enumerate(diff_cases[:5]):
        print(f"处理案例 {i+1}: {case['sub_question'][:50]}...")
        case_data = prepare_case_input(case, data, tokenizer, system_prompt)
        cases_data.append((case, case_data))

    valid_cases = sum(1 for _, cd in cases_data if cd is not None)
    print(f"\n成功准备 {valid_cases}/{len(cases_data)} 个案例")

    if valid_cases == 0:
        print("没有有效案例，退出")
        return

    # 分别用 7B 和 3B 模型分析
    results_7b = analyze_with_model('/mnt/data/models/Qwen2.5-7B-Instruct', '7B', cases_data, device)
    results_3b = analyze_with_model('/mnt/data/models/Qwen2.5-3B-Instruct', '3B', cases_data, device)

    # 对比分析
    print("\n" + "="*100)
    print("详细对比分析")
    print("="*100)

    for case_idx, (case, case_data) in enumerate(cases_data):
        if case_idx not in results_7b or case_idx not in results_3b:
            continue

        r7b = results_7b[case_idx]
        r3b = results_3b[case_idx]

        print(f"\n{'='*100}")
        print(f"案例 {case_idx + 1}: {case['sub_question'][:70]}...")
        print(f"{'='*100}")
        print(f"标准答案: {case['ground_truth'][:80]}...")
        print(f"7B 预测 (正确): {case['predicted_7b'][:80]}...")
        print(f"3B 预测 (错误): {case['predicted_3b'][:80]}...")

        print(f"\n【层选择对比】")
        entropy_7b = [f"{r7b['layer_entropies'][l]:.3f}" for l in r7b['layers']]
        entropy_3b = [f"{r3b['layer_entropies'][l]:.3f}" for l in r3b['layers']]
        print(f"  7B 选择的层: {r7b['layers']} (熵值: {entropy_7b})")
        print(f"  3B 选择的层: {r3b['layers']} (熵值: {entropy_3b})")

        set_7b = set(r7b['selected'])
        set_3b = set(r3b['selected'])
        overlap = set_7b & set_3b
        only_7b = set_7b - set_3b
        only_3b = set_3b - set_7b

        print(f"\n【Token 选择对比】")
        print(f"  7B 选择了 {len(set_7b)} 个 tokens")
        print(f"  3B 选择了 {len(set_3b)} 个 tokens")
        print(f"  重叠: {len(overlap)} 个 ({len(overlap)/len(set_7b)*100:.1f}%)")
        print(f"  仅 7B: {len(only_7b)} 个")
        print(f"  仅 3B: {len(only_3b)} 个")

        print(f"\n【Attention 分布对比】")
        print(f"  7B Gini: {r7b['gini']:.4f} (越高越集中)")
        print(f"  3B Gini: {r3b['gini']:.4f}")
        print(f"  7B max: {r7b['scores'].max().item():.4f}, mean: {r7b['scores'].mean().item():.6f}")
        print(f"  3B max: {r3b['scores'].max().item():.4f}, mean: {r3b['scores'].mean().item():.6f}")

        # 解码显示差异 tokens
        tokens = r7b['input_ids'][0].tolist()

        print(f"\n【仅 7B 选择的关键 tokens (按位置排序，前 15 个)】")
        only_7b_sorted = sorted(only_7b)[:15]
        only_7b_text = []
        for pos in only_7b_sorted:
            if pos < len(tokens):
                token_text = tokenizer.decode([tokens[pos]])
                only_7b_text.append(f"'{token_text}'")
        print(f"  {' '.join(only_7b_text)}")

        print(f"\n【仅 3B 选择的 tokens (前 15 个)】")
        only_3b_sorted = sorted(only_3b)[:15]
        only_3b_text = []
        for pos in only_3b_sorted:
            if pos < len(tokens):
                token_text = tokenizer.decode([tokens[pos]])
                only_3b_text.append(f"'{token_text}'")
        print(f"  {' '.join(only_3b_text)}")

        # 找出答案相关的 tokens
        answer = case['ground_truth']
        answer_keywords = []
        for word in answer.lower().split():
            if len(word) > 3 and word.isalpha():
                answer_keywords.append(word)

        system_len = case_data['system_len']
        doc_len = case_data['doc_len']

        answer_related_positions = []
        for i in range(system_len, system_len + doc_len):
            if i < len(tokens):
                tok_text = tokenizer.decode([tokens[i]]).lower().strip()
                for word in answer_keywords:
                    if word in tok_text or tok_text in word:
                        answer_related_positions.append(i)
                        break

        if answer_related_positions:
            answer_set = set(answer_related_positions)
            covered_7b = len(answer_set & set_7b)
            covered_3b = len(answer_set & set_3b)
            print(f"\n【答案相关 token 覆盖率】")
            print(f"  找到 {len(answer_related_positions)} 个答案相关 tokens")
            print(f"  7B 覆盖: {covered_7b}/{len(answer_related_positions)} ({covered_7b/len(answer_related_positions)*100:.1f}%)")
            print(f"  3B 覆盖: {covered_3b}/{len(answer_related_positions)} ({covered_3b/len(answer_related_positions)*100:.1f}%)")

            answer_only_7b = answer_set & only_7b
            if answer_only_7b:
                print(f"\n  7B 覆盖但 3B 未覆盖的答案 tokens:")
                for pos in sorted(answer_only_7b)[:10]:
                    tok_text = tokenizer.decode([tokens[pos]])
                    print(f"    位置 {pos}: '{tok_text}'")

    # 汇总统计
    print("\n" + "="*100)
    print("汇总统计")
    print("="*100)

    overlaps = []
    gini_7bs = []
    gini_3bs = []

    for case_idx in results_7b:
        if case_idx in results_3b:
            r7b = results_7b[case_idx]
            r3b = results_3b[case_idx]
            set_7b = set(r7b['selected'])
            set_3b = set(r3b['selected'])
            overlap = len(set_7b & set_3b) / len(set_7b) if len(set_7b) > 0 else 0
            overlaps.append(overlap)
            gini_7bs.append(r7b['gini'])
            gini_3bs.append(r3b['gini'])

    if overlaps:
        print(f"平均 token 重叠率: {np.mean(overlaps)*100:.1f}%")
        print(f"7B 平均 Gini: {np.mean(gini_7bs):.4f}")
        print(f"3B 平均 Gini: {np.mean(gini_3bs):.4f}")

        if np.mean(gini_7bs) > np.mean(gini_3bs):
            print("\n结论: 7B 的 attention 更集中，选择的 tokens 更聚焦于关键信息")
        else:
            print("\n结论: 3B 的 attention 更集中，但可能遗漏了部分关键信息")

if __name__ == '__main__':
    main()
