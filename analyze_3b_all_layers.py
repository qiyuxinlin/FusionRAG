#!/usr/bin/env python3
"""
分析 3B 模型所有 36 层的 attention 分布
目标：找出哪些层能准确识别到关键 tokens
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
        layer_stats = {}

        for layer_idx in range(num_layers):
            # attention shape: (batch, num_heads, seq_len, seq_len)
            attn = outputs.attentions[layer_idx][0, :, query_start:, doc_start:doc_end]
            # 平均所有 heads 和 query positions
            attn_avg = attn.mean(dim=(0, 1))  # (doc_len,)
            layer_attentions[layer_idx] = attn_avg.cpu()

            # 计算统计特征
            attn_norm = attn_avg / (attn_avg.sum() + 1e-10)

            # 熵
            entropy = -torch.sum(attn_norm * torch.log(attn_norm + 1e-10)).item()

            # Gini 系数
            sorted_attn = torch.sort(attn_norm)[0]
            n = len(sorted_attn)
            cumsum = torch.cumsum(sorted_attn, dim=0)
            gini = (2 * torch.sum((torch.arange(1, n+1).float() * sorted_attn.cpu()) / (n * sorted_attn.sum().cpu()))) - (n + 1) / n

            # Top-10% 占比
            top_k = max(1, int(n * 0.1))
            top_ratio = attn_norm.topk(top_k)[0].sum().item()

            # 最大值
            max_attn = attn_norm.max().item()

            layer_stats[layer_idx] = {
                'entropy': entropy,
                'gini': gini.item() if not torch.isnan(gini) else 0,
                'top_10_ratio': top_ratio,
                'max_attn': max_attn
            }

    return layer_attentions, layer_stats


def select_tokens_from_attention(attn, rate, doc_len):
    """从 attention 中选择 tokens"""
    num_select = max(1, int(doc_len * rate))
    _, indices = torch.topk(attn, num_select)
    return indices.sort()[0].tolist()


def compute_answer_coverage(selected_positions, input_ids, answer_text, tokenizer, doc_start, doc_end):
    """计算答案覆盖率"""
    # 提取答案关键词
    answer_keywords = [w.lower() for w in answer_text.split() if len(w) > 2]

    # 找出文档中与答案相关的位置
    answer_positions = []
    for i in range(doc_start, min(doc_end, len(input_ids))):
        tok_text = tokenizer.decode([input_ids[i]]).lower().strip()
        for word in answer_keywords:
            if word in tok_text or tok_text in word:
                answer_positions.append(i)
                break

    if not answer_positions:
        return 0, [], []

    # 计算覆盖率
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
                # 移除 "Intermediate queryXXX:" 前缀
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
        print(f"  警告: 未找到对应数据")
        return None

    # 获取文档
    docs = sub_item.get('retrieve docs', [])
    if not docs:
        print(f"  警告: 没有找到文档")
        return None

    doc_text = "\n\n".join(docs[:10])
    sub_q = case['sub_question']

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{doc_text}\n\nQuestion: {sub_q}"}
    ]

    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    # 计算各部分长度
    system_text = tokenizer.apply_chat_template([{"role": "system", "content": system_prompt}],
                                                 tokenize=False, add_generation_prompt=False)
    system_len = len(tokenizer.encode(system_text))

    query_text = f"Question: {sub_q}"
    query_ids = tokenizer.encode(query_text, add_special_tokens=False)

    total_len = input_ids.shape[1]
    query_start = total_len - len(query_ids) - 5  # 粗略估计
    doc_len = query_start - system_len

    return {
        'input_ids': input_ids,
        'system_len': system_len,
        'doc_len': doc_len,
        'query_start': query_start,
        'docs_text': doc_text,
        'sub_question': sub_q,
        'ground_truth': case['ground_truth']
    }


def main():
    device = "cuda:0"
    rate = 0.2  # 测试 20% 选择率

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

    # 只分析第一个案例
    case = diff_cases[0]
    print(f"\n{'='*70}")
    print(f"分析案例: {case['sub_question']}")
    print(f"标准答案: {case['ground_truth']}")
    print(f"7B 预测: {case['predicted_7b']}")
    print(f"3B 预测: {case['predicted_3b']}")
    print(f"{'='*70}\n")

    case_data = prepare_case_input(case, data, tokenizer, system_prompt)
    if case_data is None:
        print("无法准备案例数据")
        return

    input_ids = case_data['input_ids'].to(device)
    system_len = case_data['system_len']
    doc_len = case_data['doc_len']
    query_start = case_data['query_start']
    doc_start = system_len
    doc_end = system_len + doc_len

    print(f"文档长度: {doc_len} tokens")
    print(f"选择数量: {int(doc_len * rate)} tokens (rate={rate})")
    print()

    # 计算所有层的 attention
    print("计算所有层的 attention 分布...")
    layer_attentions, layer_stats = compute_attention_all_layers(
        model, input_ids, query_start, doc_start, doc_end
    )

    # 对每一层计算答案覆盖率
    print("\n" + "="*70)
    print("各层答案覆盖率分析 (rate=0.2)")
    print("="*70)
    print(f"{'Layer':>6} | {'Coverage':>10} | {'Entropy':>10} | {'Gini':>10} | {'Top10%':>10} | {'MaxAttn':>10}")
    print("-"*70)

    layer_coverages = {}
    input_ids_cpu = input_ids[0].cpu()

    for layer_idx in range(len(layer_attentions)):
        attn = layer_attentions[layer_idx]
        selected = select_tokens_from_attention(attn, rate, doc_len)
        selected_global = [s + doc_start for s in selected]

        coverage, answer_pos, covered = compute_answer_coverage(
            selected_global, input_ids_cpu, case['ground_truth'], tokenizer, doc_start, doc_end
        )

        layer_coverages[layer_idx] = {
            'coverage': coverage,
            'answer_positions': len(answer_pos),
            'covered': len(covered),
            'selected': selected_global
        }

        stats = layer_stats[layer_idx]
        print(f"{layer_idx:>6} | {coverage*100:>9.1f}% | {stats['entropy']:>10.3f} | {stats['gini']:>10.3f} | {stats['top_10_ratio']*100:>9.1f}% | {stats['max_attn']:>10.4f}")

    # 找出最佳层
    print("\n" + "="*70)
    print("最佳层分析")
    print("="*70)

    sorted_by_coverage = sorted(layer_coverages.items(), key=lambda x: x[1]['coverage'], reverse=True)

    print("\n答案覆盖率 TOP 5 层:")
    for layer_idx, info in sorted_by_coverage[:5]:
        stats = layer_stats[layer_idx]
        print(f"  Layer {layer_idx}: 覆盖率 {info['coverage']*100:.1f}% ({info['covered']}/{info['answer_positions']}), "
              f"熵={stats['entropy']:.3f}, Gini={stats['gini']:.3f}")

    # 对比熵选层选择的层
    print("\n熵最低的 4 层 (当前熵选层策略):")
    sorted_by_entropy = sorted(layer_stats.items(), key=lambda x: x[1]['entropy'])
    for layer_idx, stats in sorted_by_entropy[:4]:
        info = layer_coverages[layer_idx]
        print(f"  Layer {layer_idx}: 熵={stats['entropy']:.3f}, 覆盖率 {info['coverage']*100:.1f}%")

    # 详细分析最佳层
    best_layer = sorted_by_coverage[0][0]
    print(f"\n\n最佳层 (Layer {best_layer}) 详细分析:")
    print("-"*50)

    best_attn = layer_attentions[best_layer]
    best_selected = layer_coverages[best_layer]['selected']

    print(f"选中的 tokens ({len(best_selected)} 个):")
    selected_tokens = []
    for pos in best_selected[:30]:  # 只显示前 30 个
        tok = tokenizer.decode([input_ids_cpu[pos]])
        selected_tokens.append(f"'{tok}'")
    print("  " + ", ".join(selected_tokens))
    if len(best_selected) > 30:
        print(f"  ... 还有 {len(best_selected) - 30} 个")

    # 分析答案相关 tokens
    print(f"\n答案关键词: {case['ground_truth']}")
    answer_keywords = [w.lower() for w in case['ground_truth'].split() if len(w) > 2]
    print(f"提取的关键词: {answer_keywords}")

    # 对比最差层
    worst_layer = sorted_by_coverage[-1][0]
    print(f"\n\n最差层 (Layer {worst_layer}) 详细分析:")
    print("-"*50)
    worst_info = layer_coverages[worst_layer]
    worst_stats = layer_stats[worst_layer]
    print(f"覆盖率: {worst_info['coverage']*100:.1f}%, 熵={worst_stats['entropy']:.3f}, Gini={worst_stats['gini']:.3f}")

    worst_selected = worst_info['selected']
    print(f"选中的 tokens ({len(worst_selected)} 个):")
    selected_tokens = []
    for pos in worst_selected[:30]:
        tok = tokenizer.decode([input_ids_cpu[pos]])
        selected_tokens.append(f"'{tok}'")
    print("  " + ", ".join(selected_tokens))

    # 保存结果
    results = {
        'case': case,
        'doc_len': doc_len,
        'rate': rate,
        'layer_coverages': {str(k): {'coverage': v['coverage'], 'covered': v['covered'], 'answer_positions': v['answer_positions']}
                           for k, v in layer_coverages.items()},
        'layer_stats': {str(k): v for k, v in layer_stats.items()},
        'best_layers': [{'layer': l, 'coverage': layer_coverages[l]['coverage'], 'stats': layer_stats[l]}
                       for l, _ in sorted_by_coverage[:5]],
        'entropy_selected_layers': [l for l, _ in sorted_by_entropy[:4]]
    }

    with open('layer_analysis_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n\n结果已保存到 layer_analysis_results.json")


if __name__ == '__main__':
    main()
