#!/usr/bin/env python3
"""
完整分析 3B vs 7B DraftModel 流程差异
使用与 test_fusionrag_reflect.py 完全相同的 DraftModel 流程
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
    smart_query_selection,
    find_connected_components
)


def analyze_attention_distribution(attn, name=""):
    """分析 attention 分布特征"""
    attn_np = attn.cpu().numpy() if isinstance(attn, torch.Tensor) else attn

    # 基本统计
    mean_val = np.mean(attn_np)
    std_val = np.std(attn_np)
    max_val = np.max(attn_np)

    # Gini 系数
    sorted_attn = np.sort(attn_np)
    n = len(sorted_attn)
    cumsum = np.cumsum(sorted_attn)
    gini = (2 * np.sum((np.arange(1, n+1) * sorted_attn)) / (n * np.sum(sorted_attn))) - (n + 1) / n

    # 熵
    attn_norm = attn_np / (attn_np.sum() + 1e-10)
    entropy = -np.sum(attn_norm * np.log(attn_norm + 1e-10))

    # 高 attention 位置数量
    threshold = mean_val + 0.5 * std_val
    high_attn_count = np.sum(attn_np > threshold)

    return {
        'mean': mean_val,
        'std': std_val,
        'max': max_val,
        'gini': gini,
        'entropy': entropy,
        'high_attn_count': high_attn_count,
        'threshold': threshold
    }


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
        return 0, [], set()

    selected_set = set(selected_positions)
    covered = set(answer_positions) & selected_set
    coverage = len(covered) / len(answer_positions)

    return coverage, answer_positions, covered


def get_token_texts(positions, input_ids, tokenizer, max_show=20):
    """获取位置对应的 token 文本"""
    tokens = []
    for pos in positions[:max_show]:
        if pos < len(input_ids):
            tok = tokenizer.decode([input_ids[pos]]).replace('\n', '\\n')
            tokens.append(f"'{tok}'")
    return tokens


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
    query_start = total_len - 30  # 粗略估计 query 起始位置
    doc_len = query_start - system_len

    return {
        'input_ids': input_ids,
        'system_len': system_len,
        'doc_len': doc_len,
        'query_start': query_start,
        'ground_truth': case['ground_truth']
    }


def run_draftmodel_pipeline(model, input_ids, system_len, doc_len, query_start, rate, device, model_name):
    """运行完整的 DraftModel 流程，返回每个步骤的详细信息"""

    results = {'model_name': model_name}

    # Step 1: 计算 attention
    print(f"\n  [{model_name}] Step 1: 计算 attention...")
    draft_attention = compute_draft_model_attention(model, input_ids, query_start, device)

    results['num_layers_computed'] = len(draft_attention)
    results['layers_computed'] = list(draft_attention.keys())

    # Step 2: 收集各层的 query→doc attention
    doc_start = system_len
    doc_end = system_len + doc_len

    layer_attention_dict = {}
    for layer_idx, layer_attn in draft_attention.items():
        query_to_doc = layer_attn[:, :, doc_start:doc_end]
        doc_attention_avg = query_to_doc.mean(axis=(0, 1))
        layer_attention_dict[layer_idx] = torch.tensor(doc_attention_avg, device=device)

    # Step 3: 熵选层
    print(f"  [{model_name}] Step 2: 熵选层...")
    active_layers, layer_entropy = entropy_layer_selection(
        layer_attention_dict, top_k=4, return_entropy=True
    )

    results['entropy_selected_layers'] = active_layers
    results['layer_entropies'] = {str(k): v for k, v in layer_entropy.items()}
    print(f"    选择的层: {active_layers}")
    print(f"    层熵值: {[f'L{l}:{layer_entropy[l]:.3f}' for l in active_layers]}")

    # Step 4: 聚合 attention
    layer_attentions = [layer_attention_dict[idx] for idx in active_layers]
    multi_layer_attn = torch.stack(layer_attentions).mean(dim=0)

    # 分析聚合后的 attention 分布
    attn_stats = analyze_attention_distribution(multi_layer_attn, model_name)
    results['attention_stats'] = attn_stats
    print(f"    Attention 分布: mean={attn_stats['mean']:.4f}, std={attn_stats['std']:.4f}, "
          f"gini={attn_stats['gini']:.3f}, entropy={attn_stats['entropy']:.3f}")
    print(f"    高 attention 位置数: {attn_stats['high_attn_count']} (阈值>{attn_stats['threshold']:.4f})")

    # Step 5: 连通分量分析
    attn_np = multi_layer_attn.cpu().numpy()
    threshold = attn_stats['threshold']
    high_attn_positions = list(np.where(attn_np > threshold)[0])
    components = find_connected_components(high_attn_positions, max_gap=2)

    results['num_components'] = len(components)
    results['component_sizes'] = [len(c) for c in components]
    print(f"    连通分量数: {len(components)}, 大小分布: {sorted(results['component_sizes'], reverse=True)[:10]}")

    # Step 6: smart_query_selection
    print(f"  [{model_name}] Step 3: smart_query_selection...")
    selected_positions = smart_query_selection(
        attention_scores=multi_layer_attn,
        doc_len=doc_len,
        target_ratio=rate,
        system_len=system_len,
        device=device
    )

    results['selected_positions'] = selected_positions
    results['num_selected'] = len(selected_positions)
    results['actual_rate'] = len(selected_positions) / doc_len
    print(f"    选择了 {len(selected_positions)} 个 tokens ({results['actual_rate']*100:.1f}%)")

    # 保存 attention 用于后续对比
    results['multi_layer_attn'] = multi_layer_attn.cpu()
    results['high_attn_positions'] = high_attn_positions

    return results


def compare_selections(results_3b, results_7b, input_ids, tokenizer, doc_start, doc_end, answer_text):
    """对比 3B 和 7B 的选择差异"""

    sel_3b = set(results_3b['selected_positions'])
    sel_7b = set(results_7b['selected_positions'])

    common = sel_3b & sel_7b
    only_3b = sel_3b - sel_7b
    only_7b = sel_7b - sel_3b

    overlap_ratio = len(common) / len(sel_3b | sel_7b) if sel_3b | sel_7b else 0

    print(f"\n{'='*70}")
    print("选择对比分析")
    print(f"{'='*70}")
    print(f"  3B 选择数: {len(sel_3b)}")
    print(f"  7B 选择数: {len(sel_7b)}")
    print(f"  共同选择: {len(common)} ({overlap_ratio*100:.1f}% 重叠)")
    print(f"  仅 3B 选择: {len(only_3b)}")
    print(f"  仅 7B 选择: {len(only_7b)}")

    # 计算答案覆盖率
    cov_3b, answer_pos, covered_3b = compute_answer_coverage(
        results_3b['selected_positions'], input_ids, answer_text, tokenizer, doc_start, doc_end
    )
    cov_7b, _, covered_7b = compute_answer_coverage(
        results_7b['selected_positions'], input_ids, answer_text, tokenizer, doc_start, doc_end
    )

    print(f"\n  答案覆盖率:")
    print(f"    3B: {cov_3b*100:.1f}% ({len(covered_3b)}/{len(answer_pos)})")
    print(f"    7B: {cov_7b*100:.1f}% ({len(covered_7b)}/{len(answer_pos)})")

    # 分析仅 7B 选中的 tokens
    only_7b_tokens = get_token_texts(sorted(only_7b), input_ids, tokenizer, max_show=15)
    only_3b_tokens = get_token_texts(sorted(only_3b), input_ids, tokenizer, max_show=15)

    print(f"\n  仅 7B 选择的 tokens (前15):")
    print(f"    {', '.join(only_7b_tokens)}")
    print(f"\n  仅 3B 选择的 tokens (前15):")
    print(f"    {', '.join(only_3b_tokens)}")

    # 分析 7B 选中但 3B 未选中的答案相关 tokens
    answer_only_7b = covered_7b - covered_3b
    if answer_only_7b:
        answer_tokens = get_token_texts(sorted(answer_only_7b), input_ids, tokenizer, max_show=10)
        print(f"\n  7B 选中但 3B 未选中的答案 tokens:")
        print(f"    {', '.join(answer_tokens)}")

    return {
        'overlap_ratio': overlap_ratio,
        'common_count': len(common),
        'only_3b_count': len(only_3b),
        'only_7b_count': len(only_7b),
        'coverage_3b': cov_3b,
        'coverage_7b': cov_7b,
        'answer_positions_count': len(answer_pos)
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

    print("\n加载 3B 模型...")
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

    all_comparisons = []

    # 分析前几个差异案例
    for i, case in enumerate(diff_cases[:3]):
        print(f"\n{'#'*80}")
        print(f"# 案例 {i+1}: {case['sub_question']}")
        print(f"# 答案: {case['ground_truth']}")
        print(f"# 7B 预测: {case['predicted_7b']}")
        print(f"# 3B 预测: {case['predicted_3b']}")
        print(f"{'#'*80}")

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

        print(f"\n文档长度: {doc_len} tokens, 选择比例: {rate}")

        # 运行 3B 流程
        results_3b = run_draftmodel_pipeline(
            model_3b, input_ids, system_len, doc_len, query_start, rate, device, "3B"
        )

        torch.cuda.empty_cache()

        # 运行 7B 流程
        results_7b = run_draftmodel_pipeline(
            model_7b, input_ids, system_len, doc_len, query_start, rate, device, "7B"
        )

        torch.cuda.empty_cache()

        # 对比分析
        input_ids_cpu = input_ids[0].cpu().tolist()
        comparison = compare_selections(
            results_3b, results_7b, input_ids_cpu, tokenizer,
            doc_start, doc_end, case['ground_truth']
        )

        # 对比熵选层差异
        print(f"\n熵选层对比:")
        print(f"  3B 选择的层: {results_3b['entropy_selected_layers']}")
        print(f"  7B 选择的层: {results_7b['entropy_selected_layers']}")

        # 对比 attention 分布
        print(f"\nAttention 分布对比:")
        print(f"  3B: gini={results_3b['attention_stats']['gini']:.3f}, "
              f"entropy={results_3b['attention_stats']['entropy']:.3f}, "
              f"high_attn={results_3b['attention_stats']['high_attn_count']}")
        print(f"  7B: gini={results_7b['attention_stats']['gini']:.3f}, "
              f"entropy={results_7b['attention_stats']['entropy']:.3f}, "
              f"high_attn={results_7b['attention_stats']['high_attn_count']}")

        comparison['case_idx'] = i + 1
        comparison['question'] = case['sub_question']
        all_comparisons.append(comparison)

    # 汇总
    print(f"\n{'='*80}")
    print("汇总统计")
    print(f"{'='*80}")

    avg_overlap = np.mean([c['overlap_ratio'] for c in all_comparisons])
    avg_cov_3b = np.mean([c['coverage_3b'] for c in all_comparisons])
    avg_cov_7b = np.mean([c['coverage_7b'] for c in all_comparisons])

    print(f"平均重叠率: {avg_overlap*100:.1f}%")
    print(f"平均答案覆盖率 - 3B: {avg_cov_3b*100:.1f}%, 7B: {avg_cov_7b*100:.1f}%")
    print(f"覆盖率差距: {(avg_cov_7b - avg_cov_3b)*100:.1f}%")

    # 保存结果
    with open('3b_vs_7b_pipeline_analysis.json', 'w') as f:
        json.dump(all_comparisons, f, indent=2, ensure_ascii=False)

    print(f"\n详细结果已保存到 3b_vs_7b_pipeline_analysis.json")


if __name__ == '__main__':
    main()
