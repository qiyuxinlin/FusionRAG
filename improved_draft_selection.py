#!/usr/bin/env python3
"""
改进的 DraftModel token 选择算法
目标：让 3B 模型在更低的重算比例上达到更好的效果

问题分析：
1. 3B 模型的 attention 过度集中在格式化 token (如 <|im_start|>, Documents: 等)
2. 3B 选择的 token 位置偏向输入开头，而非语义相关内容

改进方案：
1. 过滤格式化 token - 排除 special tokens 和固定格式的 token
2. 位置偏置修正 - 对开头位置的 token 添加惩罚
3. 文档内容聚焦 - 只在实际文档内容区域选择
4. Query 相关性加权 - 用 query 与 token 的相关性加权
"""
import sys
sys.path.insert(0, '/mnt/data/wjh/FusionRAG')

import torch
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc

# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained('/mnt/data/models/Qwen2.5-7B-Instruct', trust_remote_code=True)

# 需要过滤的格式化 token patterns
FILTER_TOKENS = {
    '<|im_start|>', '<|im_end|>', 'user', 'assistant', 'system',
    'Documents', 'Document', 'Question', ':', '\n', '\n\n',
}

def get_format_token_ids(tokenizer):
    """获取格式化 token 的 ID 集合"""
    format_ids = set()
    for token in FILTER_TOKENS:
        ids = tokenizer.encode(token, add_special_tokens=False)
        format_ids.update(ids)
    # 添加 special tokens
    if tokenizer.bos_token_id:
        format_ids.add(tokenizer.bos_token_id)
    if tokenizer.eos_token_id:
        format_ids.add(tokenizer.eos_token_id)
    if tokenizer.pad_token_id:
        format_ids.add(tokenizer.pad_token_id)
    return format_ids

FORMAT_TOKEN_IDS = get_format_token_ids(tokenizer)
print(f"格式化 token IDs 数量: {len(FORMAT_TOKEN_IDS)}")

def compute_position_penalty(doc_len, decay_rate=0.1):
    """
    计算位置惩罚因子
    开头的 token 会有惩罚，越往后惩罚越小

    Args:
        doc_len: 文档长度
        decay_rate: 衰减速率，越大惩罚越强

    Returns:
        position_weights: (doc_len,) 位置权重，1.0 表示无惩罚
    """
    positions = torch.arange(doc_len, dtype=torch.float32)
    # 使用 sigmoid 函数，开头惩罚大，后面接近 1
    # 前 10% 的位置会有明显惩罚
    threshold = doc_len * 0.1
    weights = torch.sigmoid((positions - threshold) * decay_rate)
    # 归一化到 [0.5, 1.0] 范围，避免完全屏蔽开头
    weights = 0.5 + 0.5 * weights
    return weights

def improved_token_selection_v1(attention_scores, input_ids, doc_start, doc_end, rate,
                                 filter_format_tokens=True,
                                 use_position_penalty=True,
                                 position_decay=0.05):
    """
    改进版 token 选择算法 V1: 过滤格式 token + 位置惩罚

    Args:
        attention_scores: (doc_len,) attention 分数
        input_ids: 完整输入的 token IDs
        doc_start: 文档开始位置
        doc_end: 文档结束位置
        rate: 选择比例
        filter_format_tokens: 是否过滤格式化 token
        use_position_penalty: 是否使用位置惩罚
        position_decay: 位置惩罚的衰减率

    Returns:
        selected_positions: 选中的位置列表 (相对于 doc_start)
    """
    doc_len = doc_end - doc_start
    scores = attention_scores.clone()

    # 1. 过滤格式化 token
    if filter_format_tokens:
        doc_ids = input_ids[doc_start:doc_end]
        for i, token_id in enumerate(doc_ids):
            if token_id.item() in FORMAT_TOKEN_IDS:
                scores[i] = 0.0

    # 2. 应用位置惩罚
    if use_position_penalty:
        position_weights = compute_position_penalty(doc_len, position_decay)
        scores = scores * position_weights.to(scores.device)

    # 3. 选择 top-k
    num_select = max(1, int(doc_len * rate))
    _, indices = torch.topk(scores, min(num_select, (scores > 0).sum().item()))

    # 转换为绝对位置
    selected_positions = (indices + doc_start).sort()[0].tolist()

    return selected_positions

def improved_token_selection_v2(attention_scores, input_ids, doc_start, doc_end, rate,
                                 query_attention=None,
                                 query_weight=0.3):
    """
    改进版 token 选择算法 V2: 结合 query attention 加权

    如果有 query attention，则结合两者：
    final_score = (1 - query_weight) * doc_attention + query_weight * query_attention
    """
    doc_len = doc_end - doc_start
    scores = attention_scores.clone()

    # 过滤格式化 token
    doc_ids = input_ids[doc_start:doc_end]
    for i, token_id in enumerate(doc_ids):
        if token_id.item() in FORMAT_TOKEN_IDS:
            scores[i] = 0.0

    # 位置惩罚
    position_weights = compute_position_penalty(doc_len, 0.05)
    scores = scores * position_weights.to(scores.device)

    # 结合 query attention
    if query_attention is not None:
        query_scores = query_attention.clone()
        # 归一化
        if scores.sum() > 0:
            scores = scores / scores.sum()
        if query_scores.sum() > 0:
            query_scores = query_scores / query_scores.sum()
        scores = (1 - query_weight) * scores + query_weight * query_scores

    # 选择 top-k
    num_select = max(1, int(doc_len * rate))
    _, indices = torch.topk(scores, min(num_select, (scores > 0).sum().item()))
    selected_positions = (indices + doc_start).sort()[0].tolist()

    return selected_positions

def improved_token_selection_v3(attention_scores, input_ids, doc_start, doc_end, rate,
                                 min_gap=3):
    """
    改进版 token 选择算法 V3: 过滤 + 位置惩罚 + 连通性分析

    在选择时，优先选择连续区域，避免选择孤立的 token
    """
    doc_len = doc_end - doc_start
    scores = attention_scores.clone()

    # 过滤格式化 token
    doc_ids = input_ids[doc_start:doc_end]
    for i, token_id in enumerate(doc_ids):
        if token_id.item() in FORMAT_TOKEN_IDS:
            scores[i] = 0.0

    # 位置惩罚
    position_weights = compute_position_penalty(doc_len, 0.05)
    scores = scores * position_weights.to(scores.device)

    # 连通性分析：高分 token 附近的 token 也给予加分
    # 使用一个小的卷积核来平滑分数
    if doc_len > min_gap * 2:
        kernel_size = min_gap * 2 + 1
        kernel = torch.ones(kernel_size) / kernel_size
        # 手动卷积
        padded = torch.nn.functional.pad(scores.unsqueeze(0).unsqueeze(0),
                                         (min_gap, min_gap), mode='replicate')
        smoothed = torch.nn.functional.conv1d(padded, kernel.unsqueeze(0).unsqueeze(0).to(scores.device))
        smoothed = smoothed.squeeze()
        # 结合原始分数和平滑分数
        scores = 0.7 * scores + 0.3 * smoothed

    # 选择 top-k
    num_select = max(1, int(doc_len * rate))
    _, indices = torch.topk(scores, min(num_select, (scores > 0).sum().item()))
    selected_positions = (indices + doc_start).sort()[0].tolist()

    return selected_positions


def test_improvement_on_case(case_data, model, tokenizer, device, rate=0.2):
    """在单个案例上测试改进效果"""
    input_ids = case_data['input_ids'].to(device)
    system_len = case_data['system_len']
    doc_len = case_data['doc_len']
    query_start = case_data['query_start']

    # 计算 attention
    model.eval()
    num_layers = model.config.num_hidden_layers

    layer_attentions = {}

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            output_attentions=True,
            use_cache=False
        )

        attentions = outputs.attentions
        # 取最后 4 层
        for layer_idx in range(num_layers - 4, num_layers):
            attn = attentions[layer_idx][0, :, query_start:, :query_start]
            attn_avg = attn.mean(dim=(0, 1))  # 对 heads 和 query 求平均
            layer_attentions[layer_idx] = attn_avg.cpu()

    # 熵选层
    layer_entropies = {}
    for layer_idx, attn in layer_attentions.items():
        attn_norm = attn / (attn.sum() + 1e-10)
        entropy = -torch.sum(attn_norm * torch.log(attn_norm + 1e-10))
        layer_entropies[layer_idx] = entropy.item()

    sorted_layers = sorted(layer_entropies.items(), key=lambda x: x[1])
    selected_layers = [l[0] for l in sorted_layers[:4]]

    # 聚合选定层的 attention
    all_attn = []
    for layer_idx in selected_layers:
        all_attn.append(layer_attentions[layer_idx])
    combined_attn = torch.stack(all_attn).mean(dim=0)

    # 只取文档部分
    doc_attn = combined_attn[system_len:system_len + doc_len]

    results = {}

    # 原始方法
    num_select = max(1, int(doc_len * rate))
    _, indices = torch.topk(doc_attn, num_select)
    results['original'] = (indices + system_len).sort()[0].tolist()

    # 改进方法 V1
    results['v1_filter_only'] = improved_token_selection_v1(
        doc_attn, input_ids[0], system_len, system_len + doc_len, rate,
        filter_format_tokens=True, use_position_penalty=False
    )

    results['v1_position_only'] = improved_token_selection_v1(
        doc_attn, input_ids[0], system_len, system_len + doc_len, rate,
        filter_format_tokens=False, use_position_penalty=True
    )

    results['v1_both'] = improved_token_selection_v1(
        doc_attn, input_ids[0], system_len, system_len + doc_len, rate,
        filter_format_tokens=True, use_position_penalty=True
    )

    # 改进方法 V3 (带连通性)
    results['v3_connected'] = improved_token_selection_v3(
        doc_attn, input_ids[0], system_len, system_len + doc_len, rate
    )

    return results, input_ids[0].cpu()


def analyze_selection_quality(selected_positions, input_ids, answer_text, tokenizer, doc_start, doc_end):
    """分析选择质量"""
    # 统计格式 token 数量
    format_count = 0
    for pos in selected_positions:
        if pos < len(input_ids) and input_ids[pos].item() in FORMAT_TOKEN_IDS:
            format_count += 1

    # 统计答案覆盖
    answer_keywords = [w.lower() for w in answer_text.split() if len(w) > 3 and w.isalpha()]
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
        'format_token_ratio': format_count / len(selected_positions) if selected_positions else 0,
        'answer_coverage': answer_covered / len(answer_positions) if answer_positions else 0,
        'total_selected': len(selected_positions),
        'format_count': format_count,
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

    # 加载系统提示
    config_path = "./config/dataset2prompt_few-shot.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    system_prompt = config["system_prompt"]["Qwen3"]["2wikimqa"]

    print("加载 3B 模型进行测试...")
    model = AutoModelForCausalLM.from_pretrained(
        '/mnt/data/models/Qwen2.5-3B-Instruct',
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="eager"
    )

    # 准备测试案例
    from analyze_token_selection_diff import prepare_case_input, find_case_data

    all_results = {
        'original': {'format_ratio': [], 'answer_coverage': []},
        'v1_filter_only': {'format_ratio': [], 'answer_coverage': []},
        'v1_position_only': {'format_ratio': [], 'answer_coverage': []},
        'v1_both': {'format_ratio': [], 'answer_coverage': []},
        'v3_connected': {'format_ratio': [], 'answer_coverage': []},
    }

    for i, case in enumerate(diff_cases[:5]):
        print(f"\n分析案例 {i+1}: {case['sub_question'][:50]}...")

        case_data = prepare_case_input(case, data, tokenizer, system_prompt)
        if case_data is None:
            continue

        results, input_ids = test_improvement_on_case(case_data, model, tokenizer, device, rate=0.2)

        print(f"  文档长度: {case_data['doc_len']} tokens")

        for method, positions in results.items():
            quality = analyze_selection_quality(
                positions, input_ids, case['ground_truth'], tokenizer,
                case_data['system_len'], case_data['system_len'] + case_data['doc_len']
            )
            all_results[method]['format_ratio'].append(quality['format_token_ratio'])
            all_results[method]['answer_coverage'].append(quality['answer_coverage'])

            print(f"  {method:20s}: 格式token {quality['format_count']:3d}/{quality['total_selected']} ({quality['format_token_ratio']*100:5.1f}%), "
                  f"答案覆盖 {quality['answer_covered']:2d}/{quality['answer_positions']} ({quality['answer_coverage']*100:5.1f}%)")

    # 汇总
    print("\n" + "="*80)
    print("汇总统计")
    print("="*80)
    print(f"{'方法':20s} | {'平均格式token比例':15s} | {'平均答案覆盖率':15s}")
    print("-"*60)
    for method in all_results:
        avg_format = np.mean(all_results[method]['format_ratio'])
        avg_coverage = np.mean(all_results[method]['answer_coverage'])
        print(f"{method:20s} | {avg_format*100:14.1f}% | {avg_coverage*100:14.1f}%")

if __name__ == '__main__':
    main()
