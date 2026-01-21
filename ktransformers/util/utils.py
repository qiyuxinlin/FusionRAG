#!/usr/bin/env python
# coding=utf-8
'''
Description  :
Author       : Boxin Zhang, Azure-Tang
Version      : 0.1.0
Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
'''
import copy

import torch
from torch import nn
import itertools
import time
import enum
import re
import math
import string
import json
import collections
import numpy as np
from ktransformers.util.run_ppr import personalized_pagerank, get_top_tokens, highlight_tokens_compare, topk_position_dispersion, OnlineEncoder, calculate_vector_set_similarity
from ktransformers.models.custom_cache import StaticCache
from ktransformers.util.cuda_graph_runner import CUDAGraphRunner
from ktransformers.util.textstream import TextStreamer
from transformers import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    MinPLogitsWarper,
    TypicalLogitsWarper,
    EpsilonLogitsWarper,
    EtaLogitsWarper,
    GenerationConfig,
    AutoTokenizer
)
from rouge import Rouge
from filelock import FileLock


# ============================================================
# Smart Query Selection 辅助函数
# ============================================================

def find_connected_components(positions, max_gap=2, within=False):
    """
    找到位置列表中的连通分量（相邻 token 群组）

    Args:
        positions: 位置列表
        max_gap: 最大允许的间隔，小于等于这个间隔的位置被认为是连通的

    Returns:
        List of lists, 每个子列表是一个连通分量
    """
    if len(positions) == 0:
        return []

    positions = sorted(positions)
    components = []
    current_component = [positions[0]]
    connect_positions = set()

    for i in range(1, len(positions)):
        if positions[i] - positions[i-1] <= max_gap:
            if not within:
                current_component.append(positions[i])
            else:
                current_component.extend([p for p in range(positions[i - 1] + 1, positions[i] + 1)])
                for c in range(positions[i - 1] + 1, positions[i]):
                    connect_positions.add(c)
        else:
            components.append(current_component)
            current_component = [positions[i]]

    components.append(current_component)
    return components, list(connect_positions)

def find_outliers_zscore(data, threshold=2):
    """
    使用Z-score方法检测离群点
    threshold: 阈值，通常取2、2.5或3
    """
    mean = np.mean(data)
    std = np.std(data)
    z_scores = [(x - mean) / std for x in data]

    outliers = []
    for i, z in enumerate(z_scores):
        if abs(z) > threshold:
            outliers.append((i, data[i]))

    return outliers

def smart_query_selection(attention_scores, doc_len, target_ratio, system_len, device='cpu', smarter=False, eigenvalue=None, tokenizer=None, input_tokens=None, similarity=0.0):
    """
    Smart Query Selection: 使用连通性分析确保相关 token 群组被完整选中

    Args:
        attention_scores: torch.Tensor, shape [doc_len], 每个位置的 attention 分数
        doc_len: 文档长度
        target_ratio: 目标选择比例
        system_len: system prompt 长度
        device: 计算设备

    Returns:
        List of selected positions (global indices, including system_len offset)
    """
    if isinstance(attention_scores, torch.Tensor):
        attention_scores = attention_scores.float().cpu().numpy()

    target_count = int(doc_len * target_ratio)

    # Step 1: 找到高 attention 位置
    mean_attn = np.mean(attention_scores)
    std_attn = np.std(attention_scores)
    threshold = mean_attn + 0.25 * std_attn ## 1/4 std

    high_attn_positions = list(np.where(attention_scores > threshold)[0])
    if eigenvalue is not None:
        eigenvalue["attention_scores"] = [float(t.item()) for t in attention_scores]

    # 1. max_gap=5, min_len=3
    # 2. max_gap=20, min_len=20
    # 2. max_gap=min_len=5, 0.5 * std_attn, min_chosen_weight = 0.4/0.2: this is the current best, but will leftout some important infos
    # 3. max_gap=min_len=5, 0.1 * std_attn, min_chosen_weight = 0.2: let more relevant data be found, but cut the irelevant
    # 4. max_gap=min_len=5, 0.25 * std_attn, min_chosen_weight = 0.0002
    # 5. max_gap=min_len=5, 0.25 * std_attn, min_chosen_weight = 0.01
    # Step 2: 连通分量分析
    connect_positions = []
    if smarter:
        max_gap = 5
        min_len = max_gap
        min_chosen_weight = 0.2
        if eigenvalue is not None:
            eigenvalue["max_gap"] = max_gap
            eigenvalue["min_len"] = min_len
            eigenvalue["min_chosen_weight"] = min_chosen_weight
        components, connect_positions = find_connected_components(high_attn_positions, max_gap=max_gap, within=True)
    else:
        max_gap = 5 ## mengyao_debug I changed this.
        components, _ = find_connected_components(high_attn_positions, max_gap=max_gap, within=False)


    # Step 3: 计算每个分量的总 attention
    component_scores = []
    for comp in components:
        total_score = sum(attention_scores[p] for p in comp if p not in connect_positions)
        component_scores.append((comp, total_score))

    # Step 4: 按总 attention 排序
    component_scores.sort(key=lambda x: x[1], reverse=True)
    component_scores_ = copy.deepcopy(component_scores)

    if smarter:
        max_score = 1e-8
        all_tokens = []
        if tokenizer is not None:
            input_tokens_ = torch.cat(input_tokens)
            for cs, score in component_scores:
                if len(cs) > min_len:
                    if max_score == 1e-8:
                        max_score = score # set max score
                input_str = tokenizer.decode(input_tokens_[cs], skip_special_tokens=True)
                prefix = ""
                if len(cs) >= min_len and score/max_score > min_chosen_weight:
                    prefix = "【Chosen】"
                print(f"{prefix} \033[31m{input_str}\033[0m, score={score} len={len(cs)}")
                #for debug
                chosen = False
                if len(cs) >= min_len and score/max_score > min_chosen_weight:
                    chosen = True
                all_tokens.append({
                    "str": input_str,
                    "score": float(score),
                    "len": len(cs),
                    "chosen": chosen
                })
        eigenvalue["chosen_tokens"] = all_tokens
        # mengyao_debug: make sure the token we select is not a single token and has some weights on it.
        component_scores = [cs for cs in component_scores if len(cs[0]) >= min_len and cs[1]/max_score > min_chosen_weight]


    # for cs in component_scores:
    #     print(f"component len={len(cs[0])}, score={cs[1]}")
    if eigenvalue != None:
        eigenvalue["components"] = len(component_scores)

    # Step 5: 贪心选择分量 + 上下文扩展 (±1)
    selected = set()
    selected_reserved = set()

    for comp, total_score in component_scores:
        # 扩展分量边界 (±1)
        extended_comp = set()
        for p in comp:
            for offset in range(-1, 2):
                new_p = p + offset
                if 0 <= new_p < doc_len:
                    extended_comp.add(new_p)

        # 检查是否会超过目标 (允许 10% 余量)
        new_positions = extended_comp - selected
        if len(selected) + len(new_positions) <= target_count * 1.1:
            selected.update(extended_comp)

    for cs in component_scores_:
        if cs not in component_scores:
            comp, total_score = cs
            extended_comp = set()
            for p in comp:
                extended_comp.add(p)
            selected_reserved.update(extended_comp)

    # Step 6: 补充到目标数量
    if not smarter:
        if len(selected) < target_count:
            sorted_indices = np.argsort(attention_scores)[::-1]
            for pos in sorted_indices:
                if pos not in selected:
                    selected.add(int(pos))
                    if len(selected) >= target_count:
                        break

    # Step 7: 如果超过目标，移除最低分的位置
    while len(selected) > target_count:
        min_pos = min(selected, key=lambda p: attention_scores[p])
        selected.remove(min_pos)

    # 转换为全局索引 (加上 system_len 偏移)
    selected_global = [p + system_len for p in sorted(selected)]
    selected_global_reserved = [p + system_len for p in sorted(selected_reserved) if p not in connect_positions]
    # selected_global_reserved = [p + system_len for p in sorted(selected_reserved)]

    if smarter:
        return selected_global, selected_global_reserved

    return selected_global

# def smarter_query_selection(attention_scores, doc_len, target_ratio, system_len, device='cpu'):
#     """
#     Smart Query Selection: 使用连通性分析确保相关 token 群组被完整选中
#
#     Args:
#         attention_scores: torch.Tensor, shape [doc_len], 每个位置的 attention 分数
#         doc_len: 文档长度
#         target_ratio: 目标选择比例
#         system_len: system prompt 长度
#         device: 计算设备
#
#     Returns:
#         List of selected positions (global indices, including system_len offset)
#     """
#     if isinstance(attention_scores, torch.Tensor):
#         attention_scores = attention_scores.float().cpu().numpy()
#
#     target_count = int(doc_len * target_ratio)
#
#     # Step 1: 找到高 attention 位置
#     mean_attn = np.mean(attention_scores)
#     std_attn = np.std(attention_scores)
#     threshold = mean_attn + 0.5 * std_attn
#
#     high_attn_positions = list(np.where(attention_scores > threshold)[0])
#
#     # Step 2: 连通分量分析
#     components = find_connected_components(high_attn_positions, max_gap=20)
#     ## remove all short contexts.
#     components = [component for component in components if len(component) > 1]
#     #components = find_connected_components(high_attn_positions, max_gap=20)
#
#     # Step 3: 计算每个分量的总 attention
#     component_scores = []
#     for comp in components:
#         total_score = sum(attention_scores[p] for p in comp if p in high_attn_positions)
#         component_scores.append((comp, total_score))
#
#     # Step 4: 按总 attention 排序
#     component_scores.sort(key=lambda x: x[1], reverse=True)
#
#     # Step 5: 贪心选择分量 + 上下文扩展 (±1)
#     selected = set()
#     total_pieces = 0
#     # for comp, total_score in component_scores[:5]:
#     for comp, total_score in component_scores:
#         # 扩展分量边界 (±1)
#         extended_comp = set()
#         for p in comp:
#             extended_comp.add(p)
#
#         new_positions = extended_comp - selected
#         print(f"total_score={total_score}")
#         # drop too short pieces of enough info acquired.
#         if total_pieces>=2 and total_score/component_scores[0][1] < 0.5:
#             break
#         if len(selected) + len(new_positions) <= target_count * 4:
#             selected.update(extended_comp)
#             total_pieces += 1
#         else:
#             break
#
#
#
#     # # Step 6: 补充到目标数量
#     # if len(selected) < target_count:
#     #     sorted_indices = np.argsort(attention_scores)[::-1]
#     #     for pos in sorted_indices:
#     #         if pos not in selected:
#     #             selected.add(int(pos))
#     #             if len(selected) >= target_count:
#     #                 break
#
#     # Step 7: 如果超过目标，移除最低分的位置
#     # while len(selected) > target_count:
#     #     min_pos = min(selected, key=lambda p: attention_scores[p])
#     #     selected.remove(min_pos)
#
#     # outliers = find_outliers_zscore(list(selected))
#     # outliers_idx = [o[1] for o in outliers]
#     # selected = sorted(selected)
#     # selected = [i for i in selected if i not in outliers_idx]
#
#     # 转换为全局索引 (加上 system_len 偏移)
#     selected_global = [p + system_len for p in sorted(selected)]
#
#     return selected_global


def entropy_layer_selection(layer_attentions, top_k=4, return_entropy=False):
    """
    基于熵动态选择层（熵越低的层，attention 越集中，信息量可能越大）

    Args:
        layer_attentions: dict {layer_idx: attention_tensor [doc_len]}
                          或 list of (layer_idx, attention_tensor)
        top_k: 选择熵最低的 top_k 层
        return_entropy: 是否返回各层的熵值

    Returns:
        selected_layers: 选中的层索引列表
        layer_entropy: (可选) 各层的熵值字典
    """
    layer_entropy = {}

    # 处理不同输入格式
    if isinstance(layer_attentions, dict):
        items = layer_attentions.items()
    else:
        items = layer_attentions

    for layer_idx, attn in items:
        # 确保是 numpy array
        if isinstance(attn, torch.Tensor):
            attn = attn.cpu().float().numpy()

        # 归一化为概率分布
        p = attn / (attn.sum() + 1e-10)
        p = np.clip(p, 1e-10, 1.0)

        # 计算熵 H = -sum(p * log(p))
        entropy = -np.sum(p * np.log(p))
        layer_entropy[layer_idx] = entropy

    # 按熵值排序，选择熵最低的 top_k 层
    sorted_layers = sorted(layer_entropy.items(), key=lambda x: x[1])
    selected_layers = [layer_idx for layer_idx, _ in sorted_layers[:top_k]]

    if return_entropy:
        return selected_layers, layer_entropy
    return selected_layers


def compute_draft_model_attention(draft_model, input_ids, device="cuda:0"):
    """
    用 draft model 完整 prefill 获取 attention 分布

    Args:
        draft_model: 小模型
        input_ids: 输入 token ids [1, seq_len]
        device: 设备

    Returns:
        layer_attention_scores: {layer_idx: attention_matrix [num_heads, seq_len, seq_len]}
    """
    import torch.nn.functional as F

    seq_len = input_ids.shape[1]
    num_layers = draft_model.config.num_hidden_layers
    num_heads = draft_model.config.num_attention_heads
    num_kv_heads = draft_model.config.num_key_value_heads
    head_dim = draft_model.config.hidden_size // num_heads

    print(f"\n{'='*60}")
    print("Computing Draft Model Attention")
    print(f"{'='*60}")
    print(f"  Layers: {num_layers}, Heads: {num_heads}, Seq len: {seq_len}")

    layer_attention_scores = {}

    with torch.no_grad():
        inputs_embeds = draft_model.model.embed_tokens(input_ids.to(device))
        hidden_states = inputs_embeds
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

        # 获取 rotary_emb (兼容不同模型结构)
        if hasattr(draft_model.model, 'rotary_emb'):
            rotary_emb = draft_model.model.rotary_emb
            cos, sin = rotary_emb(hidden_states, position_ids)
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
            use_global_rope = True
        else:
            use_global_rope = False
            cos, sin = None, None

        for layer_idx in range(num_layers):
            layer = draft_model.model.layers[layer_idx]

            residual = hidden_states
            hidden_states = layer.input_layernorm(hidden_states)

            bsz, q_len, _ = hidden_states.size()

            # Q, K, V projections
            query_states = layer.self_attn.q_proj(hidden_states)
            key_states = layer.self_attn.k_proj(hidden_states)
            value_states = layer.self_attn.v_proj(hidden_states)

            # Reshape
            query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

            # Apply RoPE
            if not use_global_rope:
                cos, sin = layer.self_attn.rotary_emb(value_states, position_ids)
                cos = cos.unsqueeze(1)
                sin = sin.unsqueeze(1)
            query_states = (query_states * cos) + (rotate_half(query_states) * sin)
            key_states = (key_states * cos) + (rotate_half(key_states) * sin)

            # Expand K, V for GQA
            n_rep = num_heads // num_kv_heads
            key_states_expanded = key_states.repeat_interleave(n_rep, dim=1)
            value_states_expanded = value_states.repeat_interleave(n_rep, dim=1)

            # Compute attention
            attn_weights = torch.matmul(query_states.float(), key_states_expanded.float().transpose(2, 3)) / (head_dim ** 0.5)
            causal_mask = torch.triu(torch.ones(q_len, q_len, device=device), diagonal=1).bool()
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
            attn_weights = F.softmax(attn_weights, dim=-1)

            # 保存后半部分层的 attention
            if layer_idx >= num_layers // 2:
                layer_attention_scores[layer_idx] = attn_weights[0].cpu().float().numpy()

            # Continue forward
            attn_output = torch.matmul(attn_weights.to(value_states_expanded.dtype), value_states_expanded)
            attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
            attn_output = layer.self_attn.o_proj(attn_output)

            hidden_states = residual + attn_output

            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states

            if layer_idx % 8 == 0 or layer_idx == num_layers - 1:
                print(f"  Layer {layer_idx} done")

    print(f"Draft model attention computed")
    return layer_attention_scores


def prefill_and_save_kv_cache(model, tokenizer, past_key_values, inputs,
                          save_path='', example_id = 0, chunk_id = 0, system_len = 0, passage_len = 0, reprocess_method=None, device="cuda", device_map=None, hash_key="",
                          ):

    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch._dynamo.config.suppress_errors = True
    batch_size, seq_length = inputs.shape

    # Determine input device: use first GPU if device_map provided, otherwise use device
    input_device = f"cuda:{device_map['model.embed_tokens']}" if device_map is not None else device
    inputs = inputs.to(input_device)

    tokens = []
    for layer_idx in range(len(past_key_values.key_cache)):
        past_key_values.past_tokens[layer_idx] = 0
    with torch.no_grad():
        cache_position = torch.arange(seq_length, device=input_device)
        generated_ids = torch.zeros(
            batch_size, seq_length + 1, dtype=torch.int, device=input_device
        )
        generated_ids[:, cache_position] = inputs.to(input_device).to(torch.int)
        if past_key_values != None:
            past_key_values.cur_idx=cache_position
        start_time = time.time()


        inputs_embeds = model.model.embed_tokens(inputs).to(input_device)
        if reprocess_method == "Cache-Craft" and chunk_id != 0:
            passages_len = [system_len, passage_len]
            logits = model(
                inputs_embeds = inputs_embeds, cache_position=cache_position, past_key_values=past_key_values, return_dict=False, use_cache=True, reprocess_method=reprocess_method, passages_len=passages_len
            )[0][:,-1,:].unsqueeze(0).clone().to(input_device)
            cachecraft_score = past_key_values.importance_cache[-1] # [num_head, passage_len]
            cachecraft_score = torch.sum(cachecraft_score, dim=0)
            if hash_key != "":
                torch.save(cachecraft_score, f'{save_path}/cachecraftattn_{hash_key}.pt')
            else:
                torch.save(cachecraft_score, f'{save_path}/cachecraftattn_{example_id}_{chunk_id}.pt')
        else:
            logits = model(
                inputs_embeds = inputs_embeds, cache_position=cache_position, past_key_values=past_key_values, return_dict=False, use_cache=True
            )[0][:,-1,:].unsqueeze(0).clone().to(input_device)
        past_len = past_key_values.past_tokens[0]
        key_cache = []
        value_cache = []
        if chunk_id == 0:
            # Move to CPU to handle multi-GPU scenarios where different layers are on different devices
            key_cache = [past_key_values.key_cache[i][:,:,:past_len,:].cpu() for i in range(len(past_key_values.key_cache))]
            key_cache = torch.stack(key_cache)
            value_cache = [past_key_values.value_cache[i][:,:,:past_len,:].cpu() for i in range(len(past_key_values.value_cache))]
            value_cache = torch.stack(value_cache)
        else:
            # Move to CPU to handle multi-GPU scenarios where different layers are on different devices
            key_cache = [past_key_values.key_cache[i][:,:,system_len:system_len + passage_len,:].cpu() for i in range(len(past_key_values.key_cache))]
            key_cache = torch.stack(key_cache)
            value_cache = [past_key_values.value_cache[i][:,:,system_len:system_len + passage_len,:].cpu() for i in range(len(past_key_values.value_cache))]
            value_cache = torch.stack(value_cache)

        if hash_key != "":
            key_path = f'{save_path}/{hash_key}_key.pt'
            value_path = f'{save_path}/{hash_key}_value.pt'
            lock_path = f'{save_path}/{hash_key}.lock'
        else:
            key_path = f'{save_path}/{example_id}_{chunk_id}_key.pt'
            value_path = f'{save_path}/{example_id}_{chunk_id}_value.pt'
            lock_path = f'{save_path}/{example_id}_{chunk_id}.lock'
        # print(f'hashkey: {hash_key}, chunk_id: {chunk_id}')

        with FileLock(lock_path, timeout=60):
            # Double-check if file exists (another process might have created it)
            if not os.path.exists(key_path):
                torch.save(key_cache.clone(), key_path)
                torch.save(value_cache.clone(), value_path)
                # print(f'example_id: {example_id}, chunk_id: {chunk_id} (saved by current process)')
            else:
                ""
                # print(f'example_id: {example_id}, chunk_id: {chunk_id} (already exists, skipped)')
        return key_cache, value_cache

def decode_one_tokens(model, cur_token, position_ids, cache_position, past_key_values, logits_warper, inputs, rate=0.0, path=""):
    inputs_embeds = model.model.embed_tokens(cur_token)
    # with torch.cuda.stream(custom_stream):
    # path_ = path.split("question is")[1].split("?")[0].replace(" ", "")
    # import os
    # save_path = f'/data1/qy_tmp/xumengyao/attention/{path_}/{rate}'
    # os.makedirs(save_path, exist_ok=True)
    logits=model(inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                cache_position=cache_position,
                past_key_values=past_key_values,
                return_dict=False,
                 use_cache=True,
                 rate=rate,
                 )[0]
    if past_key_values != None:
        past_key_values.change_seq_length(1)
    #print(logits)
    next_token_scores = logits_warper(inputs, logits[:, -1, :])
    next_token = torch.argmax(next_token_scores, dim=-1)
    return next_token
# mistral 是这个函数，其他函数得考虑把这个函数换掉
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def prefill_with_cache_and_save_preprocess(model, tokenizer, past_key_values, passages,
                                           save_path='', example_id = 0, chunk_id=0, system_len=0,
                                           revert_rope=False, reprocess_method=None, device="cuda",
                                           device_map=None, hash_key=""):

    # load KV
    past_len = past_key_values.past_tokens[0]

    # Determine input device: use first GPU if device_map provided, otherwise use device
    input_device = f"cuda:{device_map['model.embed_tokens']}" if device_map is not None else device

    # prefill context
    inputs = passages[-1].unsqueeze(0).to(input_device)
    passage_len = passages[-1].shape[0]
    passages_len = [passages[i].shape[0] for i in range(len(passages))]
    batch_size, seq_length = inputs.shape
    cache_position = torch.arange(past_len,past_len + seq_length, device=input_device)
    # position_ids = torch.arange(system_len,system_len + seq_length, device='cuda').unsqueeze(0)
    generated_ids = torch.zeros(
        batch_size, past_len + seq_length + 1, dtype=torch.int, device=input_device
    )
    generated_ids[:, :past_len] = torch.cat(passages[:-1]).unsqueeze(0).to(input_device)
    generated_ids[:, cache_position] = inputs.to(input_device).to(torch.int)
    tokens = []
    with torch.no_grad():
        stream = TextStreamer(tokenizer)
        inputs_embeds = model.model.embed_tokens(inputs).to(input_device)
        if reprocess_method == "Cache-Craft":
            logits = model(
                inputs_embeds = inputs_embeds, cache_position=cache_position,
                past_key_values=past_key_values, return_dict=False, use_cache=True, reprocess_method=reprocess_method, passages_len=passages_len
            )[0][:,-1,:].unsqueeze(0).clone().to(input_device)
            cachecraft_score = past_key_values.importance_cache[-1] # [num_head, passage_len]
            cachecraft_score = torch.sum(cachecraft_score, dim=0)
            if hash_key != "":
                torch.save(cachecraft_score, f'{save_path}/cachecraftattn_{hash_key}.pt')
            else:
                torch.save(cachecraft_score, f'{save_path}/cachecraftattn_{example_id}_{chunk_id}.pt')
        else:
            logits = model(
                inputs_embeds = inputs_embeds, cache_position=cache_position,
                past_key_values=past_key_values, return_dict=False, use_cache=True
            )[0][:,-1,:].unsqueeze(0).clone().to(input_device)
    # Move to CPU to handle multi-GPU scenarios where different layers are on different devices
    key_cache = torch.stack([cache.cpu() for cache in past_key_values.key_cache])[:,:,:,past_len:past_len + passage_len,:]
    # Compute RoPE on CPU to match key_cache device
    position_ids = torch.full((1, key_cache[0].shape[2]), system_len - past_len, device='cpu')
    key_cache_for_rope = key_cache[0].to(input_device)
    try:
        cos, sin = model.model.layers[0].self_attn.rotary_emb(key_cache_for_rope, position_ids.to(input_device))
    except:
        cos, sin = model.model.rotary_emb(key_cache_for_rope, position_ids.to(input_device))
    # mistral 限定
    cos = cos.unsqueeze(1).cpu()
    sin = sin.unsqueeze(1).cpu()
    key_cache = (key_cache * cos) + (rotate_half(key_cache) * sin)
    if hash_key != "":
        torch.save(key_cache.clone(), f'{save_path}/{hash_key}_key.pt')
    else:
        torch.save(key_cache.clone(), f'{save_path}/{example_id}_{chunk_id}_key.pt')
    key_cache = None
    if "cuda" in input_device:
        torch.cuda.empty_cache()
    elif "npu" in input_device:
        torch.npu.empty_cache()
    # Move to CPU to handle multi-GPU scenarios where different layers are on different devices
    value_cache = torch.stack([cache.cpu() for cache in past_key_values.value_cache])[:,:,:,past_len:past_len + passage_len,:]
    if hash_key != "":
        torch.save(value_cache.clone(), f'{save_path}/{hash_key}_value.pt')
    else:
        torch.save(value_cache.clone(), f'{save_path}/{example_id}_{chunk_id}_value.pt')


def get_multilayer_attn_with_answer(passages, draft_model, draft_model_device, tokenizer, entropy_top_k=4):
    full_input = torch.cat(passages).unsqueeze(0).to(draft_model_device)
    query_start = sum([passage.shape[0] for passage in passages[:-1]])
    total_len = sum([passage.shape[0] for passage in passages])
    document_start = passages[0].shape[0]
    # 存储所有生成的token
    generated_tokens = []
    max_tokens = 50
    attention = None
    for i in range(max_tokens):  # 生成20个token
        inputs_embeds = draft_model.model.embed_tokens(full_input).to(draft_model_device)
        position_ids = torch.arange(full_input.shape[1], device=draft_model_device).unsqueeze(0)

        with torch.no_grad():
            result = draft_model(
                inputs_embeds=inputs_embeds,
                use_cache=False,
                position_ids=position_ids,
                output_attentions=False,
                return_dict=True
            )
            print("gen 1 token")

            logits = result.logits
            next_token_logits = logits[0, -1, :]
            token_id = torch.argmax(next_token_logits).item()
            if token_id == tokenizer.eos_token_id:
                max_tokens = i+1
                break
            generated_tokens.append(token_id)
            new_token_tensor = torch.tensor([[token_id]], device=draft_model_device)
            full_input = torch.cat([full_input, new_token_tensor], dim=1)

    with torch.no_grad():
        result = draft_model(
            inputs_embeds=inputs_embeds,
            use_cache=False,
            position_ids=position_ids,
            output_attentions=True,
            return_dict=True
        )
        attention = result.attentions

    output = tokenizer.decode(full_input[0])
    print(output)
    layer_attention_dict = {}
    for layer_idx, layer_attn in enumerate(attention):
        layer_attn = layer_attn.squeeze(dim=0)
        query_to_doc = layer_attn[:, total_len:total_len+max_tokens, document_start:query_start]
        doc_attention_avg = query_to_doc.mean(axis=(0, 1))  # [doc_len]
        layer_attention_dict[layer_idx] = torch.tensor(doc_attention_avg, device=draft_model_device)

    active_layers, layer_entropy = entropy_layer_selection(
        layer_attention_dict, top_k=entropy_top_k, return_entropy=True
    )
    layer_attentions = [layer_attention_dict[idx] for idx in active_layers]
    multi_layer_attn = torch.stack(layer_attentions).mean(dim=0)  # [doc_len]

    return generated_tokens, multi_layer_attn

def concentration_coefficient_v1(tensor: torch.Tensor, eps: float = 1e-8) -> float:
    mean = tensor.mean()
    std = tensor.std()

    # 变异系数 = std/mean，我们取其倒数
    if abs(mean) < eps:
        return 0.0  # 均值为0时无法计算

    cv = std / mean  # 变异系数
    concentration = 1.0 / (1.0 + abs(cv))  # 归一化到[0,1]

    return concentration.item()

def get_multilayer_attn(passages, draft_model, draft_model_device, entropy_top_k, draft_attention, query_start, system_len, doc_len, total_len, smarter=False):
    # 如果没有传入 draft_attention，需要用 draft_model 计算
    if draft_attention is None:
        if draft_model is None:
            raise ValueError("Either draft_model or draft_attention must be provided for DraftModel method")
        # 构建完整输入
        full_input = torch.cat(passages).unsqueeze(0).to(draft_model_device)
        draft_attention = compute_draft_model_attention(draft_model, full_input, draft_model_device)
    layer_attention_dict = {}
    doc_to_doc_attns = []
    for layer_idx, layer_attn in draft_attention.items():
        # layer_attn: [num_heads, seq_len, seq_len]
        query_to_doc = layer_attn[:, query_start:total_len, system_len:system_len + doc_len]
        doc_to_doc_attn = layer_attn[:, system_len:system_len + doc_len, system_len:system_len + doc_len]
        doc_to_doc_attn_avg = doc_to_doc_attn.mean(axis=(0))
        doc_to_doc_attns.append(doc_to_doc_attn_avg)
        # 对 heads 和 query positions 平均
        doc_attention_avg = query_to_doc.mean(axis=(0, 1))  # [doc_len]
        layer_attention_dict[layer_idx] = torch.tensor(doc_attention_avg, device=draft_model_device)

    if not smarter:
        # 基于熵动态选层（DraftModel 默认使用熵选层）
        active_layers, layer_entropy = entropy_layer_selection(
            layer_attention_dict, top_k=entropy_top_k, return_entropy=True
        )
        print(f"  DraftModel 熵选层: 选择了 {active_layers}")
    else:
        # I'm dumb.
        print(f"  DraftModel dumb version")
        active_layers = [k for k, v in layer_attention_dict.items()]

    # 聚合选中层的 attention
    layer_attentions = [layer_attention_dict[idx] for idx in active_layers]
    doc_to_doc_attns = [doc_to_doc_attns[idx - draft_model.config.num_hidden_layers // 2] for idx in active_layers]
    doc_to_doc_attns = np.mean(np.stack(doc_to_doc_attns, axis=0), axis=0)
    doc_to_doc_attns = torch.tensor(doc_to_doc_attns)
    multi_layer_attn = torch.stack(layer_attentions).mean(dim=0)  # [doc_len]

    return multi_layer_attn, doc_to_doc_attns


def get_multilayer_attn_sep(passages, draft_model, draft_model_device, entropy_top_k):
    system_tensor = passages[0]
    system_len = len(system_tensor)
    doc_tensors = passages[1:-1]
    query_tensor = passages[-1]
    query_len = len(query_tensor)
    multi_layer_attns = []
    for doc_idx, doc_tensor in enumerate(doc_tensors):
        layer_attention_dict = {}
        full_input = torch.cat([system_tensor, doc_tensor, query_tensor]).unsqueeze(0).to(draft_model_device)
        draft_attention = compute_draft_model_attention(draft_model, full_input, draft_model_device)
        doc_len = len(doc_tensor)
        for layer_idx, layer_attn in draft_attention.items():
            query_to_doc = layer_attn[:, system_len + doc_len:system_len + doc_len + query_len,
                           system_len:system_len + doc_len]
            doc_attention_avg = query_to_doc.mean(axis=(0, 1))
            layer_attention_dict[layer_idx] = torch.tensor(doc_attention_avg, device=draft_model_device)
        active_layers, layer_entropy = entropy_layer_selection(
            layer_attention_dict, top_k=entropy_top_k, return_entropy=True
        )
        if doc_idx == 0:
            print(f"  DraftModel 熵选层: 选择了 {active_layers}")
        layer_attentions = [layer_attention_dict[idx] for idx in active_layers]
        multi_layer_attn = torch.stack(layer_attentions).mean(dim=0)
        multi_layer_attns.append(multi_layer_attn)
    return multi_layer_attns

def load_kv(model, passages, chunk_ids, key_cache, value_cache, input_device, past_key_values, revert_rope, system_len):
    past_len = 0
    start_time = time.time()
    for layer_idx in range(len(past_key_values.key_cache)):
        past_key_values.past_tokens[layer_idx] = 0
    for idx, passage in enumerate(passages[:-1]):
        chunk_id = chunk_ids[idx]
        passage_len = passage.shape[0]
        key_cache[idx] = key_cache[idx].to(input_device)
        chunk_key_cache = key_cache[idx]
        chunk_value_cache = value_cache[idx].to(input_device)
        assert passage_len == chunk_key_cache.shape[3]
        if revert_rope and chunk_id > 0:
            all_position_ids = []
            # Get the device of the rotary embedding layer from inv_freq buffer
            rotary_emb = model.model.layers[0].self_attn.rotary_emb
            if hasattr(rotary_emb, 'inv_freq') and rotary_emb.inv_freq is not None:
                rotary_device = rotary_emb.inv_freq.device
            else:
                # Fallback: use the device of the first layer
                rotary_device = next(model.model.layers[0].parameters()).device

            position_ids = torch.full((1, chunk_key_cache[0].shape[2]), past_len - system_len, device=rotary_device)
            chunk_key_for_rope = chunk_key_cache[0].to(rotary_device)
            cos, sin = rotary_emb(chunk_key_for_rope, position_ids)
            # mistral 限定
            cos = cos.unsqueeze(1).to(input_device)
            sin = sin.unsqueeze(1).to(input_device)
            chunk_key_cache = (chunk_key_cache * cos) + (rotate_half(chunk_key_cache) * sin)
        elif chunk_id > 0:
            all_position_ids.append(torch.arange(system_len, system_len + passage_len).to(input_device).unsqueeze(0))

        for layer_idx in range(len(past_key_values.key_cache)):
            past_key_values.key_cache[layer_idx].narrow(2, past_len, passage_len).copy_(chunk_key_cache[layer_idx])
            past_key_values.value_cache[layer_idx].narrow(2, past_len, passage_len).copy_(chunk_value_cache[layer_idx])
            past_key_values.past_tokens[layer_idx] += passage_len
        past_len += passage_len
    storage_time = time.time() - start_time
    print(f'KV cache load time={storage_time}')
    return past_len

def load_kv_and_generate(model, tokenizer, past_key_values, passages,
                          load_path='', example_id = 0, max_new_tokens=1, revert_rope=False,
                          reprocess_method='normal', rate=0, preprocess=False, draft_model=None,
                          draft_attention=None, use_entropy_selection=False, entropy_top_k=4,
                          group=False, device="cuda", chunk_ids=None, device_map=None, draft_model_device="",
                         hash_keys=None, prefix_cache_path="", query="", embeddings=None, question_prefix_tensor=None, similarity=0.0):
    # Determine input device: use first GPU if device_map provided, otherwise use device
    input_device = f"cuda:{device_map['model.embed_tokens']}" if device_map is not None else device

    passages_len = [passage.shape[0] for passage in passages]
    passages_start = [sum(passages_len[:i]) for i in range(1,len(passages_len))]
    query_prefix_len = len(tokenizer.encode(tokenizer.decode(passages[-1]).split('Question: ')[0]))
    inputs = passages[-1][query_prefix_len:].unsqueeze(0).to(input_device)
    seq_length = passages[-1][query_prefix_len:].shape[0]
    eigenvalue = {}
    # load KV

    for layer_idx in range(len(past_key_values.key_cache)):
        past_key_values.past_tokens[layer_idx] = 0

    system_len = passages[0].shape[0]
    first_doc_len = passages[1].shape[0]

    key_cache = []
    value_cache = []
    all_position_ids = [torch.arange(0,system_len).unsqueeze(0).to(input_device)]

    # If chunk_ids is provided, use it; otherwise use sequential indices (backward compatible)
    if chunk_ids is None:
        chunk_ids = list(range(len(passages) - 1))

    for idx, passage in enumerate(passages[:-1]):
        chunk_id = chunk_ids[idx]
        passage_len = passage.shape[0]

        if isinstance(hash_keys, list):
            if idx == 1 and prefix_cache_path!="":
                chunk_key_cache = torch.load(f'{prefix_cache_path}/{hash_keys[idx]}_key.pt', weights_only=True).to('cpu')
                chunk_value_cache = torch.load(f'{prefix_cache_path}/{hash_keys[idx]}_value.pt', weights_only=True).to('cpu')
            else:
                chunk_key_cache = torch.load(f'{load_path}/{hash_keys[idx]}_key.pt', weights_only=True).to('cpu')
                chunk_value_cache = torch.load(f'{load_path}/{hash_keys[idx]}_value.pt', weights_only=True).to('cpu')
        else:
            chunk_key_cache = torch.load(f'{load_path}/{example_id}_{chunk_id}_key.pt',weights_only=True).to('cpu')
            chunk_value_cache = torch.load(f'{load_path}/{example_id}_{chunk_id}_value.pt',weights_only=True).to('cpu')
        key_cache.append(chunk_key_cache)
        value_cache.append(chunk_value_cache)
    ## load kv cache
    past_len = load_kv(model, passages, chunk_ids, key_cache, value_cache, input_device, past_key_values, revert_rope, system_len)

    if rate != 0:
        if reprocess_method == 'cacheBlend':
            without_attn_key = past_key_values.key_cache[1].narrow(2,0,past_len).clone()
            without_attn_value = past_key_values.value_cache[1].narrow(2,0,past_len).clone()
            inputs = torch.cat(passages[:-1]).to(input_device).unsqueeze(0)
            # 这里会在终端上多输出一次
            _, tmp_past_key_value, _ = prefill_and_generate(model, tokenizer, inputs, max_new_tokens=1, device=input_device, early_exit_layer=2,device_map=device_map)
            with_attn_key = tmp_past_key_value.key_cache[1].narrow(2,0,past_len).clone()
            with_attn_value = tmp_past_key_value.value_cache[1].narrow(2,0,past_len).clone()
            v_sub_all = without_attn_value - with_attn_value
            v_sub_all = v_sub_all.squeeze(0)
            v_sub_all = v_sub_all.transpose(0, 1)
            v_sum = torch.sum(v_sub_all**2, dim=[1,2])
            v_sum = v_sum[system_len:]
            v_need_index = torch.topk(v_sum,int(rate*(past_len - system_len))).indices.to('cpu')
            v_need_index = v_need_index + system_len

            # k_sub_all = without_attn_key - with_attn_key
            # k_sub_all = torch.abs(k_sub_all)
            # k_sub_all = k_sub_all.squeeze(0)
            # k_sub_all = k_sub_all.transpose(0, 1)
            # k_sub_all = k_sub_all.reshape(past_len,-1)
            # k_sum = torch.sum(k_sub_all,dim=1)
            # k_sum = k_sum.tolist()
            # k_sum = k_sum[system_len:]
            # k_sum = torch.tensor(k_sum,device=k_sub_all.device)
            # k_need_index = torch.topk(k_sum,int(rate*(past_len - system_len))).indices.to('cpu')
            # k_need_index = k_need_index + system_len
            k_need_index = v_need_index
        elif reprocess_method == 'Draftmodel_origin':
            with torch.no_grad():
                inputs = torch.cat(passages).unsqueeze(0).to(input_device)
                inputs_embeds = model.model.embed_tokens(inputs).to(input_device)
                cache_position = torch.arange(0, sum(passages_len), device=input_device)
                start_idx = passages_len[0] + passages_len[1]
                end_idx = sum(passages_len[:-1])
                attn_layer_idx = 1
                model(
                    inputs_embeds=inputs_embeds, past_key_values=past_key_values,
                    cache_position=cache_position, reprocess_method=reprocess_method,
                    return_dict=False, use_cache=True, passages_len=passages_len, history_key_cache=key_cache,
                    draft=True, query_start=end_idx, attn_layer_idx=attn_layer_idx
                )
                result = past_key_values.importance_cache[attn_layer_idx][:, :sum(passages_len)].sum(dim=0)
                sublist = result[start_idx:end_idx]
                tensor_sublist = torch.tensor(sublist)

                topk_values, topk_indices = torch.topk(tensor_sublist, k=int(rate*(end_idx-start_idx)))
                print(topk_values)

                k_need_index = topk_indices.to('cpu') + start_idx
                print(k_need_index)


        elif reprocess_method == 'FusionRAG':
            select_time = time.time()
            query_prefix_len = len(tokenizer.encode(tokenizer.decode(passages[-1]).split('Question: ')[0]))
            if query_prefix_len >= len(passages[-1]):
                query_prefix_len = len(tokenizer.encode(tokenizer.decode(passages[-1]).split('Question：')[0]))+1
            inputs = passages[-1][query_prefix_len:].unsqueeze(0).to(input_device)
            seq_length = passages[-1][query_prefix_len:].shape[0]

            cache_position = torch.arange(past_len, past_len+seq_length, device=input_device)
            with torch.no_grad():
                ss_time = time.perf_counter()
                inputs_embeds = model.model.embed_tokens(inputs).to(input_device)
                model(
                    inputs_embeds = inputs_embeds, past_key_values=past_key_values,
                    cache_position=cache_position, reprocess_method=reprocess_method,
                    return_dict=False, use_cache=True, passages_len=passages_len, history_key_cache=key_cache
                    )

                k_sum = torch.sum(past_key_values.importance_cache[-1], dim=0)[:cache_position[0]]
                end_time = time.perf_counter() - ss_time
                if group:
                    k_sum_relevant = k_sum[system_len:]  # 只看中间文本块的分数
                    k_sum_relevant = torch.tensor(k_sum_relevant, device=input_device)

                    # 计算需要选择的 token 数量
                    total_relevant_tokens = torch.cat(passages[1:-1]).shape[0]  # 中间文本块的总 token 数
                    k_lens = int(rate * total_relevant_tokens)

                    # === 新增：按组选择逻辑 ===
                    group_size = 16
                    num_groups = (total_relevant_tokens + group_size - 1) // group_size  # 向上取整

                    # 将分数按组重塑（最后一组可能不足 8 个）
                    # 先 pad 到能被 group_size 整除
                    padded_length = num_groups * group_size
                    if total_relevant_tokens < padded_length:
                        # 用很小的负数填充，确保不会被选中
                        padding = torch.full((padded_length - total_relevant_tokens,),
                                            -float('inf'), device=input_device)
                        k_sum_padded = torch.cat([k_sum_relevant, padding])
                    else:
                        k_sum_padded = k_sum_relevant

                    # 重塑为 [num_groups, group_size]
                    k_sum_grouped = k_sum_padded.view(num_groups, group_size)

                    # 计算每组的最大分数
                    group_max_scores, _ = torch.max(k_sum_grouped, dim=1)  # [num_groups]

                    # 根据组的最大分数选择 top-k 组
                    num_groups_to_select = (k_lens + group_size - 1) // group_size  # 向上取整
                    num_groups_to_select = min(num_groups_to_select, num_groups)  # 不超过总组数

                    top_group_indices = torch.topk(group_max_scores, num_groups_to_select).indices

                    # 将选中的组展开为 token 索引
                    selected_token_indices = []
                    for group_idx in top_group_indices.tolist():
                        start_idx = group_idx * group_size
                        end_idx = min(start_idx + group_size, total_relevant_tokens)
                        selected_token_indices.extend(range(start_idx, end_idx))

                    # 转换为 tensor 并加上 system_len 偏移
                    k_need_index = torch.tensor(selected_token_indices, device='cpu') + system_len

                    print(f"选择了 {len(selected_token_indices)} 个 tokens，"
                          f"来自 {len(top_group_indices)} 个组 (目标: {k_lens} tokens)")
                else:
                    k_sum = k_sum.tolist()
                    k_sum = k_sum[system_len:]
                    k_sum = torch.tensor(k_sum,device=input_device)
                    k_lens = int(rate*(torch.cat(passages[:-1]).shape[0] - system_len))
                    k_need_index = torch.topk(k_sum, k_lens).indices.to('cpu')
                    k_need_index = k_need_index + system_len
                    print(f'select_time: {time.time() - select_time}')
        elif reprocess_method == 'frontRow':
            k_need_index = []
            for i in range(len(passages_start[:-1])):
                k_need_index.extend(range(passages_start[i], passages_start[i] + int(passages_len[i+1]*rate)))
            k_need_index = torch.tensor(k_need_index)
        elif reprocess_method == "Cache-Craft":
            import os

            if isinstance(hash_keys, list):
                save_prefix_path_list = [f"{load_path}/cachecraftattn_{hash_keys[i]}.pt" for i in
                                         range(1, len(passages) - 1)]
            else:
                save_prefix_path_list = [f"{load_path}/cachecraftattn_{example_id}_{i}.pt" for i in
                                         range(1, len(passages) - 1)]

            chunk_score_list = []
            for file in save_prefix_path_list:
                if not os.path.exists(file):
                    raise FileNotFoundError(f"未找到 cache-craft 文件: {file}")
                tensor = torch.load(file, weights_only=True, map_location="cpu")
                chunk_score_list.append(tensor)
            chunk_score = torch.cat(chunk_score_list, dim=0)
            assert chunk_score.shape[0] == sum(passages_len[1:-1])
            k_lens = int(rate*(torch.cat(passages[:-1]).shape[0] - system_len))
            k_need_index = torch.topk(chunk_score, k_lens).indices.to('cpu')
            k_need_index = k_need_index + system_len

        elif reprocess_method == "speculative_prefill":
            inputs = torch.cat(passages).to(input_device).unsqueeze(0)
            cache_position = torch.arange(0, inputs.shape[1], device=input_device)
            # Pass device_map if multi-GPU, otherwise pass device
            cache_device = device_map if device_map is not None else input_device
            tmp_past_key_values = StaticCache(
                                    config = model.config, max_batch_size = 1,
                                    max_cache_len = inputs.shape[1], device = cache_device, dtype = model.dtype,
                                    passage_len=torch.cat(passages[:-1]).shape[0],
                                )
            with torch.no_grad():
                ss_time = time.perf_counter()
                inputs_embeds = draft_model.model.embed_tokens(inputs).to(input_device)
                draft_model(
                    inputs_embeds = inputs_embeds, past_key_values=tmp_past_key_values,
                    cache_position=cache_position, reprocess_method=reprocess_method,
                    return_dict=False, use_cache=True, passages_len=passages_len
                    )

                # 获取重要性分数
                k_sum = torch.sum(past_key_values.importance_cache[-1], dim=0)[:past_len]
                if group:
                    k_sum_relevant = k_sum[system_len:]  # 只看中间文本块的分数
                    k_sum_relevant = torch.tensor(k_sum_relevant, device=input_device)

                    # 计算需要选择的 token 数量
                    total_relevant_tokens = torch.cat(passages[1:-1]).shape[0]  # 中间文本块的总 token 数
                    k_lens = int(rate * total_relevant_tokens)

                    # === 新增：按组选择逻辑 ===
                    group_size = 16
                    num_groups = (total_relevant_tokens + group_size - 1) // group_size  # 向上取整

                    # 将分数按组重塑（最后一组可能不足 8 个）
                    # 先 pad 到能被 group_size 整除
                    padded_length = num_groups * group_size
                    if total_relevant_tokens < padded_length:
                        # 用很小的负数填充，确保不会被选中
                        padding = torch.full((padded_length - total_relevant_tokens,),
                                            -float('inf'), device=input_device)
                        k_sum_padded = torch.cat([k_sum_relevant, padding])
                    else:
                        k_sum_padded = k_sum_relevant

                    # 重塑为 [num_groups, group_size]
                    k_sum_grouped = k_sum_padded.view(num_groups, group_size)

                    # 计算每组的最大分数
                    group_max_scores, _ = torch.max(k_sum_grouped, dim=1)  # [num_groups]

                    # 根据组的最大分数选择 top-k 组
                    num_groups_to_select = (k_lens + group_size - 1) // group_size  # 向上取整
                    num_groups_to_select = min(num_groups_to_select, num_groups)  # 不超过总组数

                    top_group_indices = torch.topk(group_max_scores, num_groups_to_select).indices

                    # 将选中的组展开为 token 索引
                    selected_token_indices = []
                    for group_idx in top_group_indices.tolist():
                        start_idx = group_idx * group_size
                        end_idx = min(start_idx + group_size, total_relevant_tokens)
                        selected_token_indices.extend(range(start_idx, end_idx))

                    # 转换为 tensor 并加上 system_len 偏移
                    k_need_index = torch.tensor(selected_token_indices, device='cpu') + system_len

                    print(f"选择了 {len(selected_token_indices)} 个 tokens，"
                          f"来自 {len(top_group_indices)} 个组 (目标: {k_lens} tokens)")
                else:
                    k_sum = k_sum.tolist()
                    k_sum = k_sum[system_len:]
                    k_sum = torch.tensor(k_sum,device=input_device)
                    k_lens = int(rate*(torch.cat(passages[:-1]).shape[0] - system_len))
                    k_need_index = torch.topk(k_sum, k_lens).indices.to('cpu')
                    k_need_index = k_need_index + system_len
        elif reprocess_method == 'QueryAttention':
            # Smart Query Selection: 使用 query attention + 连通分量分析
            select_time = time.time()

            # 获取 query tokens
            inputs = passages[-1][:].unsqueeze(0).to(input_device)
            seq_length = passages[-1][:].shape[0]

            cache_position = torch.arange(past_len, past_len + seq_length, device=input_device)

            with torch.no_grad():
                inputs_embeds = model.model.embed_tokens(inputs).to(input_device)
                model(
                    inputs_embeds=inputs_embeds, past_key_values=past_key_values,
                    cache_position=cache_position, reprocess_method='QueryAttention',
                    return_dict=False, use_cache=True, passages_len=passages_len
                )

                # 获取文档部分的 attention 分数
                num_layers = len(past_key_values.importance_cache)
                doc_len = sum(passages_len[1:-1])

                # 收集所有候选层的 attention（后 1/2 的层，用于熵选层）
                candidate_start = num_layers // 2
                layer_attention_dict = {}
                for layer_idx in range(candidate_start, num_layers):
                    layer_attn = past_key_values.importance_cache[layer_idx][:, system_len:system_len + doc_len]
                    layer_attn_avg = layer_attn.mean(dim=0).to(input_device)  # [doc_len]
                    layer_attention_dict[layer_idx] = layer_attn_avg

                # 选择使用的层
                if use_entropy_selection:
                    # 基于熵动态选层
                    active_layers, layer_entropy = entropy_layer_selection(
                        layer_attention_dict, top_k=entropy_top_k, return_entropy=True
                    )
                    print(f"  熵选层: 选择了 {active_layers} (熵最低的 {entropy_top_k} 层)")
                else:
                    # 默认使用后 1/4 的层
                    start_layer = num_layers * 3 // 4
                    active_layers = list(range(start_layer, num_layers))

                # 聚合选中层的 attention
                layer_attentions = [layer_attention_dict[idx] for idx in active_layers]
                multi_layer_attn = torch.stack(layer_attentions).mean(dim=0)  # [doc_len]

                # 使用 smart_query_selection 进行选择
                selected_indices = smart_query_selection(
                    attention_scores=multi_layer_attn,
                    doc_len=doc_len,
                    target_ratio=rate,
                    system_len=system_len,
                    device=input_device
                )
                # 转成 tensor 以与后续 torch.sort 兼容
                k_need_index = torch.tensor(selected_indices, device='cpu')

                print(f"QueryAttention 选择了 {len(k_need_index)} 个 tokens ({len(k_need_index)/doc_len*100:.1f}%)")
                print(f"使用了 {len(active_layers)} 个层: {active_layers}")
                print(f'select_time: {time.time() - select_time:.3f}s')

        elif reprocess_method == "average":
            ""
            k_need_index = torch.tensor(list(range(0, sum(passages_len[:-1]), int(1/rate))))
        elif reprocess_method == "DraftModel_with_answer":
            doc_len = sum(passages_len[1:-1])
            query_start = sum(passages_len[:-1])
            total_len = sum(passages_len)
            _, multi_layer_attn = get_multilayer_attn_with_answer(passages, draft_model, draft_model_device, tokenizer)

            selected_indices = smart_query_selection(
                attention_scores=multi_layer_attn,
                doc_len=doc_len,
                target_ratio=rate,
                system_len=system_len,
                device=draft_model_device
            )
            k_need_index = torch.tensor(selected_indices, device='cpu')

        elif reprocess_method == "DraftModel_choose": ## first use the attention to score to choose the articles, and then recompute in those articles.
            doc_len = sum(passages_len[1:-1])
            query_start = sum(passages_len[:-1])
            total_len = sum(passages_len)
            multi_layer_attn, doc_to_doc_attns = get_multilayer_attn(passages, draft_model, draft_model_device,
                                                                     entropy_top_k, draft_attention, query_start,
                                                                     system_len, doc_len, total_len)
            multi_layer_attns_sum = []
            for idx in range(1, len(passages_len)-1):
                sum_segment = multi_layer_attn[sum(passages_len[:idx]):sum(passages_len[:idx+1])].sum()
                multi_layer_attns_sum.append(sum_segment/passages_len[idx])
            k = math.ceil(len(multi_layer_attns_sum) * rate)  # 向上取整
            multi_layer_attns_sum = torch.stack(multi_layer_attns_sum)
            topk_values, topk_indices = torch.topk(multi_layer_attns_sum, k)
            top30_percent_indices = topk_indices.tolist()
            top30_percent_indices = [x+1 for x in top30_percent_indices]
            ## add system prompt at front
            top30_percent_indices.insert(0, 0)
            key_cache = [key_cache[idx] for idx in top30_percent_indices]
            value_cache = [value_cache[idx] for idx in top30_percent_indices]
            ## add query at back
            top30_percent_indices.append(len(passages)-1)
            passages = [passages[idx] for idx in top30_percent_indices]
            passages_len = [passage.shape[0] for passage in passages]
            chunk_ids = list(range(len(passages) - 1))
            # kv needs to be reloaded
            load_kv(
                model, passages, chunk_ids, key_cache, value_cache, input_device, past_key_values, revert_rope, system_len
            )
            #
            doc_len = sum(passages_len[1:-1])
            query_start = sum(passages_len[:-1])
            total_len = sum(passages_len)
            # generate again
            multi_layer_attn, doc_to_doc_attns = get_multilayer_attn(passages, draft_model, draft_model_device,
                                                                     entropy_top_k, draft_attention, query_start,
                                                                     system_len, doc_len, total_len)
            ## now we choose again
            selected_indices = smart_query_selection(
                attention_scores=multi_layer_attn,
                doc_len=doc_len,
                target_ratio=rate,
                system_len=system_len,
                device=draft_model_device
            )
            k_need_index = torch.tensor(selected_indices, device='cpu')
            print(f"DraftModel 选择了 {len(k_need_index)} 个 tokens ({len(k_need_index)/doc_len*100:.1f}%)")

        elif reprocess_method == "DraftModel_sep":
            multi_layer_attns = get_multilayer_attn_sep(
                passages=passages,
                draft_model=draft_model,
                entropy_top_k=entropy_top_k,
                draft_model_device=draft_model_device
            )
            multi_layer_attns = torch.cat(multi_layer_attns)
            selected_indices = smart_query_selection(
                attention_scores=multi_layer_attns,
                doc_len=sum(passages_len[1:-1]),
                target_ratio=rate,
                system_len=system_len,
            )
            k_need_index = torch.tensor(selected_indices, device='cpu')


        elif 'DraftModel' in reprocess_method:
            # DraftModel: 用小模型 prefill 获取 attention，指导 token 选择
            select_time = time.time()

            doc_len = sum(passages_len[1:-1])
            query_start = sum(passages_len[:-1])
            total_len = sum(passages_len)
            smarter = reprocess_method == "DraftModel_smarter"
            multi_layer_attn, doc_to_doc_attns = get_multilayer_attn(passages, draft_model, draft_model_device,
                                                                     entropy_top_k, draft_attention, query_start,
                                                                     system_len, doc_len, total_len,smarter=smarter)
            eigenvalue["coefficient"] = concentration_coefficient_v1(tensor=multi_layer_attn)
            eigenvalue["dispersion"] = topk_position_dispersion(multi_layer_attn, top_percent=0.1)
            # 使用 smart_query_selection 进行选择
            if reprocess_method == "DraftModel":
                selected_indices = smart_query_selection(
                    attention_scores=multi_layer_attn,
                    doc_len=doc_len,
                    target_ratio=rate,
                    system_len=system_len,
                    device=draft_model_device
                )

            elif reprocess_method == "DraftModel_smarter":
                selected_indices, reserved_selected_indices = smart_query_selection(
                    attention_scores=multi_layer_attn,
                    input_tokens=passages[1:],
                    doc_len=doc_len,
                    target_ratio=rate,
                    system_len=system_len,
                    device=draft_model_device,
                    smarter=True,
                    tokenizer=tokenizer,
                    eigenvalue=eigenvalue,
                    similarity=similarity
                )
                # selected_indices, reserved_selected_indices = smart_query_selection(
                #     attention_scores=multi_layer_attn[first_doc_len:],
                #     input_tokens=passages[2:],
                #     doc_len=doc_len-first_doc_len,
                #     target_ratio=rate,
                #     system_len=system_len+first_doc_len,
                #     device=draft_model_device,
                #     smarter=True,
                #     tokenizer=tokenizer,
                #     eigenvalue=eigenvalue
                # )

                selected_indices = [x for x in selected_indices if x >= system_len+first_doc_len]
                reserved_selected_indices = [x for x in reserved_selected_indices if x >= system_len + first_doc_len]
                passages_len_chosen = set()
                ## add system prompt and the first document.
                passages_len_chosen.add(0)
                passages_len_chosen.add(1)
                embedding_index_chosen = set()
                ## add query
                passages_len_chosen.add(len(passages_len)-1)
                selected_indices_passage = []
                selected_indices_passage_reserve = {}
                # looks slow but actually really fast
                for selected_index in selected_indices:
                    for i in range(len(passages_len)):
                        start_idx = sum(passages_len[:i])
                        end_idx = sum(passages_len[:i+1])
                        if selected_index >= start_idx and selected_index<end_idx:
                            passages_len_chosen.add(i)
                            embedding_index_chosen.add(i-2)
                            selected_indices_passage.append(i)

                for selected_index in reserved_selected_indices:
                    for i in range(len(passages_len)):
                        start_idx = sum(passages_len[:i])
                        end_idx = sum(passages_len[:i+1])
                        if selected_index >= start_idx and selected_index<end_idx and i in selected_indices_passage:
                            reserve = selected_indices_passage_reserve.get(i, [])
                            reserve.append(selected_index)
                            selected_indices_passage_reserve[i] = reserve

                passages_len_chosen = sorted(list(passages_len_chosen))
                selected_index_new = []
                for i, index in enumerate(selected_indices):
                    passage_index = selected_indices_passage[i]
                    all_before_passage_index = list(range(passage_index))
                    all_cur_passage_index = [x for x in all_before_passage_index if x in passages_len_chosen]
                    before_len = sum(passages_len[x] for x in all_before_passage_index)
                    cur_len = sum(passages_len[x] for x in all_cur_passage_index)
                    selected_index_new.append(index-before_len+cur_len)

                reserved_indices = set()
                for passage_index, reserved_idx in selected_indices_passage_reserve.items():
                    all_before_passage_index = list(range(passage_index))
                    all_cur_passage_index = [x for x in all_before_passage_index if x in passages_len_chosen]
                    before_len = sum(passages_len[x] for x in all_before_passage_index)
                    cur_len = sum(passages_len[x] for x in all_cur_passage_index)
                    for x in reserved_idx:
                        reserved_indices.add(x-before_len+cur_len)

                selected_indices = selected_index_new
                selected_indices.extend(list(reserved_indices))

                ## insert the query prefix now
                passages.insert(-1, question_prefix_tensor)
                passages_len.insert(-1, question_prefix_tensor.shape[0])
                passages_len_chosen.append(len(passages)-1)
                ## rebuild passages_len
                passages_len = [passages_len[x] for x in passages_len_chosen]
                # systemprompt, first passage and query and query_prefix
                eigenvalue["passages_chosen"] = len(passages_len_chosen) - 4 #
                passages = [passages[x] for x in passages_len_chosen]
                # 2 is query and query prefix
                chunk_ids = list(range(len(passages) - 2))
                key_cache = [cache for i, cache in enumerate(key_cache) if i in passages_len_chosen]
                value_cache = [cache for i, cache in enumerate(value_cache) if i in passages_len_chosen]

                ## passages
                past_len = load_kv(model, passages[:-1], chunk_ids, key_cache, value_cache, input_device, past_key_values,
                                   revert_rope, system_len)

                # do it now because later will add query prefix
                eigenvalue["recompute_rate"] = len(selected_indices) / doc_len
                print(f"DraftModel 选择了 {len(selected_indices)} 个 tokens ({len(selected_indices) / doc_len * 100:.1f}%)")
                ## add query prefix
                selected_indices.extend(range(sum(passages_len[:-2]), sum(passages_len[:-1])))


            if reprocess_method != "DraftModel_smarter":
                eigenvalue["recompute_rate"] = len(selected_indices) / doc_len
            k_need_index = torch.tensor(selected_indices, device='cpu')
            if reprocess_method == 'DraftModel_ppr':
                print(f"using ppr to draft.")
                scores = personalized_pagerank(attention_matrix=doc_to_doc_attns,
                                               initial_scores=multi_layer_attn.to('cpu'))
                top_indices, top_scores = get_top_tokens(scores=scores, top_n=int(rate * doc_len))
                k_need_index = top_indices

            print(f'select_time: {time.time() - select_time:.3f}s')

        else:
            raise NotImplementedError

        # reprocess kv cache and prefill question
        k_need_index = torch.sort(k_need_index)[0].tolist()

        ## add query itself
        k_need_index.extend(range(sum(passages_len[:-1]),sum(passages_len)))
    else:
        k_need_index = range(sum(passages_len[:-1]),sum(passages_len))
    past_len = sum(passages_len)

    batch_size, seq_length = 1, len(k_need_index)

    generated_ids = torch.zeros(
        batch_size, past_len + max_new_tokens + 1, dtype=torch.int, device=input_device
    )
    generated_ids[:, :past_len] = torch.cat(passages).unsqueeze(0).to(input_device)
    tokens = []

    if reprocess_method != 'FusionRAG':
        use_sparse_attention = False
    else:
        use_sparse_attention = False
    reprocess_inputs = torch.cat(passages)[k_need_index].unsqueeze(0).to(input_device)
    cache_position = torch.tensor(k_need_index, device=input_device)
    if rate > 0.0:
        k_need_index_ = copy.deepcopy(k_need_index)
        ## first is prefix cache
        if reprocess_method == "DraftModel_smarter":
            k_need_index_.extend(range(passages_len[0], passages_len[0]+passages_len[1]))
        highlight_tokens_compare(k_need_index_, passages, tokenizer)
    with torch.no_grad():
        without_attn_value = past_key_values.value_cache[-1].narrow(2,0, sum(passages_len[:-1])).clone()
        inputs_embeds = model.model.embed_tokens(reprocess_inputs).to(input_device)

        # Don't force move to input_device - keep on the device where model output is
        # This avoids cross-GPU transfer deadlock in PP mode
        start_time = time.time()
        model_output = model(
            inputs_embeds = inputs_embeds, cache_position=cache_position,
            past_key_values=past_key_values, return_dict=False, use_cache=True, use_sparse_attention=use_sparse_attention,
        )[0]

        logits = model_output[:,-1,:].unsqueeze(0).clone()

        first_token_time = time.time() - start_time
        stream = TextStreamer(tokenizer)
        logits_warper = tf_logits_warper(temperature=0.01, top_k=1)
        next_token_scores = logits_warper(reprocess_inputs, logits[:, -1, :])
        next_token = torch.argmax(next_token_scores, dim=-1)
        prefill_count = seq_length
        prefill_time = first_token_time
        generated_ids[:, past_len+1] = next_token
        tokens.append(next_token)

        # Use the device where logits/next_token are (model output device in PP mode)
        output_device = next_token.device
        inputs = torch.cat((torch.cat(passages).unsqueeze(0).to(output_device), next_token.unsqueeze(0)), dim=-1)
        cache_position = torch.tensor([past_len], device=output_device)
        position_ids = cache_position.unsqueeze(0)
        seq_length += 1

        decode_time = time.time()
        for _ in range(1, max_new_tokens):
            next_token = decode_one_tokens(model, next_token.unsqueeze(0), position_ids, cache_position, past_key_values, logits_warper, inputs, rate=rate, path=query)
            inputs = torch.cat((inputs, next_token.unsqueeze(0)), dim=-1)
            generated_ids[:, cache_position] = next_token.int()
            tokens.append(next_token.int())
            # print(f"mengyao_debug current token is {tokenizer.decode(torch.tensor(tokens[:-1]))}")
            seq_length += 1

            if next_token[0].item() == tokenizer.eos_token_id or tokenizer.decode(next_token) == '<|im_end|>':
                break

            cache_position += 1
            position_ids = cache_position.unsqueeze(0)


    total_time = time.time() - decode_time
    tokens_generated = len(tokens)
    tokens_per_second = tokens_generated / total_time

    print(f"decode_time={total_time}")

    # print(f"prompt eval count:    {prefill_count} token(s)")
    # print(f"prompt eval duration: {prefill_time}s")
    # print(f"prompt eval rate:     {prefill_count/prefill_time} tokens/s")
    # print(f"eval count:           {tokens_generated} token(s)")
    # print(f"eval duration:        {total_time}s")
    # print(f"eval rate:            {tokens_per_second} tokens/s")

    return tokens, prefill_time, eigenvalue

def tf_logits_warper(temperature, top_k):
        """
        This class returns a [`LogitsProcessorList`] list object that contains all relevant [`LogitsWarper`] instances
        used for multinomial sampling.
        """

        # instantiate warpers list
        warpers = LogitsProcessorList()

        # In beam methods, we need to keep at least one non-eos token to explore continuations that might have a
        # better score (i.e. keep len(list(generation_config._eos_token_tensor)) + 1)
        min_tokens_to_keep = 1

        # the following idea is largely copied from this PR: https://github.com/huggingface/transformers/pull/5420/files
        # all samplers can be found in `generation_utils_samplers.py`
        warpers.append(TemperatureLogitsWarper(temperature))
        warpers.append(TopKLogitsWarper(top_k=top_k, min_tokens_to_keep=min_tokens_to_keep))
        return warpers

def prefill_and_generate(model, tokenizer, inputs, max_new_tokens=10000, use_cuda_graph: bool = False,
                         mode = 'normal', early_exit_layer=None, device='cuda', device_map=None):
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch._dynamo.config.suppress_errors = True
    batch_size, seq_length = inputs.shape

    # Determine input device: use first GPU if device_map provided, otherwise use device
    input_device = f"cuda:{device_map['model.embed_tokens']}" if device_map is not None else device
    inputs = inputs.to(input_device)

    tokens = []

    def decode_one_tokens(cuda_graph_runner, cur_token, position_ids, cache_position, past_key_values, use_cuda_graph: bool = False):
        if use_cuda_graph:
            logits = cuda_graph_runner(cur_token, position_ids, cache_position)
        else:
            # custom_stream = torch.cuda.Stream()
            inputs_embeds = model.model.embed_tokens(cur_token).to(input_device)
            # with torch.cuda.stream(custom_stream):
            logits=model(inputs_embeds=inputs_embeds,
                        position_ids=position_ids,
                        cache_position=cache_position,
                        past_key_values=past_key_values,
                        return_dict=False, use_cache=True)[0]
        if past_key_values != None:
            past_key_values.change_seq_length(1)
        #print(logits)
        next_token_scores = logits_warper(inputs, logits[:, -1, :])

        next_token = torch.argmax(next_token_scores, dim=-1)
        return next_token

    with torch.no_grad():
        stream = TextStreamer(tokenizer)

        # Pass device_map if multi-GPU, otherwise pass device
        cache_device = device_map if device_map is not None else input_device
        past_key_values = StaticCache(
            config = model.config, max_batch_size=1, max_cache_len=seq_length+max_new_tokens, device=cache_device, dtype=model.dtype
        )

        cache_position = torch.arange(seq_length, device=input_device)
        generated_ids = torch.zeros(
            batch_size, seq_length + max_new_tokens + 1, dtype=torch.int, device=input_device
        )
        generated_ids[:, cache_position] = inputs.to(input_device).to(torch.int)
        if past_key_values != None:
            past_key_values.cur_idx=cache_position
        start_time = time.time()

        if mode == "long_context":
            inputs_embeds = model.model.embed_tokens(inputs.to("cpu"))
        else:
            inputs_embeds = model.model.embed_tokens(inputs).to(input_device)

        # Pass early_exit_layer if specified (for CacheBlend)
        model_kwargs = {
            'inputs_embeds': inputs_embeds,
            'cache_position': cache_position,
            'past_key_values': past_key_values,
            'return_dict': False,
            'use_cache': True
        }
        if early_exit_layer is not None:
            model_kwargs['early_exit_layer'] = early_exit_layer

        logits = model(**model_kwargs)[0][:,-1,:].unsqueeze(0).clone().to(input_device)
        # generation_config, model_kwargs = model._prepare_generation_config(None, do_sample=False, top_k=1,  temperature=0.01)

        logits_warper = tf_logits_warper(temperature=0.01, top_k=1)
        next_token_scores = logits_warper(inputs, logits[:, -1, :])

        next_token = torch.argmax(next_token_scores, dim=-1)
        first_token_time = time.time() - start_time

        prefill_count = seq_length
        prefill_time = first_token_time
        # print(stream.put(next_token.item()), end="", flush=True)
        generated_ids[:, seq_length] = next_token
        tokens.append(next_token)
        inputs = torch.cat((inputs, next_token.unsqueeze(0)), dim=-1)
        cache_position = torch.tensor([seq_length], device=input_device)
        position_ids = cache_position.unsqueeze(0)
        seq_length += 1

        if use_cuda_graph:
            cuda_graph_runner = CUDAGraphRunner()
            cuda_graph_runner.capture(model, next_token.unsqueeze(0), position_ids, cache_position, past_key_values, device, return_dict=False, use_cache=True)
        else:
            cuda_graph_runner = None

        start_time = time.time()
        for _ in range(1, max_new_tokens):
            next_token = decode_one_tokens(cuda_graph_runner, next_token.unsqueeze(0), position_ids, cache_position, past_key_values, use_cuda_graph).to(input_device)
            inputs = torch.cat((inputs, next_token.unsqueeze(0)), dim=-1)
            generated_ids[:, cache_position] = next_token.int()
            tokens.append(next_token.int())
            seq_length += 1

            if next_token[0].item() == tokenizer.eos_token_id or tokenizer.decode(next_token) == '<|im_end|>' or tokenizer.decode(next_token) == '[unused10]':
                # print(stream.end(), end="", flush=True)
                break

            cache_position += 1
            position_ids = cache_position.unsqueeze(0)


    total_time = time.time() - start_time
    tokens_generated = len(tokens)
    tokens_per_second = tokens_generated / total_time



    return tokens, past_key_values, prefill_time



# ============================================================
# Utility functions for process_cache
# ============================================================

def remove_unused_tokens(text):
    """移除所有 [unusedXX] 格式的 token (PanGu specific)"""
    cleaned = re.sub(r'\[unused\d+\]', '', text)
    return cleaned.strip()

def parse_generation(s):
    s = s.lstrip('\n').split('\n')[0]
    if s.startswith("Yes") or s.startswith("yes"):
        s = "Yes"
    elif (s.split()[0]).startswith("No") or (s.split()[0]).startswith("no"):
        s = "No"
    return s

def compute_f1(a_pred, a_gold, tokenizer):
    a_pred = parse_generation(a_pred)
    gold_toks = tokenizer.encode(normalize_answer(a_gold))[1:]
    pred_toks = tokenizer.encode(normalize_answer(a_pred))[1:]
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def _exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def _metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

def find_group_and_index(sizes, idx):
    """
    找到list中的某个索引属于哪个组及该组中的索引
    :param sizes: 每个组的大小的列表
    :param idx: 要查找的索引
    :return: (组号, 组中的索引)
    """
    cumulative_size = 0
    for group_id, group_size in enumerate(sizes):
        if cumulative_size + group_size > idx:
            group_index = idx - cumulative_size
            return group_id, group_index
        cumulative_size += group_size
    return None, None  # 如果索引超出范围，返回None

def split_passages_by_title(text, title_marker):
    # 使用标题标记作为分割点，找到所有的位置
    titles = [i for i in range(len(text)) if text.startswith(title_marker, i)]
    # 根据标题位置分割文本为多个段落
    passages = [text[titles[i]:titles[i+1]].strip() for i in range(len(titles) - 1)]
    passages.append(text[titles[-1]:].strip())  # 添加最后一个段落
    return passages

def normalize_answer(s, model_type='default'):
    """
    Normalize answer text
    :param s: answer string
    :param model_type: 'llama' uses slightly different normalization
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        if model_type == 'llama':
            return ' '.join(text.replace('\n', ' ').split())
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def _rouge1_score(prediction, ground_truth):
    rouge = Rouge()
    try:
        scores = rouge.get_scores(normalize_answer(prediction), normalize_answer(ground_truth), avg=True)
    except ValueError:  # "Hypothesis is empty."
        return 0.0
    return scores["rouge-1"]["f"]

def _rougel_score(prediction, ground_truth):
    rouge = Rouge()
    try:
        scores = rouge.get_scores(prediction, ground_truth, avg=True)
    except ValueError:  # "Hypothesis is empty."
        return 0.0
    return scores["rouge-l"]["f"]

def save_list_to_jsonl(data_list, file_path):
    """
    保存一个字典的列表为 jsonl 文件。
    :param data_list: 要保存的字典列表
    :param file_path: 保存的文件路径
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        for item in data_list:
            json_line = json.dumps(item, ensure_ascii=False)
            file.write(json_line + '\n')

def prepare_data(model_name, data_path, data_name, cache_path, tokenizer: AutoTokenizer,
                 topk: int, revert_rope, preprocess, bge_model_path='/mnt/data/models/bge-m3-FP16'):
    """
    Prepare data for process cache experiments
    """
    import faiss
    from FlagEmbedding import FlagModel
    import random
    import os

    prompt_config = json.load(open('./config/dataset2prompt_few-shot.json'))
    data_path = data_path+data_name
    if data_name in ['2wikimqa.jsonl', 'samsum.jsonl', 'multi_news.jsonl', 'musique.jsonl', 'hotpotqa.jsonl', 'triviaqa.jsonl']:
        data_name_prefix = data_name.split('.')[0]
    else:
        data_name_prefix = data_name.split('-')[0]
    if data_name_prefix in ['hotpotqa','triviaqa','2wikimqa','musique']:
        rouge_metrics = _rouge1_score
        max_tokens_length = 50
    elif data_name_prefix in ['samsum','multi_news']:
        rouge_metrics = _rougel_score
        max_tokens_length = 512
    system_prompt = prompt_config['system_prompt'][model_name.split('-')[0]][data_name_prefix]
    system_tokens = torch.tensor(tokenizer.encode(system_prompt, add_special_tokens = False),dtype=torch.int)
    query_task = prompt_config['query_prompt'][model_name.split('-')[0]][data_name_prefix]
    local_model_config = json.load(open('./config/model_config.json'))
    stop_token_id = local_model_config[model_name.split('-')[0]]['stop_token_id']
    # 存报告
    if not os.path.exists(f"{cache_path}{data_name.split('.')[0]}/{model_name}"):
        os.makedirs(f"{cache_path}{data_name.split('.')[0]}/{model_name}")
    # 存数据
    if not os.path.exists(f"{cache_path}data"):
        os.makedirs(f"{cache_path}data")
    # reprocess 数据
    if not os.path.exists(f"{cache_path}data/{data_name.split('.')[0]}/{model_name}"):
        os.makedirs(f"{cache_path}data/{data_name.split('.')[0]}/{model_name}")
    if not os.path.exists(f"{cache_path}{data_name.split('.')[0]}/{model_name}"):
        os.makedirs(f"{cache_path}{data_name.split('.')[0]}/{model_name}")
    # preprocesss 数据
    if not os.path.exists(f"{cache_path}data/{data_name.split('.')[0]}-preprocess-{topk}-revert_rope-{revert_rope}/{model_name}"):
        os.makedirs(f"{cache_path}data/{data_name.split('.')[0]}-preprocess-{topk}-revert_rope-{revert_rope}/{model_name}")

    csv_path = f"{cache_path}{data_name.split('.')[0]}/{model_name}"
    reprocess_path = f"{cache_path}data/{data_name.split('.')[0]}/{model_name}"
    preprocess_path = f"{cache_path}data/{data_name.split('.')[0]}-preprocess-{topk}-revert_rope-{revert_rope}/{model_name}"
    data_file = open(data_path, 'r', encoding='utf-8')
    data = []
    for line in data_file.readlines():
        data.append(json.loads(line))
    if data_name_prefix in ['hotpotqa','triviaqa'] and data_name not in ['hotpotqa.jsonl', 'triviaqa.jsonl', 'hotpotqa-200.jsonl']:
        data = data[0]
        for i in range(len(data)):
            # 打乱顺序
            random.seed(1)
            random.shuffle(data[i]['output'][0]['document'])
            data[i]['passage'] = data[i]['output'][0]['document']
    else:
        if data_name == 'samsum.jsonl':
            split_mark = 'Dialogue:'
        else:
            split_mark = 'Passage'
        for i in range(len(data)):
            data[i]['passage'] = re.findall(f'({split_mark} \\d+.*?)(?={split_mark} \\d+|$)', data[i]['context'], re.DOTALL)
        if data_name == 'musique-140.jsonl':
            for i in range(len(data)):
                data[i]['passage'] = re.findall(f'Passage \\d+:\\n(.*?)(?=Passage \\d+:|$)', data[i]['context'], re.DOTALL)
                data[i]['passage'] = ['\n\n' + text for text in data[i]['passage']]
                data[i]['passage'][-1] = data[i]['passage'][-1] + '\n'

    N = len(data)
    batch_data = []
    batch_tokens = []
    question_list = []
    real_answer_list = []

    for query_id,query in enumerate(data[:N]):
        query_prompt = query_task.format(input=data[query_id]['input'])
        query_tokens = torch.tensor(tokenizer.encode(query_prompt, add_special_tokens = False),dtype=torch.int)
        question_list.append(data[query_id]['input'])
        tmp_list = []
        if data_name_prefix in ['hotpotqa','triviaqa'] and data_name not in ['hotpotqa.jsonl', 'triviaqa.jsonl', 'hotpotqa-200.jsonl']:
            for i in range(len(data[query_id]['output'])):
                if 'answer' in data[query_id]['output'][i] and \
                    data[query_id]['output'][i]['answer'] not in tmp_list:
                    tmp_list.append(data[query_id]['output'][i]['answer'])
        else:
            for i in range(len(data[query_id]['answers'])):
                if data[query_id]['answers'][i] not in tmp_list:
                    tmp_list.append(data[query_id]['answers'][i])
        real_answer_list.append(tmp_list)
        index = 0

        passage = [system_prompt]
        passage_tokens = [system_tokens]
        for bn in range(len(query['passage'])):
            if data_name_prefix in ['hotpotqa','triviaqa'] and data_name not in ['hotpotqa.jsonl', 'triviaqa.jsonl', 'hotpotqa-200.jsonl']:
                passage.append(f'Passage {index+1}:\n' + query['passage'][index] + '\n')
                passage_tokens.append(torch.tensor(tokenizer.encode(f'Passage {index+1}:\n' + query['passage'][index] + '\n', add_special_tokens = False),dtype=torch.int))
            else:
                passage.append(query['passage'][index] + '\n')
                passage_tokens.append(torch.tensor(tokenizer.encode(query['passage'][index] + '\n', add_special_tokens = False),dtype=torch.int))
            index += 1
            if index >=len(query['passage']):
                break
        passage.append(query_prompt)
        passage_tokens.append(query_tokens)
        batch_tokens.append(passage_tokens)
        batch_data.append(passage)

    if preprocess == True:
        bgem3 = FlagModel(bge_model_path,
                      query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                      use_fp16=True)
        corpus = []
        corpus_lens = []
        for batch in batch_data:
            corpus.extend(batch[1:-1])
            corpus_lens.append(len(batch[1:-1]))
        path = f"{cache_path}data/{data_name.split('.')[0]}.bin"
        start_time = time.time()
        if os.path.exists(path):
            index = faiss.read_index(path)
        else:
            corpus_embeddings = bgem3.encode(corpus)
            print("shape of the corpus embeddings:", corpus_embeddings.shape)
            print("data type of the embeddings: ", corpus_embeddings.dtype)
            dim = corpus_embeddings.shape[-1]
            index = faiss.index_factory(dim, 'Flat', faiss.METRIC_INNER_PRODUCT)
            corpus_embeddings = corpus_embeddings.astype(np.float32)
            index.train(corpus_embeddings)
            index.add(corpus_embeddings)
            print(f"total number of vectors: {index.ntotal}")

            faiss.write_index(index, path)
        corpus = np.asarray(corpus)
        corpus_embeddings = bgem3.encode_queries(corpus)
        corpus_embeddings = corpus_embeddings[:].astype(np.float32)
        score, idx = index.search(corpus_embeddings, k=topk)
        context_rank = idx
        bgem3 = None
        duration_time = time.time() - start_time
        print(f"embedding time: {duration_time}")
    context_rank = context_rank if preprocess == True else []
    corpus_lens = corpus_lens if preprocess == True else []
    return batch_data, batch_tokens,question_list, real_answer_list, stop_token_id, \
        reprocess_path, preprocess_path, csv_path,\
            data_name_prefix, rouge_metrics, context_rank, corpus_lens
