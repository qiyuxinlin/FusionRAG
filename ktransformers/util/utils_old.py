#!/usr/bin/env python
# coding=utf-8
'''
Description  :
Author       : Boxin Zhang, Azure-Tang
Version      : 0.1.0
Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
'''
import torch
from torch import nn
import itertools
import time
import enum
import re
import string
import json
import collections
import numpy as np
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
# RoPE 辅助函数
# ============================================================

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb_single(x, cos, sin):
    """Apply Rotary Position Embedding to a single tensor (key or value).

    Args:
        x: Input tensor (key or value)
        cos: Cosine part of rotary embedding
        sin: Sine part of rotary embedding

    Returns:
        Tensor with rotary position embedding applied
    """
    cos = cos.unsqueeze(1)  # Add head dimension
    sin = sin.unsqueeze(1)
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed


def load_kv_and_generate(model, tokenizer, past_key_values, passages,
                          load_path='', doc_ids=None, max_new_tokens=1, revert_rope=False,
                          reprocess_method='normal', rate=0, preprocess=False, draft_model=None,
                          draft_attention=None, use_entropy_selection=False, entropy_top_k=4,
                          draft_layer_selection='entropy',  # 'entropy', 'last', 'fixed', or 'middle'
                          draft_fixed_layer=3,  # 固定使用哪一层 (当 draft_layer_selection='fixed' 时生效)
                          draft_threshold_factor=0.5,  # smart_query_selection 阈值因子 (default 0.5)
                          use_similarity_rerank=False,  # 使用 query-doc 相似度重排序改进选择
                          rerank_multiplier=2.0,  # 重排序时先选择多少倍候选
                          group=False, device="cuda", device_map=None,
                          vattention_topk_ratio=0.5,  # vAttention: top-k 占总 budget 的比例
                          # OracleDynamic 参数
                          epsilon=0.1,  # 误差容忍度 (如 0.1 = 10% 相对误差)
                          delta=0.05,   # 置信度 (如 0.05 = 95% 置信)
                          min_rate=0.05,  # 动态 budget 的最小比例
                          max_rate=0.5,  # 动态 budget 的最大比例
                          query_text='',  # 用于 DraftModelDynamic 的问题文本
                          # DraftModelLayerwise 参数
                          layerwise_decay='linear',  # 'linear', 'exponential', 'cosine', 'step'
                          layerwise_final_rate=0.05,  # 最后一层的 rate
                          # 文本块1用原始KV cache (prefix cache)
                          original_kv_path=None):  # 原始KV cache路径，用于文本块1 (doc_id=first document)
    import os

    # Determine input device: use first GPU if device_map provided, otherwise use device
    input_device = "cuda:0" if device_map is not None else device

    passages_len = [passage.shape[0] for passage in passages]
    passages_start = [sum(passages_len[:i]) for i in range(1,len(passages_len))]
    query_prefix_len = len(tokenizer.encode(tokenizer.decode(passages[-1]).split('Question: ')[0]))
    inputs = passages[-1][query_prefix_len:].unsqueeze(0).to(input_device)
    seq_length = passages[-1][query_prefix_len:].shape[0]

    # 用于存储额外信息（如 OracleDynamic 的动态 rate）
    extra_info = {
        'dynamic_rate': None,
        'topk_coverage': None,
        'topk_count_for_coverage': None,
        'normalized_entropy': None,
        'total_budget': None,
        'doc_len': None
    }

    # load KV

    for layer_idx in range(len(past_key_values.key_cache)):
        past_key_values.past_tokens[layer_idx] = 0
    past_len = 0
    system_len = passages[0].shape[0]

    # Define global system prompt ID
    SYSTEM_PROMPT_ID = -1

    # Check doc_ids parameter
    if doc_ids is None:
        raise ValueError("doc_ids parameter is required")

    key_cache = []
    value_cache = []
    all_position_ids = [torch.arange(0,system_len).unsqueeze(0).to(input_device)]

    # ========== ONLINE LAZY LOADING: Check which documents have no KV cache ==========
    missing_chunks = []  # Track documents without KV cache: [(idx, doc_id, passage)]

    for idx, passage in enumerate(passages[:-1]):
        doc_id = doc_ids[idx]
        passage_len = passage.shape[0]

        # 对于第一个文档（索引1），如果提供了 original_kv_path，则从原始路径加载
        # 这样第一个文档可以使用没有 preprocess 的 KV cache (prefix cache hit)
        if idx == 1 and original_kv_path is not None:
            kv_path = original_kv_path
        else:
            kv_path = load_path

        # New path format using global doc_id
        key_path = f'{kv_path}/doc_{doc_id}_key.pt'
        value_path = f'{kv_path}/doc_{doc_id}_value.pt'

        if os.path.exists(key_path) and os.path.exists(value_path):
            # KV cache exists - try to load it
            # Note: Saved as list of layers, each layer is [1, num_heads, seq_len, head_dim]
            try:
                chunk_key_cache = torch.load(key_path, weights_only=True)
                chunk_value_cache = torch.load(value_path, weights_only=True)

                # Move to CPU if not already
                if isinstance(chunk_key_cache, list):
                    chunk_key_cache = [k.to('cpu') if k.device.type != 'cpu' else k for k in chunk_key_cache]
                    chunk_value_cache = [v.to('cpu') if v.device.type != 'cpu' else v for v in chunk_value_cache]
                else:
                    # Old format (single tensor) - shouldn't happen after clearing cache
                    chunk_key_cache = chunk_key_cache.to('cpu')
                    chunk_value_cache = chunk_value_cache.to('cpu')

                key_cache.append(chunk_key_cache)
                value_cache.append(chunk_value_cache)
            except Exception as e:
                # Corrupted cache file - delete and treat as missing
                print(f"  ⚠ Doc {doc_id}: KV cache corrupted ({e}), deleting and will regenerate")
                try:
                    if os.path.exists(key_path):
                        os.remove(key_path)
                    if os.path.exists(value_path):
                        os.remove(value_path)
                except:
                    pass
                key_cache.append(None)
                value_cache.append(None)
                missing_chunks.append((idx, doc_id, passage))
        else:
            # KV cache missing - will generate during forward pass
            print(f"  ⚠ Doc {doc_id}: KV cache not found, will generate during answer generation")
            key_cache.append(None)  # Placeholder
            value_cache.append(None)
            missing_chunks.append((idx, doc_id, passage))

    start_time = time.time()
    for idx, passage in enumerate(passages[:-1]):
        doc_id = doc_ids[idx]
        passage_len = passage.shape[0]

        # Skip missing chunks - they will be generated during forward pass
        if key_cache[idx] is None:
            continue

        # chunk_key_cache and chunk_value_cache are lists (one per layer)
        chunk_key_cache = key_cache[idx]
        chunk_value_cache = value_cache[idx]

        # Verify it's the correct format
        if isinstance(chunk_key_cache, list):
            # New format: list of layers
            assert passage_len == chunk_key_cache[0].shape[2], f"passage_len={passage_len}, but KV shape={chunk_key_cache[0].shape}"

            # Apply RoPE adjustment if needed
            if revert_rope and doc_id != SYSTEM_PROMPT_ID:
                for layer_idx in range(len(chunk_key_cache)):
                    # Get the device of the rotary embedding layer
                    rotary_emb = model.model.layers[layer_idx].self_attn.rotary_emb
                    if hasattr(rotary_emb, 'inv_freq') and rotary_emb.inv_freq is not None:
                        rotary_device = rotary_emb.inv_freq.device
                    else:
                        rotary_device = next(model.model.layers[layer_idx].parameters()).device

                    layer_chunk_key = chunk_key_cache[layer_idx].to(rotary_device)
                    layer_chunk_value = chunk_value_cache[layer_idx].to(rotary_device)

                    position_ids = torch.full((1, layer_chunk_key.shape[2]), past_len - system_len, device=rotary_device)
                    cos, sin = rotary_emb(layer_chunk_value, position_ids)
                    cos = cos.unsqueeze(1)
                    sin = sin.unsqueeze(1)

                    # Apply RoPE
                    layer_chunk_key = (layer_chunk_key * cos) + (rotate_half(layer_chunk_key) * sin)

                    # Update in list
                    chunk_key_cache[layer_idx] = layer_chunk_key.to(input_device)
                    chunk_value_cache[layer_idx] = layer_chunk_value.to(input_device)
            else:
                # Just move to input_device
                chunk_key_cache = [k.to(input_device) for k in chunk_key_cache]
                chunk_value_cache = [v.to(input_device) for v in chunk_value_cache]

            # Copy to past_key_values
            for layer_idx in range(len(past_key_values.key_cache)):
                past_key_values.key_cache[layer_idx].narrow(2, past_len, passage_len).copy_(chunk_key_cache[layer_idx])
                past_key_values.value_cache[layer_idx].narrow(2, past_len, passage_len).copy_(chunk_value_cache[layer_idx])
                past_key_values.past_tokens[layer_idx] += passage_len
        else:
            # Old format (shouldn't happen after clearing cache) - keep for compatibility
            chunk_key_cache = chunk_key_cache.to(input_device)
            chunk_value_cache = chunk_value_cache.to(input_device)
            assert passage_len == chunk_key_cache.shape[3]

            for layer_idx in range(len(past_key_values.key_cache)):
                past_key_values.key_cache[layer_idx].narrow(2, past_len, passage_len).copy_(chunk_key_cache[layer_idx])
                past_key_values.value_cache[layer_idx].narrow(2, past_len, passage_len).copy_(chunk_value_cache[layer_idx])
                past_key_values.past_tokens[layer_idx] += passage_len
        past_len += passage_len
    storage_time = time.time() - start_time 
    print(f'storage_time: {storage_time}')
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

            k_need_index = v_need_index
        elif reprocess_method == 'FusionRAG':
            select_time = time.time()
            query_prefix_len = len(tokenizer.encode(tokenizer.decode(passages[-1]).split('Question: ')[0]))
            if query_prefix_len >= len(passages[-1]):
                query_prefix_len = len(tokenizer.encode(tokenizer.decode(passages[-1]).split('Question：')[0]))+1

            # Calculate importance using question
            # ONLINE LAZY: For importance calculation, use a temporary past_key_values
            # to avoid writing question KV at wrong position
            inputs = passages[-1][query_prefix_len:].unsqueeze(0).to(input_device)
            seq_length = passages[-1][query_prefix_len:].shape[0]

            with torch.no_grad():
                ss_time = time.perf_counter()
                inputs_embeds = model.model.embed_tokens(inputs).to(input_device)

                if missing_chunks:
                    # When there are missing chunks, use a temporary cache for importance calculation
                    # Create a temporary past_key_values that only contains loaded docs
                    temp_past_key_values = StaticCache(
                        config=past_key_values.config,
                        max_batch_size=past_key_values.max_batch_size,
                        max_cache_len=past_key_values.max_cache_len,
                        device=past_key_values.key_cache[0].device,
                        dtype=past_key_values.key_cache[0].dtype
                    )

                    # Copy loaded KV to temp cache
                    for layer_idx in range(len(past_key_values.key_cache)):
                        temp_past_key_values.key_cache[layer_idx][:, :, :past_len, :] = \
                            past_key_values.key_cache[layer_idx][:, :, :past_len, :].clone()
                        temp_past_key_values.value_cache[layer_idx][:, :, :past_len, :] = \
                            past_key_values.value_cache[layer_idx][:, :, :past_len, :].clone()

                    # Use temp cache for forward (question KV written to temp, not original)
                    cache_position = torch.arange(past_len, past_len + seq_length, device=input_device)
                    model(
                        inputs_embeds = inputs_embeds, past_key_values=temp_past_key_values,
                        cache_position=cache_position, reprocess_method=reprocess_method,
                        return_dict=False, use_cache=True, passages_len=passages_len, history_key_cache=key_cache
                    )

                    # Extract importance from temp cache
                    k_sum = torch.sum(temp_past_key_values.importance_cache[-1], dim=0)[:past_len]
                else:
                    # No missing chunks, use original logic
                    cache_position = torch.arange(past_len, past_len + seq_length, device=input_device)
                    model(
                        inputs_embeds = inputs_embeds, past_key_values=past_key_values,
                        cache_position=cache_position, reprocess_method=reprocess_method,
                        return_dict=False, use_cache=True, passages_len=passages_len, history_key_cache=key_cache
                    )

                    # Calculate k_sum for LOADED documents only
                    k_sum = torch.sum(past_key_values.importance_cache[-1], dim=0)[:past_len]


            k_sum = k_sum.tolist()
            k_sum = k_sum[system_len:]
            k_sum = torch.tensor(k_sum,device=input_device)

            # FIX: 基于实际加载的文档长度，而不是所有文档（包括 missing_chunks）
            loaded_relevant_tokens = past_len - system_len  # 实际已加载的文档 tokens
            k_lens = int(rate * loaded_relevant_tokens)

            if missing_chunks:
                # 日志：说明在有 missing_chunks 时的策略
                all_relevant_tokens = torch.cat(passages[:-1]).shape[0] - system_len
                missing_docs_len = sum([chunk[2].shape[0] for chunk in missing_chunks])
                print(f"    → ONLINE_LAZY mode: {loaded_relevant_tokens} tokens loaded, {missing_docs_len} tokens missing")
                print(f"    → Importance-based selection: {k_lens} tokens from loaded docs (rate={rate:.1f})")
                print(f"    → Missing docs will be added separately (100% coverage)")
            else:
                # 没有 missing_chunks，loaded 就是全部
                pass

            k_need_index = torch.topk(k_sum, k_lens).indices.to('cpu')
            k_need_index = k_need_index + system_len
            print(f'select_time: {time.time() - select_time}')
        elif reprocess_method == 'frontRow':
            k_need_index = []
            for i in range(len(passages_start[:-1])):
                k_need_index.extend(range(passages_start[i], passages_start[i] + int(passages_len[i+1]*rate)))
            k_need_index = torch.tensor(k_need_index)

        
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
                    # FIX: 基于实际加载的文档长度（speculative_prefill 模式下全部加载，但保持一致性）
                    loaded_relevant_tokens = past_len - system_len
                    k_lens = int(rate * loaded_relevant_tokens)

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
                    # FIX: 基于实际加载的文档长度（speculative_prefill 模式下全部加载，但保持一致性）
                    loaded_relevant_tokens = past_len - system_len
                    k_lens = int(rate * loaded_relevant_tokens)
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
                # 文本块1 长度 (用于 prefix cache，不参与重算)
                text_block1_len = passages_len[1]
                # 从文本块2开始选择 (文本块1直接用原始KV cache)
                doc_len = sum(passages_len[2:-1])
                selection_start = system_len + text_block1_len

                # 收集所有候选层的 attention（后 1/2 的层，用于熵选层）
                # 注意: 从 selection_start 开始，长度为 doc_len（只包括文本块2到文本块n）
                candidate_start = num_layers // 2
                layer_attention_dict = {}
                for layer_idx in range(candidate_start, num_layers):
                    layer_attn = past_key_values.importance_cache[layer_idx][:, selection_start:selection_start + doc_len]
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
                # 注意: selection_start 是选择区域的起始位置（跳过了 system 和 文本块1）
                selected_indices = smart_query_selection(
                    attention_scores=multi_layer_attn,
                    doc_len=doc_len,
                    target_ratio=rate,
                    system_len=selection_start,  # 使用 selection_start 作为偏移量
                    device=input_device
                )
                # 转成 tensor 以与后续 torch.sort 兼容
                k_need_index = torch.tensor(selected_indices, device='cpu')

                print(f"QueryAttention 选择了 {len(k_need_index)} 个 tokens ({len(k_need_index)/doc_len*100:.1f}%)")
                print(f"使用了 {len(active_layers)} 个层: {active_layers}")
                print(f'select_time: {time.time() - select_time:.3f}s')

        elif reprocess_method == 'DraftModel':
            # DraftModel: 用小模型 prefill 获取 attention，指导 token 选择
            select_time = time.time()

            # query_start 用于 compute_draft_model_attention
            query_start = sum(passages_len[:-1])
            # 文本块1 长度 (用于 prefix cache，不参与重算)
            text_block1_len = passages_len[1]
            # 从文本块2开始选择 (文本块1直接用原始KV cache，不参与重算选择)
            # doc_len = 文本块2 + 文本块3 + ... + 文本块n
            doc_len = sum(passages_len[1:-1])
            # selection_start 跳过 system_prompt 和 文本块1
            selection_start = system_len 

            # 如果没有传入 draft_attention，需要用 draft_model 计算
            if draft_attention is None:
                if draft_model is None:
                    raise ValueError("Either draft_model or draft_attention must be provided for DraftModel method")
                # 构建完整输入
                full_input = torch.cat(passages).unsqueeze(0).to(input_device)
                # 如果使用固定层且层号在前 50%，需要额外计算该层的 attention
                extra_layers = None
                if draft_layer_selection == 'fixed':
                    num_layers = draft_model.config.num_hidden_layers
                    if draft_fixed_layer < num_layers // 2:
                        extra_layers = [draft_fixed_layer]
                        print(f"  固定层 {draft_fixed_layer} 在前半部分，额外计算其 attention")
                draft_attention = compute_draft_model_attention(draft_model, full_input, query_start, input_device, extra_layers=extra_layers)
                torch.cuda.empty_cache()

            # draft_attention 是 {layer_idx: attention [num_heads, query_len, seq_len]} 格式（内存优化版本）
            # 已经只包含 query positions 的 attention，不需要再切片 query 维度

            # 收集各层的 query→doc attention
            # 注意: 从 selection_start 开始，长度为 doc_len（包括文本块1到文本块n）
            layer_attention_dict = {}
            for layer_idx, layer_attn in draft_attention.items():
                # layer_attn: [num_heads, query_len, seq_len]
                # 提取 query→doc attention（只切片 key 维度，跳过 system 和 文本块1）
                query_to_doc = layer_attn[:, :, selection_start:selection_start + doc_len]
                # 对 heads 和 query positions 平均
                doc_attention_avg = query_to_doc.mean(axis=(0, 1))  # [doc_len]
                layer_attention_dict[layer_idx] = torch.tensor(doc_attention_avg, device=input_device)

            # 选择用于聚合的层
            if draft_layer_selection == 'entropy':
                # 基于熵动态选层
                active_layers, layer_entropy = entropy_layer_selection(
                    layer_attention_dict, top_k=entropy_top_k, return_entropy=True
                )
                print(f"  DraftModel 熵选层: 选择了 {active_layers}")
                # 聚合选中层的 attention
                layer_attentions = [layer_attention_dict[idx] for idx in active_layers]
                multi_layer_attn = torch.stack(layer_attentions).mean(dim=0)  # [doc_len]
            elif draft_layer_selection == 'last':
                # 只使用最后一层
                last_layer_idx = max(layer_attention_dict.keys())
                active_layers = [last_layer_idx]
                print(f"  DraftModel 使用最后一层: Layer {last_layer_idx}")
                multi_layer_attn = layer_attention_dict[last_layer_idx]
            elif draft_layer_selection == 'fixed':
                # 使用固定层
                if draft_fixed_layer in layer_attention_dict:
                    active_layers = [draft_fixed_layer]
                    print(f"  DraftModel 使用固定层: Layer {draft_fixed_layer}")
                    multi_layer_attn = layer_attention_dict[draft_fixed_layer]
                else:
                    # 如果指定层不在 attention dict 中，回退到使用最接近的层
                    available_layers = sorted(layer_attention_dict.keys())
                    closest_layer = min(available_layers, key=lambda x: abs(x - draft_fixed_layer))
                    active_layers = [closest_layer]
                    print(f"  DraftModel 固定层 {draft_fixed_layer} 不可用, 使用最接近的层: Layer {closest_layer}")
                    multi_layer_attn = layer_attention_dict[closest_layer]
            elif draft_layer_selection == 'middle':
                # 使用中间层 (40%-60% 位置的层)
                # 实验发现中间层与 7B 模型选择更相似
                available_layers = sorted(layer_attention_dict.keys())
                num_layers = draft_model.config.num_hidden_layers if draft_model is not None else max(available_layers) + 1
                mid_start = int(0.4 * num_layers)
                mid_end = int(0.6 * num_layers)
                middle_layers = [l for l in range(mid_start, mid_end + 1) if l in available_layers]
                if len(middle_layers) >= entropy_top_k:
                    active_layers = middle_layers[:entropy_top_k]
                elif len(middle_layers) > 0:
                    active_layers = middle_layers
                else:
                    # fallback: 如果中间层不可用，使用可用层中最接近中间的
                    mid_point = num_layers // 2
                    active_layers = sorted(available_layers, key=lambda x: abs(x - mid_point))[:entropy_top_k]
                print(f"  DraftModel 使用中间层: {active_layers}")
                layer_attentions = [layer_attention_dict[idx] for idx in active_layers]
                multi_layer_attn = torch.stack(layer_attentions).mean(dim=0)  # [doc_len]
            else:
                raise ValueError(f"Unknown draft_layer_selection: {draft_layer_selection}, expected 'entropy', 'last', 'fixed', or 'middle'")

            # 使用 smart_query_selection 进行选择
            # 注意: selection_start 是选择区域的起始位置（跳过了 system 和 文本块1）

            if use_similarity_rerank and draft_model is not None:
                # 使用相似度重排序改进选择
                # 关键改进: 先用 smart_query_selection 选候选，保留连通分量和边界扩展
                print(f"  使用相似度重排序 (multiplier={rerank_multiplier})...")

                # 计算 query-doc 相似度
                similarity_scores = compute_query_doc_similarity(
                    draft_model, full_input, selection_start, selection_start + doc_len,
                    query_start, input_device
                )

                target_count = int(doc_len * rate)

                # 先用 smart_query_selection 选择 rerank_multiplier 倍候选
                # 这样保留了连通分量分析和边界扩展的优势
                candidate_ratio = min(rate * rerank_multiplier, 1.0)
                candidates_global = smart_query_selection(
                    attention_scores=multi_layer_attn,
                    doc_len=doc_len,
                    target_ratio=candidate_ratio,
                    system_len=selection_start,
                    device=input_device,
                    threshold_factor=draft_threshold_factor
                )

                # 转换为相对于 doc 的位置
                candidates_local = [pos - selection_start for pos in candidates_global]

                # 在候选中按相似度排序
                candidate_sim = [(pos, similarity_scores[pos].item()) for pos in candidates_local]
                candidate_sim.sort(key=lambda x: x[1], reverse=True)

                # 选择相似度最高的 target_count 个
                selected_local = [pos for pos, _ in candidate_sim[:target_count]]
                selected_indices = [pos + selection_start for pos in sorted(selected_local)]

                print(f"  相似度重排序完成: {len(candidates_global)} 候选 -> {len(selected_indices)} 最终选择")
            else:
                # 原始方法 (使用可配置的 threshold_factor)
                selected_indices = smart_query_selection(
                    attention_scores=multi_layer_attn,
                    doc_len=doc_len,
                    target_ratio=rate,
                    system_len=selection_start,  # 使用 selection_start 作为偏移量
                    device=input_device,
                    threshold_factor=draft_threshold_factor
                )
            k_need_index = torch.tensor(selected_indices, device='cpu')

            print(f"DraftModel 选择了 {len(k_need_index)} 个 tokens (从文本块2-n中选{len(k_need_index)/doc_len*100:.1f}%), 文本块1({text_block1_len}tokens)用prefix cache, threshold={draft_threshold_factor}")
            print(f'select_time: {time.time() - select_time:.3f}s')

        elif reprocess_method == 'DynamicDraftModel':
            # DynamicDraftModel: 基于 3B 模型 attention 分布特征的动态重算比例
            #
            # 核心发现（来自 Type A vs Type B 分析，effect size > 0.4）：
            # - Type B（需要高rate）: attention 更分散，peak 更低，concentration 更低
            # - Type A（不需要高rate）: attention 更集中，peak 更高
            #
            # 正确策略：
            # - 当 attention 分散（peak 低，concentration 低，entropy 高）→ 需要高 rate
            # - 当 attention 集中 → 可以用低 rate
            #
            # 关键指标（按 effect size 排序）：
            # 1. top10_concentration: Type A=0.1448, Type B=0.1284 (effect=0.615)
            # 2. coverage_50: Type A=0.0659, Type B=0.0760 (effect=0.590)
            # 3. peak_strength: Type A=0.0044, Type B=0.0038 (effect=0.526)

            select_time = time.time()

            # query_start 用于 compute_draft_model_attention
            query_start = sum(passages_len[:-1])
            # 文本块1 长度 (用于 prefix cache，不参与重算)
            text_block1_len = passages_len[1]
            # 从文本块2开始选择 (文本块1直接用原始KV cache)
            doc_len = sum(passages_len[2:-1])
            selection_start = system_len + text_block1_len

            # 如果没有传入 draft_attention，需要用 draft_model 计算
            if draft_attention is None:
                if draft_model is None:
                    raise ValueError("Either draft_model or draft_attention must be provided for DynamicDraftModel method")
                full_input = torch.cat(passages).unsqueeze(0).to(input_device)
                draft_attention = compute_draft_model_attention(draft_model, full_input, query_start, input_device)
                torch.cuda.empty_cache()

            # 收集各层的 query→doc attention
            layer_attention_dict = {}
            all_layer_attentions = []

            for layer_idx, layer_attn in draft_attention.items():
                query_to_doc = layer_attn[:, :, selection_start:selection_start + doc_len]
                doc_attention_avg = query_to_doc.mean(axis=(0, 1))
                layer_attention_dict[layer_idx] = torch.tensor(doc_attention_avg, device=input_device)
                all_layer_attentions.append(doc_attention_avg)

            # 基于熵选层
            if draft_layer_selection == 'entropy':
                active_layers, layer_entropy = entropy_layer_selection(
                    layer_attention_dict, top_k=entropy_top_k, return_entropy=True
                )
                print(f"  DynamicDraftModel 熵选层: 选择了 {active_layers}")
                layer_attentions = [layer_attention_dict[idx] for idx in active_layers]
                multi_layer_attn = torch.stack(layer_attentions).mean(dim=0)
            elif draft_layer_selection == 'last':
                last_layer_idx = max(layer_attention_dict.keys())
                active_layers = [last_layer_idx]
                print(f"  DynamicDraftModel 使用最后一层: Layer {last_layer_idx}")
                multi_layer_attn = layer_attention_dict[last_layer_idx]
            else:
                # 默认使用熵选层
                active_layers, _ = entropy_layer_selection(layer_attention_dict, top_k=entropy_top_k)
                layer_attentions = [layer_attention_dict[idx] for idx in active_layers]
                multi_layer_attn = torch.stack(layer_attentions).mean(dim=0)

            # =========================================================================
            # 计算 3B 模型的不确定性特征
            # =========================================================================
            aggregated_attn = multi_layer_attn.cpu()
            uncertainty_features = {}

            # Feature 1: Peak strength (attention 峰值强度)
            # 低 peak = attention 分散 → 需要高 rate (Type B: 0.0038, Type A: 0.0044)
            peak_strength = aggregated_attn.max().item()
            uncertainty_features['peak_strength'] = peak_strength

            # Feature 2: Top-10 concentration (前10个token占总attention的比例)
            # 低 concentration = attention 分散 → 需要高 rate (Type B: 0.1284, Type A: 0.1448)
            sorted_attn, _ = torch.sort(aggregated_attn, descending=True)
            total = sorted_attn.sum()
            top10_concentration = sorted_attn[:10].sum().item() / total.item() if total > 0 else 0
            uncertainty_features['top10_concentration'] = top10_concentration

            # Feature 3: Coverage 50% ratio (覆盖50% attention所需token比例)
            # 高 coverage = attention 分散 → 需要高 rate (Type B: 0.0760, Type A: 0.0659)
            cumsum = torch.cumsum(sorted_attn, dim=0)
            coverage_50 = (cumsum >= 0.5 * total).nonzero(as_tuple=True)[0]
            if len(coverage_50) > 0:
                tokens_for_50 = coverage_50[0].item() + 1
            else:
                tokens_for_50 = len(aggregated_attn)
            coverage_50_ratio = tokens_for_50 / len(aggregated_attn)
            uncertainty_features['coverage_50_ratio'] = coverage_50_ratio

            # Feature 4: Normalized entropy
            # 高 entropy = attention 分散 → 需要高 rate (Type B: 0.8610, Type A: 0.8504)
            p = aggregated_attn / (aggregated_attn.sum() + 1e-10)
            p = torch.clamp(p, min=1e-10)
            entropy = -(p * torch.log(p)).sum()
            max_entropy = np.log(len(aggregated_attn))
            normalized_entropy = (entropy / max_entropy).item()
            uncertainty_features['normalized_entropy'] = normalized_entropy

            # 保留 layer consistency 用于分析
            top_k_per_layer = min(50, doc_len)
            layer_top_tokens = []
            for idx in active_layers:
                top_indices = torch.topk(layer_attention_dict[idx], top_k_per_layer).indices
                layer_top_tokens.append(set(top_indices.tolist()))

            layer_consistency_scores = []
            for i in range(len(layer_top_tokens) - 1):
                intersection = len(layer_top_tokens[i] & layer_top_tokens[i+1])
                union = len(layer_top_tokens[i] | layer_top_tokens[i+1])
                if union > 0:
                    layer_consistency_scores.append(intersection / union)

            layer_consistency = np.mean(layer_consistency_scores) if layer_consistency_scores else 0.5
            uncertainty_features['layer_consistency'] = layer_consistency

            # =========================================================================
            # 基于 attention 分散程度计算动态 rate
            # =========================================================================
            # 阈值（来自 Type A vs Type B 分析，取建议阈值）
            # Type A (不需要高rate): peak=0.0044, top10=0.1448, coverage_50=0.0659, entropy=0.8504
            # Type B (需要高rate):   peak=0.0038, top10=0.1284, coverage_50=0.0760, entropy=0.8610

            peak_threshold = 0.0041  # 低于此值 → 需要高 rate
            top10_threshold = 0.1366  # 低于此值 → 需要高 rate
            coverage_50_threshold = 0.0710  # 高于此值 → 需要高 rate
            entropy_threshold = 0.8557  # 高于此值 → 需要高 rate

            # 计算 dispersion score (0-1)
            # 越高表示 attention 越分散，需要越高的 rate
            dispersion = 0.0
            reasons = []

            # Peak strength contribution (低 peak = 分散)
            # effect size: 0.526, weight: 0.30
            if peak_strength < peak_threshold:
                peak_factor = min((peak_threshold - peak_strength) / 0.001, 1.0)
                dispersion += 0.30 * peak_factor
                reasons.append(f"low_peak({peak_strength:.4f})")

            # Top-10 concentration contribution (低 concentration = 分散)
            # effect size: 0.615 (最高), weight: 0.35
            if top10_concentration < top10_threshold:
                conc_factor = min((top10_threshold - top10_concentration) / 0.02, 1.0)
                dispersion += 0.35 * conc_factor
                reasons.append(f"low_top10({top10_concentration:.4f})")

            # Coverage 50% contribution (高 coverage = 分散)
            # effect size: 0.590, weight: 0.25
            if coverage_50_ratio > coverage_50_threshold:
                cov_factor = min((coverage_50_ratio - coverage_50_threshold) / 0.02, 1.0)
                dispersion += 0.25 * cov_factor
                reasons.append(f"high_cov50({coverage_50_ratio:.4f})")

            # Normalized entropy contribution (高 entropy = 分散)
            # effect size: 0.470, weight: 0.10
            if normalized_entropy > entropy_threshold:
                ent_factor = min((normalized_entropy - entropy_threshold) / 0.02, 1.0)
                dispersion += 0.10 * ent_factor
                reasons.append(f"high_entropy({normalized_entropy:.4f})")

            # Map dispersion to rate
            # rate 作为 base_rate，min_rate 和 max_rate 作为范围
            base_rate = rate
            dynamic_rate = base_rate + dispersion * (max_rate - base_rate)
            dynamic_rate = max(min_rate, min(max_rate, dynamic_rate))

            reason_str = ", ".join(reasons) if reasons else "concentrated"

            print(f"\n  DynamicDraftModel Attention Dispersion Features:")
            print(f"    peak_strength: {peak_strength:.5f} (threshold: <{peak_threshold})")
            print(f"    top10_concentration: {top10_concentration:.4f} (threshold: <{top10_threshold})")
            print(f"    coverage_50_ratio: {coverage_50_ratio:.4f} (threshold: >{coverage_50_threshold})")
            print(f"    normalized_entropy: {normalized_entropy:.4f} (threshold: >{entropy_threshold})")
            print(f"    dispersion_score: {dispersion:.3f}")
            print(f"  Dynamic rate: {dynamic_rate:.3f} ({reason_str})")
            print(f"  Rate range: base={base_rate}, min={min_rate}, max={max_rate}")

            # 使用动态计算的 rate 进行 token 选择
            selected_indices = smart_query_selection(
                attention_scores=multi_layer_attn,
                doc_len=doc_len,
                target_ratio=dynamic_rate,
                system_len=selection_start,
                device=input_device,
                threshold_factor=draft_threshold_factor
            )
            k_need_index = torch.tensor(selected_indices, device='cpu')

            # 保存动态 rate 信息
            extra_info['dynamic_rate'] = dynamic_rate
            extra_info['dispersion_score'] = dispersion
            extra_info['peak_strength'] = peak_strength
            extra_info['top10_concentration'] = top10_concentration
            extra_info['coverage_50_ratio'] = coverage_50_ratio
            extra_info['normalized_entropy'] = normalized_entropy
            extra_info['layer_consistency'] = layer_consistency
            extra_info['doc_len'] = doc_len
            extra_info['total_budget'] = len(k_need_index)

            print(f"DynamicDraftModel 选择了 {len(k_need_index)} 个 tokens ({len(k_need_index)/doc_len*100:.1f}%)")
            print(f'select_time: {time.time() - select_time:.3f}s')

        elif reprocess_method == 'DraftModelDynamic':
            # DraftModelDynamic: 用小模型 prefill 获取 attention，动态计算重算比例
            select_time = time.time()

            # query_start 用于 compute_draft_model_attention
            query_start = sum(passages_len[:-1])
            # 文本块1 长度 (用于 prefix cache，不参与重算)
            text_block1_len = passages_len[1]
            # 从文本块2开始选择 (文本块1直接用原始KV cache)
            doc_len = sum(passages_len[2:-1])
            selection_start = system_len + text_block1_len

            # 如果没有传入 draft_attention，需要用 draft_model 计算
            if draft_attention is None:
                if draft_model is None:
                    raise ValueError("Either draft_model or draft_attention must be provided for DraftModelDynamic method")
                full_input = torch.cat(passages).unsqueeze(0).to(input_device)
                draft_attention = compute_draft_model_attention(draft_model, full_input, query_start, input_device)
                torch.cuda.empty_cache()

            # 收集各层的 query→doc attention
            layer_attention_dict = {}
            all_layer_attentions = []  # 用于动态rate计算

            for layer_idx, layer_attn in draft_attention.items():
                query_to_doc = layer_attn[:, :, selection_start:selection_start + doc_len]
                doc_attention_avg = query_to_doc.mean(axis=(0, 1))
                layer_attention_dict[layer_idx] = torch.tensor(doc_attention_avg, device=input_device)
                all_layer_attentions.append(doc_attention_avg)

            # 基于熵选层
            if draft_layer_selection == 'entropy':
                active_layers, layer_entropy = entropy_layer_selection(
                    layer_attention_dict, top_k=entropy_top_k, return_entropy=True
                )
                print(f"  DraftModelDynamic 熵选层: 选择了 {active_layers}")
                layer_attentions = [layer_attention_dict[idx] for idx in active_layers]
                multi_layer_attn = torch.stack(layer_attentions).mean(dim=0)
            elif draft_layer_selection == 'last':
                last_layer_idx = max(layer_attention_dict.keys())
                active_layers = [last_layer_idx]
                print(f"  DraftModelDynamic 使用最后一层: Layer {last_layer_idx}")
                multi_layer_attn = layer_attention_dict[last_layer_idx]
            elif draft_layer_selection == 'middle':
                # 使用中间层 (40%-60% 位置的层)
                available_layers = sorted(layer_attention_dict.keys())
                num_layers = draft_model.config.num_hidden_layers if draft_model is not None else max(available_layers) + 1
                mid_start = int(0.4 * num_layers)
                mid_end = int(0.6 * num_layers)
                middle_layers = [l for l in range(mid_start, mid_end + 1) if l in available_layers]
                if len(middle_layers) >= entropy_top_k:
                    active_layers = middle_layers[:entropy_top_k]
                elif len(middle_layers) > 0:
                    active_layers = middle_layers
                else:
                    mid_point = num_layers // 2
                    active_layers = sorted(available_layers, key=lambda x: abs(x - mid_point))[:entropy_top_k]
                print(f"  DraftModelDynamic 使用中间层: {active_layers}")
                layer_attentions = [layer_attention_dict[idx] for idx in active_layers]
                multi_layer_attn = torch.stack(layer_attentions).mean(dim=0)
            elif draft_layer_selection == 'fixed':
                if draft_fixed_layer in layer_attention_dict:
                    active_layers = [draft_fixed_layer]
                    print(f"  DraftModelDynamic 使用固定层: Layer {draft_fixed_layer}")
                    multi_layer_attn = layer_attention_dict[draft_fixed_layer]
                else:
                    available_layers = sorted(layer_attention_dict.keys())
                    closest_layer = min(available_layers, key=lambda x: abs(x - draft_fixed_layer))
                    active_layers = [closest_layer]
                    print(f"  DraftModelDynamic 固定层 {draft_fixed_layer} 不可用, 使用最接近的层: Layer {closest_layer}")
                    multi_layer_attn = layer_attention_dict[closest_layer]
            else:
                raise ValueError(f"Unknown draft_layer_selection: {draft_layer_selection}")

            # =========================================================================
            # 动态计算 rate
            # =========================================================================
            aggregated_attn = multi_layer_attn.cpu().numpy()
            attn_features = {}

            # Feature 1: 归一化熵
            p_agg = aggregated_attn / (aggregated_attn.sum() + 1e-10)
            p_agg = np.clip(p_agg, 1e-10, 1.0)
            attn_entropy = -np.sum(p_agg * np.log(p_agg))
            max_entropy = np.log(doc_len) if doc_len > 0 else 1
            attn_features['normalized_entropy'] = attn_entropy / max_entropy

            # Feature 2: Coverage ratio
            sorted_indices = np.argsort(aggregated_attn)[::-1]
            sorted_attn = aggregated_attn[sorted_indices]
            cumsum = np.cumsum(sorted_attn) / (sorted_attn.sum() + 1e-10)
            coverage_count = np.searchsorted(cumsum, 0.85) + 1
            attn_features['coverage_85_ratio'] = coverage_count / doc_len

            # Feature 3: Gini coefficient
            sorted_attn_asc = np.sort(aggregated_attn)
            n = len(sorted_attn_asc)
            index = np.arange(1, n + 1)
            gini = ((2 * index - n - 1) * sorted_attn_asc).sum() / (n * sorted_attn_asc.sum() + 1e-10)
            attn_features['gini_coefficient'] = gini

            # Feature 4: 连通分量数量
            mean_attn = np.mean(aggregated_attn)
            std_attn = np.std(aggregated_attn)
            threshold_positions = list(np.where(aggregated_attn > mean_attn + 0.5 * std_attn)[0])
            components = find_connected_components(threshold_positions, max_gap=2)
            attn_features['num_components'] = len(components)

            # Feature 5: 高attention位置跨度
            high_positions = np.where(aggregated_attn > mean_attn + std_attn)[0]
            if len(high_positions) > 1:
                attn_features['high_attn_span'] = (high_positions.max() - high_positions.min()) / doc_len
            else:
                attn_features['high_attn_span'] = 0

            # =========================================================================
            # 动态计算 rate（范围 min_rate ~ max_rate）
            # =========================================================================
            #
            # 经过大量分析发现：
            # - Attention 特征（gini, coverage, cross_doc_entropy 等）区分力都很弱
            # - Easy 和 Medium 问题的特征分布高度重叠（分离度 < 0.5）
            # - 问题的"难度"并不反映在 attention 分布中
            #
            # 因此采用简单优雅的公式，基于 coverage 的物理意义：
            #
            #   rate = coverage_90_ratio
            #
            # 物理意义：
            #   "达到 90% attention 覆盖需要多少比例的 token，就用多少比例去重算"
            #
            # 这个公式简单、有理论依据，且不依赖于复杂的特征工程

            # 计算 coverage_90
            coverage_90_count = np.searchsorted(cumsum, 0.90) + 1
            coverage_90_ratio = coverage_90_count / doc_len
            attn_features['coverage_90_ratio'] = coverage_90_ratio

            # 简洁公式：rate = coverage_90_ratio
            # 但由于 coverage 计算的范围和实际需要的范围有差异，需要缩放
            # 经验缩放系数：0.7（因为 coverage_90 平均约 40%，而最优 rate 约 20-30%）
            scale_factor = 0.7

            dynamic_rate = coverage_90_ratio * scale_factor

            # 确保在范围内
            dynamic_rate = np.clip(dynamic_rate, min_rate, max_rate)

            print(f"\n  DraftModelDynamic Features:")
            print(f"    normalized_entropy: {attn_features['normalized_entropy']:.3f}")
            print(f"    coverage_85_ratio: {attn_features['coverage_85_ratio']:.3f}")
            print(f"    coverage_90_ratio: {coverage_90_ratio:.3f}")
            print(f"    gini_coefficient: {attn_features['gini_coefficient']:.3f}")
            print(f"    num_components: {attn_features['num_components']}")
            print(f"    high_attn_span: {attn_features['high_attn_span']:.3f}")
            print(f"  Dynamic rate: {dynamic_rate:.3f} (min={min_rate}, max={max_rate})")

            # 使用动态计算的 rate 进行 token 选择
            selected_indices = smart_query_selection(
                attention_scores=multi_layer_attn,
                doc_len=doc_len,
                target_ratio=dynamic_rate,
                system_len=selection_start,
                device=input_device
            )
            k_need_index = torch.tensor(selected_indices, device='cpu')

            # 保存动态 rate 信息
            extra_info['dynamic_rate'] = dynamic_rate
            extra_info['normalized_entropy'] = attn_features['normalized_entropy']
            extra_info['topk_coverage'] = attn_features['coverage_85_ratio']
            extra_info['topk_count_for_coverage'] = coverage_count
            extra_info['doc_len'] = doc_len
            extra_info['total_budget'] = len(k_need_index)

            print(f"DraftModelDynamic 选择了 {len(k_need_index)} 个 tokens ({len(k_need_index)/doc_len*100:.1f}%)")
            print(f'select_time: {time.time() - select_time:.3f}s')

        elif reprocess_method == 'DraftModelLayerwise':
            # DraftModelLayerwise: 用小模型 attention 指导选择，但每层使用不同的重算比例
            # 逐层递减策略: 第0层用 initial_rate，逐层递减到 final_rate
            select_time = time.time()

            # 逐层参数已通过函数参数传入: layerwise_decay, layerwise_final_rate
            initial_rate = rate  # rate 参数作为 initial_rate

            query_start = sum(passages_len[:-1])
            # 文本块1 长度 (用于 prefix cache，不参与重算)
            text_block1_len = passages_len[1]
            # 从文本块2开始选择 (文本块1直接用原始KV cache)
            doc_len = sum(passages_len[2:-1])
            selection_start = system_len + text_block1_len

            print(f"\n{'='*60}")
            print("DraftModelLayerwise: Layer-wise Dynamic Rate")
            print(f"{'='*60}")
            print(f"  Initial rate: {initial_rate:.1%}")
            print(f"  Final rate: {layerwise_final_rate:.1%}")
            print(f"  Decay type: {layerwise_decay}")

            # 计算 draft model attention
            if draft_attention is None:
                if draft_model is None:
                    raise ValueError("Either draft_model or draft_attention must be provided")
                full_input = torch.cat(passages).unsqueeze(0).to(input_device)
                draft_attention = compute_draft_model_attention(draft_model, full_input, query_start, input_device)
                torch.cuda.empty_cache()

            # 收集各层的 query→doc attention
            layer_attention_dict = {}
            for layer_idx, layer_attn in draft_attention.items():
                query_to_doc = layer_attn[:, :, selection_start:selection_start + doc_len]
                doc_attention_avg = query_to_doc.mean(axis=(0, 1))
                layer_attention_dict[layer_idx] = torch.tensor(doc_attention_avg, device=input_device)

            # 基于熵选层
            if draft_layer_selection == 'entropy':
                active_layers, _ = entropy_layer_selection(
                    layer_attention_dict, top_k=entropy_top_k, return_entropy=True
                )
                print(f"  Draft 熵选层: {active_layers}")
                layer_attentions = [layer_attention_dict[idx] for idx in active_layers]
                multi_layer_attn = torch.stack(layer_attentions).mean(dim=0)
            else:
                last_layer_idx = max(layer_attention_dict.keys())
                active_layers = [last_layer_idx]
                print(f"  使用最后一层: Layer {last_layer_idx}")
                multi_layer_attn = layer_attention_dict[last_layer_idx]

            # 计算每层的 rate (使用递减策略)
            num_layers = model.config.num_hidden_layers

            def compute_layer_rate(layer_idx, num_layers, initial, final, decay_type):
                progress = layer_idx / (num_layers - 1) if num_layers > 1 else 0
                if decay_type == "linear":
                    return initial - (initial - final) * progress
                elif decay_type == "exponential":
                    ratio = final / initial if initial > 0 else 1
                    return initial * (ratio ** progress)
                elif decay_type == "cosine":
                    return final + (initial - final) * (1 + np.cos(np.pi * progress)) / 2
                else:  # step
                    if progress < 0.25: return initial
                    elif progress < 0.5: return initial * 0.7
                    elif progress < 0.75: return initial * 0.4
                    else: return final

            # 计算每层的 token selections
            per_layer_selections = []
            layer_rates = []
            for layer_idx in range(num_layers):
                layer_rate = compute_layer_rate(
                    layer_idx, num_layers, initial_rate, layerwise_final_rate, layerwise_decay
                )
                layer_rates.append(layer_rate)

                # 使用 smart_query_selection 为该层选择 tokens
                selected_indices = smart_query_selection(
                    attention_scores=multi_layer_attn,
                    doc_len=doc_len,
                    target_ratio=layer_rate,
                    system_len=selection_start,
                    device=input_device
                )
                per_layer_selections.append(selected_indices)

            # 计算统计信息
            avg_rate = np.mean(layer_rates)
            total_selected = sum(len(s) for s in per_layer_selections)
            total_possible = doc_len * num_layers

            print(f"\n  Per-layer rate statistics:")
            print(f"    Layer 0:  {layer_rates[0]:.1%} ({len(per_layer_selections[0])} tokens)")
            print(f"    Layer {num_layers//2}:  {layer_rates[num_layers//2]:.1%} ({len(per_layer_selections[num_layers//2])} tokens)")
            print(f"    Layer {num_layers-1}: {layer_rates[-1]:.1%} ({len(per_layer_selections[-1])} tokens)")
            print(f"    Average rate: {avg_rate:.1%}")
            print(f"    Total recomputation: {total_selected}/{total_possible} = {total_selected/total_possible*100:.1f}%")

            # 对于 DraftModelLayerwise，我们需要使用自定义的逐层 prefill
            # 但为了保持与现有框架兼容，我们用第 0 层的 selections 作为 k_need_index
            # 然后在 prefill 时使用自定义的逐层处理
            # 将 per_layer_selections 存储在 extra_info 中供后续使用
            k_need_index = torch.tensor(per_layer_selections[0], device='cpu')

            extra_info['layerwise_mode'] = True
            extra_info['per_layer_selections'] = per_layer_selections
            extra_info['layer_rates'] = layer_rates
            extra_info['dynamic_rate'] = avg_rate

            print(f"\nDraftModelLayerwise 完成逐层选择")
            print(f'select_time: {time.time() - select_time:.3f}s')

        elif reprocess_method == 'Oracle':
            # Oracle: 用主模型本身 prefill 获取 attention，指导 token 选择
            # 与 DraftModel 方法相同，唯一区别是使用主模型而非小模型
            select_time = time.time()

            # query_start 用于 compute_draft_model_attention
            query_start = sum(passages_len[:-1])
            # 文本块1 长度 (用于 prefix cache，不参与重算)
            text_block1_len = passages_len[1]
            # 从文本块2开始选择 (文本块1直接用原始KV cache)
            doc_len = sum(passages_len[1:-1])
            selection_start = system_len

            # 使用主模型计算 attention（复用 compute_draft_model_attention 函数）
            print(f"\n{'='*60}")
            print("Oracle: Using main model for attention computation")
            print(f"{'='*60}")
            full_input = torch.cat(passages).unsqueeze(0).to(input_device)
            oracle_attention = compute_draft_model_attention(model, full_input, query_start, input_device)
            torch.cuda.empty_cache()

            # oracle_attention 是 {layer_idx: attention [num_heads, query_len, seq_len]} 格式
            # 已经只包含 query positions 的 attention

            # 收集各层的 query→doc attention
            # 注意: 从 selection_start 开始，长度为 doc_len（只包括文本块2到文本块n）
            layer_attention_dict = {}
            for layer_idx, layer_attn in oracle_attention.items():
                # layer_attn: [num_heads, query_len, seq_len]
                # 提取 query→doc attention（只切片 key 维度，跳过 system 和 文本块1）
                query_to_doc = layer_attn[:, :, selection_start:selection_start + doc_len]
                # 对 heads 和 query positions 平均
                doc_attention_avg = query_to_doc.mean(axis=(0, 1))  # [doc_len]
                layer_attention_dict[layer_idx] = torch.tensor(doc_attention_avg, device=input_device)

            # 选择用于聚合的层
            if draft_layer_selection == 'entropy':
                # 基于熵动态选层
                active_layers, layer_entropy = entropy_layer_selection(
                    layer_attention_dict, top_k=entropy_top_k, return_entropy=True
                )
                print(f"  Oracle 熵选层: 选择了 {active_layers}")
                # 聚合选中层的 attention
                layer_attentions = [layer_attention_dict[idx] for idx in active_layers]
                multi_layer_attn = torch.stack(layer_attentions).mean(dim=0)  # [doc_len]
            elif draft_layer_selection == 'last':
                # 只使用最后一层
                last_layer_idx = max(layer_attention_dict.keys())
                active_layers = [last_layer_idx]
                print(f"  Oracle 使用最后一层: Layer {last_layer_idx}")
                multi_layer_attn = layer_attention_dict[last_layer_idx]
            else:
                raise ValueError(f"Unknown draft_layer_selection: {draft_layer_selection}, expected 'entropy' or 'last'")

            # 使用 smart_query_selection 进行选择
            # 注意: selection_start 是选择区域的起始位置（跳过了 system 和 文本块1）
            selected_indices = smart_query_selection(
                attention_scores=multi_layer_attn,
                doc_len=doc_len,
                target_ratio=rate,
                system_len=selection_start,  # 使用 selection_start 作为偏移量
                device=input_device
            )
            k_need_index = torch.tensor(selected_indices, device='cpu')

            print(f"Oracle 选择了 {len(k_need_index)} 个 tokens ({len(k_need_index)/doc_len*100:.1f}%)")
            print(f'select_time: {time.time() - select_time:.3f}s')

        elif reprocess_method == 'OracleAdaptive':
            # OracleAdaptive: Oracle 选择方式 + 动态比例计算
            # 与 Oracle 相同的选择逻辑 (smart_query_selection: 连通分量 + 边界扩展)
            # 但比例是动态计算的，而非固定值
            select_time = time.time()

            # query_start 用于 compute_draft_model_attention
            query_start = sum(passages_len[:-1])
            # 文本块1 长度 (用于 prefix cache，不参与重算)
            text_block1_len = passages_len[1]
            # 从文本块2开始选择 (文本块1直接用原始KV cache)
            doc_len = sum(passages_len[2:-1])
            selection_start = system_len + text_block1_len

            # 使用主模型计算 attention
            print(f"\n{'='*60}")
            print("OracleAdaptive: Oracle selection + Dynamic rate")
            print(f"{'='*60}")
            full_input = torch.cat(passages).unsqueeze(0).to(input_device)
            oracle_attention = compute_draft_model_attention(model, full_input, query_start, input_device)
            torch.cuda.empty_cache()

            # 收集各层的 query→doc attention
            layer_attention_dict = {}
            for layer_idx, layer_attn in oracle_attention.items():
                query_to_doc = layer_attn[:, :, selection_start:selection_start + doc_len]
                doc_attention_avg = query_to_doc.mean(axis=(0, 1))
                layer_attention_dict[layer_idx] = torch.tensor(doc_attention_avg, device=input_device)

            # 选择用于聚合的层
            if draft_layer_selection == 'entropy':
                active_layers, layer_entropy = entropy_layer_selection(
                    layer_attention_dict, top_k=entropy_top_k, return_entropy=True
                )
                print(f"  熵选层: 选择了 {active_layers}")
                layer_attentions = [layer_attention_dict[idx] for idx in active_layers]
                multi_layer_attn = torch.stack(layer_attentions).mean(dim=0)
            elif draft_layer_selection == 'last':
                last_layer_idx = max(layer_attention_dict.keys())
                active_layers = [last_layer_idx]
                print(f"  使用最后一层: Layer {last_layer_idx}")
                multi_layer_attn = layer_attention_dict[last_layer_idx]
            else:
                raise ValueError(f"Unknown draft_layer_selection: {draft_layer_selection}")

            # 动态计算比例 (使用综合多特征方法)
            budget_info = compute_dynamic_ratio_comprehensive(
                attention_scores=multi_layer_attn,
                doc_len=doc_len,
                base_ratio=rate,  # 使用 rate 作为 base_ratio
                min_ratio=min_rate,
                max_ratio=max_rate
            )

            dynamic_rate = budget_info['dynamic_ratio']

            print(f"  Parameters: base_ratio={rate:.0%}")
            print(f"  Budget range: [{min_rate:.1%}, {max_rate:.1%}]")
            print(f"\n  Attention 分布特征分析:")
            print(f"    Coverage Analysis:")
            for cov, ratio_val in budget_info['coverage_analysis'].items():
                print(f"      {cov} coverage: {ratio_val:.2%}")
            print(f"    Concentration (Top-k):")
            print(f"      Top-1: {budget_info['concentration']['top1_ratio']:.2%}")
            print(f"      Top-3: {budget_info['concentration']['top3_ratio']:.2%}")
            print(f"      Top-5: {budget_info['concentration']['top5_ratio']:.2%}")
            print(f"      → Concentration Factor: {budget_info['concentration']['concentration_factor']:.2f}")
            print(f"    Connected Components: {budget_info['num_components']}")
            print(f"    Position Span: {budget_info['position_span']} tokens")
            print(f"    Spread Ratio: {budget_info['spread_ratio']:.4f}")
            print(f"    Gini Coefficient: {budget_info['gini_coefficient']:.4f}")
            print(f"\n  动态比例计算:")
            print(f"    Base coverage ratio (80%): {budget_info['base_coverage_ratio']:.2%}")
            print(f"    After concentration factor: {budget_info['base_coverage_ratio'] * budget_info['concentration']['concentration_factor']:.2%}")
            print(f"    Adjustments:")
            for adj_name, adj_value in budget_info['adjustments'].items():
                print(f"      {adj_name}: {adj_value:+.4f}")
            print(f"    Raw computed ratio: {budget_info['raw_ratio']:.2%}")
            print(f"    FINAL DYNAMIC RATIO: {dynamic_rate:.2%}")

            # 使用 smart_query_selection 进行选择 (与 Oracle 相同)
            selected_indices = smart_query_selection(
                attention_scores=multi_layer_attn,
                doc_len=doc_len,
                target_ratio=dynamic_rate,  # 使用动态计算的比例
                system_len=selection_start,
                device=input_device
            )
            k_need_index = torch.tensor(selected_indices, device='cpu')

            print(f"\n  OracleAdaptive 选择结果:")
            print(f"    选中 {len(k_need_index)} tokens ({len(k_need_index)/doc_len*100:.1f}%)")
            print(f"    (smart_query_selection: 连通分量 + 边界扩展)")
            print(f"  select_time: {time.time() - select_time:.3f}s")

            # 保存动态信息用于后续统计
            extra_info['dynamic_rate'] = dynamic_rate
            extra_info['base_coverage_ratio'] = budget_info['base_coverage_ratio']
            extra_info['num_components'] = budget_info['num_components']
            extra_info['gini_coefficient'] = budget_info['gini_coefficient']
            extra_info['spread_ratio'] = budget_info['spread_ratio']
            extra_info['concentration_factor'] = budget_info['concentration']['concentration_factor']
            extra_info['top1_ratio'] = budget_info['concentration']['top1_ratio']
            extra_info['total_budget'] = len(k_need_index)
            extra_info['doc_len'] = doc_len

        elif reprocess_method == 'vAttention':
            # vAttention: 结合 top-k 选择和随机采样
            # 参考论文 "vAttention: Verified Sparse Attention" (arXiv:2510.05688)
            # 核心思想：一部分 budget 用于 top-k，一部分用于随机采样
            select_time = time.time()

            # query_start 用于 compute_draft_model_attention
            query_start = sum(passages_len[:-1])
            # 文本块1 长度 (用于 prefix cache，不参与重算)
            text_block1_len = passages_len[1]
            # 从文本块2开始选择 (文本块1直接用原始KV cache)
            doc_len = sum(passages_len[2:-1])
            selection_start = system_len + text_block1_len

            # 计算总的需要选择的 token 数量
            total_budget = int(rate * doc_len)

            # vAttention 参数：topk_ratio 控制 top-k 和随机采样的比例
            # 默认各占 50%（论文中推荐的配置）
            topk_ratio = vattention_topk_ratio

            topk_budget = int(total_budget * topk_ratio)
            random_budget = total_budget - topk_budget

            print(f"\n{'='*60}")
            print("vAttention: Combining Top-k and Random Sampling")
            print(f"{'='*60}")
            print(f"  Total budget: {total_budget} tokens (rate={rate:.2%})")
            print(f"  Top-k budget: {topk_budget} tokens ({topk_ratio:.0%})")
            print(f"  Random budget: {random_budget} tokens ({1-topk_ratio:.0%})")

            # 使用主模型计算 attention（复用 Oracle 的计算方式）
            full_input = torch.cat(passages).unsqueeze(0).to(input_device)
            oracle_attention = compute_draft_model_attention(model, full_input, query_start, input_device)
            torch.cuda.empty_cache()

            # 收集各层的 query→doc attention
            # 注意: 从 selection_start 开始，长度为 doc_len（只包括文本块2到文本块n）
            layer_attention_dict = {}
            for layer_idx, layer_attn in oracle_attention.items():
                query_to_doc = layer_attn[:, :, selection_start:selection_start + doc_len]
                doc_attention_avg = query_to_doc.mean(axis=(0, 1))
                layer_attention_dict[layer_idx] = torch.tensor(doc_attention_avg, device=input_device)

            # 选择用于聚合的层（使用熵选层）
            if draft_layer_selection == 'entropy':
                active_layers, layer_entropy = entropy_layer_selection(
                    layer_attention_dict, top_k=entropy_top_k, return_entropy=True
                )
                print(f"  vAttention 熵选层: 选择了 {active_layers}")
                layer_attentions = [layer_attention_dict[idx] for idx in active_layers]
                multi_layer_attn = torch.stack(layer_attentions).mean(dim=0)
            elif draft_layer_selection == 'last':
                last_layer_idx = max(layer_attention_dict.keys())
                active_layers = [last_layer_idx]
                print(f"  vAttention 使用最后一层: Layer {last_layer_idx}")
                multi_layer_attn = layer_attention_dict[last_layer_idx]
            else:
                raise ValueError(f"Unknown draft_layer_selection: {draft_layer_selection}")

            # Step 1: Top-k 选择
            # 获取 attention 分数最高的 topk_budget 个 token
            if topk_budget > 0:
                topk_values, topk_indices = torch.topk(multi_layer_attn, min(topk_budget, doc_len))
                topk_selected = set(topk_indices.cpu().tolist())
            else:
                topk_selected = set()

            # Step 2: 随机采样
            # 从剩余的 token 中随机采样 random_budget 个
            all_indices = set(range(doc_len))
            remaining_indices = list(all_indices - topk_selected)

            if random_budget > 0 and len(remaining_indices) > 0:
                # 设置随机种子以确保可复现性
                torch.manual_seed(42)
                random_sample_size = min(random_budget, len(remaining_indices))
                random_indices = torch.randperm(len(remaining_indices))[:random_sample_size]
                random_selected = set([remaining_indices[i] for i in random_indices.tolist()])
            else:
                random_selected = set()

            # Step 3: 合并两者
            combined_selected = topk_selected | random_selected

            # 确保不超过总 budget
            if len(combined_selected) > total_budget:
                # 如果超过了，按 attention 分数排序，保留分数最高的
                combined_list = list(combined_selected)
                combined_scores = [(idx, multi_layer_attn[idx].item()) for idx in combined_list]
                combined_scores.sort(key=lambda x: x[1], reverse=True)
                combined_selected = set([idx for idx, _ in combined_scores[:total_budget]])

            # 转换为全局索引（使用 selection_start 作为偏移量，跳过 system 和 文本块1）
            selected_indices = [idx + selection_start for idx in sorted(combined_selected)]
            k_need_index = torch.tensor(selected_indices, device='cpu')

            print(f"\n  vAttention 选择结果:")
            print(f"    Top-k 选中: {len(topk_selected)} tokens")
            print(f"    随机采样选中: {len(random_selected)} tokens")
            print(f"    总计选中: {len(combined_selected)} tokens ({len(combined_selected)/doc_len*100:.1f}%)")
            print(f'  select_time: {time.time() - select_time:.3f}s')

        elif reprocess_method == 'OracleDynamic':
            # OracleDynamic: 动态计算重算比例
            # 参考论文 "vAttention: Verified Sparse Attention" (arXiv:2510.05688)
            # 核心思想：根据 attention 分布的方差动态决定需要重算多少 token
            # 用户指定 epsilon (误差容忍度) 和 delta (置信度)，系统自动计算 budget
            select_time = time.time()

            # query_start 用于 compute_draft_model_attention
            query_start = sum(passages_len[:-1])
            # 文本块1 长度 (用于 prefix cache，不参与重算)
            text_block1_len = passages_len[1]
            # 从文本块2开始选择 (文本块1直接用原始KV cache)
            doc_len = sum(passages_len[2:-1])
            selection_start = system_len + text_block1_len

            print(f"\n{'='*60}")
            print("OracleDynamic: Adaptive Budget Computation")
            print(f"{'='*60}")
            print(f"  Parameters: epsilon={epsilon}, delta={delta}")
            print(f"  Budget range: [{min_rate:.1%}, {max_rate:.1%}]")

            # Step 1: 使用主模型计算 attention
            full_input = torch.cat(passages).unsqueeze(0).to(input_device)
            oracle_attention = compute_draft_model_attention(model, full_input, query_start, input_device)
            torch.cuda.empty_cache()

            # 收集各层的 query→doc attention
            # 注意: 从 selection_start 开始，长度为 doc_len（只包括文本块2到文本块n）
            layer_attention_dict = {}
            for layer_idx, layer_attn in oracle_attention.items():
                query_to_doc = layer_attn[:, :, selection_start:selection_start + doc_len]
                doc_attention_avg = query_to_doc.mean(axis=(0, 1))
                layer_attention_dict[layer_idx] = torch.tensor(doc_attention_avg, device=input_device)

            # 选择用于聚合的层
            if draft_layer_selection == 'entropy':
                active_layers, layer_entropy = entropy_layer_selection(
                    layer_attention_dict, top_k=entropy_top_k, return_entropy=True
                )
                print(f"  熵选层: 选择了 {active_layers}")
                layer_attentions = [layer_attention_dict[idx] for idx in active_layers]
                multi_layer_attn = torch.stack(layer_attentions).mean(dim=0)
            elif draft_layer_selection == 'last':
                last_layer_idx = max(layer_attention_dict.keys())
                active_layers = [last_layer_idx]
                print(f"  使用最后一层: Layer {last_layer_idx}")
                multi_layer_attn = layer_attention_dict[last_layer_idx]
            else:
                raise ValueError(f"Unknown draft_layer_selection: {draft_layer_selection}")

            # Step 2: 动态计算 budget
            budget_info = compute_dynamic_budget(
                attention_scores=multi_layer_attn,
                doc_len=doc_len,
                epsilon=epsilon,
                delta=delta,
                base_sample_ratio=0.05,
                topk_ratio=vattention_topk_ratio,
                min_rate=min_rate,
                max_rate=max_rate
            )

            total_budget = budget_info['total_budget']
            topk_budget = budget_info['topk_budget']
            random_budget = budget_info['random_budget']
            dynamic_rate = budget_info['dynamic_rate']

            print(f"\n  Attention 分布分析:")
            print(f"    覆盖阈值: {budget_info['coverage_threshold']:.1%}")
            print(f"    达到覆盖需要的 top-k 数量: {budget_info['topk_count_for_coverage']} tokens ({budget_info['topk_count_for_coverage']/doc_len*100:.2f}%)")
            print(f"    Top-k 实际覆盖权重: {budget_info['topk_coverage']:.2%}")
            print(f"    残差权重: {budget_info['residual_weight']:.2%}")
            print(f"    残差变异系数 (CV): {budget_info['residual_cv']:.4f}")
            print(f"    归一化熵: {budget_info['normalized_entropy']:.4f} (0=极度集中, 1=均匀分布)")
            print(f"\n  动态 Budget 计算结果:")
            print(f"    原始覆盖率: {budget_info['coverage_based_rate']:.2%}")
            print(f"    限制后比例: {dynamic_rate:.2%} (min={min_rate:.0%}, max={max_rate:.0%})")
            print(f"    总 budget: {total_budget} tokens")

            # Step 3: 纯 Top-k 选择 (和 Oracle 一样，只选 attention 最高的)
            # 注意：不再使用随机采样，因为随机采样的低 attention token 会影响生成质量
            if total_budget > 0:
                topk_values, topk_indices = torch.topk(multi_layer_attn, min(total_budget, doc_len))
                combined_selected = set(topk_indices.cpu().tolist())
            else:
                combined_selected = set()

            # 记录选择信息（保持兼容性）
            topk_selected = combined_selected
            random_selected = set()  # 不再使用随机采样

            # 转换为全局索引（使用 selection_start 作为偏移量，跳过 system 和 文本块1）
            selected_indices = [idx + selection_start for idx in sorted(combined_selected)]
            k_need_index = torch.tensor(selected_indices, device='cpu')

            print(f"\n  OracleDynamic 选择结果:")
            print(f"    选中 Top-{len(combined_selected)} tokens ({len(combined_selected)/doc_len*100:.1f}%)")
            print(f"    (纯 top-k 选择，和 Oracle 相同策略，只是动态计算比例)")
            print(f"  select_time: {time.time() - select_time:.3f}s")

            # 保存动态信息用于后续统计
            extra_info['dynamic_rate'] = dynamic_rate
            extra_info['topk_coverage'] = budget_info['topk_coverage']
            extra_info['topk_count_for_coverage'] = budget_info['topk_count_for_coverage']
            extra_info['normalized_entropy'] = budget_info['normalized_entropy']
            extra_info['total_budget'] = len(combined_selected)
            extra_info['doc_len'] = doc_len

        else:
            raise NotImplementedError

        # reprocess kv cache and prefill question
        # 对于 DraftModelLayerwise，使用所有层选择的并集
        if extra_info.get('layerwise_mode') and extra_info.get('per_layer_selections'):
            per_layer_selections = extra_info['per_layer_selections']
            # 使用所有层选择的并集 (即第 0 层的选择，因为它包含最多 tokens)
            all_positions = set()
            for layer_sel in per_layer_selections:
                all_positions.update(layer_sel)
            k_need_index = sorted(list(all_positions))
            print(f"\n  Layerwise union positions: {len(all_positions)} unique doc positions")
            print(f"  (Note: Using union for forward pass; per-layer rates documented in extra_info)")
        else:
            k_need_index = torch.sort(k_need_index)[0].tolist()

        k_need_index.extend(range(sum(passages_len[:-1]),sum(passages_len)))
    else:
        k_need_index = range(sum(passages_len[:-1]),sum(passages_len))

    # ========== ONLINE LAZY: Add ALL missing chunks tokens to k_need_index ==========
    # This must be done AFTER the if/else block above, for both rate=0 and rate!=0
    if missing_chunks:
        # Convert k_need_index to list if it's a range
        if not isinstance(k_need_index, list):
            k_need_index = list(k_need_index)

        passages_len_cumsum = [sum([p.shape[0] for p in passages[:i+1]]) for i in range(len(passages)-1)]
        for idx, doc_id, passage in missing_chunks:
            chunk_start = passages_len_cumsum[idx-1] if idx > 0 else 0
            chunk_end = passages_len_cumsum[idx]
            k_need_index.extend(range(chunk_start, chunk_end))
        k_need_index = sorted(k_need_index)  # Keep sorted
        print(f"    → Added {sum([chunk[2].shape[0] for chunk in missing_chunks])} missing document tokens to reprocess")

    # ONLINE LAZY: Don't reset past_len - it should remain as the actual loaded length
    # past_len = sum(passages_len)  # This line is for offline mode only
    # In offline mode, all docs are loaded, so past_len == sum(passages_len)
    # In online lazy mode, only loaded docs are in past_key_values
    final_len = sum(passages_len)  # Total length after reprocess

    batch_size, seq_length = 1, len(k_need_index)

    # Use final_len for generated_ids size (includes all passages)
    generated_ids = torch.zeros(
        batch_size, final_len + max_new_tokens + 1, dtype=torch.int, device=input_device
    )
    generated_ids[:, :final_len] = torch.cat(passages).unsqueeze(0).to(input_device)
    tokens = []

    if reprocess_method != 'FusionRAG':
        use_sparse_attention = False
    else:
        use_sparse_attention = False
    reprocess_inputs = torch.cat(passages)[k_need_index].unsqueeze(0).to(input_device)
    cache_position = torch.tensor(k_need_index, device=input_device)

    # Debug: Print cache position info for Online Lazy
    # if missing_chunks:
    #     print(f"  [DEBUG] past_len from load stage: {past_len}")
    #     print(f"  [DEBUG] sum(passages_len): {sum(passages_len)}")
    #     print(f"  [DEBUG] k_need_index length: {len(k_need_index)}")
    #     print(f"  [DEBUG] k_need_index range: [{min(k_need_index)}, {max(k_need_index)}]")
    #     print(f"  [DEBUG] cache_position range: [{cache_position.min().item()}, {cache_position.max().item()}]")

    with torch.no_grad():
        # ONLINE LAZY: Only extract loaded docs, not missing chunks
        # past_len is the actual loaded length (may be < sum(passages_len[:-1]))
        if missing_chunks:
            # Extract only loaded part
            without_attn_value = past_key_values.value_cache[-1].narrow(2, 0, past_len).clone()
        else:
            # All docs loaded
            without_attn_value = past_key_values.value_cache[-1].narrow(2, 0, sum(passages_len[:-1])).clone()

        inputs_embeds = model.model.embed_tokens(reprocess_inputs).to(input_device)

        # Don't force move to input_device - keep on the device where model output is
        # This avoids cross-GPU transfer deadlock in PP mode

        model_output = model(
            inputs_embeds = inputs_embeds, cache_position=cache_position,
            past_key_values=past_key_values, return_dict=False, use_cache=True, use_sparse_attention=use_sparse_attention,
        )[0]

        # ========== ONLINE LAZY: Extract and save missing chunks KV after reprocess ==========
        if missing_chunks:
            print(f"    ⚡ Extracting and saving KV for {len(missing_chunks)} new document(s)...")
            passages_len_cumsum = [sum([p.shape[0] for p in passages[:i+1]]) for i in range(len(passages)-1)]

            for idx, doc_id, passage in missing_chunks:
                # Calculate document position range
                chunk_start = passages_len_cumsum[idx-1] if idx > 0 else 0
                chunk_end = passages_len_cumsum[idx]
                chunk_len = chunk_end - chunk_start

                # Extract key and value for this document (ALL LAYERS)
                num_layers = len(past_key_values.key_cache)
                chunk_key_all_layers = []
                chunk_value_all_layers = []

                for layer_idx in range(num_layers):
                    layer_chunk_key = past_key_values.key_cache[layer_idx][:, :, chunk_start:chunk_end, :].clone()
                    layer_chunk_value = past_key_values.value_cache[layer_idx][:, :, chunk_start:chunk_end, :].clone()

                    # Apply RoPE adjustment: revert to relative position 0 (only for keys)
                    if revert_rope and doc_id != SYSTEM_PROMPT_ID:
                        # Get rotary_emb from this layer
                        rotary_emb = model.model.layers[layer_idx].self_attn.rotary_emb

                        # Original absolute positions [chunk_start, chunk_start+1, ..., chunk_end-1]
                        original_position_ids = torch.arange(chunk_start, chunk_end, device=layer_chunk_key.device).unsqueeze(0)

                        # Target relative positions [0, 1, 2, ..., chunk_len-1]
                        target_position_ids = torch.arange(0, chunk_len, device=layer_chunk_key.device).unsqueeze(0)

                        # Get cos/sin for both positions
                        original_cos, original_sin = rotary_emb(layer_chunk_value, original_position_ids)
                        target_cos, target_sin = rotary_emb(layer_chunk_value, target_position_ids)

                        # Revert original RoPE
                        layer_chunk_key = apply_rotary_pos_emb_single(layer_chunk_key, original_cos, -original_sin)
                        # Apply new RoPE at position 0
                        layer_chunk_key = apply_rotary_pos_emb_single(layer_chunk_key, target_cos, target_sin)

                    chunk_key_all_layers.append(layer_chunk_key)
                    chunk_value_all_layers.append(layer_chunk_value)

                # Save to disk (all layers as a list) using atomic write
                key_save_path = f'{load_path}/doc_{doc_id}_key.pt'
                value_save_path = f'{load_path}/doc_{doc_id}_value.pt'
                key_temp_path = f'{key_save_path}.tmp'
                value_temp_path = f'{value_save_path}.tmp'

                try:
                    # Write to temporary files first
                    torch.save([k.to('cpu') for k in chunk_key_all_layers], key_temp_path)
                    torch.save([v.to('cpu') for v in chunk_value_all_layers], value_temp_path)

                    # Atomic rename (replaces existing file if any)
                    os.rename(key_temp_path, key_save_path)
                    os.rename(value_temp_path, value_save_path)

                    print(f"      ✓ Doc {doc_id} KV generated and saved ({chunk_len} tokens)")
                except Exception as e:
                    # Clean up temporary files on error
                    if os.path.exists(key_temp_path):
                        os.remove(key_temp_path)
                    if os.path.exists(value_temp_path):
                        os.remove(value_temp_path)
                    print(f"      ✗ Failed to save Doc {doc_id} KV: {e}")
                    raise

        logits = model_output[:,-1,:].unsqueeze(0).clone()

        # ONLINE LAZY: Extract same length as without_attn_value
        if missing_chunks:
            # Extract only loaded part (same as without_attn_value)
            with_attn_value = past_key_values.value_cache[-1].narrow(2, 0, past_len).clone()
        else:
            # All docs loaded
            with_attn_value = past_key_values.value_cache[-1].narrow(2, 0, sum(passages_len[:-1])).clone()

        v_sub_all = without_attn_value - with_attn_value
        v_sub_all = v_sub_all.squeeze(0)
        v_sub_all = v_sub_all.transpose(0, 1)
        v_sum = torch.sum(v_sub_all**2, dim=[1,2])   
        
        first_token_time = time.time() - start_time
        stream = TextStreamer(tokenizer)
        logits_warper = tf_logits_warper(temperature=0.01, top_k=1)
        next_token_scores = logits_warper(reprocess_inputs, logits[:, -1, :])
        next_token = torch.argmax(next_token_scores, dim=-1)
        prefill_count = seq_length
        prefill_time = first_token_time
        # print(stream.put(next_token.item()), end="", flush=True)
        generated_ids[:, final_len+1] = next_token
        tokens.append(next_token)

        # Use the device where logits/next_token are (model output device in PP mode)
        output_device = next_token.device
        inputs = torch.cat((torch.cat(passages).unsqueeze(0).to(output_device), next_token.unsqueeze(0)), dim=-1)
        cache_position = torch.tensor([final_len], device=output_device)
        position_ids = cache_position.unsqueeze(0)
        seq_length += 1
        
        decode_time = time.time()
        for _ in range(1, max_new_tokens):
            next_token = decode_one_tokens(model, next_token.unsqueeze(0), position_ids, cache_position, past_key_values, logits_warper, inputs)
            inputs = torch.cat((inputs, next_token.unsqueeze(0)), dim=-1)
            generated_ids[:, cache_position] = next_token.int()
            tokens.append(next_token.int())
            seq_length += 1
            
            if next_token[0].item() == tokenizer.eos_token_id or tokenizer.decode(next_token) == '<|im_end|>':
                break

            cache_position += 1
            position_ids = cache_position.unsqueeze(0)
        

    total_time = time.time() - decode_time
    tokens_generated = len(tokens)
    tokens_per_second = tokens_generated / total_time

    print("")

    # print(f"prompt eval count:    {prefill_count} token(s)")
    # print(f"prompt eval duration: {prefill_time}s")
    # print(f"prompt eval rate:     {prefill_count/prefill_time} tokens/s")
    # print(f"eval count:           {tokens_generated} token(s)")
    # print(f"eval duration:        {total_time}s")
    # print(f"eval rate:            {tokens_per_second} tokens/s")

    return tokens, prefill_time, extra_info

