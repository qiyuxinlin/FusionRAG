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
from ktransformers.util.custom_gguf import translate_name_to_gguf
from ktransformers.util.custom_gguf import GGUFLoader
from ktransformers.operators import base_operator
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

def set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        if hasattr(cur_mod, s):
            cur_mod = getattr(cur_mod, s)
        else: # nn.ModuleList or nn.ModuleList
            cur_mod=cur_mod[int(s)]
    if hasattr(cur_mod, tokens[-1]):
        setattr(cur_mod, tokens[-1], module)
    else: # nn.ModuleList or nn.ModuleList
        cur_mod[int(tokens[-1])] = module

def set_param(module: nn.Module, name: str, weights: torch.Tensor):
    
    param=nn.parameter.Parameter(weights, requires_grad=False)
    if isinstance(module, nn.Linear) and len(weights.shape)==1:
        param.unsqueeze_(0)
    setattr(module, name, param)

def get_device(gguf_module_key:str, device_map:dict):
    if gguf_module_key in device_map:
        return device_map[gguf_module_key]["generate_device"]
    else:
        return "cuda"

def get_all_used_cuda_device(device_map:dict):
    all_device_list = set()
    for key in device_map:
        all_device_list.add(device_map[key]["generate_device"]) if "generate_device" in device_map[key] else None
        all_device_list.add(device_map[key]["prefill_device"]) if "prefill_device" in device_map[key] else None
    if "cpu" in all_device_list:
        all_device_list.remove("cpu")
    all_device_list = list(all_device_list)
    return all_device_list

def load_cur_state_dict(module: nn.Module, gguf_loader: GGUFLoader, prefix: str = ""):
    prefix = prefix.replace("orig_module.", "")
    persistent_buffers = {k: v for k, v in module._buffers.items() if k not in module._non_persistent_buffers_set}
    local_name_params = itertools.chain(module._parameters.items(), persistent_buffers.items())
    local_state = {k: v for k, v in local_name_params if v is not None}
    for name, param in local_state.items():
        key = prefix + name
        translated_key = translate_name_to_gguf(key)
        if translated_key in gguf_loader.tensor_file_map:
            target_dtype = torch.get_default_dtype()
            device = get_device(translated_key[:translated_key.rfind(".")], gguf_loader.tensor_device_map)
            print(f"loading {translated_key} to {device}")
            # device = "cpu" if "embd" in translated_key else "cuda"
            weights = gguf_loader.load_gguf_tensor(translated_key, device = device).to(dtype = target_dtype)
            set_param(module, name, weights)
            del weights
        else:
            #print(load_config.tensor_file_map.keys())
            raise Exception(f"can't find {translated_key} in GGUF file!")
        
def load_weights(module:nn.Module, gguf_loader:GGUFLoader, prefix=''):
    # print(f"recursively loading weights {prefix},{return_when_injected=}, {only_load_injected=}")
    if not isinstance(module, base_operator.BaseInjectedModule):
        load_cur_state_dict(module, gguf_loader, prefix)
        for name, child in module._modules.items():
            load_weights(child, gguf_loader, prefix+name+".")
    else:
        module.load()

def prefill_and_save_kv_cache(model, tokenizer, past_key_values, inputs,
                          save_path='', example_id = 0, chunk_id = 0, system_len = 0, passage_len = 0, reprocess_method=None
                          ):
    
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch._dynamo.config.suppress_errors = True
    batch_size, seq_length = inputs.shape

    torch_device = "cuda:0"
    inputs = inputs.to(torch_device)

    tokens = []
    for layer_idx in range(len(past_key_values.key_cache)):
        past_key_values.past_tokens[layer_idx] = 0
    torch.cuda.set_device(torch_device)
    with torch.no_grad():
        cache_position = torch.arange(seq_length, device=torch_device)
        generated_ids = torch.zeros(
            batch_size, seq_length  + 1, dtype=torch.int, device=torch_device
        )
        generated_ids[:, cache_position] = inputs.to(torch_device).to(torch.int)
        if past_key_values != None:
            past_key_values.cur_idx=cache_position
        start_time = time.time()


        inputs_embeds = model.model.embed_tokens(inputs).to(torch_device)
        if reprocess_method == "Cache-Craft" and chunk_id != 0:
            passages_len = [system_len, passage_len]
            logits = model(
                inputs_embeds = inputs_embeds, cache_position=cache_position, past_key_values=past_key_values, return_dict=False, use_cache=True, reprocess_method=reprocess_method, passages_len=passages_len
            )[0][:,-1,:].unsqueeze(0).clone().to(torch_device)
            cachecraft_score = past_key_values.importance_cache[-1] # [num_head, passage_len]
            cachecraft_score = torch.sum(cachecraft_score, dim=0)
            torch.save(cachecraft_score, f'{save_path}/cachecraftattn_{example_id}_{chunk_id}.pt')
        else:
            logits = model(
                inputs_embeds = inputs_embeds, cache_position=cache_position, past_key_values=past_key_values, return_dict=False, use_cache=True
            )[0][:,-1,:].unsqueeze(0).clone().to(torch_device)
        past_len = past_key_values.past_tokens[0]
        key_cache = []
        value_cache = []
        if chunk_id == 0:
            key_cache = [past_key_values.key_cache[i][:,:,:past_len,:] for i in range(len(past_key_values.key_cache))]
            key_cache = torch.stack(key_cache)
            value_cache = [past_key_values.value_cache[i][:,:,:past_len,:] for i in range(len(past_key_values.value_cache))]
            value_cache = torch.stack(value_cache)
        else:
            key_cache = [past_key_values.key_cache[i][:,:,system_len:system_len + passage_len,:] for i in range(len(past_key_values.key_cache))]
            key_cache = torch.stack(key_cache)
            value_cache = [past_key_values.value_cache[i][:,:,system_len:system_len + passage_len,:] for i in range(len(past_key_values.value_cache))]
            value_cache = torch.stack(value_cache)
        torch.save(key_cache.clone(), f'{save_path}/{example_id}_{chunk_id}_key.pt')
        torch.save(value_cache.clone(), f'{save_path}/{example_id}_{chunk_id}_value.pt')
        print(f'example_id: {example_id}, chunk_id: {chunk_id}')
        return key_cache, value_cache

def decode_one_tokens(model, cur_token, position_ids, cache_position, past_key_values, logits_warper, inputs):
    inputs_embeds = model.model.embed_tokens(cur_token)
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
# mistral 是这个函数，其他函数得考虑把这个函数换掉
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def prefill_with_cache_and_save_preprocess(model, tokenizer, past_key_values, passages,
                          save_path='', example_id = 0, chunk_id=0, system_len=0, revert_rope=False, reprocess_method=None):

    # load KV
    torch_device = 'cuda'
    past_len = past_key_values.past_tokens[0]

    # prefill context
    inputs = passages[-1].unsqueeze(0).to(torch_device)
    passage_len = passages[-1].shape[0]
    passages_len = [passages[i].shape[0] for i in range(len(passages))]
    batch_size, seq_length = inputs.shape
    cache_position = torch.arange(past_len,past_len + seq_length, device='cuda')
    # position_ids = torch.arange(system_len,system_len + seq_length, device='cuda').unsqueeze(0)
    generated_ids = torch.zeros(
        batch_size, past_len + seq_length + 1, dtype=torch.int, device='cuda'
    )
    generated_ids[:, :past_len] = torch.cat(passages[:-1]).unsqueeze(0).to('cuda')
    generated_ids[:, cache_position] = inputs.to('cuda').to(torch.int)
    tokens = []
    with torch.no_grad():
        stream = TextStreamer(tokenizer)
        inputs_embeds = model.model.embed_tokens(inputs).to(torch_device)
        if reprocess_method == "Cache-Craft":
            logits = model(
                inputs_embeds = inputs_embeds, cache_position=cache_position,
                past_key_values=past_key_values, return_dict=False, use_cache=True, reprocess_method=reprocess_method, passages_len=passages_len
            )[0][:,-1,:].unsqueeze(0).clone().to(torch_device)
            cachecraft_score = past_key_values.importance_cache[-1] # [num_head, passage_len]
            cachecraft_score = torch.sum(cachecraft_score, dim=0)
            torch.save(cachecraft_score, f'{save_path}/cachecraftattn_{example_id}_{chunk_id}.pt')
        else:
            logits = model(
                inputs_embeds = inputs_embeds, cache_position=cache_position,
                past_key_values=past_key_values, return_dict=False, use_cache=True
            )[0][:,-1,:].unsqueeze(0).clone().to(torch_device)
    key_cache = torch.stack(past_key_values.key_cache)[:,:,:,past_len:past_len + passage_len,:]
    position_ids = torch.full((1, key_cache[0].shape[2]), system_len - past_len, device=torch_device)
    try:
        cos, sin = model.model.layers[0].self_attn.rotary_emb(key_cache[0], position_ids)
    except:
        cos, sin = model.model.rotary_emb(key_cache[0], position_ids)
    # mistral 限定
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    key_cache = (key_cache * cos) + (rotate_half(key_cache) * sin)
    torch.save(key_cache.clone(), f'{save_path}/{example_id}_{chunk_id}_key.pt')
    key_cache = None
    torch.cuda.empty_cache()
    value_cache = torch.stack(past_key_values.value_cache)[:,:,:,past_len:past_len + passage_len,:]
    torch.save(value_cache.clone(), f'{save_path}/{example_id}_{chunk_id}_value.pt')
    # if revert_rope:


    

def load_kv_and_generate(model, tokenizer, past_key_values, passages,
                          load_path='', example_id = 0, max_new_tokens=1, revert_rope=False, 
                          reprocess_method='normal', rate=0, dense=2, preprocess=False, draft_model=None, group=False):
    passages_len = [passage.shape[0] for passage in passages]
    passages_start = [sum(passages_len[:i]) for i in range(1,len(passages_len))]
    torch.set_printoptions(threshold=50_000)
    torch_device = 'cuda'
    query_prefix_len = len(tokenizer.encode(tokenizer.decode(passages[-1]).split('Question: ')[0]))
    inputs = passages[-1][query_prefix_len:].unsqueeze(0).to(torch_device)
    seq_length = passages[-1][query_prefix_len:].shape[0]

    # load KV
    
    for layer_idx in range(len(past_key_values.key_cache)):
        past_key_values.past_tokens[layer_idx] = 0
    past_len = 0
    system_len = passages[0].shape[0]

    key_cache = []
    value_cache = []
    all_position_ids = [torch.arange(0,system_len).unsqueeze(0).to(torch_device)]
    
    for chunk_id, passage in enumerate(passages[:-1]):
        passage_len = passage.shape[0]
        
        chunk_key_cache = torch.load(f'{load_path}/{example_id}_{chunk_id}_key.pt',weights_only=True).to('cpu')
        chunk_value_cache = torch.load(f'{load_path}/{example_id}_{chunk_id}_value.pt',weights_only=True).to('cpu')
        key_cache.append(chunk_key_cache)
        value_cache.append(chunk_value_cache)
    start_time = time.time()
    for chunk_id, passage in enumerate(passages[:-1]):
        passage_len = passage.shape[0]
        key_cache[chunk_id] = key_cache[chunk_id].to(torch_device)
        chunk_key_cache = key_cache[chunk_id]
        chunk_value_cache = value_cache[chunk_id].to(torch_device)
        assert passage_len == chunk_key_cache.shape[3]
        if revert_rope and chunk_id > 1:
            all_position_ids = []
            position_ids = torch.full((1, chunk_key_cache[layer_idx].shape[2]), past_len - system_len, device=torch_device)
            try:
                cos, sin = model.model.layers[0].self_attn.rotary_emb(chunk_key_cache[layer_idx], position_ids)
            except:
                cos, sin = model.model.rotary_emb(chunk_key_cache[layer_idx], position_ids)
            # mistral 限定
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
            chunk_key_cache = (chunk_key_cache * cos) + (rotate_half(chunk_key_cache) * sin)
        elif chunk_id > 0:
            all_position_ids.append(torch.arange(system_len,system_len+passage_len).to(torch_device).unsqueeze(0))

        for layer_idx in range(len(past_key_values.key_cache)):
            past_key_values.key_cache[layer_idx].narrow(2,past_len,passage_len).copy_(chunk_key_cache[layer_idx])
            past_key_values.value_cache[layer_idx].narrow(2,past_len,passage_len).copy_(chunk_value_cache[layer_idx])
            past_key_values.past_tokens[layer_idx] += passage_len
        past_len += passage_len
    storage_time = time.time() - start_time 
    print(f'storage_time: {storage_time}')
    if rate != 0:
        if reprocess_method == 'cacheBlend':
            without_attn_key = past_key_values.key_cache[1].narrow(2,0,past_len).clone()
            without_attn_value = past_key_values.value_cache[1].narrow(2,0,past_len).clone()
            inputs = torch.cat(passages[:-1]).to('cuda').unsqueeze(0)
            # 这里会在终端上多输出一次
            _, tmp_past_key_value = prefill_and_generate(model, tokenizer, inputs, max_new_tokens=1)
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
        elif reprocess_method == 'processCache':
            select_time = time.time()
            query_prefix_len = len(tokenizer.encode(tokenizer.decode(passages[-1]).split('Question: ')[0]))
            if query_prefix_len >= len(passages[-1]):
                query_prefix_len = len(tokenizer.encode(tokenizer.decode(passages[-1]).split('Question：')[0]))+1
            inputs = passages[-1][query_prefix_len:].unsqueeze(0).to(torch_device)
            seq_length = passages[-1][query_prefix_len:].shape[0]
            
            cache_position = torch.arange(past_len, past_len+seq_length, device='cuda')
            with torch.no_grad():
                ss_time = time.perf_counter()
                inputs_embeds = model.model.embed_tokens(inputs).to(torch_device)
                model(
                    inputs_embeds = inputs_embeds, past_key_values=past_key_values,
                    cache_position=cache_position, reprocess_method=reprocess_method, 
                    return_dict=False, use_cache=True, passages_len=passages_len, history_key_cache=key_cache
                    )
                
                k_sum = torch.sum(past_key_values.importance_cache[-1], dim=0)[:cache_position[0]]
                end_time = time.perf_counter() - ss_time
                if group:
                    k_sum_relevant = k_sum[system_len:]  # 只看中间文本块的分数
                    k_sum_relevant = torch.tensor(k_sum_relevant, device=torch_device)

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
                                            -float('inf'), device=torch_device)
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
                    k_sum = torch.tensor(k_sum,device=torch_device)
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
            save_prefix_path_list = [f"{load_path}/cachecraftattn_{example_id}_{i}.pt" for i in range(1,len(passages)-1)]
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
            inputs = torch.cat(passages).to('cuda').unsqueeze(0)
            cache_position = torch.arange(0, inputs.shape[1], device='cuda')
            tmp_past_key_values = StaticCache(
                                    config = model.config, max_batch_size = 1, 
                                    max_cache_len = inputs.shape[1], device = 'cuda', dtype = model.dtype,
                                    passage_len=torch.cat(passages[:-1]).shape[0],
                                )
            with torch.no_grad():
                ss_time = time.perf_counter()
                inputs_embeds = draft_model.model.embed_tokens(inputs).to(torch_device)
                draft_model(
                    inputs_embeds = inputs_embeds, past_key_values=tmp_past_key_values,
                    cache_position=cache_position, reprocess_method=reprocess_method, 
                    return_dict=False, use_cache=True, passages_len=passages_len
                    )

                # 获取重要性分数
                k_sum = torch.sum(past_key_values.importance_cache[-1], dim=0)[:past_len]
                if group:
                    k_sum_relevant = k_sum[system_len:]  # 只看中间文本块的分数
                    k_sum_relevant = torch.tensor(k_sum_relevant, device=torch_device)

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
                                            -float('inf'), device=torch_device)
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
                    k_sum = torch.tensor(k_sum,device=torch_device)
                    k_lens = int(rate*(torch.cat(passages[:-1]).shape[0] - system_len))
                    k_need_index = torch.topk(k_sum, k_lens).indices.to('cpu')
                    k_need_index = k_need_index + system_len
        else:
            raise NotImplementedError

        # reprocess kv cache and prefill question
        k_need_index = torch.sort(k_need_index)[0].tolist()
        k_need_index.extend(range(sum(passages_len[:-1]),sum(passages_len)))
    else:
        k_need_index = range(sum(passages_len[:-1]),sum(passages_len))
    past_len = sum(passages_len)

    batch_size, seq_length = 1, len(k_need_index)

    generated_ids = torch.zeros(
        batch_size, past_len + max_new_tokens + 1, dtype=torch.int, device='cuda'
    )
    generated_ids[:, :past_len] = torch.cat(passages).unsqueeze(0).to('cuda')
    tokens = []

    if reprocess_method != 'processCache' or reprocess_method != 'frontRow':
        dense = 0
    reprocess_inputs = torch.cat(passages)[k_need_index].unsqueeze(0).to(torch_device)
    cache_position = torch.tensor(k_need_index, device=torch_device)
    with torch.no_grad():
        without_attn_value = past_key_values.value_cache[-1].narrow(2,0, sum(passages_len[:-1])).clone()
        inputs_embeds = model.model.embed_tokens(reprocess_inputs).to(torch_device)
        logits = model(
        inputs_embeds = inputs_embeds, cache_position=cache_position, 
        past_key_values=past_key_values, return_dict=False, use_cache=True, dense = dense,
        )[0][:,-1,:].unsqueeze(0).clone().to(torch_device)
        with_attn_value = past_key_values.value_cache[-1].narrow(2,0, sum(passages_len[:-1])).clone()
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
        print(stream.put(next_token.item()), end="", flush=True)
        generated_ids[:, past_len+1] = next_token
        tokens.append(next_token)
        inputs = torch.cat((torch.cat(passages).unsqueeze(0).to(torch_device), next_token.unsqueeze(0)), dim=-1)
        cache_position = torch.tensor([past_len], device=torch_device)
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
                print(stream.end(), end="", flush=True)
                break
            else:
                print(stream.put(next_token.item()), end="", flush=True)
            cache_position += 1
            position_ids = cache_position.unsqueeze(0)
        

    total_time = time.time() - decode_time
    tokens_generated = len(tokens)
    tokens_per_second = tokens_generated / total_time

    print("")

    print(f"prompt eval count:    {prefill_count} token(s)")
    print(f"prompt eval duration: {prefill_time}s")
    print(f"prompt eval rate:     {prefill_count/prefill_time} tokens/s")
    print(f"eval count:           {tokens_generated} token(s)")
    print(f"eval duration:        {total_time}s")
    print(f"eval rate:            {tokens_per_second} tokens/s")

    return tokens

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
                         mode = 'normal', reprocess_method=None):
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch._dynamo.config.suppress_errors = True
    batch_size, seq_length = inputs.shape

    torch_device = "cuda:0"
    inputs = inputs.to(torch_device)

    tokens = []
    
    def decode_one_tokens(cuda_graph_runner, cur_token, position_ids, cache_position, past_key_values, use_cuda_graph: bool = False):
        if use_cuda_graph:
            logits = cuda_graph_runner(cur_token, position_ids, cache_position)
        else:
            # custom_stream = torch.cuda.Stream()
            torch.cuda.set_device(torch_device)
            inputs_embeds = model.model.embed_tokens(cur_token).to(torch_device)
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
    
    torch.cuda.set_device(torch_device)
    with torch.no_grad():
        stream = TextStreamer(tokenizer)
        if mode != 'long_context':
            past_key_values = StaticCache(
                config = model.config, max_batch_size = 1, max_cache_len = seq_length + max_new_tokens, device = torch_device, dtype = model.dtype
            )
        else:
            past_key_values = None
        cache_position = torch.arange(seq_length, device=torch_device)
        generated_ids = torch.zeros(
            batch_size, seq_length + max_new_tokens + 1, dtype=torch.int, device=torch_device
        )
        generated_ids[:, cache_position] = inputs.to(torch_device).to(torch.int)
        if past_key_values != None:
            past_key_values.cur_idx=cache_position
        start_time = time.time()

        if mode == "long_context":
            inputs_embeds = model.model.embed_tokens(inputs.to("cpu"))
        else:
            inputs_embeds = model.model.embed_tokens(inputs).to(torch_device)
        logits = model(
            inputs_embeds = inputs_embeds, cache_position=cache_position, 
            past_key_values=past_key_values, return_dict=False, use_cache=True
        )[0][:,-1,:].unsqueeze(0).clone().to(torch_device)
        # generation_config, model_kwargs = model._prepare_generation_config(None, do_sample=False, top_k=1,  temperature=0.01)

        logits_warper = tf_logits_warper(temperature=0.01, top_k=1)
        next_token_scores = logits_warper(inputs, logits[:, -1, :])

        next_token = torch.argmax(next_token_scores, dim=-1)
        first_token_time = time.time() - start_time

        prefill_count = seq_length
        prefill_time = first_token_time
        print(stream.put(next_token.item()), end="", flush=True)
        generated_ids[:, seq_length] = next_token
        tokens.append(next_token)
        inputs = torch.cat((inputs, next_token.unsqueeze(0)), dim=-1)
        cache_position = torch.tensor([seq_length], device=torch_device)
        position_ids = cache_position.unsqueeze(0)
        seq_length += 1
        
        if use_cuda_graph:
            cuda_graph_runner = CUDAGraphRunner()
            cuda_graph_runner.capture(model, next_token.unsqueeze(0), position_ids, cache_position, past_key_values, torch_device, return_dict=False, use_cache=True)
        else:
            cuda_graph_runner = None
            
        start_time = time.time()
        for _ in range(1, max_new_tokens):
            next_token = decode_one_tokens(cuda_graph_runner, next_token.unsqueeze(0), position_ids, cache_position, past_key_values, use_cuda_graph).to(torch_device)
            inputs = torch.cat((inputs, next_token.unsqueeze(0)), dim=-1)
            generated_ids[:, cache_position] = next_token.int()
            tokens.append(next_token.int())
            seq_length += 1
            
            if next_token[0].item() == tokenizer.eos_token_id or tokenizer.decode(next_token) == '<|im_end|>' or tokenizer.decode(next_token) == '[unused10]':
                print(stream.end(), end="", flush=True)
                break
            else:
                print(stream.put(next_token.item()), end="", flush=True)
            cache_position += 1
            position_ids = cache_position.unsqueeze(0)
        

    total_time = time.time() - start_time
    tokens_generated = len(tokens)
    tokens_per_second = tokens_generated / total_time

    print("")

    print(f"prompt eval count:    {prefill_count} token(s)")
    print(f"prompt eval duration: {prefill_time}s")
    print(f"prompt eval rate:     {prefill_count/prefill_time} tokens/s")
    print(f"eval count:           {tokens_generated} token(s)")
    print(f"eval duration:        {total_time}s")
    print(f"eval rate:            {tokens_per_second} tokens/s")

    return tokens, past_key_values



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
