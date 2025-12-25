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








    

def prefill_and_save_kv_cache(model, tokenizer, past_key_values, inputs,
                          save_path='', example_id = 0, chunk_id = 0, system_len = 0, passage_len = 0, reprocess_method=None, device="cuda"
                          ):
    
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch._dynamo.config.suppress_errors = True
    batch_size, seq_length = inputs.shape


    inputs = inputs.to(device)

    tokens = []
    for layer_idx in range(len(past_key_values.key_cache)):
        past_key_values.past_tokens[layer_idx] = 0
    with torch.no_grad():
        cache_position = torch.arange(seq_length, device=device)
        generated_ids = torch.zeros(
            batch_size, seq_length  + 1, dtype=torch.int, device=device
        )
        generated_ids[:, cache_position] = inputs.to(device).to(torch.int)
        if past_key_values != None:
            past_key_values.cur_idx=cache_position

        inputs_embeds = model.model.embed_tokens(inputs).to(device)
        if reprocess_method == "Cache-Craft" and chunk_id != 0:
            passages_len = [system_len, passage_len]
            logits = model(
                inputs_embeds = inputs_embeds, cache_position=cache_position, past_key_values=past_key_values, return_dict=False, use_cache=True, reprocess_method=reprocess_method, passages_len=passages_len
            )[0][:,-1,:].unsqueeze(0).clone().to(device)
            cachecraft_score = past_key_values.importance_cache[-1] # [num_head, passage_len]
            cachecraft_score = torch.sum(cachecraft_score, dim=0)
            torch.save(cachecraft_score, f'{save_path}/cachecraftattn_{example_id}_{chunk_id}.pt')
        else:
            logits = model(
                inputs_embeds = inputs_embeds, cache_position=cache_position, past_key_values=past_key_values, return_dict=False, use_cache=True
            )[0][:,-1,:].unsqueeze(0).clone().to(device)
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
    next_token_scores = logits_warper(inputs, logits[:, -1, :])
    next_token = torch.argmax(next_token_scores, dim=-1)
    return next_token

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def prefill_with_cache_and_save_preprocess(model, tokenizer, past_key_values, passages,
                          save_path='', example_id = 0, chunk_id=0, system_len=0, revert_rope=False, reprocess_method=None, device="cuda"):

    # load KV
    past_len = past_key_values.past_tokens[0]

    # prefill context
    inputs = passages[-1].unsqueeze(0).to(device)
    passage_len = passages[-1].shape[0]
    passages_len = [passages[i].shape[0] for i in range(len(passages))]
    batch_size, seq_length = inputs.shape
    cache_position = torch.arange(past_len,past_len + seq_length, device=device)
    generated_ids = torch.zeros(
        batch_size, past_len + seq_length + 1, dtype=torch.int, device=device
    )
    generated_ids[:, :past_len] = torch.cat(passages[:-1]).unsqueeze(0).to(device)
    generated_ids[:, cache_position] = inputs.to(device).to(torch.int)
    tokens = []
    with torch.no_grad():
        stream = TextStreamer(tokenizer)
        inputs_embeds = model.model.embed_tokens(inputs).to(device)
        if reprocess_method == "Cache-Craft":
            logits = model(
                inputs_embeds = inputs_embeds, cache_position=cache_position,
                past_key_values=past_key_values, return_dict=False, use_cache=True, reprocess_method=reprocess_method, passages_len=passages_len
            )[0][:,-1,:].unsqueeze(0).clone().to(device)
            cachecraft_score = past_key_values.importance_cache[-1]
            cachecraft_score = torch.sum(cachecraft_score, dim=0)
            torch.save(cachecraft_score, f'{save_path}/cachecraftattn_{example_id}_{chunk_id}.pt')
        else:
            logits = model(
                inputs_embeds = inputs_embeds, cache_position=cache_position,
                past_key_values=past_key_values, return_dict=False, use_cache=True
            )[0][:,-1,:].unsqueeze(0).clone().to(device)
    key_cache = torch.stack(past_key_values.key_cache)[:,:,:,past_len:past_len + passage_len,:]
    position_ids = torch.full((1, key_cache[0].shape[2]), system_len - past_len, device=device)
    try:
        cos, sin = model.model.layers[0].self_attn.rotary_emb(key_cache[0], position_ids)
    except:
        cos, sin = model.model.rotary_emb(key_cache[0], position_ids)
    # Apply rotary embeddings
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    key_cache = (key_cache * cos) + (rotate_half(key_cache) * sin)
    torch.save(key_cache.clone(), f'{save_path}/{example_id}_{chunk_id}_key.pt')
    key_cache = None
    if "cuda" in device:
        torch.cuda.empty_cache()
    elif "npu" in device:
        torch.npu.empty_cache()
    value_cache = torch.stack(past_key_values.value_cache)[:,:,:,past_len:past_len + passage_len,:]
    torch.save(value_cache.clone(), f'{save_path}/{example_id}_{chunk_id}_value.pt')



def load_kv_and_generate(model, tokenizer, past_key_values, passages,
                          load_path='', example_id = 0, max_new_tokens=1, revert_rope=False,
                          reprocess_method='normal', rate=0, preprocess=False, draft_model=None, device="cuda", chunk_ids=None):
    passages_len = [passage.shape[0] for passage in passages]
    passages_start = [sum(passages_len[:i]) for i in range(1,len(passages_len))]
    query_prefix_len = len(tokenizer.encode(tokenizer.decode(passages[-1]).split('Question: ')[0]))
    inputs = passages[-1][query_prefix_len:].unsqueeze(0).to(device)
    seq_length = passages[-1][query_prefix_len:].shape[0]

    # load KV
    
    for layer_idx in range(len(past_key_values.key_cache)):
        past_key_values.past_tokens[layer_idx] = 0
    past_len = 0
    system_len = passages[0].shape[0]

    key_cache = []
    value_cache = []
    all_position_ids = [torch.arange(0,system_len).unsqueeze(0).to(device)]

    # If chunk_ids is provided, use it; otherwise use sequential indices (backward compatible)
    if chunk_ids is None:
        chunk_ids = list(range(len(passages) - 1))

    for idx, passage in enumerate(passages[:-1]):
        chunk_id = chunk_ids[idx]
        passage_len = passage.shape[0]

        chunk_key_cache = torch.load(f'{load_path}/{example_id}_{chunk_id}_key.pt',weights_only=True).to('cpu')
        chunk_value_cache = torch.load(f'{load_path}/{example_id}_{chunk_id}_value.pt',weights_only=True).to('cpu')
        key_cache.append(chunk_key_cache)
        value_cache.append(chunk_value_cache)
    for idx, passage in enumerate(passages[:-1]):
        chunk_id = chunk_ids[idx]
        passage_len = passage.shape[0]
        key_cache[idx] = key_cache[idx].to(device)
        chunk_key_cache = key_cache[idx]
        chunk_value_cache = value_cache[idx].to(device)
        assert passage_len == chunk_key_cache.shape[3]
        if revert_rope and chunk_id > 0:
            all_position_ids = []
            position_ids = torch.full((1, chunk_key_cache[layer_idx].shape[2]), past_len - system_len, device=device)
            try:
                cos, sin = model.model.layers[0].self_attn.rotary_emb(chunk_key_cache[layer_idx], position_ids)
            except:
                cos, sin = model.model.rotary_emb(chunk_key_cache[layer_idx], position_ids)
            # Apply rotary embeddings
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
            chunk_key_cache = (chunk_key_cache * cos) + (rotate_half(chunk_key_cache) * sin)
        elif chunk_id > 0:
            all_position_ids.append(torch.arange(system_len,system_len+passage_len).to(device).unsqueeze(0))

        for layer_idx in range(len(past_key_values.key_cache)):
            past_key_values.key_cache[layer_idx].narrow(2,past_len,passage_len).copy_(chunk_key_cache[layer_idx])
            past_key_values.value_cache[layer_idx].narrow(2,past_len,passage_len).copy_(chunk_value_cache[layer_idx])
            past_key_values.past_tokens[layer_idx] += passage_len
        past_len += passage_len
    if rate != 0:
        if reprocess_method == 'cacheBlend':
            without_attn_key = past_key_values.key_cache[1].narrow(2,0,past_len).clone()
            without_attn_value = past_key_values.value_cache[1].narrow(2,0,past_len).clone()
            inputs = torch.cat(passages[:-1]).to(device).unsqueeze(0)
            _, tmp_past_key_value = prefill_and_generate(model, tokenizer, inputs, max_new_tokens=1, early_exit_layer=2)
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
            query_prefix_len = len(tokenizer.encode(tokenizer.decode(passages[-1]).split('Question: ')[0]))
            if query_prefix_len >= len(passages[-1]):
                query_prefix_len = len(tokenizer.encode(tokenizer.decode(passages[-1]).split('Question：')[0]))+1
            inputs = passages[-1][query_prefix_len:].unsqueeze(0).to(device)
            seq_length = passages[-1][query_prefix_len:].shape[0]
            
            cache_position = torch.arange(past_len, past_len+seq_length, device=device)
            with torch.no_grad():
                inputs_embeds = model.model.embed_tokens(inputs).to(device)
                model(
                    inputs_embeds = inputs_embeds, past_key_values=past_key_values,
                    cache_position=cache_position, reprocess_method=reprocess_method, 
                    return_dict=False, use_cache=True, passages_len=passages_len, history_key_cache=key_cache
                    )
                k_sum = torch.sum(past_key_values.importance_cache[-1], dim=0)[:cache_position[0]]
                k_sum = k_sum.tolist()
                k_sum = k_sum[system_len:]
                k_sum = torch.tensor(k_sum,device=device)
                k_lens = int(rate*(torch.cat(passages[:-1]).shape[0] - system_len))
                k_need_index = torch.topk(k_sum, k_lens).indices.to('cpu')
                k_need_index = k_need_index + system_len

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
        batch_size, past_len + max_new_tokens + 1, dtype=torch.int, device=device
    )
    generated_ids[:, :past_len] = torch.cat(passages).unsqueeze(0).to(device)
    tokens = []

    if reprocess_method != 'FusionRAG':
        use_sparse_attention = False
    else:
        use_sparse_attention = True
    reprocess_inputs = torch.cat(passages)[k_need_index].unsqueeze(0).to(device)
    cache_position = torch.tensor(k_need_index, device=device)
    with torch.no_grad():
        without_attn_value = past_key_values.value_cache[-1].narrow(2,0, sum(passages_len[:-1])).clone()
        inputs_embeds = model.model.embed_tokens(reprocess_inputs).to(device)
        logits = model(
        inputs_embeds = inputs_embeds, cache_position=cache_position, 
        past_key_values=past_key_values, return_dict=False, use_cache=True, use_sparse_attention=use_sparse_attention,
        )[0][:,-1,:].unsqueeze(0).clone().to(device)
        with_attn_value = past_key_values.value_cache[-1].narrow(2,0, sum(passages_len[:-1])).clone()
        v_sub_all = without_attn_value - with_attn_value
        v_sub_all = v_sub_all.squeeze(0)
        v_sub_all = v_sub_all.transpose(0, 1)
        v_sum = torch.sum(v_sub_all**2, dim=[1,2])

        stream = TextStreamer(tokenizer)
        logits_warper = tf_logits_warper(temperature=0.01, top_k=1)
        next_token_scores = logits_warper(reprocess_inputs, logits[:, -1, :])
        next_token = torch.argmax(next_token_scores, dim=-1)
        print(stream.put(next_token.item()), end="", flush=True)
        generated_ids[:, past_len+1] = next_token
        tokens.append(next_token)
        inputs = torch.cat((torch.cat(passages).unsqueeze(0).to(device), next_token.unsqueeze(0)), dim=-1)
        cache_position = torch.tensor([past_len], device=device)
        position_ids = cache_position.unsqueeze(0)
        seq_length += 1
        
 
        
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

def prefill_and_generate(model, tokenizer, inputs, max_new_tokens=10000,
                         mode = 'normal', early_exit_layer=None, device='cuda'):
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch._dynamo.config.suppress_errors = True
    batch_size, seq_length = inputs.shape


    inputs = inputs.to(device)

    tokens = []
    
    def decode_one_tokens(cur_token, position_ids, cache_position, past_key_values):
        inputs_embeds = model.model.embed_tokens(cur_token).to(device)
        logits=model(inputs_embeds=inputs_embeds,
                    position_ids=position_ids,
                    cache_position=cache_position,
                    past_key_values=past_key_values,
                    return_dict=False, use_cache=True)[0]
        if past_key_values != None:
            past_key_values.change_seq_length(1)
        next_token_scores = logits_warper(inputs, logits[:, -1, :])
        next_token = torch.argmax(next_token_scores, dim=-1)
        return next_token
    
    with torch.no_grad():
        stream = TextStreamer(tokenizer)

        past_key_values = StaticCache(
            config = model.config, max_batch_size=1, max_cache_len=seq_length+max_new_tokens, device=device, dtype=model.dtype
        )

        cache_position = torch.arange(seq_length, device=device)
        generated_ids = torch.zeros(
            batch_size, seq_length + max_new_tokens + 1, dtype=torch.int, device=device
        )
        generated_ids[:, cache_position] = inputs.to(device).to(torch.int)
        if past_key_values != None:
            past_key_values.cur_idx=cache_position

        if mode == "long_context":
            inputs_embeds = model.model.embed_tokens(inputs.to("cpu"))
        else:
            inputs_embeds = model.model.embed_tokens(inputs).to(device)

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

        logits = model(**model_kwargs)[0][:,-1,:].unsqueeze(0).clone().to(device)
        # generation_config, model_kwargs = model._prepare_generation_config(None, do_sample=False, top_k=1,  temperature=0.01)

        logits_warper = tf_logits_warper(temperature=0.01, top_k=1)
        next_token_scores = logits_warper(inputs, logits[:, -1, :])

        next_token = torch.argmax(next_token_scores, dim=-1)
        print(stream.put(next_token.item()), end="", flush=True)
        generated_ids[:, seq_length] = next_token
        tokens.append(next_token)
        inputs = torch.cat((inputs, next_token.unsqueeze(0)), dim=-1)
        cache_position = torch.tensor([seq_length], device=device)
        position_ids = cache_position.unsqueeze(0)
        seq_length += 1

        for _ in range(1, max_new_tokens):
            next_token = decode_one_tokens(next_token.unsqueeze(0), position_ids, cache_position, past_key_values).to(device)
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
    context_rank = context_rank if preprocess == True else []
    corpus_lens = corpus_lens if preprocess == True else []
    return batch_data, batch_tokens,question_list, real_answer_list, stop_token_id, \
        reprocess_path, preprocess_path, csv_path,\
            data_name_prefix, rouge_metrics, context_rank, corpus_lens
