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
from ktransformers.util.custom_gguf import translate_name_to_gguf
from ktransformers.util.custom_gguf import GGUFLoader
from ktransformers.operators import base_operator
from ktransformers.models.custom_cache import StaticCache
from ktransformers.util.cuda_graph_runner import CUDAGraphRunner
from ktransformers.util.textstream import TextStreamer

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
                          save_path='', example_id = 0, chunk_id = 0, system_len = 0, passage_len = 0
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
        logits = model(
            inputs_embeds = inputs_embeds, cache_position=cache_position, past_key_values=past_key_values, return_dict=False, use_cache=True
        )[0][:,-1,:].unsqueeze(0).clone().to(torch_device)
        past_len = past_key_values.past_tokens[0]
        if chunk_id == 0:
            key_cache = torch.stack(past_key_values.key_cache)[:,:,:,:past_len,:]
            value_cache = torch.stack(past_key_values.value_cache)[:,:,:,:past_len,:]
        else:
            key_cache = torch.stack(past_key_values.key_cache)[:,:,:,system_len:system_len + passage_len,:]
            value_cache = torch.stack(past_key_values.value_cache)[:,:,:,system_len:system_len + passage_len,:]
        torch.save(key_cache.clone(), f'{save_path}/{example_id}_{chunk_id}_key.pt')
        torch.save(value_cache.clone(), f'{save_path}/{example_id}_{chunk_id}_value.pt')
        print(f'example_id: {example_id}, chunk_id: {chunk_id}')
        return key_cache, value_cache

def decode_one_tokens(model, cur_token, position_ids, cache_position, past_key_values, logits_warper, generation_config, inputs):
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
    if generation_config.do_sample:
        probs = nn.functional.softmax(next_token_scores, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
    else:
        next_token = torch.argmax(next_token_scores, dim=-1)
    return next_token
# mistral 是这个函数，其他函数得考虑把这个函数换掉
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def prefill_with_cache_and_save_preprocess(model, tokenizer, past_key_values, passages,
                          save_path='', example_id = 0, chunk_id=0, system_len=0, revert_rope=False):

    # load KV
    torch_device = 'cuda'
    past_len = past_key_values.past_tokens[0]

    # prefill context
    inputs = passages[-1].unsqueeze(0).to(torch_device)
    passage_len = passages[-1].shape[0]
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
        logits = model(
            inputs_embeds = inputs_embeds, cache_position=cache_position,
            past_key_values=past_key_values, return_dict=False, use_cache=True
        )[0][:,-1,:].unsqueeze(0).clone().to(torch_device)
    key_cache = torch.stack(past_key_values.key_cache)[:,:,:,past_len:past_len + passage_len,:]
    value_cache = torch.stack(past_key_values.value_cache)[:,:,:,past_len:past_len + passage_len,:]
    # if revert_rope:
    position_ids = torch.full((1, key_cache[0].shape[2]), system_len - past_len, device=torch_device)
    cos, sin = model.model.layers[0].self_attn.rotary_emb(key_cache[0], position_ids)
    # mistral 限定
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    key_cache = (key_cache * cos) + (rotate_half(key_cache) * sin)
    torch.save(key_cache.clone(), f'{save_path}/{example_id}_{chunk_id}_key.pt')
    torch.save(value_cache.clone(), f'{save_path}/{example_id}_{chunk_id}_value.pt')

def load_kv_and_generate(model, tokenizer, past_key_values, passages,
                          load_path='', example_id = 0, max_new_tokens=1, revert_rope=False, 
                          reprocess_method='normal', rate=0, dense=2):
    # load KV
    torch_device = 'cuda'
    for layer_idx in range(len(past_key_values.key_cache)):
        past_key_values.past_tokens[layer_idx] = 0
    past_len = 0
    system_len = passages[0].shape[0]

    key_cache = []
    value_cache = []
    
        
    for chunk_id, passage in enumerate(passages[:-1]):
        passage_len = passage.shape[0]
        chunk_key_cache = torch.load(f'{load_path}/{example_id}_{chunk_id}_key.pt',weights_only=True).to('cpu')
        chunk_value_cache = torch.load(f'{load_path}/{example_id}_{chunk_id}_value.pt',weights_only=True).to('cpu')
        key_cache.append(chunk_key_cache)
        value_cache.append(chunk_value_cache)
    start_time = time.time()
    for chunk_id, passage in enumerate(passages[:-1]):
        passage_len = passage.shape[0]
        chunk_key_cache = key_cache[chunk_id].to(torch_device)
        chunk_value_cache = value_cache[chunk_id].to(torch_device)
        assert passage_len == chunk_key_cache.shape[3]
        if revert_rope and chunk_id > 1:
            position_ids = torch.full((1, chunk_key_cache[layer_idx].shape[2]), past_len - system_len, device=torch_device)
            cos, sin = model.model.layers[0].self_attn.rotary_emb(chunk_key_cache[layer_idx], position_ids)
            # mistral 限定
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
            chunk_key_cache = (chunk_key_cache * cos) + (rotate_half(chunk_key_cache) * sin)
        for layer_idx in range(len(past_key_values.key_cache)):
            past_key_values.key_cache[layer_idx].narrow(2,past_len,passage_len).copy_(chunk_key_cache[layer_idx])
            past_key_values.value_cache[layer_idx].narrow(2,past_len,passage_len).copy_(chunk_value_cache[layer_idx])
            past_key_values.past_tokens[layer_idx] += passage_len
        past_len += passage_len 

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

        k_sub_all = without_attn_key - with_attn_key
        k_sub_all = torch.abs(k_sub_all)
        k_sub_all = k_sub_all.squeeze(0)
        k_sub_all = k_sub_all.transpose(0, 1)
        k_sub_all = k_sub_all.reshape(past_len,-1)
        k_sum = torch.sum(k_sub_all,dim=1)
        k_sum = k_sum.tolist()
        k_sum = k_sum[system_len:]
        k_sum = torch.tensor(k_sum,device=k_sub_all.device)
        k_need_index = torch.topk(k_sum,int(rate*(past_len - system_len))).indices.to('cpu')
        k_need_index = k_need_index + system_len
        k_need_index = v_need_index
    elif reprocess_method == 'processCache':
        start_time = time.time()
        query_prefix_len = len(tokenizer.encode(tokenizer.decode(passages[-1]).split('Question: ')[0]))
        inputs = passages[-1][query_prefix_len:].unsqueeze(0).to(torch_device)
        seq_length = passages[-1][query_prefix_len:].shape[0]
        passages_len = [passage.shape[0] for passage in passages[:-1]]
        passages_start = [sum(passages_len[:i]) for i in range(1,len(passages_len))]
        cache_position = torch.arange(0, seq_length, device='cuda')
        with torch.no_grad():
            tmp_past_key_values = StaticCache(
                                config = model.config, max_batch_size = 1, 
                                max_cache_len = seq_length, device = 'cuda', dtype = model.dtype,
                                passage_len=torch.cat(passages[:-1]).shape[0]
                            )
            inputs_embeds = model.model.embed_tokens(inputs).to(torch_device)
            model(
                inputs_embeds = inputs_embeds, past_key_values=tmp_past_key_values,
                cache_position=cache_position,
                return_dict=False, use_cache=True, reprocess_method=reprocess_method, 
                passages_len=passages_len, load_path=load_path, example_id=example_id)
            k_sum = torch.sum(tmp_past_key_values.importance_cache[-1], dim=0)
            k_sum = k_sum.tolist()
            k_sum = k_sum[system_len:]
            k_sum = torch.tensor(k_sum,device=torch_device)
            k_need_index = torch.topk(k_sum,int(rate*(torch.cat(passages[:-1]).shape[0] - system_len))).indices.to('cpu')
            k_need_index = k_need_index + system_len
    else:
        raise NotImplementedError

    # reprocess kv cache
    if reprocess_method != 'normal' and rate != 0:
        if reprocess_method != 'processCache:':
            dense = 0
        reprocess_inputs = torch.cat(passages[:-1])[k_need_index].unsqueeze(0).to(torch_device)
        cache_position = k_need_index.to(torch_device)
        with torch.no_grad():
            inputs_embeds = model.model.embed_tokens(reprocess_inputs).to(torch_device)
            model(
            inputs_embeds = inputs_embeds, cache_position=cache_position, past_key_values=past_key_values, return_dict=False, use_cache=True, dense = dense,
            )
            
    # prefill question
    past_len = torch.cat(passages[:-1]).shape[0]
    inputs = passages[-1].unsqueeze(0).to(torch_device)
    batch_size, seq_length = inputs.shape
    cache_position = torch.arange(past_len,past_len + seq_length, device='cuda')
    generated_ids = torch.zeros(
        batch_size, past_len + seq_length + max_new_tokens + 1, dtype=torch.int, device='cuda'
    )
    generated_ids[:, :past_len] = torch.cat(passages[:-1]).unsqueeze(0).to('cuda')
    generated_ids[:, cache_position] = inputs.to('cuda').to(torch.int)
    tokens = []
    with torch.no_grad():
        stream = TextStreamer(tokenizer)
        inputs_embeds = model.model.embed_tokens(inputs).to(torch_device)
        logits = model(
            inputs_embeds = inputs_embeds, cache_position=cache_position, past_key_values=past_key_values, return_dict=False, use_cache=True
        )[0][:,-1,:].unsqueeze(0).clone().to(torch_device)
        generation_config, model_kwargs = model._prepare_generation_config(
            None, max_length=max_new_tokens,
            do_sample=True, top_k=1, temperature=0.01 # change this to modify generate config
        )
        try: # transformers==4.43
            logits_warper = (
                model._get_logits_warper(generation_config,device=inputs.device)
            )
        except: 
            logits_warper = (
                model._get_logits_warper(generation_config)
            )
        next_token_scores = logits_warper(inputs, logits[:, -1, :])
        if generation_config.do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_token = torch.argmax(next_token_scores, dim=-1)
        first_token_time = time.time() - start_time

        prefill_count = seq_length
        prefill_time = first_token_time
        print(stream.put(next_token.item()), end="", flush=True)
        generated_ids[:, past_len+seq_length] = next_token
        tokens.append(next_token)
        inputs = torch.cat((torch.cat(passages).unsqueeze(0).to(torch_device), next_token.unsqueeze(0)), dim=-1)
        cache_position = torch.tensor([past_len+seq_length], device=torch_device)
        position_ids = cache_position.unsqueeze(0)
        seq_length += 1
        
            
        start_time = time.time()
        for _ in range(1, max_new_tokens):
            next_token = decode_one_tokens(model, next_token.unsqueeze(0), position_ids, cache_position, past_key_values, logits_warper, generation_config, inputs)
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

    return tokens

def prefill_and_generate(model, tokenizer, inputs, max_new_tokens=10000, use_cuda_graph: bool = True,
                         mode = 'normal'):
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch._dynamo.config.suppress_errors = True
    batch_size, seq_length = inputs.shape

    torch_device = "cuda:0"
    inputs = inputs.to(torch_device)

    tokens = []
    
    def decode_one_tokens(cuda_graph_runner, cur_token, position_ids, cache_position, past_key_values, use_cuda_graph: bool = True):
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
        if generation_config.do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
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
            inputs_embeds = inputs_embeds, cache_position=cache_position, past_key_values=past_key_values, return_dict=False, use_cache=True
        )[0][:,-1,:].unsqueeze(0).clone().to(torch_device)
        generation_config, model_kwargs = model._prepare_generation_config(
            None, max_length=max_new_tokens,
            do_sample=True, top_k=1, temperature=0.01 # change this to modify generate config
        )
        try: # transformers==4.43
            logits_warper = (
                model._get_logits_warper(generation_config,device=inputs.device)
            )
        except: 
            logits_warper = (
                model._get_logits_warper(generation_config)
            )
        next_token_scores = logits_warper(inputs, logits[:, -1, :])
        if generation_config.do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
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
            
            if next_token[0].item() == tokenizer.eos_token_id or tokenizer.decode(next_token) == '<|im_end|>':
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

class InferenceState(enum.Enum):
    UNLOAD = 0
    PREFILL = 1
    GENERATE = 2
    RESTORE = 3
