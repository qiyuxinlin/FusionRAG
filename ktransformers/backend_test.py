from transformers import AutoTokenizer, StaticCache, logging, AutoConfig
from model.modeling_deepseek import DeepseekV2ForCausalLM
from model.modeling_qwen2_moe import Qwen2MoeForCausalLM
from gguf_injected_loader import optimize_model_using_optimization_dict
import torch
import torch.nn as nn
import time
import json
prompts = [
    {"role": 'system', "content": "you are a helpful assistant. 你是一个乐于助人的助手"},
    {"role": 'user', "content": "你好，请给我介绍一下秦始皇"}
]

import torch._dynamo
torch.set_grad_enabled(False)

torch._dynamo.config.suppress_errors = True
NUM_TOKENS_TO_GENERATE = 1000
torch_device = "cuda:0"
with open("optimize_config/Qwen2-57B-A14B-Instruct.json", 'r', encoding='utf-8') as file:
    optimize_config = json.load(file)
tokenizer = AutoTokenizer.from_pretrained("/mnt/default/data/Qwen2-57B-A14B-Instruct/", use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
config=AutoConfig.from_pretrained("/mnt/default/data/Qwen2-57B-A14B-Instruct/", trust_remote_code=True)
config._attn_implementation="flash_attention_2"
config.skip_init_experts = True
torch.set_default_dtype(config.torch_dtype)
with torch.device("meta"):
    model = Qwen2MoeForCausalLM(config)
optimize_model_using_optimization_dict(model, optimize_config, "/mnt/default/data/Qwen2-57B-A14B-Instruct-GGUF/q8_0/qwen2-57b-a14b-instruct-q8_0-00001-of-00002.gguf")
inputs = tokenizer.apply_chat_template(prompts, tokenize=True, add_generation_prompt=True)
print(tokenizer.apply_chat_template(prompts, tokenize=False, add_generation_prompt=True))

inputs = torch.tensor([inputs]).to(torch_device)
print("inputs lenght:", inputs.shape)
def decode_one_tokens(model, cur_token, input_pos, cache_position, past_key_values):
    logits = model(
        cur_token,
        position_ids=input_pos,
        cache_position=cache_position,
        past_key_values=past_key_values,
        return_dict=False,
        use_cache=True
    )[0].clone()
    next_token_scores = logits_warper(inputs, logits[:, -1, :])
    if generation_config.do_sample:
        probs = nn.functional.softmax(next_token_scores, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
    else:
        next_token = torch.argmax(next_token_scores, dim=-1)
    
    return next_token

batch_size, seq_length = inputs.shape
tokens = []
with torch.no_grad():
    past_key_values = StaticCache(
        config=model.config, max_batch_size=1, max_cache_len=4096, device=torch_device, dtype=model.dtype
    )
    cache_position = torch.arange(seq_length, device=torch_device)
    generated_ids = torch.zeros(
        batch_size, seq_length + NUM_TOKENS_TO_GENERATE + 1, dtype=torch.int, device=torch_device
    )
    generated_ids[:, cache_position] = inputs.to(torch_device).to(torch.int)
    start_time = time.time()
    logits = model(
        inputs, cache_position=cache_position, past_key_values=past_key_values, return_dict=False, use_cache=True
    )[0].clone()
    generation_config, model_kwargs = model._prepare_generation_config(
        None, max_length=NUM_TOKENS_TO_GENERATE,
        do_sample=True, top_k=5, top_p=0.85, temperature=0.01  # 调整这些参数以增加输出截止符的概率
    )
    logits_warper = (
        model._get_logits_warper(generation_config) if generation_config.do_sample else None
    )
    next_token_scores = logits_warper(inputs, logits[:, -1, :])
    if generation_config.do_sample:
        probs = nn.functional.softmax(next_token_scores, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
    else:
        next_token = torch.argmax(next_token_scores, dim=-1)
    first_token_time = time.time() - start_time  # 计算生成第一个token的时间
    print(f"Time to generate first token: {first_token_time} seconds")
    generated_ids[:, seq_length] = next_token
    tokens.append(next_token)
    inputs = torch.cat((inputs, next_token.unsqueeze(0)), dim=-1)
    
    cache_position = torch.tensor([seq_length + 1], device=torch_device)
    position_ids = cache_position.unsqueeze(0)
    temp_token = []
    start_time = time.time()
    # decode_one_tokens = torch.compile(decode_one_tokens, mode="reduce-overhead", fullgraph=True)
    for _ in range(1, NUM_TOKENS_TO_GENERATE):
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
            next_token = decode_one_tokens(model, next_token.unsqueeze(0), position_ids, cache_position, past_key_values)
            inputs = torch.cat((inputs, next_token.unsqueeze(0)), dim=-1)
            generated_ids[:, cache_position] = next_token.int()
            tokens.append(next_token.int())
            
            if next_token[0].item() == tokenizer.eos_token_id:
                break
            elif next_token[0].item() in range(3,259):
                temp_token.append(next_token[0])
            else:
                if len(temp_token) == 0:
                    temp_token.append(next_token[0])
                print(tokenizer.decode(torch.tensor(temp_token), skip_special_tokens=True))
                temp_token = []
            cache_position += 1
            position_ids = cache_position.unsqueeze(0)

total_time = time.time() - start_time  # 计算生成阶段的总时间
tokens_generated = len(tokens)
tokens_per_second = tokens_generated / total_time  # 计算每秒生成的token数量
print(f"Tokens generated: {tokens_generated}")
print(f"Total time for generation: {total_time} seconds")
print(f"Tokens per second: {tokens_per_second}")
# generate_prompts = torch.tensor(tokens).unsqueeze(0)
# tokenizer.decode(tokens, skip_special_tokens=True)
text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
print(text)