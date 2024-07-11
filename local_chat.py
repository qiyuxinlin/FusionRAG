# Copyright 2024 Shaoyuan Chen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from contextlib import contextmanager
import torch
from torch import nn
import logging
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    GenerationConfig,
    TextStreamer,
)
import json
import time
import fire
from ktransformers.tools.prepare_optimize_config import gen_optimize_config
from ktransformers.optimize.optimize import optimize_via_injection
from ktransformers.models.custom_cache import StaticCache
from ktransformers.models.modeling_deepseek import DeepseekV2ForCausalLM
from ktransformers.models.modeling_qwen2_moe import Qwen2MoeForCausalLM
from typing import Dict

custom_models = {
    "DeepseekV2ForCausalLM": DeepseekV2ForCausalLM,
    "Qwen2MoeForCausalLM": Qwen2MoeForCausalLM,
}

class CUDAGraphRunner:

    def __init__(self):
        self.graph = None
        self.input_buffers: Dict[str, torch.Tensor] = {}
        self.output_buffers: Dict[str, torch.Tensor] = {}

    def capture(
        self,
        model,
        cur_token,
        position_ids,
        cache_position,
        past_key_values,
        **kwargs,
    ) -> None:
        assert self.graph is None
        # Capture the graph.
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        #self.graph.enable_debug_mode()
        self.model = model
        with torch.cuda.graph(self.graph):
            logits=model(cur_token, 
                         position_ids=position_ids,
                         cache_position=cache_position,
                         past_key_values=past_key_values,
                         **kwargs)[0]
        past_key_values.past_tokens -= 1
        torch.cuda.synchronize()
        #self.graph.debug_dump("cuda_graph_hooked.dot")

        # Save the input and output buffers.
        self.input_buffers = {
            "cur_token": cur_token,
            "position_ids": position_ids,
            "cache_position": cache_position,
        }
        self.output_buffers = {"logits": logits}
        return

    def forward(
        self,
        cur_token,
        position_ids,
        cache_position,
    ) -> torch.Tensor:
        # Copy the input tensors to the input buffers.
        self.input_buffers["cur_token"].copy_(cur_token)
        self.input_buffers["position_ids"].copy_(position_ids)
        self.input_buffers["cache_position"].copy_(cache_position)

        # Run the graph.
        #print("begin replay")
        #time.sleep(1)
        self.graph.replay()
        torch.cuda.synchronize()
        # Return the output tensor.
        return self.output_buffers["logits"]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

def prefill_and_generate(model, tokenizer, inputs, max_new_tokens=10000):
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch._dynamo.config.suppress_errors = True
    batch_size, seq_length = inputs.shape
    torch_device = inputs.device
    tokens = []
    
    def decode_one_tokens(cuda_graph_runner, cur_token, position_ids, cache_position, past_key_values):
        logits = cuda_graph_runner(cur_token, position_ids, cache_position)
        """
        with torch.cuda.stream(custom_stream):
            logits=model(cur_token, 
                         position_ids=position_ids,
                         cache_position=cache_position,
                         past_key_values=past_key_values,
                         return_dict=False, use_cache=True)[0]
        """            
        torch.cuda.synchronize()
        #print(logits)
        next_token_scores = logits_warper(inputs, logits[:, -1, :])
        if generation_config.do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_token = torch.argmax(next_token_scores, dim=-1)
        return next_token

    with torch.no_grad():
        past_key_values = StaticCache(
            config=model.config, max_batch_size=1, max_cache_len=4096, device=torch_device, dtype=model.dtype
        )
        cache_position = torch.arange(seq_length, device=torch_device)
        generated_ids = torch.zeros(
            batch_size, seq_length + max_new_tokens + 1, dtype=torch.int, device=torch_device
        )
        generated_ids[:, cache_position] = inputs.to(torch_device).to(torch.int)
        past_key_values.cur_idx=cache_position
        start_time = time.time()
        custom_stream = torch.cuda.Stream()
        with torch.cuda.stream(custom_stream):
            logits = model(
                inputs, cache_position=cache_position, past_key_values=past_key_values, return_dict=False, use_cache=True
            )[0].clone()
        generation_config, model_kwargs = model._prepare_generation_config(
            None, max_length=max_new_tokens,
            do_sample=True, top_k=5, top_p=0.85, temperature=0.5 # change this to modify generate config
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
        first_token_time = time.time() - start_time
        print(f"Time to generate first token: {first_token_time} seconds")
        print(f"Prefill sepeed: {seq_length/first_token_time} tokens/s")
        generated_ids[:, seq_length] = next_token
        tokens.append(next_token)
        inputs = torch.cat((inputs, next_token.unsqueeze(0)), dim=-1)
        seq_length += 1
        cache_position = torch.tensor([seq_length], device=torch_device)
        position_ids = cache_position.unsqueeze(0)
        temp_token = []
        #decode_one_tokens = torch.compile(decode_one_tokens)
        cuda_graph_runner = CUDAGraphRunner()
        cuda_graph_runner.capture(model, next_token.unsqueeze(0), position_ids, cache_position, past_key_values, return_dict=False, use_cache=True)
        print("finish capture")
        start_time = time.time()
        for _ in range(1, max_new_tokens):
            next_token = decode_one_tokens(cuda_graph_runner, next_token.unsqueeze(0), position_ids, cache_position, past_key_values)
            inputs = torch.cat((inputs, next_token.unsqueeze(0)), dim=-1)
            generated_ids[:, cache_position] = next_token.int()
            tokens.append(next_token.int())
            seq_length += 1
            
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

    total_time = time.time() - start_time
    tokens_generated = len(tokens)
    tokens_per_second = tokens_generated / total_time
    print(f"Tokens generated: {tokens_generated}")
    print(f"Total time for generation: {total_time} seconds.")
    print(f"Generate sepeed: {tokens_per_second} tokens/s.")
    # generate_prompts = torch.tensor(tokens).unsqueeze(0)
    # tokenizer.decode(tokens, skip_special_tokens=True)
    text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    print(text)
    return tokens

def main(
    model_name: str,
    optimize_config_path: str = None,
    gguf_path: str = None,
    use_generate:bool = False,
):

    torch.set_grad_enabled(False)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    torch.set_default_dtype(config.torch_dtype)

    with torch.device("meta"):
        if config.architectures[0] in custom_models:
            print("using custom modeling_xxx.py.")
            if "Qwen2Moe" in config.architectures[0]: # Qwen2Moe must use flash_attention_2 to avoid overflow.
                config._attn_implementation = "flash_attention_2"
            model = custom_models[config.architectures[0]](config)
        else:
            model = AutoModelForCausalLM.from_config(
                config, trust_remote_code=True, attn_implementation="flash_attention_2"
            )

    if optimize_config_path is not None:
        with open(optimize_config_path, "r", encoding="utf-8") as file:
            optimize_config = json.load(file)
    else:
        print("optimize_config_path is not set, generating it automatically.")
        optimize_config = {}
        gen_optimize_config(model, optimize_config)
        # print(optimize_config)

    if gguf_path is None:
        gguf_path = input(
            "please input the path of your gguf file(gguf file in the dir containing input gguf file must all belong to current model):"
        )
    optimize_via_injection(model, optimize_config, gguf_path, config)

    model.generation_config = GenerationConfig.from_pretrained(model_name)
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
    model.eval()

    logging.basicConfig(level=logging.INFO)

    while True:
        content = input("Chat: ")
        if content == "":
            content = "Please write a piece of quicksort code in C++."

        messages = [{"role": "user", "content": content}]
        input_tensor = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        torch.set_default_dtype(torch.bfloat16) # TODO: Remove this, replace dtype using config
        if use_generate: # does not optimized by cuda graph
            generated = model.generate(input_tensor.cuda(), max_new_tokens=10000, streamer=TextStreamer(tokenizer, skip_prompt=True), cache_implementation="static")#
        else:
            #generated = model.generate(input_tensor.cuda(), max_new_tokens=10000, streamer=TextStreamer(tokenizer, skip_prompt=True), cache_implementation="static")#
            generated = prefill_and_generate(model, tokenizer, input_tensor.cuda())
        #print(generated.numel())

fire.Fire(main)
