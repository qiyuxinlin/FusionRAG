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
    path = "./test_input.txt"
    # read file
    with open (path, "r", encoding='utf-8') as f:
        long_context = f.read()

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
            # content = "Please write a piece of quicksort code in C++."
            content = long_context 
            # content = """
            # <|im_start|>system

            # 你的名字叫小A，是一名诚实、有帮助的、无害的科幻小说《三体》的智能问答助手。你的任务是学习和理解下文中提供的《三体》部分章节的内容，回答用户提出的关于《三体》的问题。
            # 回答时要求不能杜撰，必须实事求是，不能回答超出材料范围的内容，回答过程需要参考材料内容一步一步的思考，在回答时必须给出参考材料的引用。回答内容形式符合符号@@@中的输出规范要求，同时必须仿照符号===中给出的输出样例。
            # @@@
            # 首先回答用户提出的问题，然后给出是从材料中的那些内容推理出来的，最后总结结论
            # @@@

            # ===
            # 丁仪和杨冬是情侣关系。在原文中提到：“那个一直低着头沉默的人半天才有所反应，掏出一个白色的信封隔着桌子递给汪淼，大史在旁边低声说：‘他是杨冬的男友。’” 这句话说明了丁仪是杨冬的男朋友，因此他们是情侣关系
            # 。
            # 参考材料内容如下：
            #     那个一直低着头沉默的人半天才有所反应，掏出一个白色的信封隔着桌子递给汪淼，大史在旁边低声说：“他是杨冬的男友。”汪淼这才想起自己在良湘的高能加速器工地中也见过丁仪，他是理论组的成员，这名物理学家因>在对球状闪电的研究中发现宏原子而闻名于世。
            # 因此可以确定，丁仪和杨冬是情侣关系。
            # ===
            # <|im_end|>
            # <|im_start|>context
            # “顺山倒咧——”
            # 随着这声嘹亮的号子，一棵如巴特农神庙的巨柱般高大的落叶松轰然倒下，叶文洁感到大地抖动了一下。她拿起斧头和短锯，开始去除巨大树身上的枝丫。每到这时，她总觉得自己是在为一个巨人整理遗体。她甚至常常有这样的想象：这巨人就是自己的父亲。两年前那个凄惨的夜晚，她在太平间为父亲整理遗容时的感觉就在这时重现。巨松上那绽开的树皮，似乎就是父亲躯体上累累的伤痕。
            # 内蒙古生产建设兵团的六个师四十一个团十多万人就分布在这辽阔的森林和草原之间。刚从城市来到这陌生的世界时，很多兵团知青都怀着一个浪漫的期望：当苏修帝国主义的坦克集群越过中蒙边境时，他们将飞快地武装起来，用自己的血肉构成共和国的第一道屏障。事实上，这也确实是兵团组建时的战略考虑之一。但他们渴望的战争就像草原天边那跑死马的远山，清晰可见，但到不了眼前，于是他们只有垦荒、放牧和砍伐。这些曾在“大串联”中燃烧青春的年轻人很快发现，与这广阔天地相比，内地最大的城市不过是个羊圈；在这寒冷无际的草原和森林间，燃烧是无意义的，一腔热血喷出来，比一堆牛粪凉得更快，还不如后者有使用价值。但燃烧是他们的命运，他们是燃烧的一代。于是，在他们的油锯和电锯下，大片的林海化为荒山秃岭；在他们的拖拉机和康拜因（联合收割机）下，大片的草原被犁成粮田，然后变成沙漠。
            # 叶文洁看到的砍伐只能用疯狂来形容，高大挺拔的兴安岭落叶松、四季常青的樟子松、亭亭玉立的白桦、耸入云天的山杨、西伯利亚冷杉，以及黑桦、柞树、山榆、水曲柳、钻天柳、蒙古栎，见什么伐什么，几百把油锯如同一群钢铁蝗虫，她的连队所过之处，只剩下一片树桩。
            # <|im_end|>
            # <|im_start|>user
            # 上文是小说《三体》中部分章节的参考材料内容,部分材料与问题可能并不相关，需要你进行甄别，请根据材料内容回答符号###中的用户的提问，不能杜撰，要实事求是，不能回答超出材料范围的内容，回答过程需要参考材料内容一步一步的思考，在回答时必须给出参考材料的引用。
            # ###
            # 叶文洁用什么去除巨大树木身上的枝丫？
            # ###
            # <|im_end|>
            # <|im_start|>assistant
            # """
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
