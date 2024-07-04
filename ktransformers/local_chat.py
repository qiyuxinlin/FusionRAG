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

from tools.prepare_optimize_config import gen_optimize_config
from util.gguf_injected_loader import optimize_model_using_optimization_dict
from models.modeling_deepseek import DeepseekV2ForCausalLM
from models.modeling_qwen2_moe import Qwen2MoeForCausalLM


class Profiler:
    total_moe_time: float = 0.0
    total_time: float = 0.0
    start_instant: float

    @contextmanager
    def measure_moe_time(self):
        start = time.perf_counter()
        yield
        self.total_moe_time += time.perf_counter() - start

    @contextmanager
    def __call__(self):
        self.total_moe_time = 0.0
        self.total_time = 0.0
        self.start_instant = time.perf_counter()
        yield
        self.total_time = time.perf_counter() - self.start_instant




custom_models = {
    "DeepseekV2ForCausalLM": DeepseekV2ForCausalLM,
    "Qwen2MoeForCausalLM": Qwen2MoeForCausalLM,
}


def main(
    model_name: str,
    optimize_config_path: str = None,
    gguf_path: str = None,
    profile: bool = False,
):

    torch.set_grad_enabled(False)

    profiler = Profiler()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    torch.set_default_dtype(config.torch_dtype)

    config._attn_implementation = "flash_attention_2"

    with torch.device("meta"):
        if config.architectures[0] in custom_models:
            print("using custom modeling_xxx.py.")
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
    optimize_model_using_optimization_dict(model, optimize_config, gguf_path, config)

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
        torch.set_default_dtype(torch.bfloat16)
        with profiler():
            generated = model.generate(
                input_tensor.cuda(),
                max_new_tokens=10000,
                streamer=TextStreamer(tokenizer, skip_prompt=True),
                cache_implementation="static",
            )  #

        if profile:
            print(
                f"total time: {profiler.total_time} s, MoE time: {profiler.total_moe_time} s, {generated.numel()} iterations"
            )
        else:
            print(
                f"total time: {profiler.total_time} s, {generated.numel()} iterations"
            )


fire.Fire(main)
