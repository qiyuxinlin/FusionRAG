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
import sys, os
sys.path.append(os.path.dirname(__file__) + '/ktransformers')
from tools.prepare_optimize_config import gen_optimize_config
from optimize.optimize import optimize_via_injection
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
    path = "./test_input.txt"
    # read file
    with open (path, "r", encoding='utf-8') as f:
        long_context = f.read()

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
        torch.set_default_dtype(torch.bfloat16)
        with profiler():
            generated = model.generate(
                input_tensor.cuda(),
                max_new_tokens=1,
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
