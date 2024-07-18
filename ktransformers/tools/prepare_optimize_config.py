from ast import Dict
import json
from typing import Mapping, List
import re
import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM
from ktransformers.models.modeling_deepseek import DeepseekV2ForCausalLM, DeepseekV2MoE, DeepseekV2Model, DeepseekV2Attention
from ktransformers.models.modeling_qwen2_moe import Qwen2MoeForCausalLM, Qwen2MoeSparseMoeBlock, Qwen2MoeModel

from ktransformers.operators.linear import GPTQ_MARLIN_MIN_THREAD_N
custom_models={
    "DeepseekV2ForCausalLM":DeepseekV2ForCausalLM,
    "Qwen2MoeForCausalLM":Qwen2MoeForCausalLM
    }


if __name__ == "__main__":
    model_name=input("please input the path of your model in safetensors format(don't need the big *.safetensors files)")
    output_path=input("please input the output path of your optimize config json file:")
    torch.set_grad_enabled(False)
    config=AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    torch.set_default_dtype(config.torch_dtype)
    config._attn_implementation="flash_attention_2"
    with torch.device("meta"):
        if config.architectures[0] in custom_models:
            print("using custom modeling_xxx.py.")
            model=custom_models[config.architectures[0]](config)
        else:
            model=AutoModelForCausalLM.from_config(config, trust_remote_code=True, attn_implementation="flash_attention_2")
    out_data={}
    gen_optimize_config(model, out_data)
    with open(output_path,"w") as f:
        json.dump(out_data, f)
    print("finish generate optimize config json file.")