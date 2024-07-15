import json
from typing import Mapping
from numpy import rec
import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM
from ktransformers.models.modeling_deepseek import DeepseekV2ForCausalLM, DeepseekV2MoE, DeepseekV2Model, DeepseekV2Attention
from ktransformers.models.modeling_qwen2_moe import Qwen2MoeForCausalLM, Qwen2MoeSparseMoeBlock, Qwen2MoeModel
from ktransformers.util.custom_gguf import translate_name_to_gguf
from ktransformers.operators.linear import GPTQ_MARLIN_MIN_THREAD_N
custom_models={
    "DeepseekV2ForCausalLM":DeepseekV2ForCausalLM,
    "Qwen2MoeForCausalLM":Qwen2MoeForCausalLM
    }

def gen_optimize_config(module:nn.Module, out_data:Mapping, prefix="", device='cuda:0'):
    module_name = prefix[:-1]
    translated_name = translate_name_to_gguf(prefix)[:-1]
    recursive = True
    if isinstance(module, nn.Linear) and module.out_features%GPTQ_MARLIN_MIN_THREAD_N==0 and module.in_features%GPTQ_MARLIN_MIN_THREAD_N==0:
        out_data[module_name]={"key": translated_name,
            "module_name": "ktransformers.operators.linear",
            "class_name": "KTransformerLinear",
            "gpu_linear_type": "QuantizedLinearMarlin",
            "cpu_linear_type": "QuantizedLinearTorch",
            "device": "cuda"}
        recursive = False
    # if isinstance(module, nn.Linear) and module.out_features%GPTQ_MARLIN_MIN_THREAD_N==0 and module.in_features%GPTQ_MARLIN_MIN_THREAD_N==0:
    #     out_data[module_name]={"key": translated_name,
    #         "module_name": "operators.linear",
    #         "class_name": "KTransformerLinear",
    #         "gpu_linear_type": "QuantizedLinearMarlin",
    #         "cpu_linear_type": "QuantizedLinearTorch",
    #         "device": "cuda"}
    #     recursive = False
    if isinstance(module, nn.Linear) and module.out_features%GPTQ_MARLIN_MIN_THREAD_N==0 and module.in_features%GPTQ_MARLIN_MIN_THREAD_N==0:
        out_data[module_name]={"key": translated_name,
            "module_name": "ktransformers.operators.linear",
            "class_name": "QuantizedLinearMarlin",
            "device_idx": device}
        recursive = False
    if isinstance(module, DeepseekV2Attention):
        out_data[module_name]={"key": translated_name,
            "module_name": "ktransformers.operators.attention",
            "class_name": "DeepseekV2AttentionInjected",
            "device_idx": device}
        gen_optimize_config(module.rotary_emb, out_data, prefix + "rotary_emb.")
        recursive = False
    if isinstance(module, DeepseekV2MoE):
        out_data[module_name]={"key": translated_name,
            "module_name": "ktransformers.operators.experts",
            "class_name": "DeepseekV2MoEInjected",
            "device_idx": device}
    # if isinstance(module, DeepseekV2Model):
    #     out_data[module_name]={"key": translated_name,
    #         "module_name": "operators.layer_wise_prefill_deepseek",
    #         "class_name": "DeepseekV2ModelPerLayerPrefill",
    #         "device_idx": device}
    # if isinstance(module, Qwen2MoeModel):
    #     out_data[module_name]={"key": translated_name,
    #         "module_name": "ktransformers.operators.layer_wise_prefill_qwen_moe",
    #         "class_name": "Qwen2MoeModelPerLayerPrefill",
    #         "device_idx": device}
    if isinstance(module, Qwen2MoeSparseMoeBlock):
        out_data[module_name]={"key": translated_name,
            "module_name": "ktransformers.operators.experts",
            "class_name": "Qwen2MoeSparseMoeBlockInjected",
            "device_idx": "cuda:0"}
    # if isinstance(module, nn.ModuleList) and "expert" in module_name:
    #     out_data[module_name]={"key": translated_name,
    #         "module_name": "operators.experts",
    #         "file_name": "experts",
    #         "class_name": "MLPExperts",
    #         # "class_name": "MLPExpertsMarlin",
    #         # "class_name": "MLPExpertsTorch",
    #         "device_idx": "cuda:0"}
    #     recursive = False
    if isinstance(module, nn.ModuleList) and "expert" in module_name:
        out_data[module_name]={"key": translated_name,
            "module_name": "ktransformers.operators.experts",
            "file_name": "experts",
            "class_name": "KTransformersMLPExpert",
            "gpu_mlp_type": "MLPExpertsTorch",
            "cpu_mlp_type": "MLPCPUExperts",
            # "cpu_mlp_type": "MLPExpertsTorch",
            "device": "cpu",
            "out_device": "cuda"}
        recursive = False
    if "YarnRotaryEmbedding" in module.__class__.__name__:
        out_data[module_name]={"key": translated_name,
            "module_name": "ktransformers.operators.RoPE",
            "class_name": "YarnRotaryEmbedding",
            "device_idx": device}
        recursive = False
    elif "RotaryEmbedding" in module.__class__.__name__:
        out_data[module_name]={"key": translated_name,
            "module_name": "ktransformers.operators.RoPE",
            "class_name": "RotaryEmbedding",
            "device_idx": device}
        recursive = False
    if module_name not in out_data:
        out_data[module_name]="default"
        
    if recursive:
        for name, child in module._modules.items():
            if child is not None:
                child_prefix = prefix + name + "."
                gen_optimize_config(child, out_data, child_prefix)

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