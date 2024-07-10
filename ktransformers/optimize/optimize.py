from typing import Any, Mapping
import torch
from torch import nn
from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
# from operators import BaseInjectedModule
from util.custom_gguf import GGUFLoader
from util.utils import _set_module, _set_param, load_weights
import itertools

def inject(module, local_optimization_dict, model_config:AutoConfig ,gguf_loader:GGUFLoader, prefix=''):
    for name, child in module._modules.items():
        if child is not None:
            child_prefix = prefix + name
            if child_prefix in local_optimization_dict:
                inject_module_meta=local_optimization_dict[child_prefix]
                if isinstance(inject_module_meta, Mapping):
                    module_cls=getattr(__import__(inject_module_meta["module_name"], fromlist=[""]), inject_module_meta["class_name"])
                    print(f"Injecting {child_prefix} as", inject_module_meta["module_name"], ".", inject_module_meta["class_name"])
                    inject_module=module_cls(gguf_loader=gguf_loader, config=model_config, orig_module=child, **inject_module_meta)
                    _set_module(module, name, inject_module)
                elif isinstance(inject_module_meta, str):
                    assert inject_module_meta=="default", "for str inject_module_meta, only support \"default\"."
                else:
                    raise Exception("inject_module_meta must be a dict or str")
                child_prefix += "."
                child_optimization_dict = {k: v for k, v in local_optimization_dict.items() if k.startswith(child_prefix)}
                inject(child, child_optimization_dict, model_config, gguf_loader, child_prefix)

def del_meta(module:nn.Module):
    #print("default loading weights", prefix)
    persistent_buffers = {k: v for k, v in module._buffers.items() if k not in module._non_persistent_buffers_set}
    local_name_params = itertools.chain(module._parameters.items(), persistent_buffers.items())
    local_state = {k: v for k, v in local_name_params if v is not None}
    for name, param in local_state.items():
        if param.device == "meta" or param.device == torch.device("meta"):
            module.__delattr__(name)
    for name, child in module._modules.items():
        del_meta(child)

def optimize_via_injection(module:nn.Module, optimization_dict: Mapping[str, Any], gguf_path: str, model_config: PretrainedConfig) -> None:
    
    if not isinstance(optimization_dict, Mapping):
        raise TypeError(f"Expected optimization_dict to be dict-like, got {type(optimization_dict)}.")

    gguf_loader=GGUFLoader(gguf_path)
    inject(module, optimization_dict, model_config, gguf_loader)
    load_weights(module, gguf_loader)
    del_meta(module)