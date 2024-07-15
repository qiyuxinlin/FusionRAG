import torch
from torch import nn
import itertools
from ktransformers.util.custom_gguf import translate_name_to_gguf
from ktransformers.util.custom_gguf import GGUFLoader
from ktransformers.operators import base_operator

def _set_module(model, submodule_key, module):
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

def _set_param(module: nn.Module, name: str, weights: torch.Tensor):
    
    param=nn.parameter.Parameter(weights, requires_grad=False)
    if isinstance(module, nn.Linear) and len(weights.shape)==1:
        param.unsqueeze_(0)
    setattr(module, name, param)

def load_weight_default(module: nn.Module, gguf_loader: GGUFLoader, prefix: str = ""):
    persistent_buffers = {k: v for k, v in module._buffers.items() if k not in module._non_persistent_buffers_set}
    local_name_params = itertools.chain(module._parameters.items(), persistent_buffers.items())
    local_state = {k: v for k, v in local_name_params if v is not None}
    for name, param in local_state.items():
        key = prefix + name
        translated_key = translate_name_to_gguf(key)
        print("default loading weights", key, translated_key)
        if translated_key in gguf_loader.tensor_file_map:
            target_dtype = torch.get_default_dtype()
            device = "cpu" if "embed_tokens" in key or "lm_head" in key else "cuda"
            weights = torch.tensor(gguf_loader.load_gguf_tensor(translated_key)).to(device=device).to(dtype=target_dtype)
            _set_param(module, name, weights)
            del weights
        else:
            #print(load_config.tensor_file_map.keys())
            raise Exception(f"can't fand {translated_key} in GGUF file!")
        
def load_weights(module:nn.Module, gguf_loader:GGUFLoader, prefix='', return_when_injected:bool = False, only_load_injected:bool = False):
    # print(f"recursively loading weights {prefix},{return_when_injected=}, {only_load_injected=}")
    for name, child in module._modules.items():
        if child is not None:
            if isinstance(child, base_operator.BaseInjectedModule) and return_when_injected:
                pass
            elif isinstance(child, base_operator.BaseInjectedModule):
                child.load()
                load_weights(child, gguf_loader, prefix+name+"." if not isinstance(module, base_operator.BaseInjectedModule) else prefix, return_when_injected, True)
            else:
                if not isinstance(module, base_operator.BaseInjectedModule) and not only_load_injected:
                    load_weight_default(child, gguf_loader, prefix+name+".")
                load_weights(child, gguf_loader, prefix+name+"." if not isinstance(module, base_operator.BaseInjectedModule) else prefix, return_when_injected, only_load_injected)
