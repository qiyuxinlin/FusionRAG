from torch import nn
from models.modeling_deepseek import DeepseekV2YarnRotaryEmbedding, DeepseekV2RotaryEmbedding
from operators.base_operator import BaseInjectedModule
from util.custom_gguf import GGUFLoader
from transformers.configuration_utils import PretrainedConfig
# Copied from transformers.models.mixtral.modeling_mixtral.MixtralRotaryEmbedding with Mixtral->Qwen2Moe
class RotaryEmbedding(BaseInjectedModule, DeepseekV2RotaryEmbedding):
    def __init__(self,
                 key: str,
                 gguf_loader : GGUFLoader,
                 config: PretrainedConfig,
                 orig_module: nn.Module,
                 device: str = "cuda",
                 **kwargs):
        BaseInjectedModule.__init__(self, key, gguf_loader, config, orig_module, device, **kwargs)
        self.orig_module.__init__(orig_module.dim,
            orig_module.max_position_embeddings,
            orig_module.base,
            device)

class YarnRotaryEmbedding(BaseInjectedModule, DeepseekV2YarnRotaryEmbedding):
    def __init__(self,
                 key: str,
                 gguf_loader : GGUFLoader,
                 config: PretrainedConfig,
                 orig_module: nn.Module,
                 device: str = "cuda",
                 **kwargs):
        BaseInjectedModule.__init__(self, key, gguf_loader, config, orig_module, device, **kwargs)
        self.orig_module.__init__(orig_module.dim,
            orig_module.max_position_embeddings,
            orig_module.base,
            device,
            orig_module.scaling_factor,
            orig_module.original_max_position_embeddings,
            orig_module.beta_fast,
            orig_module.beta_slow,
            orig_module.mscale,
            orig_module.mscale_all_dim)