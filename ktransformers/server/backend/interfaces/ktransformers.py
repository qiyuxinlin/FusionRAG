
import json
import torch
from transformers import LlamaTokenizer,AutoTokenizer, AutoConfig, LlamaForCausalLM,GenerationConfig, Cache, AutoModelForCausalLM,BitsAndBytesConfig
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers.configuration_utils import PretrainedConfig
from .transformers import TransformersInterface,ConfigArgs, TransformersThreadContext,default_args,TextStreamer
from ktransformers.server.config.log import logger
from ktransformers.models.modeling_deepseek import DeepseekV2ForCausalLM
from ktransformers.models.modeling_qwen2_moe import Qwen2MoeForCausalLM
from ktransformers.optimize.optimize import optimize_via_injection
from ktransformers.tools.prepare_optimize_config import gen_optimize_config

class StaticCache(Cache):
    """
    Static Cache class to be used with `torch.compile(model)`.

    Parameters:
        config (`PretrainedConfig):
            The configuration file defining the shape-related attributes required to initialize the static cache.
        max_batch_size (`int`):
            The maximum batch size with which the model will be used.
        max_cache_len (`int`):
            The maximum sequence length with which the model will be used.
        device (`torch.device`):
            The device on which the cache should be initialized. Should be the same as the layer.
        dtype (*optional*, defaults to `torch.float32`):
            The default `dtype` to use when initializing the layer.
    """

    def __init__(self, config: PretrainedConfig, max_batch_size: int, max_cache_len: int, device, dtype=None) -> None:
        super().__init__()
        self.max_batch_size = max_batch_size
        self.max_cache_len = config.max_position_embeddings if max_cache_len is None else max_cache_len
        # Some model define a custom `head_dim` != config.hidden_size // config.num_attention_heads
        self.head_dim = (
            config.head_dim if hasattr(config, "head_dim") else config.hidden_size // config.num_attention_heads
        )

        self.dtype = dtype if dtype is not None else torch.float32
        self.num_key_value_heads = (
            config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        )

        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        cache_shape = (max_batch_size, self.num_key_value_heads, self.max_cache_len, self.head_dim)
        if config.architectures[0] == "DeepseekV2ForCausalLM":
            key_shape = (max_batch_size, 1, self.max_cache_len, config.qk_rope_head_dim)
            value_shape = (max_batch_size, 1, self.max_cache_len, config.kv_lora_rank)
        else:
            key_shape = cache_shape
            value_shape = cache_shape
        for _ in range(config.num_hidden_layers):
            # Note: `mark_static_address` is used to tag the cache as an fixed data pointer, preventing cuda graph
            # breaks when updating the cache.
            new_layer_key_cache = torch.zeros(key_shape, dtype=self.dtype, device=device)
            new_layer_value_cache = torch.zeros(value_shape, dtype=self.dtype, device=device)
            torch._dynamo.mark_static_address(new_layer_key_cache)
            torch._dynamo.mark_static_address(new_layer_value_cache)
            self.key_cache.append(new_layer_key_cache)
            self.value_cache.append(new_layer_value_cache)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.
        It is VERY important to index using a tensor, otherwise you introduce a copy to the device.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. The `StaticCache` needs the `cache_position` input
                to know how where to write in the cache.

        Return:
            A tuple containing the updated key and value states.
        """
        cache_position = cache_kwargs.get("cache_position")
        k_out = self.key_cache[layer_idx]
        v_out = self.value_cache[layer_idx]

        k_out[:, :, cache_position] = key_states
        v_out[:, :, cache_position] = value_states

        return k_out, v_out

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states that were seen by the model."""
        # Occupied cache == any slot in the 3rd dim (sequence length) holds a non-zero value. To save on compute, let's
        # limit the check to the first batch member and head dimension.
        # TODO: deprecate this function in favor of `cache_position`
        return (self.key_cache[layer_idx][0, 0].any(dim=-1)).sum()

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states."""
        return self.max_cache_len

    def reset(self):
        """Resets the cache values while preserving the objects"""
        for layer_idx in range(len(self.key_cache)):
            # In-place ops prevent breaking the static address
            self.key_cache[layer_idx].zero_()
            self.value_cache[layer_idx].zero_()
custom_models={
    "DeepseekV2ForCausalLM":DeepseekV2ForCausalLM,
    "Qwen2MoeForCausalLM":Qwen2MoeForCausalLM
    }


class KTransformersThreadContext(TransformersThreadContext):
    pass


class KTransformersInterface(TransformersInterface):
    def __init__(self,args:ConfigArgs= default_args):
        self.args = args
        torch.set_default_dtype(torch.bfloat16)
        torch.set_grad_enabled(False)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_dir,device = args.device)
        config=AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
        if config.architectures[0] == "Qwen2MoeForCausalLM":
            config._attn_implementation="flash_attention_2"

        with torch.device("meta"):
            self.model=custom_models[config.architectures[0]](config)

        print("optimize_config_path is not set, generating it automatically.")
        optimize_config = {}
        gen_optimize_config(self.model, optimize_config,device=args.device)
        # print(optimize_config)

        gguf_path = args.gguf_path
        if gguf_path is None:
            gguf_path = input(
            "please input the path of your gguf file(gguf file in the dir containing input gguf file must all belong to current model):"
            )
        optimize_via_injection(self.model, optimize_config, gguf_path, config)

        
    
        logger.info(f'{args.model_name} loaded from {args.model_dir} to {args.device}')
        self.cache = StaticCache(config=self.model.config, max_batch_size=args.batch_size, max_cache_len=args.cache_lens, device=args.device, dtype=self.model.dtype)
        logger.info(f'StaticCache (length={args.cache_lens}) created at {args.device}, batch size:{args.batch_size}')
        self.model.generation_config = GenerationConfig.from_pretrained(args.model_dir)
        if self.model.generation_config.pad_token_id is None:
            self.model.generation_config.pad_token_id = self.model.generation_config.eos_token_id
        self.streamer = TextStreamer(self.tokenizer)
    

