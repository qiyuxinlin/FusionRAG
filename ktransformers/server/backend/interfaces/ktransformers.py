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
from ktransformers.models.custom_cache import StaticCache
from ktransformers.util.cuda_graph_runner import CUDAGraphRunner

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
        gen_optimize_config(self.model, optimize_config, device=args.device)
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
    
    def decode_one_tokens(self):
        if not hasattr(self, "cuda_graph_runner"):
            self.cuda_graph_runner = CUDAGraphRunner()
            self.cuda_graph_runner.capture(self.model, self.current_ids, self.active_cache_position.unsqueeze(0), self.active_cache_position, self.cache, return_dict=False, use_cache=True)
        
        if hasattr(self, "cuda_graph_runner"):
            logits = self.cuda_graph_runner(self.current_ids, self.active_cache_position.unsqueeze(0), self.active_cache_position)
            self.cache.change_seq_length(1)
            torch.cuda.synchronize()
            logits = logits[0,-1,:]
            return self.logits_to_token(logits)
        
        if self.use_static_cache:
            mask = torch.ones((1,self.seq_length)).to(self.args.device)
            logits = self.model(
                self.current_ids,
                cache_position=self.active_cache_position,
                past_key_values=self.cache,
                attention_mask=mask,
                return_dict=False,
                use_cache=True
            )[0]
        else:
            logits = self.model(
                self.current_ids,
                return_dict=False
            )[0]
        logits = logits[0,-1,:]

        return self.logits_to_token(logits)
