
import json
import torch
from transformers import LlamaTokenizer,AutoTokenizer, AutoConfig, LlamaForCausalLM,GenerationConfig, StaticCache, AutoModelForCausalLM,BitsAndBytesConfig

from server.backend.text_streamer import TextStreamer
from .transformers import TransformersInterface,ConfigArgs,default_args
from server.config.log import logger


class KTransformersInterface(TransformersInterface):
    def __init__(self,args:ConfigArgs= default_args):
        self.args = args
        torch.set_default_dtype(torch.bfloat16)
        torch.set_grad_enabled(False)
        with open(args.optimize_config_path, 'r', encoding='utf-8') as file:
            optimize_config = json.load(file)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_dir,device = args.device)
        config=AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
        config._attn_implementation="flash_attention_2"
        config.skip_init_experts = True
        # with torch.device("meta"):
        #     self.model=custom_models[config.architectures[0]](config)
        # optimize_model_using_optimization_dict(self.model, optimize_config, args.gguf_path)
    
        logger.info(f'{args.model_name} loaded from {args.model_dir} to {args.device}')
        self.cache = StaticCache(config=self.model.config, max_batch_size=args.batch_size, max_cache_len=args.cache_lens, device=args.device, dtype=self.model.dtype)
        logger.info(f'StaticCache (length={args.cache_lens}) created at {args.device}, batch size:{args.batch_size}')
        self.model.generation_config = GenerationConfig.from_pretrained(args.model_dir)
        if self.model.generation_config.pad_token_id is None:
            self.model.generation_config.pad_token_id = self.model.generation_config.eos_token_id
        self.streamer = TextStreamer(self.tokenizer)
        