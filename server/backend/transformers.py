from typing import Any, List, Optional, Set
from transformers import LlamaTokenizer,AutoTokenizer, AutoConfig, LlamaForCausalLM,GenerationConfig, StaticCache, AutoModelForCausalLM,BitsAndBytesConfig

from server.schemas.base import ObjectID
from server.utils.multi_timer import MultiTimer
# from ..transformersInject.modeling_qwen2 import Qwen2ForCausalLM
import torch
import sys, os
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "KTransformers",
    )
)
from KTransformers.model.modeling_deepseek import DeepseekV2ForCausalLM
from KTransformers.model.modeling_qwen2_moe import Qwen2MoeForCausalLM
from KTransformers.gguf_injected_loader import optimize_model_using_optimization_dict
from .text_streamer import TextStreamer
from server.config.log import logger

from .args import ConfigArgs,default_args
import json
custom_models={
    "DeepseekV2ForCausalLM":DeepseekV2ForCausalLM,
    "Qwen2MoeForCausalLM":Qwen2MoeForCausalLM
    }

class TransformersInterface:
    args: ConfigArgs
    use_static_cache : bool = False


    model: Any
    tokenizer: AutoTokenizer
  
    cache: StaticCache
    generated_ids:torch.Tensor 
    seq_length:int
 
    streamer: TextStreamer

    # thread_related
    thread_id: Optional[str] = None
    ever_generated_ids: Set[int] = set()

    # profile
    timer:MultiTimer = MultiTimer()
    
    
    def __init__(self, args:ConfigArgs = default_args):
        self.args = args
        torch.set_default_dtype(torch.bfloat16)
        torch.set_grad_enabled(False)
        with open(args.optimize_config_path, 'r', encoding='utf-8') as file:
            optimize_config = json.load(file)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_dir,device = args.device)
        config=AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
        # config._attn_implementation="flash_attention_2"
        config.skip_init_experts = True
        with torch.device("meta"):
            self.model=custom_models[config.architectures[0]](config)
        optimize_model_using_optimization_dict(self.model, optimize_config, args.gguf_path)
    
        logger.info(f'{args.model_name} loaded from {args.model_dir} to {args.device}')
        self.cache = StaticCache(config=self.model.config, max_batch_size=args.batch_size, max_cache_len=args.cache_lens, device=args.device, dtype=self.model.dtype)
        logger.info(f'StaticCache (length={args.cache_lens}) created at {args.device}, batch size:{args.batch_size}')
        self.model.generation_config = GenerationConfig.from_pretrained(args.model_dir)
        if self.model.generation_config.pad_token_id is None:
            self.model.generation_config.pad_token_id = self.model.generation_config.eos_token_id
        self.streamer = TextStreamer(self.tokenizer)





    @property
    def current_ids(self):
        return self.generated_ids[:,self.seq_length-1].unsqueeze(1)
    
    @property
    def active_cache_position(self):
        return  torch.tensor([self.seq_length], device=self.args.device)


    def tokenize_prompt(self,prompt:str):
        input_ids = self.tokenizer.encode(prompt,return_tensors='pt').to(self.args.device)
        return input_ids

    def format_and_tokenize_input_ids(self,thread_id:ObjectID,messages:List):
        for m in messages:
            if m['role']=='system':
                logger.warn(f'change {m["role"]} to user')
                m['role'] = 'user'

        new_messages = [messages[0]]
        for m in messages[1:]:     
            if m['role'] == 'user' and new_messages[-1]['role']=='user':
                logger.warn('merge two adjacent user messages')
                new_messages[-1]['content']+=m['content']
            else:
                new_messages.append(m)   


        input_ids = self.tokenizer.apply_chat_template(new_messages,return_tensors='pt',add_generation_prompt=True).to(self.args.device)
        
        if (self.thread_id is not None) and self.thread_id == thread_id:
            x = self.generated_ids[:,:self.seq_length]
            y = input_ids[:,:self.seq_length]
            # We can only hope that the input_ids are the same
            unequal_mask = torch.ne(x,y)
            unequal_positions = torch.nonzero(unequal_mask)
            num_unequal_elements = unequal_mask.sum().item()
            logger.warn(f'num_unequal_elements: {num_unequal_elements}') 

            input_ids = input_ids[:,self.seq_length:]
        logger.debug(f'get input ids of shape {input_ids.shape}')
        return input_ids
    
    def append_new_tokens(self,new_tokens:int)->Optional[str]:    
        self.generated_ids[0,self.seq_length] = new_tokens
        self.seq_length+=1
        return self.streamer.put(new_tokens)

    def logits_to_token(self,logits:torch.Tensor):
        logits = logits/self.args.temperature

        for token_idx in self.ever_generated_ids:
            if logits[token_idx] < 0:
                logits[token_idx] *= self.args.repetition_penalty
            else:
                logits[token_idx] /= self.args.repetition_penalty

        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        sample = True
        if sample:
            last = torch.multinomial(probs, num_samples=1)
        else:
            _, last = torch.topk(probs, k=1, dim=-1)

        last = last.item()
        self.ever_generated_ids.add(last)
        return last



    def decode_one_tokens(self):
        if self.use_static_cache:
            logits = self.model(
                self.current_ids,
                cache_position=self.active_cache_position,
                past_key_values=self.cache,
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

    @torch.no_grad
    def prefill(self,input_ids:torch.Tensor,is_new:bool):
        input_ids_length = input_ids.shape[-1]
        logger.debug(f'input_ids: {input_ids.shape}')

        
        if is_new:
            self.cache.reset()
            self.ever_generated_ids.clear()
            former_seq_length = 0
            self.seq_length = input_ids_length
            self.generated_ids = torch.zeros(
                self.args.batch_size, self.seq_length + self.args.max_new_tokens + 1, dtype=torch.int, device=self.args.device
            )            
        else:
            logger.debug(f'generate_ids: {self.generated_ids.shape}')
            former_seq_length = self.seq_length
            self.seq_length += input_ids_length
            expected_length = self.seq_length + self.args.max_new_tokens+1
            delta_length = expected_length - self.generated_ids.shape[-1]
            if delta_length>0:
                new_generate_ids = torch.zeros(
                    self.args.batch_size, delta_length, dtype=torch.int, device=self.args.device
                )
                self.generated_ids = torch.cat([self.generated_ids,new_generate_ids],dim=-1)
        logger.debug(f'cache position: {former_seq_length} to {self.seq_length}')
        cache_position = torch.arange(former_seq_length,self.seq_length, device=self.args.device)
        self.generated_ids[:,cache_position] = input_ids.to(self.args.device).to(torch.int)

        self.timer.create_and_start_timer('prefill')

        if self.use_static_cache:
            logits = self.model(
                input_ids=input_ids, cache_position=cache_position, past_key_values=self.cache,return_dict=False, use_cache=True
            )[0]
        else:
            logits = self.model(
                input_ids=input_ids,return_dict=False
            )[0]

        t = self.timer.get_timer_sec('prefill')
        ave =  input_ids.shape[-1]/t
        logger.info(f'prefill time: {t:.5f}s, input id len:{input_ids.shape[-1]} ,average: {ave:.5f} token/s')


        next_token = self.logits_to_token(logits[0,-1,:])
        yield self.append_new_tokens(next_token)

    @torch.no_grad
    def generate(self):
        for _ in range(1, self.args.max_new_tokens):
            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
                next_token = self.decode_one_tokens()
                if next_token == self.tokenizer.eos_token_id:
                    assert self.args.batch_size == 1
                    break
                yield self.append_new_tokens(next_token)
        yield self.streamer.end()

    def check_is_new(self,thread_id:str):
        if not self.use_static_cache:
            return True


        if self.thread_id is None:
            self.thread_id = thread_id
            return True
        else:
            if self.thread_id==thread_id:
                return False
            else:
                self.thread_id = thread_id
                return True

    async def work(self,thread_id:str,input_ids:torch.Tensor):
        for t in self.prefill(input_ids,self.check_is_new(thread_id)):
            if t is not None:
                print(t,end='')
                yield t

        self.timer.create_and_start_timer('decode')
        token_count = 0
        for t in self.generate():
            token_count+=1
            if t is not None:
                print(t,end='')
                yield t
        print('')
        t = self.timer.get_timer_sec('decode')
        ave = token_count/t
        logger.info(f'decode time: {t:.5f}s, token count:{token_count} ,average: {ave:.5f} token/s, seq_length to {self.seq_length}')

class globalInterface:
    interface:TransformersInterface   
def get_interface()->TransformersInterface:
    return globalInterface.interface