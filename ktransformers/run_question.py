import json
import os
import sys
import csv
import shutil
import time

import torch
import numpy as np
from typing import List, Dict, Any, Tuple
from openai import OpenAI
from transformers import AutoTokenizer, AutoConfig
from .models.custom_cache import StaticCache
from .util.utils import (
    prefill_and_save_kv_cache,
    load_kv_and_generate,
    prefill_with_cache_and_save_preprocess,
    rotate_half,
    find_group_and_index
)
import hashlib
import faiss
from FlagEmbedding import FlagModel

DEFAULT_SYSTEM_PROMPT = "<|im_start|>system\nYou are a helpful assistant.\nWrite a high-quality answer for the given question using only the provided search results. The answer process requires reference to the material content and step-by-step thinking."

question_test = {
    "question": """
    Are director of film Move (1970 Film) and director of film M\u00e9diterran\u00e9e (1963 Film) from the same country? please output using json format
    please output using json format:
    {{
    "reason": "",
    "sub_query": ""
    }}
    """,
    "gold_docs": [
        "Stuart Rosenberg (August 11, 1927 \u2013 March 15, 2007) was an American film and television director whose motion pictures include \"Cool Hand Luke\" (1967), \"Voyage of the Damned\" (1976), \"The Amityville Horror\" (1979), and \"The Pope of Greenwich Village\" (1984). He was noted for his work with actor Paul Newman.",
        "M\u00e9diterran\u00e9e (1963 film): M\u00e9diterran\u00e9e is a 1963 French experimental film directed by Jean-Daniel Pollet with assistance from Volker Schl\u00f6ndorff. It was written by Philippe Sollers and produced by Barbet Schroeder, with music by Antione Duhamel. The 45 minute film is cited as one of Pollet's most influential films, which according to Jonathan Rosenbaum directly influenced Jean-Luc Goddard's \"Contempt\", released later the same year. Footage for the film was shot around the Mediterranean, including at a Greek temple, a Sicilian garden, the sea, and also features a fisherman, a bullfighter, and a girl on an operating table.",
        "Move (1970 film): Move is a 1970 American comedy film starring Elliott Gould, Paula Prentiss and Genevi\u00e8ve Wa\u00efte, and directed by Stuart Rosenberg. The screenplay was written by Joel Lieber and Stanley Hart, adapted from a novel by Lieber.",
        "Jean-Daniel Pollet (1936\u20132004) was a French film director and screenwriter who was most active in the 1960s and 1970s. He was associated with two approaches to filmmaking: comedies which blended burlesque and melancholic elements, and poetic films based on texts by writers such as the French poet Francis Ponge."
    ],
    "answer": "no"
}

class RerankModel:
    def __init__(self,
                 bge_model_path: str):
        self.bgem3 = FlagModel(bge_model_path, use_fp16=True, device="cuda:0")

    def preprocess_build_faiss_index(self, all_documents: list[str], topk: int):
        corpus_embeddings = self.bgem3.encode(all_documents)
        # Build FAISS index
        dim = corpus_embeddings.shape[-1]
        index = faiss.index_factory(dim, 'Flat', faiss.METRIC_INNER_PRODUCT)
        corpus_embeddings = corpus_embeddings.astype(np.float32)
        index.train(corpus_embeddings)
        index.add(corpus_embeddings)
        print(f"FAISS index built with {index.ntotal} vectors")

        # Search for similar documents globally
        print(f"Searching for top-{topk} similar documents for each document...")
        corpus_embeddings_query = self.bgem3.encode_queries(all_documents)
        corpus_embeddings_query = corpus_embeddings_query.astype(np.float32)
        score, idx = index.search(corpus_embeddings_query, k=topk)
        context_rank = idx  # Shape: [total_docs, topk]
        return context_rank

    def clean_(self):
        self.bgem3.model = self.bgem3.model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        del self.bgem3
        import gc
        gc.collect()

class FusionRAGModel:
    def __init__(
            self,
            model_path: str,
            use_multi_gpu: bool,
            model_type='qwen',
            device="cuda:0",
            max_cache_len=32768,
            cache_path='/mnt/data3/reflect/',
            model_name='Qwen2.5-7B-Instruct',
    ):
        self.model_cache_root = os.path.join(cache_path, model_name)
        self.save_path = os.path.join(self.model_cache_root, 'kv_cache')
        self.preprocess_save_path = os.path.join(self.model_cache_root, 'preprocess_kv_cache')
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(self.preprocess_save_path, exist_ok=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        config._attn_implementation = "sdpa"
        print(f"Loading {model_type} model...")
        if use_multi_gpu:
            print("Using multi-GPU with device_map='auto'")
        self.model, self.device_map = self.load_model(model_type, model_path, config, device, use_multi_gpu)

        cache_device = self.device_map if use_multi_gpu else device
        self.past_key_values = StaticCache(
            config=self.model.config,
            max_batch_size=1,
            max_cache_len=max_cache_len,
            device=cache_device,
            dtype=self.model.dtype,
            passage_len=32768
        )
        if use_multi_gpu:
            self.input_device = "cuda:0"  # First GPU for inputs
        else:
            self.input_device = device


    def preprocess_all_documents(self, system_prompt: str, context_rank, all_documents: list[str], reprocess_method: str, revert_rope: bool):
        system_tokens = self.tokenizer.encode(system_prompt, add_special_tokens=True)
        system_tensor = torch.tensor(system_tokens, dtype=torch.long)
        system_len = system_tensor.shape[0]
        hash_key = hashlib.md5(system_tensor.cpu().numpy().tobytes()).hexdigest()
        system_cache_path = f'{self.save_path}/{hash_key}_key.pt'
        ## first generate system cache, this should already be there.
        if not os.path.exists(system_cache_path):
            print(f"Generating system KV cache...")
            input_tensor = system_tensor.unsqueeze(0)
            prefill_and_save_kv_cache(
                model=self.model,
                tokenizer=self.tokenizer,
                past_key_values=self.past_key_values,
                inputs=input_tensor.to(self.input_device),
                save_path=self.save_path,
                chunk_id=0,
                hash_key=hash_key,
                system_len=system_len,
                passage_len=system_len,
                reprocess_method=reprocess_method,
                device=self.input_device,
                device_map=self.device_map
            )

        ## also put in preprocess_save_path
        system_cache_path = f'{self.preprocess_save_path}/{hash_key}_key.pt'
        if not os.path.exists(system_cache_path):
            input_tensor = system_tensor.unsqueeze(0)
            prefill_and_save_kv_cache(
                model=self.model,
                tokenizer=self.tokenizer,
                past_key_values=self.past_key_values,
                inputs=input_tensor.to(self.input_device),
                save_path=self.preprocess_save_path,
                chunk_id=0,
                hash_key=hash_key,
                system_len=system_len,
                passage_len=system_len,
                reprocess_method=reprocess_method,
                device=self.input_device,
                device_map=self.device_map
            )
        time_start = time.time()
        ## process all docs.
        for current_doc_index in range(len(all_documents)):
            print(f"[preprocess_all_documents] takes {time.time()-time_start} seconds")
            time_start = time.time()
            print(f"[preprocess_all_documents] processing doc={current_doc_index}")
            current_doc = all_documents[current_doc_index]
            current_doc_tokens = self.tokenizer.encode(current_doc, add_special_tokens=False)
            current_doc_tensor = torch.tensor(current_doc_tokens, dtype=torch.long)
            current_hash_key = hashlib.md5(current_doc_tensor.cpu().numpy().tobytes()).hexdigest()
            print(f"for doc={current_doc}\n current_hash_key={current_hash_key}")
            if os.path.exists(f'{self.preprocess_save_path}/{current_hash_key}_value.pt') \
                    and os.path.exists(f'{self.preprocess_save_path}/{current_hash_key}_key.pt'):
                print(f"preprocess_all_documents skipping doc {current_doc}.")
                continue

            similar_doc_indeces = context_rank[current_doc_index]
            all_doc_tensors = [system_tensor]
            all_doc_len = [len(system_tensor)]


            # 1. compute all kv.
            for similar_doc_index in similar_doc_indeces:
                if similar_doc_index < 0:
                    continue
                if similar_doc_index == current_doc_index:
                    continue
                similar_doc_text = all_documents[similar_doc_index]
                doc_tokens = self.tokenizer.encode(similar_doc_text, add_special_tokens=False)
                doc_tensor = torch.tensor(doc_tokens, dtype=torch.long)
                all_doc_tensors.append(doc_tensor)
                all_doc_len.append(len(doc_tensor))
                hash_key = hashlib.md5(doc_tensor.cpu().numpy().tobytes()).hexdigest()
                cache_key_path = f'{self.save_path}/{hash_key}_key.pt'
                cache_value_path = f'{self.save_path}/{hash_key}_value.pt'

                if not os.path.exists(cache_key_path):
                    passage_len = doc_tensor.shape[0]
                    ## system_prompt + document_text
                    input_tensor = torch.cat((system_tensor, doc_tensor)).unsqueeze(0)
                    prefill_and_save_kv_cache(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        past_key_values=self.past_key_values,
                        inputs=input_tensor.to(self.input_device),
                        hash_key=hash_key,
                        save_path=self.save_path,
                        chunk_id=1,
                        system_len=system_len,
                        passage_len=passage_len,
                        reprocess_method=reprocess_method,
                        device=self.input_device,
                        device_map=self.device_map
                    )

            # clean all past tokens
            for layer_idx in range(len(self.past_key_values.key_cache)):
                self.past_key_values.past_tokens[layer_idx] = 0
            ## 2. load all kv caches
            for doc_idx, doc_tensor in enumerate(all_doc_tensors):
                hash_key = hashlib.md5(doc_tensor.cpu().numpy().tobytes()).hexdigest()
                cache_key_path = f'{self.save_path}/{hash_key}_key.pt'
                cache_value_path = f'{self.save_path}/{hash_key}_value.pt'
                print(f"cache_key_path = {cache_key_path}")
                chunk_key_cache = torch.load(cache_key_path, weights_only=True)
                chunk_value_cache = torch.load(cache_value_path, weights_only=True)
                past_len = sum(all_doc_len[:doc_idx])
                ## revert rope, skip system prompt and the first doc, I did this anyway.
                if revert_rope and doc_idx > 1:
                    all_position_ids = []
                    position_ids = torch.full((1, chunk_key_cache[0].shape[2]), past_len - system_len,
                                              device=self.input_device)
                    try:
                        cos, sin = self.model.model.layers[0].self_attn.rotary_emb(chunk_key_cache[0], position_ids)
                    except:
                        cos, sin = self.model.model.rotary_emb(chunk_key_cache[0], position_ids)
                    # mistral 限定
                    cos = cos.unsqueeze(1).cpu()
                    sin = sin.unsqueeze(1).cpu()
                    chunk_key_cache = (chunk_key_cache * cos) + (rotate_half(chunk_key_cache) * sin)

                ## load all kv caches
                for layer_idx in range(len(self.past_key_values.key_cache)):
                    self.past_key_values.key_cache[layer_idx].narrow(2, past_len, all_doc_len[doc_idx]).copy_(
                        chunk_key_cache[layer_idx])
                    self.past_key_values.value_cache[layer_idx].narrow(2, past_len, all_doc_len[doc_idx]).copy_(
                        chunk_value_cache[layer_idx])
                    self.past_key_values.past_tokens[layer_idx] += all_doc_len[doc_idx]
                    if layer_idx == 0:
                        print(f"past_tokens += {all_doc_len[doc_idx]}, ={self.past_key_values.past_tokens[layer_idx]}")

            all_doc_tensors.append(current_doc_tensor)
            prefill_with_cache_and_save_preprocess(
                self.model, self.tokenizer, self.past_key_values, all_doc_tensors,
                self.preprocess_save_path, example_id=0, chunk_id=0,
                system_len=system_len, revert_rope=revert_rope,
                reprocess_method=reprocess_method, device=self.input_device, device_map=self.device_map,
                hash_key=current_hash_key
            )


    def load_model(self, model_type, model_path, config, device="cuda:0", use_multi_gpu=False):
        """
        Load model based on model type (same as unified_process_cache.py)

        Args:
            model_type: Type of model ('qwen', 'qwen2', 'qwen3', 'mistral', 'llama', 'pangu')
            model_path: Path to the model
            config: Model configuration
            device: Device to load model on (single GPU)
            use_multi_gpu: If True, use device_map="auto" for multi-GPU

        Returns:
            model: Loaded model
            device_map: Device map if multi-GPU, else None
        """
        load_kwargs = {
            'config': config,
            'torch_dtype': config.torch_dtype
        }

        # Add device_map for multi-GPU
        if use_multi_gpu:
            load_kwargs['device_map'] = 'auto'

        if model_type == 'mistral':
            from .models.modeling_mistral import MistralForCausalLM
            with torch.no_grad():
                model = MistralForCausalLM.from_pretrained(model_path, **load_kwargs)
        elif model_type == 'pangu':
            from .models.modeling_openpangu_dense import PanguEmbeddedForCausalLM
            torch.set_default_dtype(config.torch_dtype)
            with torch.no_grad():
                model = PanguEmbeddedForCausalLM.from_pretrained(model_path, **load_kwargs)
        elif model_type == 'qwen' or model_type == 'qwen2':
            from .models.modeling_qwen2 import Qwen2ForCausalLM
            torch.set_default_dtype(config.torch_dtype)
            with torch.no_grad():
                model = Qwen2ForCausalLM.from_pretrained(model_path, **load_kwargs)
        elif model_type == 'qwen3':
            from .models.modeling_qwen3 import Qwen3ForCausalLM
            torch.set_default_dtype(config.torch_dtype)
            with torch.no_grad():
                model = Qwen3ForCausalLM.from_pretrained(model_path, **load_kwargs)
        elif model_type == 'llama':
            from .models.modeling_llama import LlamaForCausalLM
            torch.set_default_dtype(config.torch_dtype)
            with torch.no_grad():
                model = LlamaForCausalLM.from_pretrained(model_path, **load_kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Get device_map if using multi-GPU
        device_map = None
        if use_multi_gpu:
            device_map = model.hf_device_map
            print(f"\nModel loaded with device_map across GPUs:")
            for name, dev in device_map.items():
                print(f"  {name}: {dev}")
        else:
            model = model.to(device)

        return model, device_map

    def run_one_question(
            self,
            query: str,
            retrieved_docs: list[str],
            system_prompt="",
            model_type='qwen',
            rate=0.2,
            reprocess_method='FusionRAG',
            revert_rope=True,
            preprocess=False,
            max_new_tokens=150,
    ) -> (int, int, int, int, str, list[int]):
        if system_prompt == "":
            system_prompt=DEFAULT_SYSTEM_PROMPT
        system_tokens = self.tokenizer.encode(system_prompt, add_special_tokens=True)
        system_tensor = torch.tensor(system_tokens, dtype=torch.long)
        system_len = system_tensor.shape[0]
        doc_tensors = []
        doc_tensors_total_length = 0
        doc_tensors_len = []
        hash_keys = [hashlib.md5(system_tensor.cpu().numpy().tobytes()).hexdigest()]
        for doc_text in retrieved_docs:
            doc_tokens = self.tokenizer.encode(doc_text, add_special_tokens=False)
            doc_tensor = torch.tensor(doc_tokens, dtype=torch.long)
            doc_tensors_total_length += len(doc_tensor)
            doc_tensors.append(doc_tensor)
            doc_tensors_len.append(len(doc_tensor))
            hash_keys.append(hashlib.md5(doc_tensor.cpu().numpy().tobytes()).hexdigest())
            print(f"for doc={doc_text}\n current_hash_key={hash_keys[-1]}")


        if rate != 1:  # Skip if full recompute
            # Generate system KV cache (chunk_id=0)
            hash_key = hashlib.md5(system_tensor.cpu().numpy().tobytes()).hexdigest()
            system_cache_path = f'{self.save_path}/{hash_key}_key.pt'
            #fixme: mengyao_debug
            if not os.path.exists(system_cache_path):
                print(f"Generating system KV cache...")
                input_tensor = system_tensor.unsqueeze(0)
                prefill_and_save_kv_cache(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    past_key_values=self.past_key_values,
                    inputs=input_tensor.to(self.input_device),
                    save_path=self.save_path,
                    chunk_id=0, ## for the system tensor
                    hash_key=hash_key,
                    system_len=system_len,
                    passage_len=system_len,
                    reprocess_method=reprocess_method,
                    device=self.input_device,
                    device_map=self.device_map
                )

            # Generate KV cache for each document in THIS main question
            for doc_idx, doc_tensor in enumerate(doc_tensors):
                doc_text = retrieved_docs[doc_idx]
                hash_key = hashlib.md5(doc_tensor.cpu().numpy().tobytes()).hexdigest()
                chunk_id = doc_idx + 1
                cache_key_path = f'{self.save_path}/{hash_key}_key.pt'

                if not os.path.exists(cache_key_path):
                    passage_len = doc_tensor.shape[0]
                    ## system_prompt + document_text
                    input_tensor = torch.cat((system_tensor, doc_tensor)).unsqueeze(0)

                    prefill_and_save_kv_cache(
                        model=self.model,
                        tokenizer=self.tokenizer,
                        past_key_values=self.past_key_values,
                        inputs=input_tensor.to(self.input_device),
                        hash_key=hash_key,
                        save_path=self.save_path,
                        chunk_id=chunk_id, ## chunk_id is bigger then 0 is fine.
                        system_len=system_len,
                        passage_len=passage_len,
                        reprocess_method=reprocess_method,
                        device=self.input_device,
                        device_map=self.device_map
                    )
                    print(f"  Generated KV cache for document {chunk_id}/{len(doc_tensors)}")

        if model_type == 'qwen3':
            question_text = f"<|im_end|>\n<|im_start|>user\n/no_think\nQuestion: {query}<|im_end|>\n<|im_start|>assistant\nAnswer: "
        else:
            question_text = f"<|im_end|>\n<|im_start|>user\nQuestion: {query}<|im_end|>\n<|im_start|>assistant\nAnswer: "
        question_tokens = self.tokenizer.encode(question_text, add_special_tokens=False)
        question_tensor = torch.tensor(question_tokens, dtype=torch.long)
        query_len = len(question_tensor)

        iter_tokens = [system_tensor] + doc_tensors + [question_tensor]
        iter_token_len = len(torch.cat(iter_tokens))

        if rate == 1:
            # Full recompute
            inputs = torch.cat(iter_tokens).to(self.input_device).unsqueeze(0)
            from .util.utils import prefill_and_generate
            generated_tokens, _, _ = prefill_and_generate(
                self.model,
                self.tokenizer,
                inputs,
                max_new_tokens=max_new_tokens,
                device=self.input_device,
                device_map=self.device_map
            )
        else:
            # Load preprocessed KV cache and generate
            load_path = self.preprocess_save_path if preprocess else self.save_path
            ## check if all preprocess cache is there.
            for hash_key in hash_keys:
                key_cache_path = f'{load_path}/{hash_key}_key.pt'
                value_cache_path = f'{load_path}/{hash_key}_value.pt'
                if preprocess and (not os.path.exists(key_cache_path) or not os.path.exists(value_cache_path)):
                    load_path = self.save_path
                    break
            generated_tokens, _ = load_kv_and_generate(
                self.model,
                self.tokenizer,
                self.past_key_values,
                passages=iter_tokens,
                hash_keys=hash_keys,
                load_path=load_path,
                max_new_tokens=max_new_tokens,
                revert_rope=revert_rope,
                reprocess_method=reprocess_method,
                rate=rate,
                preprocess=preprocess,
                device=self.input_device,
                device_map=self.device_map
            )

        # Decode answer
        answer = self.tokenizer.decode(torch.tensor(generated_tokens[:-1]), skip_special_tokens=True)
        return system_len, doc_tensors_total_length, query_len, len(generated_tokens), answer, doc_tensors_len

def preprocess_all_docs(file_input: str):
    with open(file_input, "r") as f:
        all_input = json.load(f)
        all_texts = [input["text"] for input in all_input]
        rerank_model = RerankModel(bge_model_path="/data2/qy_tmp/xumengyao/bge-m3")
        similar_idx = rerank_model.preprocess_build_faiss_index(
            all_documents=all_texts,
            topk=10
        )
        rerank_model.clean_()
        fusion_rag_model = FusionRAGModel(
            model_path='/data2/qy_tmp/xumengyao/Qwen3-32B',
            use_multi_gpu=True,
            model_type="qwen3",
            model_name="Qwen3-32B",
            device="cuda:0",
            cache_path='/data2/qy_tmp/xumengyao/fusionrag/',
        )
        fusion_rag_model.preprocess_all_documents(
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            context_rank=similar_idx,
            all_documents=all_texts,
            reprocess_method='FusionRAG',
            revert_rope=True,
        )

def test_question(fusion_rag_model):
    similar_idx = fusion_rag_model.preprocess_build_faiss_index(
        bge_model_path="/data2/qy_tmp/xumengyao/bge-m3",
        all_documents=question_test["gold_docs"],
        topk=10
    )

    fusion_rag_model.preprocess_all_documents(
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        context_rank=similar_idx,
        all_documents=question_test["gold_docs"],
        reprocess_method='FusionRAG',
        revert_rope=True,
    )

    system_len, doc_tensors_total_length, query_len, decode_len, answer, docs_lens = fusion_rag_model.run_one_question(
        query=question_test["question"],
        retrieved_docs=question_test["gold_docs"],
        model_type='qwen3',
        rate=0.3,
        reprocess_method='FusionRAG',
        revert_rope=True,
        preprocess=False,
        max_new_tokens=250,
    )
    print(f"answer={answer}")
    print(f"system_len={system_len}")
    print(f"doc_tensors_total_length={doc_tensors_total_length}")
    print(f"query_len={query_len}")
    print(f"decode_len={decode_len}")

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="1,2"


    preprocess_all_docs(file_input="/home/qy_tmp/xumengyao/all_data/musique_input.json")


