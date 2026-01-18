import copy
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
from ktransformers.models.custom_cache import StaticCache
from ktransformers.util.utils import (
    prefill_and_save_kv_cache,
    load_kv_and_generate,
    prefill_with_cache_and_save_preprocess,
    rotate_half,
    find_group_and_index
)
from ktransformers.util.run_ppr import OnlineEncoder, calculate_vector_set_similarity
import hashlib
import faiss
from FlagEmbedding import FlagModel

DEFAULT_SYSTEM_PROMPT = "<|im_start|>system\nYou are a helpful assistant.\nWrite a high-quality answer for the given question using only the provided search results. The answer process requires reference to the material content and step-by-step thinking."

question_test = {
    "question": """
    Who is the spouse of the Green performer? please output using json format
    please output using json format:
    {{
    "reason": "",
    "sub_query": ""
    }}
    """,
    "gold_docs": [
        "Miquette Giraudy (born 9 February 1953, Nice, France) is a keyboard player and vocalist, best known for her work in Gong and with her partner Steve Hillage. She and Hillage currently form the core of the ambient band System 7. In addition to her performances in music, she has also worked as an actress, film editor and writer. In each role, she has used different stage names.",
        "Green (Steve Hillage album): Green is the fourth studio album by British progressive rock musician Steve Hillage. Written in spring 1977 at the same time as his previous album, the funk-inflected \"Motivation Radio\" (1977), \"Green\" was originally going to be released as \"The Green Album\" as a companion to \"The Red Album\" (the originally intended name for \"Motivation Radio\"). However, this plan was dropped and after a US tour in late 1977, \"Green\" was recorded alone, primarily in Dorking, Surrey, and in London."
    ],
    "answer": "Miquette Giraudy"
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
            draft_model_path: str,
            use_multi_gpu: bool,
            model_type='qwen',
            draft_model_type='qwen',
            device="cuda:0",
            draft_model_device="cuda:0",
            max_cache_len=32768,
            cache_path='/mnt/data3/reflect/',
            model_name='Qwen2.5-7B-Instruct',
            preprocess=False,
            preprocess_method="default",
            file_input="",
            preprocess_model_path="/data2/qy_tmp/xumengyao/bge-m3",
            max_memory=None,
            use_origin_draft_model=False,
            apikey="",
    ):
        print(f"init FusionRAGModel")
        self.model_name=model_name
        self.model_cache_root = os.path.join(cache_path, model_name)
        self.save_path = os.path.join(self.model_cache_root, 'kv_cache')
        self.preprocess_save_path = os.path.join(self.model_cache_root, 'preprocess_kv_cache')
        self.preprocess_empty_prefix_save_path = os.path.join(self.model_cache_root, 'empty_prefix_preprocess_kv_cache')
        self.preprocess_method=preprocess_method
        self.encoder = OnlineEncoder(llm_api_key=apikey)
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(self.preprocess_save_path, exist_ok=True)
        os.makedirs(self.preprocess_empty_prefix_save_path, exist_ok=True)
        self.preprocess=preprocess
        if preprocess and self.preprocess_method == "default":
            print(f"file_input={file_input}")
            dataset_name = os.path.basename(file_input).split(".")[0]
            similar_index_save_path = os.path.join(self.preprocess_save_path, "similar_index")
            self.similar_index_file_path = os.path.join(similar_index_save_path, f"{dataset_name}.npy")
            os.makedirs(similar_index_save_path, exist_ok=True)
            with open(file_input, "r") as f:
                all_input = json.load(f)
                self.all_texts = [input["text"] for input in all_input]
                ## fixme: mengyao_debug locomo quick fix
                if "locomo" in dataset_name :
                    self.all_texts = [f"Document: {text}\n" for text in self.all_texts if not text.startswith("Document:")]
            if os.path.exists(self.similar_index_file_path):
                self.similar_idx = np.load(self.similar_index_file_path)
                print(f"index load from {self.similar_index_file_path}")
            else:
                rerank_model = RerankModel(bge_model_path=preprocess_model_path)
                self.similar_idx = rerank_model.preprocess_build_faiss_index(
                    all_documents=self.all_texts,
                    topk=10
                )
                np.save(self.similar_index_file_path, self.similar_idx)
                rerank_model.clean_()
                del rerank_model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        config._attn_implementation = "sdpa"
        print(f"Loading {model_type} model...")
        if use_multi_gpu:
            print("Using multi-GPU with device_map='auto'")
        self.model, self.device_map = self.load_model(model_type, model_path, config, device, use_multi_gpu, max_memory)
        if draft_model_path != "":
            print(f"Initialize draft model.")
            draft_config = AutoConfig.from_pretrained(draft_model_path, trust_remote_code=True)
            draft_config._attn_implementation = "sdpa"
            self.draft_model, _ = self.load_model(draft_model_type, draft_model_path, draft_config, draft_model_device, use_multi_gpu=False, use_origin_model=use_origin_draft_model)
            self.draft_model.eval()
            self.draft_model_device=draft_model_device
        else:
            print(f"Skipping draft model.")
            self.draft_model = None
            self.draft_model_device = ""

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

    def levenshtein_distance(self, s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]

            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)

                current_row.append(min(insertions, deletions, substitutions))

            previous_row = current_row

        return previous_row[-1]

    def find_closest_by_edit_distance(self, texts: list[str], target: str, return_all_min: bool = False) -> int or list[int]:
        import re
        for idx, text in enumerate(texts):
            if re.sub(r'\s+', '', text) == re.sub(r'\s+', '', target):
                print(f"find_closest_by_edit_distance found {idx}")
                return idx
        if not texts:
            raise ValueError("字符串列表不能为空")

        # for old 2wiki docs, sometimes the title doesn't exist in it.
        for idx, text in enumerate(texts):
            if re.sub(r'\s+', '', target) in re.sub(r'\s+', '', text):
                print(f"find_closest_by_edit_distance within found {idx}")
                return idx

        min_distance = float('inf')
        min_indices = []

        for i, text in enumerate(texts):
            distance = self.levenshtein_distance(text, target)

            if distance < min_distance:
                min_distance = distance
                min_indices = [i]
            elif distance == min_distance:
                min_indices.append(i)

        if return_all_min:
            return min_indices
        else:
            return min_indices[0]

    def clean_kv_cache(self):
        # clean all past tokens
        for layer_idx in range(len(self.past_key_values.key_cache)):
            self.past_key_values.past_tokens[layer_idx] = 0
            self.past_key_values.key_cache[layer_idx].zero_()
            self.past_key_values.value_cache[layer_idx].zero_()

    def prepare_system_kvcache(self, system_prompt: str, reprocess_method: str):
        system_tokens = self.tokenizer.encode(system_prompt, add_special_tokens=True)
        system_tensor = torch.tensor(system_tokens, dtype=torch.long)
        system_len = system_tensor.shape[0]
        hash_key = hashlib.md5(system_tensor.cpu().numpy().tobytes()).hexdigest()
        system_cache_paths = [self.save_path, self.preprocess_save_path, self.preprocess_empty_prefix_save_path]
        ## first generate system cache, this should already be there.
        for system_cache_path in system_cache_paths:
            if not os.path.exists(f'{system_cache_path}/{hash_key}_key.pt'):
                print(f"Generating system KV cache...")
                input_tensor = system_tensor.unsqueeze(0)
                prefill_and_save_kv_cache(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    past_key_values=self.past_key_values,
                    inputs=input_tensor.to(self.input_device),
                    save_path=system_cache_path,
                    chunk_id=0,
                    hash_key=hash_key,
                    system_len=system_len,
                    passage_len=system_len,
                    reprocess_method=reprocess_method,
                    device=self.input_device,
                    device_map=self.device_map
                )

        return system_len, system_tensor

    def preprocess_one_document_with_empty(self, system_prompt: str, document: str, all_document: list[str], reprocess_method: str, revert_rope: bool):
        system_len, system_tensor = self.prepare_system_kvcache(system_prompt=system_prompt, reprocess_method=reprocess_method)
        document_index = all_document.index(document)
        prefill_len = 0
        system_tokens = self.tokenizer.encode(system_prompt, add_special_tokens=True)
        irrelevant_tokens = self.tokenizer.encode(" ."*10000, add_special_tokens=True)
        system_tensor = torch.tensor(system_tokens, dtype=torch.long)
        system_len = system_tensor.shape[0]
        for i in range(document_index):
            tokens = self.tokenizer.encode(all_document[i], add_special_tokens=True)
            prefill_len += len(tokens)
        doc_tokens = self.tokenizer.encode(document, add_special_tokens=False)
        doc_tensor = torch.tensor(doc_tokens, dtype=torch.long)
        prefill_space_tensor = torch.tensor(irrelevant_tokens[:prefill_len], dtype=torch.long)
        input_tensor = torch.cat((system_tensor, prefill_space_tensor, doc_tensor)).unsqueeze(0)
        hash_key = hashlib.md5(doc_tensor.cpu().numpy().tobytes()).hexdigest()
        empty_prefix_cache_path = f'{self.preprocess_empty_prefix_save_path}/{hash_key}_key.pt'
        prefill_space_len = prefill_space_tensor.shape[0]
        passage_len = doc_tensor.shape[0]
        ## this has to be computed everytime.
        if True or not os.path.exists(empty_prefix_cache_path):
            print(f"[preprocess_one_document_with_empty] generate for hash={hash_key}")
            prefill_and_save_kv_cache(
                model=self.model,
                tokenizer=self.tokenizer,
                past_key_values=self.past_key_values,
                inputs=input_tensor.to(self.input_device),
                hash_key=hash_key,
                save_path=self.preprocess_empty_prefix_save_path,
                chunk_id=1,
                system_len=system_len + prefill_space_len,
                passage_len=passage_len,
                reprocess_method=reprocess_method,
                device=self.input_device,
                device_map=self.device_map
            )
            self.clean_kv_cache()


    def preprocess_one_document(self, system_prompt: str, document: str, reprocess_method: str, revert_rope: bool):
        system_len, system_tensor = self.prepare_system_kvcache(system_prompt=system_prompt, reprocess_method=reprocess_method)
        time_start = time.time()
        current_doc = document
        try:
            current_doc_index = self.all_texts.index(current_doc)
        except ValueError:
            print(f"字符串不存在, try to find replacement")
            current_doc_index = self.find_closest_by_edit_distance(texts=self.all_texts, target=current_doc, return_all_min=False)
            print(f"字符串不存在, found replacement.")

        current_doc_tokens = self.tokenizer.encode(current_doc, add_special_tokens=False)
        current_doc_tensor = torch.tensor(current_doc_tokens, dtype=torch.long)
        current_hash_key = hashlib.md5(current_doc_tensor.cpu().numpy().tobytes()).hexdigest()
        # print(f"for doc={current_doc}\n current_hash_key={current_hash_key}")
        if os.path.exists(f'{self.preprocess_save_path}/{current_hash_key}_value.pt') \
                and os.path.exists(f'{self.preprocess_save_path}/{current_hash_key}_key.pt'):
            # print(f"preprocess_all_documents skipping doc {current_doc}.")
            return

        similar_doc_indeces = self.similar_idx[current_doc_index]
        all_doc_tensors = [system_tensor]
        all_doc_len = [len(system_tensor)]


        # 1. compute all kv.
        for similar_doc_index in similar_doc_indeces:
            if similar_doc_index < 0:
                continue
            if similar_doc_index == current_doc_index:
                continue
            similar_doc_text = self.all_texts[similar_doc_index]
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

            self.clean_kv_cache()
        ## 2. load all kv caches
        for doc_idx, doc_tensor in enumerate(all_doc_tensors):
            hash_key = hashlib.md5(doc_tensor.cpu().numpy().tobytes()).hexdigest()
            cache_key_path = f'{self.save_path}/{hash_key}_key.pt'
            cache_value_path = f'{self.save_path}/{hash_key}_value.pt'
            # print(f"cache_key_path = {cache_key_path}")
            chunk_key_cache = torch.load(cache_key_path, weights_only=True)
            chunk_value_cache = torch.load(cache_value_path, weights_only=True)
            past_len = sum(all_doc_len[:doc_idx])

            ## load all kv caches
            for layer_idx in range(len(self.past_key_values.key_cache)):
                self.past_key_values.key_cache[layer_idx].narrow(2, past_len, all_doc_len[doc_idx]).copy_(
                    chunk_key_cache[layer_idx])
                self.past_key_values.value_cache[layer_idx].narrow(2, past_len, all_doc_len[doc_idx]).copy_(
                    chunk_value_cache[layer_idx])
                self.past_key_values.past_tokens[layer_idx] += all_doc_len[doc_idx]
                # if layer_idx == 0:
                    # print(f"past_tokens += {all_doc_len[doc_idx]}, ={self.past_key_values.past_tokens[layer_idx]}")

        all_doc_tensors.append(current_doc_tensor)
        prefill_with_cache_and_save_preprocess(
            self.model, self.tokenizer, self.past_key_values, all_doc_tensors,
            self.preprocess_save_path, example_id=0, chunk_id=0,
            system_len=system_len, revert_rope=revert_rope,
            reprocess_method=reprocess_method, device=self.input_device, device_map=self.device_map,
            hash_key=current_hash_key
        )
        self.clean_kv_cache()
        print(f"[preprocess_all_documents] takes {time.time()-time_start} seconds")


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
            # print(f"for doc={current_doc}\n current_hash_key={current_hash_key}")
            if os.path.exists(f'{self.preprocess_save_path}/{current_hash_key}_value.pt') \
                    and os.path.exists(f'{self.preprocess_save_path}/{current_hash_key}_key.pt'):
                # print(f"preprocess_all_documents skipping doc {current_doc}.")
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
                # print(f"cache_key_path = {cache_key_path}")
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


    def load_model(self, model_type, model_path, config, device="cuda:0", use_multi_gpu=False, max_memory=None, use_origin_model=False):
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
        if max_memory is not None:
            print(f"using max_memory={max_memory}")
            load_kwargs["max_memory"] = max_memory

        # Add device_map for multi-GPU
        if use_multi_gpu:
            load_kwargs['device_map'] = 'auto'

        if model_type == 'mistral':
            from ktransformers.models.modeling_mistral import MistralForCausalLM
            with torch.no_grad():
                model = MistralForCausalLM.from_pretrained(model_path, **load_kwargs)
        elif model_type == 'pangu':
            from ktransformers.models.modeling_openpangu_dense import PanguEmbeddedForCausalLM
            torch.set_default_dtype(config.torch_dtype)
            with torch.no_grad():
                model = PanguEmbeddedForCausalLM.from_pretrained(model_path, **load_kwargs)
        elif model_type == 'qwen' or model_type == 'qwen2':
            from ktransformers.models.modeling_qwen2 import Qwen2ForCausalLM
            torch.set_default_dtype(config.torch_dtype)
            with torch.no_grad():
                model = Qwen2ForCausalLM.from_pretrained(model_path, **load_kwargs)
        elif model_type == 'qwen3':
            from ktransformers.models.modeling_qwen3 import Qwen3ForCausalLM
            torch.set_default_dtype(config.torch_dtype)
            with torch.no_grad():
                model = Qwen3ForCausalLM.from_pretrained(model_path, **load_kwargs)
        elif model_type == 'llama':
            from ktransformers.models.modeling_llama import LlamaForCausalLM
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
            max_new_tokens=150,
            use_entropy_selection=False,
            entropy_top_k=4,
            question_prefix=""
    ) -> (int, int, int, int, str, list[int]):

        embeddings = self.encoder.encode(text=retrieved_docs, normalize_embeddings=True)
        eigenvalue = {}
        print(f"recomputing using recomputation_rate={rate}, doc_len={len(retrieved_docs)}, reprocess_method={reprocess_method}")
        if system_prompt == "":
            system_prompt=DEFAULT_SYSTEM_PROMPT
        empty_token = self.tokenizer.encode(" ", add_special_tokens=True)
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
            # print(f"for doc={doc_text}\n current_hash_key={hash_keys[-1]}")


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
                self.clean_kv_cache()

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
                    self.clean_kv_cache()
                    print(f"  Generated KV cache for document {chunk_id}/{len(doc_tensor)}")
            if self.preprocess:
                if self.preprocess_method == "default":
                    for doc_text in retrieved_docs:
                        self.preprocess_one_document(
                            system_prompt=DEFAULT_SYSTEM_PROMPT,
                            document=doc_text,
                            reprocess_method=reprocess_method,
                            revert_rope=revert_rope,
                        )
                elif self.preprocess_method == "space":
                    for doc_text in retrieved_docs:
                        self.preprocess_one_document_with_empty(
                            system_prompt=DEFAULT_SYSTEM_PROMPT,
                            document=doc_text,
                            all_document=retrieved_docs,
                            reprocess_method=reprocess_method,
                            revert_rope=revert_rope,
                        )
                else:
                    print(f"no method={self.preprocess_method}")
                    exit(1)


        if model_type == 'qwen3':
            question_text = f"<|im_end|>\n<|im_start|>user\n\nQuestion: /no_think {query}<|im_end|>\n<|im_start|>assistant\nAnswer: "
        else:
            question_text = f"<|im_end|>\n<|im_start|>user\nQuestion: {query}<|im_end|>\n<|im_start|>assistant\nAnswer: "
        question_tokens = self.tokenizer.encode(question_text, add_special_tokens=False)
        question_prefix_tokens = self.tokenizer.encode(question_prefix, add_special_tokens=False)
        question_with_prefix_tokens = self.tokenizer.encode(question_prefix + question_text, add_special_tokens=False)
        question_tensor = torch.tensor(question_tokens, dtype=torch.long)
        question_prefix_tensor = torch.tensor(question_prefix_tokens, dtype=torch.long)
        query_len = len(question_tensor)

        if reprocess_method == "DraftModel_smarter":
            iter_tokens = [system_tensor] + doc_tensors + [question_tensor]
        else:
            iter_tokens = [system_tensor] + doc_tensors + [question_prefix_tensor, question_tensor]
        iter_token_len = len(torch.cat(iter_tokens))
        if rate == 1:
            print(f"full recompute")
            # Full recompute
            inputs = torch.cat(iter_tokens).to(self.input_device).unsqueeze(0)
            from ktransformers.util.utils import prefill_and_generate
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
            if self.preprocess:
                if self.preprocess_method == "default":
                    load_path = self.preprocess_save_path
                elif self.preprocess_method == "space":
                    load_path = self.preprocess_empty_prefix_save_path
                else:
                    print(f"no method={self.preprocess_method}")
                    exit(1)
            else:
                load_path = self.save_path
            ## check if all preprocess cache is there.
            for doc_index, hash_key in enumerate(hash_keys):
                key_cache_path = f'{load_path}/{hash_key}_key.pt'
                value_cache_path = f'{load_path}/{hash_key}_value.pt'
                if self.preprocess and (not os.path.exists(key_cache_path) or not os.path.exists(value_cache_path)):
                    if doc_index > 0:
                        print(f"retrieved_docs {retrieved_docs[doc_index]} not preprocessed before. 字符串不存在")
                    load_path = self.save_path
                    break
            print(f"load_path={load_path}")
            generated_tokens, _, eigenvalue_ = load_kv_and_generate(
                self.model,
                self.tokenizer,
                self.past_key_values,
                passages=iter_tokens,
                hash_keys=hash_keys,
                load_path=load_path,
                max_new_tokens=max_new_tokens,
                revert_rope=revert_rope,
                reprocess_method=reprocess_method,
                use_entropy_selection=use_entropy_selection,
                rate=rate,
                entropy_top_k=entropy_top_k,
                draft_model=self.draft_model,
                preprocess=self.preprocess,
                device=self.input_device,
                device_map=self.device_map,
                draft_model_device=self.draft_model_device,
                prefix_cache_path=self.save_path,
                query=query,
                embeddings=embeddings,
                question_prefix_tensor=question_prefix_tensor,
            )
            eigenvalue.update(eigenvalue_)

        # Decode answer
        answer = self.tokenizer.decode(torch.tensor(generated_tokens[:-1]), skip_special_tokens=True)
        return system_len, doc_tensors_total_length, query_len, len(generated_tokens), answer, doc_tensors_len, eigenvalue

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

    system_len, doc_tensors_total_length, query_len, decode_len, answer, docs_lens = fusion_rag_model.run_one_question(
        query=question_test["question"],
        retrieved_docs=question_test["gold_docs"],
        model_type='qwen3',
        rate=0,
        reprocess_method='DraftModel',
        revert_rope=True,
        max_new_tokens=250,
    )
    print(f"answer={answer}")
    print(f"system_len={system_len}")
    print(f"doc_tensors_total_length={doc_tensors_total_length}")
    print(f"query_len={query_len}")
    print(f"decode_len={decode_len}")

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"
    print(f"start testing run_question")


    # preprocess_all_docs(file_input="/home/qy_tmp/xumengyao/all_data/musique_input.json")


    fusion_rag_model = FusionRAGModel(
        model_path='/data2/qy_tmp/xumengyao/Qwen3-32B',
        use_multi_gpu=True,
        model_type="qwen3",
        model_name="Qwen3-32B",
        device="cuda:0",
        cache_path='/data2/qy_tmp/xumengyao/fusionrag/',
        draft_model_device="cuda:0",
        draft_model_path='/data2/qy_tmp/xumengyao/Qwen2.5-3B-Instruct',
        draft_model_type="qwen",
        preprocess=True,
        file_input="/home/qy_tmp/xumengyao/all_data/musique_input.json",
        preprocess_model_path="/data2/qy_tmp/xumengyao/bge-m3",
        preprocess_method="space"
    )
    test_question(fusion_rag_model)


