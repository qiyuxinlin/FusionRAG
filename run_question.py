import copy
import json
import os
import sys
import csv
import shutil
import time
import torch
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Tuple
from openai import OpenAI
from transformers import AutoTokenizer, AutoConfig
from ktransformers.models.custom_cache import StaticCache
from ktransformers.util.utils import (
    prefill_and_save_kv_cache,
    load_kv_and_generate,
    prefill_with_cache_and_save_preprocess,
    rotate_half,
    find_group_and_index,
    find_all_substr_needs_recompute
)
from ktransformers.util.run_ppr import OnlineEncoder, calculate_vector_set_similarity
import hashlib
import faiss
from FlagEmbedding import FlagModel

DEFAULT_SYSTEM_PROMPT = "<|im_start|>system\nYou are a helpful assistant.\nWrite a brief and high-quality answer for the given question using only the provided search results.\n"

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
            use_multi_gpu=True,
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
            use_local_draft_model=True,
            draft_model_url="",
            apikey="",
    ):
        print(f"init FusionRAGModel")
        self.model_name=model_name
        self.model_type = model_type
        self.draft_model_type=draft_model_type
        self.model_cache_root = os.path.join(cache_path, model_name)
        self.save_path = os.path.join(self.model_cache_root, 'kv_cache')
        self.preprocess_save_path = os.path.join(self.model_cache_root, 'preprocess_kv_cache')
        self.preprocess_empty_prefix_save_path = os.path.join(self.model_cache_root, 'empty_prefix_preprocess_kv_cache')
        self.preprocess_method=preprocess_method
        self.encoder = OnlineEncoder(llm_api_key=apikey)
        self.file_input = file_input
        if cache_path != "":
            os.makedirs(self.save_path, exist_ok=True)
            os.makedirs(self.preprocess_save_path, exist_ok=True)
            os.makedirs(self.preprocess_empty_prefix_save_path, exist_ok=True)
        self.preprocess=preprocess
        print(f"file_input={file_input}")
        dataset_name = os.path.basename(file_input).split(".")[0]
        self.dataset_name = dataset_name
        if preprocess and self.preprocess_method == "default":
            similar_index_save_path = os.path.join(self.model_cache_root, "similar_index")
            self.similar_index_file_path = os.path.join(similar_index_save_path, f"{dataset_name}.npy")
            print(f"self.similar_index_file_path = {self.similar_index_file_path}")
            os.makedirs(similar_index_save_path, exist_ok=True)
            with open(file_input, "r") as f:
                all_input = json.load(f)
                self.all_texts = [input["text"] for input in all_input]
                ## fixme: mengyao_debug locomo quick fix
                if "locomo" in dataset_name :
                    self.all_texts = [f" {text}\n" if not text.startswith(" ") else text for text in self.all_texts]
                ##add \n in the end.
                self.all_texts = [f"{text}\n" if not text.endswith("\n") else text for text in self.all_texts]
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
        if model_path != "":
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            config._attn_implementation = "sdpa"
            print(f"Loading {model_type} model...")
            if use_multi_gpu:
                print("Using multi-GPU with device_map='auto'")
            self.model, self.device_map = self.load_model(model_type, model_path, config, device, use_multi_gpu, max_memory)
        if draft_model_path != "":
            self.use_local_draft_model = use_local_draft_model
            if use_local_draft_model:
                print(f"Initialize draft model from {draft_model_path}...")
                draft_config = AutoConfig.from_pretrained(draft_model_path, trust_remote_code=True)
                print(f"draft_config={draft_config}")
                draft_config._attn_implementation = "sdpa"
                self.draft_model, _ = self.load_model(draft_model_type, draft_model_path, draft_config, draft_model_device, use_multi_gpu=False)
                self.draft_model.eval()
                self.draft_model_device=draft_model_device
                self.draft_model_url = ""
            else:
                print(f"using remote draftmodel.")
                assert draft_model_url != "" "draft_model_url must not be empty."
                self.draft_model_url = draft_model_url
                self.draft_model = None
                self.draft_model_device = None
            self.draft_model_tokenizer = AutoTokenizer.from_pretrained(draft_model_path, trust_remote_code=True)
        else:
            print(f"Skipping draft model.")
            self.draft_model = None
            self.draft_model_device = ""

        if model_path != "":
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
                return idx
        if not texts:
            raise ValueError("字符串列表不能为空")

        # for old 2wiki docs, sometimes the title doesn't exist in it.
        for idx, text in enumerate(texts):
            if re.sub(r'\s+', '', target) in re.sub(r'\s+', '', text):
                return idx

        print(f"find_closest_by_edit_distance start the old method.")

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
            current_doc_index = self.find_closest_by_edit_distance(texts=self.all_texts, target=current_doc, return_all_min=False)

        current_doc_tokens = self.tokenizer.encode(current_doc, add_special_tokens=False)
        current_doc_tensor = torch.tensor(current_doc_tokens, dtype=torch.long)
        current_hash_key = hashlib.md5(current_doc_tensor.cpu().numpy().tobytes()).hexdigest()
        print(f"for doc={current_doc}\n current_hash_key={current_hash_key}")
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

            if not os.path.exists(cache_key_path) or not os.path.exists(cache_value_path):
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
            print(f"cache_key_path = {cache_key_path}")
            chunk_key_cache = torch.load(cache_key_path, weights_only=True)
            chunk_value_cache = torch.load(cache_value_path, weights_only=True)
            past_len = sum(all_doc_len[:doc_idx])

            # import re
            # dir_name = copy.deepcopy(current_doc[:60])
            # dir_name = re.sub(r'[^a-zA-Z]', '', dir_name)
            # save_path = f"/mnt/data/xmy/mengyao_debug/torch/{dir_name}"
            # os.makedirs(save_path, exist_ok=True)
            # key_cache_path = f"{save_path}/key_cache_{doc_idx}.pt"
            # value_cache_path = f"{save_path}/value_cache_{doc_idx}.pt"
            # torch.save(chunk_key_cache, key_cache_path)
            # torch.save(chunk_value_cache, value_cache_path)

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
                    # if layer_idx == 0:
                    #     print(f"past_tokens += {all_doc_len[doc_idx]}, ={self.past_key_values.past_tokens[layer_idx]}")

            all_doc_tensors.append(current_doc_tensor)
            prefill_with_cache_and_save_preprocess(
                self.model, self.tokenizer, self.past_key_values, all_doc_tensors,
                self.preprocess_save_path, example_id=0, chunk_id=0,
                system_len=system_len, revert_rope=revert_rope,
                reprocess_method=reprocess_method, device=self.input_device, device_map=self.device_map,
                hash_key=current_hash_key
            )


    def load_model(self, model_type, model_path, config, device="cuda:0", use_multi_gpu=False, max_memory=None):
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

    def find_all_must_recompute(self, retrieved_docs: list[str], tokenizer) -> List[int]:
        must_choose = []
        for retrieved_doc in retrieved_docs:
            retrieved_doc_prefix = retrieved_doc[:retrieved_doc.index(".")+1]
            keyword_tokens = tokenizer.encode(retrieved_doc_prefix, add_special_tokens=False)
            must_choose.append(len(keyword_tokens))
        return must_choose

    def find_all_must_recompute_indices_highlight_time(self, system_prompt: str, retrieved_docs: list[str], tokenizer) -> List[int]:
        must_choose_token_indices = []
        for idx, retrieved_doc in enumerate(retrieved_docs):
            retrieved_doc_prefix = retrieved_doc[:retrieved_doc.index(".")+1]
            previous_str = system_prompt + "".join(retrieved_docs[:idx])
            previous_str_and_recompute = system_prompt + "".join(retrieved_docs[:idx]) + retrieved_doc_prefix
            previous_token = tokenizer.encode(previous_str, add_special_tokens=False)
            previous_token_and_recompute = tokenizer.encode(previous_str_and_recompute, add_special_tokens=False)
            must_choose_token_indices.extend([i for i in range(len(previous_token), len(previous_token_and_recompute))])
        return must_choose_token_indices

    def find_all_must_recompute_indices_prefix(self, system_prompt: str, retrieved_docs: list[str], tokenizer) -> List[int]:
        must_choose_token_indices = []
        for idx, retrieved_doc in enumerate(retrieved_docs):
            previous_str = system_prompt + "".join(retrieved_docs[:idx])
            previous_token = tokenizer.encode(previous_str, add_special_tokens=False)
            must_choose_token_indices.extend([i for i in range(len(previous_token), len(previous_token) + 10)])
        return must_choose_token_indices

    def sort_docs(self, retrieved_docs: list[str]):
        """
        对对话文本进行排序，格式: "Conversation Time: 2:32 pm on 29 January, 2023. ..."
        提取时间部分进行排序，如果有不符合格式的文档直接返回原列表
        """
        if not retrieved_docs:
            return retrieved_docs

        # 检查所有文档是否都符合对话时间格式
        def is_conversation_format(doc: str) -> bool:
            if not doc.startswith("Conversation Time: "):
                return False

            # 检查是否有时间部分和句点
            time_part_end = doc.find('.', len("Conversation Time: "))
            if time_part_end == -1:
                return False

            # 提取时间字符串
            time_str = doc[len("Conversation Time: "):time_part_end].strip()

            # 尝试解析时间
            try:
                time_part, date_part = time_str.split(" on ")
                datetime.strptime(time_part, "%I:%M %p")
                datetime.strptime(date_part, "%d %B, %Y")
                return True
            except (ValueError, AttributeError, IndexError):
                return False

        # 如果有任何一个文档不符合格式，直接返回
        if not all(is_conversation_format(doc) for doc in retrieved_docs):
            print(f"fail to sort doc!")
            return retrieved_docs

        # 从文档中提取时间并解析为datetime对象
        def parse_conversation_time(doc: str) -> datetime:
            # 找到第一个句点的位置
            dot_index = doc.find('.', len("Conversation Time: "))
            # 提取时间字符串
            time_str = doc[len("Conversation Time: "):dot_index].strip()

            # 解析时间
            time_part, date_part = time_str.split(" on ")
            time_obj = datetime.strptime(time_part, "%I:%M %p")
            date_obj = datetime.strptime(date_part, "%d %B, %Y")

            return datetime.combine(date_obj.date(), time_obj.time())

        # 排序并返回
        return sorted(retrieved_docs, key=parse_conversation_time)

    def draft_one_question(self,
                           system_prompt: str,
                           passages: list[str],
                           query: str,
                           rate: float,
                           keyword: str="",
                           reverse_attn=False,
                           ):
        must_choose_token_indices = []
        if "sort" in keyword:
            passages = self.sort_docs(passages)
            print(f"sorting passages!")
        if "highlight_time" in keyword:
            must_choose_token_indices = self.find_all_must_recompute_indices_highlight_time(
                retrieved_docs=passages,
                system_prompt=system_prompt,
                tokenizer=self.draft_model_tokenizer
            )
        elif "highlight_prefix" in keyword:
            must_choose_token_indices = self.find_all_must_recompute_indices_prefix(
                retrieved_docs=passages,
                system_prompt=system_prompt,
                tokenizer=self.draft_model_tokenizer
            )
        recompute_tokens, recompute_tokens_list = find_all_substr_needs_recompute(
            draft_model=self.draft_model,
            draft_model_device=self.draft_model_device,
            tokenizer=self.draft_model_tokenizer,
            system_prompt=system_prompt,
            passages=passages,
            query=query,
            rate=rate,
            must_choose_token_indices=must_choose_token_indices,
            reverse_attn=reverse_attn,
            use_local_draft_model=self.use_local_draft_model,
            draft_model_url=self.draft_model_url
        )
        return recompute_tokens, recompute_tokens_list, passages

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
            question_prefix="",
            keyword=""
    ) -> (int, int, int, int, str, list[int]):
        must_choose = None
        ## fixme: mengyao_debug locomo quick fix
        if "locomo" in self.dataset_name:
            if "sort" in keyword:
                retrieved_docs = self.sort_docs(retrieved_docs)
            retrieved_docs = [f" {text}\n" for text in retrieved_docs if not text.startswith(" ")]
            if "highlight_time" in keyword:
                must_choose = self.find_all_must_recompute(retrieved_docs=retrieved_docs, tokenizer=self.draft_model_tokenizer)
        # print(f"run_one_question query={query}\n retrieved_docs={retrieved_docs}")
        sim = 0
        try:
            if len(retrieved_docs) > 0:
                embeddings = self.encoder.encode(text=retrieved_docs, normalize_embeddings=True)
                sim = calculate_vector_set_similarity(embeddings)
            eigenvalue = {
                "similarity": float(sim)
            }
        except Exception as e:
            embeddings = None
            eigenvalue = {
                "similarity": -1
            }
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


        if self.model_type == 'qwen3':
            question_text = f"<|im_end|>\n<|im_start|>user\n\nQuestion: /no_think {query}<|im_end|>\n<|im_start|>assistant\nAnswer: "
        else:
            question_text = f"<|im_end|>\n<|im_start|>user\n\nQuestion: /no_think {query}<|im_end|>\n<|im_start|>assistant\nAnswer: "
            # question_text = f"<|im_end|>\n<|im_start|>user\nQuestion: {query}<|im_end|>\n<|im_start|>assistant\n"
        question_tokens = self.tokenizer.encode(question_text, add_special_tokens=False)
        question_tokens_ = self.tokenizer.encode(question_text, add_special_tokens=True)
        question_prefix_tokens = self.tokenizer.encode(question_prefix, add_special_tokens=False)
        question_with_prefix_tokens = self.tokenizer.encode(question_prefix + question_text, add_special_tokens=False)
        question_with_prefix_tensor = torch.tensor(question_with_prefix_tokens, dtype=torch.long)
        question_tensor = torch.tensor(question_tokens, dtype=torch.long)
        question_prefix_tensor = torch.tensor(question_prefix_tokens, dtype=torch.long)
        query_len = len(question_tensor)

        iter_tokens = [system_tensor] + doc_tensors + [question_tensor]
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
            print(f"full prompt = {system_prompt}{''.join(retrieved_docs)}{question_text}")
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
                similarity=sim,
                must_choose=must_choose
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

import os
import requests
from typing import List, Dict, Any, Optional

def rerank(
    query: str,
    documents: List[str],
    model: str = "qwen3-rerank",
    top_n: Optional[int] = None,
    instruct: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    调用阿里云 DashScope 的 rerank 接口，根据查询对文档进行相关性排序。

    Args:
        query: 查询文本
        documents: 待排序的文档列表
        model: 使用的模型名称，默认为 "qwen3-rerank"
        top_n: 返回的 top N 个最相关文档，不指定则返回所有
        instruct: 任务指令，用于指导模型如何排序（例如：Given a web search query, retrieve relevant passages...）
        api_key: DashScope API Key，如果不提供则从环境变量 DASHSCOPE_API_KEY 读取

    Returns:
        API 响应的 JSON 数据（字典格式）

    Raises:
        ValueError: 当 API Key 未提供且环境变量中不存在时
        requests.RequestException: 当请求失败时
    """
    if api_key is None:
        api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("API Key must be provided or set in environment variable DASHSCOPE_API_KEY")

    url = "https://dashscope.aliyuncs.com/compatible-api/v1/reranks"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload: Dict[str, Any] = {
        "model": model,
        "query": query,
        "documents": documents,
    }
    if top_n is not None:
        payload["top_n"] = top_n
    if instruct is not None:
        payload["instruct"] = instruct

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()  # 如果状态码不是 2xx 则抛出异常
        return response.json()
    except requests.exceptions.RequestException as e:
        # 可以在这里添加更详细的错误处理，例如打印响应内容
        raise requests.RequestException(f"Rerank API request failed: {e}") from e


import matplotlib.pyplot as plt
import seaborn as sns

def plot_distribution_comparison(list1, list2,
                                 labels=('List 1', 'List 2'),
                                 colors=('blue', 'orange'),
                                 plot_type='hist',
                                 bins=30,
                                 alpha=0.5,
                                 density=True,
                                 figsize=(10, 5),
                                 title=None,
                                 save_path=None):
    # 检查输入
    if not isinstance(list1, (list, np.ndarray)) or not isinstance(list2, (list, np.ndarray)):
        raise TypeError("list1 and list2 must be list-like objects")

    # 转换为 numpy 数组以便处理
    arr1 = np.asarray(list1)
    arr2 = np.asarray(list2)

    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)

    if plot_type == 'hist':
        # 叠加直方图
        ax.hist(arr1, bins=bins, alpha=alpha, label=labels[0],
                color=colors[0], density=density)
        ax.hist(arr2, bins=bins, alpha=alpha, label=labels[1],
                color=colors[1], density=density)
        ax.set_ylabel('Density' if density else 'Frequency')
        if title is None:
            title = 'Overlaid Histogram Comparison'

    elif plot_type == 'kde':
        # 核密度估计图
        sns.kdeplot(arr1, label=labels[0], color=colors[0],
                    shade=True, ax=ax)
        sns.kdeplot(arr2, label=labels[1], color=colors[1],
                    shade=True, ax=ax)
        ax.set_ylabel('Density')
        if title is None:
            title = 'Kernel Density Estimation (KDE) Comparison'

    elif plot_type == 'box':
        # 并排箱线图
        bp = ax.boxplot([arr1, arr2], labels=labels, patch_artist=True)
        # 为箱体着色
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        ax.set_ylabel('Value')
        if title is None:
            title = 'Boxplot Comparison'

    else:
        raise ValueError("plot_type must be 'hist', 'kde' or 'box'")

    ax.set_xlabel('Value')
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.show()

    # 可选保存
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, ax

def check_text_distribution():

    def within_list(retrieve_doc: str, gold_docs: list[str]):
        for gold_doc in gold_docs:
            if gold_doc in retrieve_doc:
                return True
        return False

    # with open("../../../data/example_data/musique_pages/result_/result_fusion_rag_sglang_recompute_draft_qwen_None_recompute_rate_0.3_kimi-k2.5_musique.json") as f:
    relevant_scores = []
    irrelevant_scores = []

    result_files = [
        "../../../data/example_data/locomo_pages/result_/result_fusion_rag_sglang_recompute_draft_qwen_highlight_time_None_recompute_rate_0.3_Kimi-K2.5_locomo_category_1_no_detail.json",
        "../../../data/example_data/locomo_pages/result_/result_fusion_rag_sglang_recompute_draft_qwen_highlight_time_None_recompute_rate_0.3_Kimi-K2.5_locomo_category_2_no_detail.json",
        "../../../data/example_data/locomo_pages/result_/result_fusion_rag_sglang_recompute_draft_qwen_highlight_time_None_recompute_rate_0.3_Kimi-K2.5_locomo_category_3_no_detail.json",
        "../../../data/example_data/locomo_pages/result_/result_fusion_rag_sglang_recompute_draft_qwen_highlight_time_None_recompute_rate_0.3_Kimi-K2.5_locomo_category_4_no_detail.json",
        "../../../data/example_data/musique_pages/result_/result_fusion_rag_sglang_recompute_draft_qwen_None_recompute_rate_1.0_Kimi-K2.5_musique_no_detail.json",
        "../../../data/example_data/2wiki_pages/result_/result_fusion_rag_sglang_recompute_draft_qwen_None_recompute_rate_1.0_Kimi-K2.5_2wiki_no_detail.json"
    ]

    for result_file in result_files:
        with open(result_file) as f:
            result_json = json.load(f)
            title = result_file.split("Kimi-K2.5_")[1]
            for question in result_json[:50]:
                query = question["question"]
                gold_docs = question["gold_docs"]
                gold_docs = [doc.split("】")[1] for doc in gold_docs]
                for ic in question["intermediate_context"]:
                    # query = ic['query']
                    retrieve_docs_ = ic['retrieve docs']
                    retrieve_docs = [doc for doc in retrieve_docs_ if within_list(doc, gold_docs)]
                    irrelevant_retrieve_docs = [doc for doc in retrieve_docs_ if not within_list(doc, gold_docs)]

                    if len(retrieve_docs) > 0:
                        result = rerank(
                            query=query,
                            documents=retrieve_docs,
                            api_key=api_key
                        )
                        relevant_scores.extend([result["results"][result_id]['relevance_score'] for result_id in range(len(result["results"]))])

                    if len(irrelevant_retrieve_docs) > 0:
                        result = rerank(
                            query=query,
                            documents=irrelevant_retrieve_docs,
                            api_key=api_key
                        )
                        irrelevant_scores.extend(
                            [result["results"][result_id]['relevance_score'] for result_id in range(len(result["results"]))])

            print(relevant_scores)
            print(irrelevant_scores)
            plot_distribution_comparison(relevant_scores, irrelevant_scores, title=title, labels=("relevant", "irrelevant"))


def check_code_distribution(path: str):
    from pathlib import Path
    folder = Path(path)
    json_files = list(folder.glob('*.json'))
    relevant_scores = []
    irrelevant_scores = []

    for json_file in json_files:
        if "tmp.json" in str(json_file):
            continue
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            try:
                data["messages"] = data["messages"][2:]
                for idx, message in enumerate(data["messages"]):
                    if message.get("extra", {}).get("fusionrag_message", {}).get("is_read", False) is True:
                        # print(f"message={message}\n")
                        # print(f"message -1={data['messages'][idx - 1]}\n")
                        content_result = message["content"]
                        if idx>0:
                            content = data["messages"][idx-1]["content"]
                            arguments = data["messages"][idx-1]["tool_calls"][0]["function"]["arguments"]
                            query = f"{content}\n{arguments}"
                            result = rerank(
                                query=query,
                                documents=[content_result],
                                api_key=api_key
                            )
                            relevant_scores.append(result["results"][0]['relevance_score'])
                            if relevant_scores[-1]<0.5:
                                # print(f"json_file={json_file}")
                                print(f"query={query}")
                                # print(f"content_result={content_result}")
            except Exception as e:
                print(e)
        if len(relevant_scores) > 250:
            break

    plot_distribution_comparison(relevant_scores, irrelevant_scores)




if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"
    print(f"start testing run_question")


    # preprocess_all_docs(file_input="/home/qy_tmp/xumengyao/all_data/musique_input.json")

    # fusion_rag_model = FusionRAGModel(
    #     # model_path='/data2/qy_tmp/xumengyao/Qwen3-32B',
    #     model_path="",
    #     use_multi_gpu=True,
    #     model_type="qwen3",
    #     model_name="Qwen3-32B",
    #     device="cuda:0",
    #     cache_path='/tmp/fusionrag/',
    #     draft_model_device="cuda:0",
    #     draft_model_path='/mnt/data/models/Qwen2.5-3B-Instruct',
    #     draft_model_type="qwen",
    #     preprocess=False,
    #     file_input="/home/qy_tmp/xumengyao/all_data/musique_input.json",
    #     preprocess_model_path="",
    #     preprocess_method="space",
    #     apikey="xxx"
    # )

    api_key = "sk-92fdf4b662d446078e9b4f71e0e4608f"

    # check_text_distribution()
    check_code_distribution("/Users/xumengyao/work/QIYUAN/tests/agent_runs_data/mengyao_debug_test_1.0")
