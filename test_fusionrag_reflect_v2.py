#!/usr/bin/env python3
"""
Test FusionRAG on result_reflect.json dataset

Following the same workflow as unified_process_cache.py:
1. Load all documents from all sub-questions as independent text chunks
2. Generate independent KV cache for each document
3. Use BGE model to compute document similarity (context_rank)
4. If preprocess=True, perform FusionRAG preprocess (fuse related documents' KV cache)
5. For each sub-question, generate answer using preprocessed KV cache
6. Use OpenAI API to judge if answer is correct
7. A question is correct only if all sub-questions are correct
"""

import json
import os
import sys
import csv
import shutil
import torch
import numpy as np
from typing import List, Dict, Any, Tuple
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from transformers import AutoTokenizer, AutoConfig
from FlagEmbedding import BGEM3FlagModel

# Add project directory to path
project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_dir)

from ktransformers.util.utils import (
    prefill_and_save_kv_cache,
    load_kv_and_generate,
    prefill_with_cache_and_save_preprocess,
    rotate_half,
    find_group_and_index,
    compute_f1,
    _exact_match_score
)
from ktransformers.models.custom_cache import StaticCache
import torch.nn.functional as F


class DocumentPoolLoader:
    """按需加载全局文档池"""
    def __init__(self, pool_path: str):
        self.pool_path = pool_path
        self.pool = {}  # {doc_id: document_text}
        if not os.path.exists(pool_path):
            raise FileNotFoundError(f"Document pool not found: {pool_path}")
        self._load_pool()

    def _load_pool(self):
        """加载整个文档池到内存（文件很小，约3.5MB）"""
        print(f"Loading document pool from {self.pool_path}...")
        with open(self.pool_path, 'r', encoding='utf-8') as f:
            docs = json.load(f)
        for idx, doc in enumerate(docs):
            # 验证文档格式
            if 'id' not in doc:
                print(f"  Warning: Document at index {idx} missing 'id' field: {doc}")
                continue
            if 'text' not in doc:
                print(f"  Warning: Document {doc.get('id', 'unknown')} missing 'text' field")
                continue
            self.pool[doc['id']] = doc['text']
        print(f"  Loaded {len(self.pool)} documents")

    def get_document(self, doc_id: int) -> str:
        """通过全局 ID 获取文档文本"""
        if doc_id not in self.pool:
            raise ValueError(f"Document ID {doc_id} not found in pool")
        return self.pool[doc_id]


class PreprocessScope(Enum):
    """
    Enum to control the scope of document retrieval during preprocessing

    GLOBAL: Retrieve similar documents from ALL examples globally (original behavior)
    PER_EXAMPLE: Retrieve similar documents only within each example's documents
    SKIP_UNTESTED: Skip retrieval for documents from untested examples (should_test=False)
    """
    GLOBAL = "global"
    PER_EXAMPLE = "per_example"
    SKIP_UNTESTED = "skip_untested"


class RecallMethod(Enum):
    """
    Enum to control the method of document recall during preprocessing

    BGE: Use BGE similarity to recall similar documents (original behavior)
    RANDOM: Randomly sample documents from the pool
    REPEAT_SELF: Repeat the current document K times (for ablation study)
    FIXED_DOC: Use a fixed document for all recalls (for ablation study)
    RANDOM_TEXT: Use BGE to get document lengths, but fuse with random unrelated text KV (for ablation study)
    BGE_SHUFFLED: Use BGE recall, but shuffle KV positions within each recalled document (for ablation study)
    RANDOM_DOCS: Randomly pick topk unrelated texts from library, use original KV without length adjustment (for ablation study)
    NO_PREPROCESS_WITH_BIAS: Use no_preprocess KV cache with distribution bias correction (BatchNorm-like)
    ONLINE_LAZY: Online mode - no preprocessing, generate KV on-demand during answer generation
    """
    BGE = "bge"
    RANDOM = "random"
    REPEAT_SELF = "repeat_self"
    FIXED_DOC = "fixed_doc"
    RANDOM_TEXT = "random_text"
    BGE_SHUFFLED = "bge_shuffled"
    RANDOM_DOCS = "random_docs"  # Pick topk random texts from library, use original KV without trimming
    NO_PREPROCESS_WITH_BIAS = "no_preprocess_with_bias"  # Use no_preprocess KV with distribution alignment
    ONLINE_LAZY = "online_lazy"  # Online lazy loading mode
    SELF_SUPERVISED = "self-supervised"  # Online lazy loading mode


def load_model(model_type, model_path, config, device="cuda:0", use_multi_gpu=False):
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


def load_system_prompt(model_family: str, dataset_type: str = "2wikimqa") -> str:
    """
    Load system prompt from config file
    """
    config_path = "./config/dataset2prompt_few-shot.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Get system prompt
    if model_family in config["system_prompt"]:
        if dataset_type in config["system_prompt"][model_family]:
            return config["system_prompt"][model_family][dataset_type]

    # Default to Qwen2.5 2wikimqa
    return config["system_prompt"]["Qwen3"]["2wikimqa"]


def scan_kv_cache_and_load_documents(
    kv_cache_dir: str,
    dataset_name: str = 'musique',
) -> List[Dict]:
    """
    扫描 KV cache 目录并从文档池提取对应文档原文

    Args:
        kv_cache_dir: KV cache 目录路径 (例如: /mnt/data3/tmp/.../Qwen2.5-7B-Instruct/musique/kv_cache)
        dataset_name: 数据集名称 ('musique' 或 '2wikimqa')

    Returns:
        List[Dict]: 每个元素包含 {
            'doc_id': int,
            'doc_text': str,
            'kv_cache_key': str,  # KV cache key 文件路径
            'kv_cache_value': str  # KV cache value 文件路径
        }
    """
    import re
    import os

    # 1. 扫描 KV cache 目录获取所有 doc_ids
    print(f"扫描 KV cache 目录: {kv_cache_dir}")
    doc_ids = []

    if not os.path.exists(kv_cache_dir):
        raise FileNotFoundError(f"KV cache 目录不存在: {kv_cache_dir}")

    # 获取所有 _key.pt 文件
    key_files = [f for f in os.listdir(kv_cache_dir) if f.endswith('_key.pt')]

    for filename in key_files:
        # 提取 doc_id (格式: doc_{doc_id}_key.pt)
        match = re.match(r'doc_(-?\d+)_key\.pt', filename)
        if match:
            doc_id = int(match.group(1))
            if doc_id != -1:  # 排除 system prompt (doc_id=-1)
                doc_ids.append(doc_id)

    doc_ids = sorted(set(doc_ids))
    print(f"  找到 {len(doc_ids)} 个文档 KV cache (排除 system prompt)")

    # 2. 加载文档池
    project_root = os.path.dirname(os.path.abspath(__file__))

    if '2wikimqa' in dataset_name.lower() or '2wiki' in dataset_name.lower():
        corpus_filename = '2wiki_input_rebuilt.json'
    else:
        corpus_filename = 'musique_input_rebuilt.json'

    pool_path = os.path.join(project_root, 'data', corpus_filename)
    print(f"加载文档池: {pool_path}")

    doc_pool = DocumentPoolLoader(pool_path)

    # 3. 提取每个文档的原文
    documents = []
    for doc_id in doc_ids:
        try:
            doc_text = doc_pool.get_document(doc_id)

            documents.append({
                'doc_id': doc_id,
                'doc_text': doc_text,
                'kv_cache_key': os.path.join(kv_cache_dir, f"doc_{doc_id}_key.pt"),
                'kv_cache_value': os.path.join(kv_cache_dir, f"doc_{doc_id}_value.pt"),
            })
        except ValueError as e:
            print(f"  警告: 跳过 doc_id={doc_id}: {e}")
            continue

    print(f"  成功加载 {len(documents)} 个文档")
    return documents




import json
import torch
import numpy as np
import random
from typing import List, Tuple
from tqdm import tqdm

def prepare_reflect_data(
    data_path: str,
    tokenizer,
    bge_model_path: str,
    model_type: str = 'qwen2',
    topk: int = 10,
    max_main_questions: int = None,
    preprocess: bool = True,
    recall_method: RecallMethod = RecallMethod.BGE,  # 召回方法：BGE/RANDOM/REPEAT_SELF/FIXED_DOC
    random_seed: int = 42,  # 随机种子（当 recall_method=RANDOM 时生效）
    fixed_doc_idx: int = 0,  # 固定文档索引（当 recall_method=FIXED_DOC 时生效）
    preprocess_scope: PreprocessScope = PreprocessScope.GLOBAL,
    dataset_name: str = 'musique'  # 数据集名称，用于选择正确的文档池
) -> Tuple[List, torch.Tensor, List, List]:
    """
    Prepare data from result_reflect.json with configurable document corpus scope

    Args:
        model_type: Type of model to determine system prompt
        topk: Top-k similar documents for each document
        preprocess: Whether to compute context_rank
        use_random_recall: If True, use random sampling instead of BGE similarity
        random_seed: Random seed for reproducibility (when use_random_recall=True)
        preprocess_scope: Scope of document retrieval
            - GLOBAL: All documents from all questions (original behavior)
            - PER_EXAMPLE: Only retrieve within each example's documents
            - SKIP_UNTESTED: Exclude documents from untested examples (should_test=False)

    Returns:
        questions_data: List of dicts for each main question
        system_tensor: Tokenized system prompt
        context_rank: [total_docs x topk] array of similar document indices
        corpus_lens: List of document counts per question
    """
    print(f"Loading dataset from {data_path}...")
    with open(data_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    if max_main_questions:
        dataset = dataset[:max_main_questions]
        print(f"Limited to first {max_main_questions} main questions")

    # Initialize document pool
    # Support both result_reflect_optimized.json and result_reflect.json paths
    # Auto-select corpus file based on dataset_name
    import os
    data_dir = os.path.dirname(data_path)
    
    # 根据数据集名称选择文档池文件
    if '2wikimqa' in dataset_name.lower() or '2wiki' in dataset_name.lower():
        corpus_filename = '2wiki_input_rebuilt.json'
    elif 'musique' in dataset_name.lower():
        corpus_filename = 'musique_input_rebuilt.json'
    else:
        # 默认使用 musique
        corpus_filename = 'musique_input_rebuilt.json'
        print(f"  Warning: Unknown dataset '{dataset_name}', using default corpus: {corpus_filename}")
    
    pool_path = os.path.join(data_dir, corpus_filename)
    print(f"  Using corpus: {pool_path}")

    # Verify dataset format (new format uses doc IDs, not text)
    if dataset and len(dataset) > 0:
        first_item = dataset[0]
        if 'intermediate_context' in first_item and len(first_item['intermediate_context']) > 0:
            first_retrieve = first_item['intermediate_context'][0].get('retrieve docs', [])
            if first_retrieve and isinstance(first_retrieve[0], str):
                raise ValueError(
                    f"\n{'='*80}\n"
                    f"ERROR: This code requires the NEW optimized dataset format!\n"
                    f"  Current: {data_path}\n"
                    f"  Expected: result_reflect_optimized.json (with doc IDs, not text)\n"
                    f"\n"
                    f"  The 'retrieve docs' field should contain integers (doc IDs),\n"
                    f"  not document text strings.\n"
                    f"\n"
                    f"  Please use: --data_path data/result_reflect_optimized.json\n"
                    f"{'='*80}"
                )

    doc_pool = DocumentPoolLoader(pool_path)

    # Map model_type to model_family for system prompt
    model_family_map = {
        'qwen': 'Qwen2.5',
        'qwen2': 'Qwen2.5',
        'qwen3': 'Qwen3',
        'mistral': 'Mistral',
        'llama': 'Llama',
        'pangu': 'Pangu'
    }
    model_family = model_family_map.get(model_type, 'Qwen2.5')

    # Tokenize system prompt (shared across all questions)
    # system_prompt = """"<|im_start|>system\n "You are a deterministic text repeater.

    # Your task is to output EXACTLY and ONLY the text that appears between the tags <content> and </content>.

    # Rules:
    # - Repeat the text word-for-word, character-for-character.
    # - Do NOT add, remove, reorder, or modify anything.
    # - Do NOT add explanations, summaries, comments, or extra text.
    # - Do NOT repeat the text more than once.
    # - Stop immediately after the last character of the content.

    # Below are examples.

    # Example 1:
    # Input:
    # <content>
    # Hello world.
    # </content>
    # END

    # Output:
    # Hello world.

    # The real input begins below and follows this exact format:
    # <content>"""
    system_prompt= "<|im_start|>system\nYou are a helpful assistant. Based on the provided conversation history, answer the question accurately and concisely.\n" # load_system_prompt(model_family, "2wikimqa")
    system_tokens = tokenizer.encode(system_prompt, add_special_tokens=True)
    system_tensor = torch.tensor(system_tokens, dtype=torch.long)

    # STEP 1: Build document corpus based on preprocess_scope
    print("\n" + "="*80)
    print(f"Building document corpus with scope: {preprocess_scope.value}")
    print("="*80)

    global_corpus = []  # Documents based on scope
    corpus_lens = []  # Number of docs per question
    questions_data = []

    # First pass: collect all documents globally and build question metadata
    for main_q_idx, data_item in enumerate(dataset):
        main_question = data_item["question"]
        main_answer = data_item["answer"]
        intermediate_context = data_item.get("intermediate_context", [])

        doc_ids_for_question = []  # Global doc IDs for THIS question
        doc_id_set = set()  # Track unique doc IDs for this question
        sub_questions_info = []

        # Check if this main question should be tested
        # Skip if main question's llm_judge is False
        should_test_main_question = True
        # if data_item.get('llm_judge', True) is False:
        #     should_test_main_question = False

        for sub_q_idx, sub_q in enumerate(intermediate_context):
            retrieve_doc_ids = sub_q.get("retrieve docs", [])  # Now these are global doc IDs (integers)

            # Collect unique doc IDs for this question
            for doc_id in retrieve_doc_ids:
                # Skip error dictionaries (docs not found in pool)
                if isinstance(doc_id, dict):
                    if 'error' in doc_id:
                        print(f"  ⚠ Skipping document with error: {doc_id.get('error', 'unknown')}")
                    continue
                # Only process integer doc IDs
                if not isinstance(doc_id, (int, str)):
                    print(f"  ⚠ Skipping invalid doc_id type: {type(doc_id)}")
                    continue
                # Convert string to int if needed
                if isinstance(doc_id, str):
                    try:
                        doc_id = int(doc_id)
                    except ValueError:
                        print(f"  ⚠ Skipping non-integer doc_id: {doc_id}")
                        continue
                    
                if doc_id not in doc_id_set:
                    doc_ids_for_question.append(doc_id)
                    doc_id_set.add(doc_id)

            # Remove "Intermediate queryXXX:" prefix from query
            query = sub_q['query']
            if query.startswith("Intermediate query"):
                # Find the colon and extract text after it
                colon_pos = query.find(":")
                if colon_pos != -1:
                    query = query[colon_pos + 1:].strip()

            # Remove "Intermediate answerXXX:" prefix from answer
            answer = sub_q['answer']
            if answer.startswith("Intermediate answer"):
                # Find the colon and extract text after it
                colon_pos = answer.find(":")
                if colon_pos != -1:
                    answer = answer[colon_pos + 1:].strip()

            # Check if any sub-question has problematic answer
            # If so, skip the entire main question
            if "No relevant information found" in answer or "没有相关信息" in answer:
                should_test_main_question = False

            sub_questions_info.append({
                'query': query,
                'answer': answer,
                'doc_ids': retrieve_doc_ids,  # Store global doc IDs (not chunk_ids)
            })

        print(f"  Main question {main_q_idx + 1}: {len(doc_ids_for_question)} unique documents, {len(sub_questions_info)} sub-questions")

        # Tokenize documents for this main question using document pool
        doc_tensors = []
        doc_id_to_tensor_idx = {}  # Mapping: global_doc_id -> index in doc_tensors

        for idx, doc_id in enumerate(doc_ids_for_question):
            try:
                doc_text = doc_pool.get_document(doc_id)
            except ValueError as e:
                print(f"  Error: {e}, skipping document {doc_id}")
                continue

            doc_text_formatted = f"Document: {doc_text}\n"
            doc_tokens = tokenizer.encode(doc_text_formatted, add_special_tokens=False)
            doc_tensor = torch.tensor(doc_tokens, dtype=torch.long)
            doc_tensors.append(doc_tensor)
            doc_id_to_tensor_idx[doc_id] = idx

        # Get document texts for global corpus (for BGE encoding)
        question_doc_texts = []
        for doc_id in doc_ids_for_question:
            try:
                doc_text = doc_pool.get_document(doc_id)
                question_doc_texts.append(doc_text)
            except ValueError as e:
                print(f"  Error: {e}, skipping from corpus")

        # Add this question's docs to global corpus based on scope
        # For SKIP_UNTESTED, only add docs if should_test is True
        if preprocess_scope == PreprocessScope.SKIP_UNTESTED:
            if should_test_main_question:
                global_corpus.extend(question_doc_texts)
                corpus_lens.append(len(question_doc_texts))
            else:
                corpus_lens.append(0)  # No docs added for this question
        else:
            # GLOBAL and PER_EXAMPLE: add all docs
            global_corpus.extend(question_doc_texts)
            corpus_lens.append(len(question_doc_texts))

        # 获取 gold_docs（用于 long_decode 模式的支撑材料评估）
        gold_docs = data_item.get('gold_docs', [])

        questions_data.append({
            'main_question': main_question,
            'main_answer': main_answer,
            'sub_questions': sub_questions_info,
            'doc_ids': doc_ids_for_question,  # Global doc IDs instead of texts
            'doc_id_to_tensor_idx': doc_id_to_tensor_idx,  # Mapping for retrieval
            'doc_tensors': doc_tensors,
            'should_test': should_test_main_question,  # Whether to test this main question
            'gold_docs': gold_docs,  # 用于 long_decode 模式的支撑材料评估
        })

    # Statistics
    total_main_q = len(questions_data)
    testable_main_q = sum(1 for q in questions_data if q['should_test'])
    skipped_main_q = total_main_q - testable_main_q

    total_sub_q = sum(len(q['sub_questions']) for q in questions_data)
    testable_sub_q = sum(len(q['sub_questions']) for q in questions_data if q['should_test'])
    skipped_sub_q = total_sub_q - testable_sub_q

    total_docs = sum(len(q['doc_ids']) for q in questions_data)



    # STEP 2: Build FAISS index and compute context_rank based on scope
    context_rank = []
    if preprocess and len(global_corpus) > 0:
        if recall_method == RecallMethod.RANDOM:
            # ========== Random Recall Mode ==========
            print("\n" + "="*80)
            print(f"Using RANDOM recall (seed={random_seed}, scope: {preprocess_scope.value})...")
            print("="*80)

            import random
            random.seed(random_seed)

            total_docs_count = sum(corpus_lens)
            all_global_indices = list(range(total_docs_count))

            for q_idx, q_data in enumerate(questions_data):
                n_docs = len(q_data['docs'])
                if n_docs == 0:
                    continue

                global_offset = sum(corpus_lens[:q_idx])
                q_context_rank = []

                # 确定随机抽取的候选池
                if preprocess_scope == PreprocessScope.PER_EXAMPLE:
                    # 只在当前问题的文档范围内抽
                    candidate_pool = list(range(global_offset, global_offset + n_docs))
                else:
                    # 在全局所有文档范围内抽
                    candidate_pool = all_global_indices

                for i in range(n_docs):
                    current_doc_global_idx = global_offset + i

                    # 除掉文档自己本身
                    others = [idx for idx in candidate_pool if idx != current_doc_global_idx]

                    # 如果候选不够，允许重复采样；否则不重复采样
                    if len(others) < topk:
                        sampled = random.choices(others, k=topk)  # 允许重复
                    else:
                        sampled = random.sample(others, k=topk)  # 不重复抽样

                    q_context_rank.append(sampled)
                context_rank.append(np.array(q_context_rank))

            if len(context_rank) > 0:
                context_rank = np.vstack(context_rank)
                print(f"Random context_rank computed: {context_rank.shape}")

        elif recall_method == RecallMethod.REPEAT_SELF:
            # ========== Repeat Self Mode ==========
            print("\n" + "="*80)
            print(f"Using REPEAT_SELF recall (repeat current document {topk} times)")
            print("="*80)

            for q_idx, q_data in enumerate(questions_data):
                n_docs = len(q_data['docs'])
                if n_docs == 0:
                    continue

                global_offset = sum(corpus_lens[:q_idx])
                q_context_rank = []

                for i in range(n_docs):
                    current_doc_global_idx = global_offset + i
                    # 重复当前文档 topk 次
                    sampled = [current_doc_global_idx] * topk
                    q_context_rank.append(sampled)

                context_rank.append(np.array(q_context_rank))

            if len(context_rank) > 0:
                context_rank = np.vstack(context_rank)
                print(f"Repeat-self context_rank computed: {context_rank.shape}")
                print(f"Each document repeats itself {topk} times")

        elif recall_method == RecallMethod.FIXED_DOC:
            # ========== Fixed Document Mode ==========
            print("\n" + "="*80)
            print(f"Using FIXED_DOC recall (use document index {fixed_doc_idx} for all recalls)")
            print("="*80)

            total_docs_count = sum(corpus_lens)

            # 验证固定文档索引是否有效
            if fixed_doc_idx < 0 or fixed_doc_idx >= total_docs_count:
                print(f"⚠ Warning: fixed_doc_idx={fixed_doc_idx} is out of range [0, {total_docs_count-1}]")
                print(f"Using document 0 as fallback")
                fixed_doc_idx = 0

            print(f"Fixed document global index: {fixed_doc_idx}")

            for q_idx, q_data in enumerate(questions_data):
                n_docs = len(q_data['docs'])
                if n_docs == 0:
                    continue

                q_context_rank = []

                for i in range(n_docs):
                    sampled = [fixed_doc_idx] * topk
                    q_context_rank.append(sampled)

                context_rank.append(np.array(q_context_rank))

            if len(context_rank) > 0:
                context_rank = np.vstack(context_rank)
                print(f"Fixed-doc context_rank computed: {context_rank.shape}")
                print(f"All documents recall fixed document {fixed_doc_idx} for {topk} times")

        elif recall_method in (RecallMethod.BGE, RecallMethod.BGE_SHUFFLED):
            # ========== BGE Similarity Mode (and BGE_SHUFFLED) ==========
            mode_name = "BGE Similarity" if recall_method == RecallMethod.BGE else "BGE Shuffled"
            print("\n" + "="*80)
            print(f"Computing document similarity with BGE + FAISS (scope: {preprocess_scope.value})...")
            if recall_method == RecallMethod.BGE_SHUFFLED:
                print("Note: KV positions will be shuffled during fusion")
            print("="*80)

            import faiss
            from FlagEmbedding import FlagModel

            # Load BGE model
            print(f"Loading BGE model from {bge_model_path}...")
            bgem3 = FlagModel(bge_model_path, use_fp16=True)

            if preprocess_scope == PreprocessScope.PER_EXAMPLE:
                # Build separate FAISS index for EACH example
                print("Building per-example FAISS indices...")
                context_rank = []

                for q_idx, q_data in enumerate(questions_data):
                    example_docs = q_data['docs']

                    if len(example_docs) == 0:
                        continue

                    print(f"  Example {q_idx + 1}: {len(example_docs)} documents")

                    # Encode this example's documents
                    example_embeddings = bgem3.encode(example_docs)
                    example_embeddings = example_embeddings.astype(np.float32)

                    # Build FAISS index for this example
                    dim = example_embeddings.shape[-1]
                    index = faiss.index_factory(dim, 'Flat', faiss.METRIC_INNER_PRODUCT)
                    index.train(example_embeddings)
                    index.add(example_embeddings)

                    # Search within this example only
                    example_embeddings_query = bgem3.encode_queries(example_docs)
                    example_embeddings_query = example_embeddings_query.astype(np.float32)
                    actual_k = min(topk, len(example_docs))
                    score, idx = index.search(example_embeddings_query, k=actual_k)

                    # Convert local indices to global indices
                    global_offset = sum(corpus_lens[:q_idx])
                    global_idx = idx + global_offset

                    # Pad to topk if needed
                    if actual_k < topk:
                        pad_width = ((0, 0), (0, topk - actual_k))
                        global_idx = np.pad(global_idx, pad_width, mode='constant', constant_values=-1)

                    context_rank.append(global_idx)

                if len(context_rank) > 0:
                    context_rank = np.vstack(context_rank)
                    print(f"Per-example context rank computed: {context_rank.shape}")

            else:
                # GLOBAL or SKIP_UNTESTED: Build single FAISS index for all corpus
                print(f"Encoding {len(global_corpus)} documents for FAISS index...")
                corpus_embeddings = bgem3.encode(global_corpus)
                print(f"Corpus embeddings shape: {corpus_embeddings.shape}")

                # Build FAISS index
                dim = corpus_embeddings.shape[-1]
                index = faiss.index_factory(dim, 'Flat', faiss.METRIC_INNER_PRODUCT)
                corpus_embeddings = corpus_embeddings.astype(np.float32)
                index.train(corpus_embeddings)
                index.add(corpus_embeddings)
                print(f"FAISS index built with {index.ntotal} vectors")

                # Search for similar documents
                print(f"Searching for top-{topk} similar documents for each document...")
                corpus_embeddings_query = bgem3.encode_queries(global_corpus)
                corpus_embeddings_query = corpus_embeddings_query.astype(np.float32)
                score, idx = index.search(corpus_embeddings_query, k=topk)
                context_rank = idx  # Shape: [total_docs, topk]

                print(f"Context rank computed: {context_rank.shape}")

            bgem3 = None  # Free memory

        elif recall_method == RecallMethod.RANDOM_TEXT:
            # ========== Random Text Mode ==========
            # Use BGE to get recalled document lengths, but fuse with random unrelated text KV
            print("\n" + "="*80)
            print(f"Using RANDOM_TEXT mode: BGE recall for lengths, but fuse with random text KV")
            print("="*80)

            import faiss
            from FlagEmbedding import FlagModel

            # Load random text library
            random_text_lib_path = "./data/random_text_library.json"
            if not os.path.exists(random_text_lib_path):
                raise FileNotFoundError(f"Random text library not found: {random_text_lib_path}")

            with open(random_text_lib_path, 'r', encoding='utf-8') as f:
                random_text_lib = json.load(f)

            random_texts = [item['content'] for item in random_text_lib['texts']]
            print(f"Loaded {len(random_texts)} random texts from library")

            # Tokenize random texts to get their lengths
            random_text_tensors = []
            random_text_lengths = []
            for text in random_texts:
                # Add system-style formatting to random text
                formatted_text = f"<|im_start|>system\n{text}<|im_end|>\n"
                tokens = tokenizer.encode(formatted_text, add_special_tokens=False)
                random_text_tensors.append(torch.tensor(tokens, dtype=torch.long))
                random_text_lengths.append(len(tokens))

            print(f"Random text token lengths: min={min(random_text_lengths)}, max={max(random_text_lengths)}, mean={sum(random_text_lengths)/len(random_text_lengths):.1f}")

            # Step 1: Use BGE to get topk recalled documents (same as BGE mode)
            print(f"Loading BGE model from {bge_model_path}...")
            bgem3 = FlagModel(bge_model_path, use_fp16=True)

            if preprocess_scope == PreprocessScope.GLOBAL:
                # Build FAISS index for BGE similarity
                print(f"Encoding {len(global_corpus)} documents for FAISS index...")
                corpus_embeddings = bgem3.encode(global_corpus)
                print(f"Corpus embeddings shape: {corpus_embeddings.shape}")

                # Build FAISS index
                dim = corpus_embeddings.shape[-1]
                index = faiss.index_factory(dim, 'Flat', faiss.METRIC_INNER_PRODUCT)
                corpus_embeddings = corpus_embeddings.astype(np.float32)
                index.train(corpus_embeddings)
                index.add(corpus_embeddings)
                print(f"FAISS index built with {index.ntotal} vectors")

                # Search for similar documents
                print(f"Searching for top-{topk} similar documents for each document...")
                corpus_embeddings_query = bgem3.encode_queries(global_corpus)
                corpus_embeddings_query = corpus_embeddings_query.astype(np.float32)
                score, bge_idx = index.search(corpus_embeddings_query, k=topk)

                # bge_idx contains the BGE recalled document indices
                # Now replace them with random text indices based on length matching
                print(f"Mapping BGE recalls to random texts based on length...")

                context_rank = []
                # Store target lengths for each document's recalls
                random_text_target_lengths = {}  # {global_doc_idx: [len1, len2, ..., len_topk]}

                for doc_idx in range(len(global_corpus)):
                    doc_recalls = bge_idx[doc_idx]  # topk BGE recalled documents
                    random_text_recalls = []
                    target_lengths = []

                    for recalled_doc_idx in doc_recalls:
                        if recalled_doc_idx < 0:  # Skip padding
                            random_text_recalls.append(-1)
                            target_lengths.append(0)
                            continue

                        # Get recalled document's length (TARGET length)
                        corpus_i, c_id = find_group_and_index(corpus_lens, recalled_doc_idx)
                        if corpus_i < len(questions_data) and c_id < len(questions_data[corpus_i]['doc_tensors']):
                            recalled_doc_len = questions_data[corpus_i]['doc_tensors'][c_id].shape[0]
                        else:
                            recalled_doc_len = 500  # fallback

                        # Find random text with closest length
                        best_random_idx = min(range(len(random_text_lengths)),
                                            key=lambda i: abs(random_text_lengths[i] - recalled_doc_len))

                        # Store negative index to indicate random text: -(idx+1)
                        # -1 reserved for padding, so use -(idx+2) for text_idx
                        random_text_recalls.append(-(best_random_idx + 2))
                        target_lengths.append(recalled_doc_len)  # Store target length

                    context_rank.append(random_text_recalls)
                    random_text_target_lengths[doc_idx] = target_lengths

                context_rank = np.array(context_rank)
                print(f"Context rank computed with random text mapping: {context_rank.shape}")
                print(f"Random text indices encoded as negative values: -(text_id + 2)")

                # Store target lengths in questions_data for later use
                for q_data in questions_data:
                    q_data['random_text_target_lengths'] = random_text_target_lengths

            else:
                # PER_EXAMPLE mode not implemented for RANDOM_TEXT yet
                raise NotImplementedError(f"RANDOM_TEXT mode not yet implemented for scope={preprocess_scope.value}")

            bgem3 = None  # Free memory

            # Store random text info in questions_data for later use
            for q_data in questions_data:
                q_data['random_text_tensors'] = random_text_tensors
                q_data['random_text_lengths'] = random_text_lengths

        elif recall_method == RecallMethod.RANDOM_DOCS:
            # ========== Random Docs Mode ==========
            # Simply pick topk random texts, use original KV without length adjustment
            print("\n" + "="*80)
            print(f"Using RANDOM_DOCS mode: randomly pick {topk} unrelated texts")
            print("="*80)

            # Load random text library
            random_text_lib_path = "./data/random_text_library.json"
            if not os.path.exists(random_text_lib_path):
                raise FileNotFoundError(f"Random text library not found: {random_text_lib_path}")

            with open(random_text_lib_path, 'r', encoding='utf-8') as f:
                random_text_lib = json.load(f)

            random_texts = [item['content'] for item in random_text_lib['texts']]
            print(f"Loaded {len(random_texts)} random texts from library")

            # Tokenize random texts
            random_text_tensors = []
            random_text_lengths = []
            for text in random_texts:
                # Add system-style formatting to random text
                formatted_text = f"<|im_start|>system\n{text}<|im_end|>\n"
                tokens = tokenizer.encode(formatted_text, add_special_tokens=False)
                random_text_tensors.append(torch.tensor(tokens, dtype=torch.long))
                random_text_lengths.append(len(tokens))

            print(f"Random text token lengths: min={min(random_text_lengths)}, max={max(random_text_lengths)}, mean={sum(random_text_lengths)/len(random_text_lengths):.1f}")

            # For each document, randomly select topk random texts
            import random
            random.seed(random_seed)

            total_docs_count = sum(corpus_lens)
            context_rank = []

            for doc_idx in range(total_docs_count):
                # Randomly select topk indices from random_text_library
                actual_k = min(topk, len(random_texts))
                selected_text_ids = random.sample(range(len(random_texts)), actual_k)

                # Pad to topk if needed
                if actual_k < topk:
                    selected_text_ids.extend([-1] * (topk - actual_k))

                # Encode as negative indices: -(text_id + 2)
                encoded_indices = []
                for text_id in selected_text_ids:
                    if text_id == -1:
                        encoded_indices.append(-1)  # padding
                    else:
                        encoded_indices.append(-(text_id + 2))  # encode random text

                context_rank.append(encoded_indices)

            context_rank = np.array(context_rank)
            print(f"Context rank computed: {context_rank.shape}")
            print(f"Each document will use {topk} random texts (original KV lengths, no trimming)")

            # Store random text info in questions_data for later use
            for q_data in questions_data:
                q_data['random_text_tensors'] = random_text_tensors
                q_data['random_text_lengths'] = random_text_lengths
                q_data['use_original_length'] = True  # Flag to indicate no length adjustment

    return questions_data, system_tensor, context_rank, corpus_lens


# 全局缓存：存储评判结果
_judge_cache = {}
_judge_cache_file = None

def _load_judge_cache(cache_path: str):
    """加载评判缓存"""
    global _judge_cache, _judge_cache_file
    _judge_cache_file = os.path.join(cache_path, 'judge_cache_v2.json')  # v2 版本，使用修复后的解析逻辑
    if os.path.exists(_judge_cache_file):
        try:
            with open(_judge_cache_file, 'r', encoding='utf-8') as f:
                _judge_cache = json.load(f)
            print(f"Loaded {len(_judge_cache)} cached judgments from {_judge_cache_file}")
        except:
            _judge_cache = {}

def _save_judge_cache():
    """保存评判缓存"""
    global _judge_cache, _judge_cache_file
    if _judge_cache_file:
        try:
            with open(_judge_cache_file, 'w', encoding='utf-8') as f:
                json.dump(_judge_cache, f, ensure_ascii=False, indent=2)
        except:
            pass


def judge_answer_with_openai(
    openai_client: OpenAI,
    openai_model: str,
    question: str,
    predicted_answer: str,
    ground_truth_answer: str
) -> Tuple[bool, str]:
    """
    Use OpenAI API to judge if the predicted answer is correct

    Returns:
        Tuple[bool, str]: (is_correct, reason)
    """
    # 对输入做 strip() 预处理，避免空格造成的不一致
    question = question.strip()
    predicted_answer = predicted_answer.strip()
    ground_truth_answer = ground_truth_answer.strip()

    # 生成缓存 key 并查询缓存
    cache_key = f"{question}|||{predicted_answer}|||{ground_truth_answer}"
    if cache_key in _judge_cache:
        cached = _judge_cache[cache_key]
        return cached['is_correct'], cached['reason']

    judge_prompt = f"""你是一个答案评估专家。你的任务是判断预测答案是否正确地回答了问题。

问题: {question}

标准答案: {ground_truth_answer}

预测答案: {predicted_answer}

请判断预测答案是否正确回答了问题。判断标准：
1. 预测答案包含了标准答案的关键信息
2. 预测答案与标准答案在语义上等价
3. 允许措辞上的细微差异，只要意思保持一致即可

请按照以下格式回答：
判断: [正确/错误]
原因: [详细说明为什么正确或错误，至少30字]"""

    try:
        response = openai_client.chat.completions.create(
            model=openai_model,
            messages=[
                {"role": "system", "content": "你是一个专业的答案评估专家。"},
                {"role": "user", "content": judge_prompt}
            ],
            temperature=0,
            max_tokens=300
        )

        result = response.choices[0].message.content.strip()

        # 解析返回结果
        is_correct = False
        reason = result

        # 尝试解析格式化的回答 - 只提取判断值，避免原因文本干扰
        lines = result.split('\n')
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            # 查找判断行：必须以"判断"开头，避免匹配到原因文本中的"判断标准"等词
            if (line_stripped.startswith('判断') or line_stripped.lower().startswith('judgment')) and (':' in line or '：' in line):
                # 提取冒号后的判断值部分
                judgment_value = line.split(':', 1)[-1].split('：', 1)[-1].strip()
                # 只检查判断值部分（通常只有"正确"或"错误"几个字）
                if '正确' in judgment_value or '对' in judgment_value:
                    is_correct = True
                elif '错误' in judgment_value or '错' in judgment_value:
                    is_correct = False
                # 找到判断后继续找原因
                continue
            if line_stripped.startswith('原因') or line_stripped.lower().startswith('reason'):
                # 获取原因部分
                if ':' in line or '：' in line:
                    reason_start = line.split(':', 1)[-1].split('：', 1)[-1].strip()
                    # 如果原因在下一行
                    if len(lines) > i + 1 and not reason_start:
                        reason = '\n'.join(lines[i+1:]).strip()
                    else:
                        reason = reason_start + '\n' + '\n'.join(lines[i+1:]).strip()
                    reason = reason.strip()
                    break

        # 如果没有找到格式化的原因，使用整个回答
        if not reason or len(reason) < 10:
            reason = result

        # 保存到缓存
        _judge_cache[cache_key] = {'is_correct': is_correct, 'reason': reason}
        _save_judge_cache()

        return is_correct, reason

    except Exception as e:
        error_msg = f"调用 OpenAI API 时出错: {e}"
        print(error_msg)
        return False, error_msg


def judge_evidence_with_openai(
    openai_client: OpenAI,
    openai_model: str,
    question: str,
    predicted_evidence: str,
    gold_docs: List[str],
    retrieve_docs: List[str] = None
) -> Tuple[bool, str]:
    """
    Use OpenAI API to judge if the predicted evidence matches gold_docs

    Args:
        question: The question being answered
        predicted_evidence: The evidence/supporting material extracted from model output
        gold_docs: List of gold documents that contain the correct information
        retrieve_docs: List of all retrieved documents (optional, for context)

    Returns:
        Tuple[bool, str]: (is_matched, reason)
    """
    question = question.strip()
    predicted_evidence = predicted_evidence.strip()

    # 将 gold_docs 拼接成字符串
    gold_docs_text = "\n\n".join([f"[标准文档{i+1}] {doc}" for i, doc in enumerate(gold_docs)])

    # 将 retrieve_docs 拼接成字符串（如果提供）
    retrieve_docs_text = ""
    if retrieve_docs:
        retrieve_docs_text = "\n\n".join([f"[检索文档{i+1}] {doc[:500]}..." if len(doc) > 500 else f"[检索文档{i+1}] {doc}" for i, doc in enumerate(retrieve_docs)])

    # 生成缓存 key
    cache_key = f"evidence_v2|||{question}|||{predicted_evidence[:200]}|||{gold_docs_text[:300]}|||{retrieve_docs_text[:200] if retrieve_docs_text else ''}"
    if cache_key in _judge_cache:
        cached = _judge_cache[cache_key]
        return cached['is_correct'], cached['reason']

    # 构建 prompt，包含检索文档上下文
    if retrieve_docs_text:
        judge_prompt = f"""你是一个支撑材料评估专家。你的任务是判断模型输出的支撑材料是否正确。

问题: {question}

标准文档 (Gold Docs，包含正确答案的文档):
{gold_docs_text}

完整检索上下文 (模型可见的所有检索文档):
{retrieve_docs_text}

模型输出的支撑材料:
{predicted_evidence}

请判断模型输出的支撑材料是否正确。判断标准：
1. 支撑材料是否包含标准文档中的关键信息（最重要）
2. 支撑材料中引用的其他内容是否来自检索文档（而非幻觉）
3. 如果支撑材料包含检索文档中不存在的内容，则为幻觉，应判断为不匹配
4. 允许措辞上的细微差异和信息的合理简化

请按照以下格式回答：
判断: [匹配/不匹配]
原因: [详细说明为什么匹配或不匹配，至少30字]"""
    else:
        judge_prompt = f"""你是一个支撑材料评估专家。你的任务是判断模型输出的支撑材料是否与标准文档匹配。

问题: {question}

标准文档 (Gold Docs):
{gold_docs_text}

模型输出的支撑材料:
{predicted_evidence}

请判断模型输出的支撑材料是否正确引用了标准文档中的关键信息。判断标准：
1. 支撑材料是否包含标准文档中的关键信息
2. 支撑材料是否与标准文档在语义上一致
3. 允许措辞上的细微差异和信息的合理简化

请按照以下格式回答：
判断: [匹配/不匹配]
原因: [详细说明为什么匹配或不匹配，至少30字]"""

    try:
        response = openai_client.chat.completions.create(
            model=openai_model,
            messages=[
                {"role": "system", "content": "你是一个专业的支撑材料评估专家。"},
                {"role": "user", "content": judge_prompt}
            ],
            temperature=0,
            max_tokens=300
        )

        result = response.choices[0].message.content.strip()

        # 解析返回结果
        is_matched = False
        reason = result

        lines = result.split('\n')
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if (line_stripped.startswith('判断') or line_stripped.lower().startswith('judgment')) and (':' in line or '：' in line):
                judgment_value = line.split(':', 1)[-1].split('：', 1)[-1].strip()
                if '匹配' in judgment_value and '不匹配' not in judgment_value:
                    is_matched = True
                elif '不匹配' in judgment_value:
                    is_matched = False
                continue
            if line_stripped.startswith('原因') or line_stripped.lower().startswith('reason'):
                if ':' in line or '：' in line:
                    reason_start = line.split(':', 1)[-1].split('：', 1)[-1].strip()
                    if len(lines) > i + 1 and not reason_start:
                        reason = '\n'.join(lines[i+1:]).strip()
                    else:
                        reason = reason_start + '\n' + '\n'.join(lines[i+1:]).strip()
                    reason = reason.strip()
                    break

        if not reason or len(reason) < 10:
            reason = result

        # 保存到缓存
        _judge_cache[cache_key] = {'is_correct': is_matched, 'reason': reason}
        _save_judge_cache()

        return is_matched, reason

    except Exception as e:
        error_msg = f"调用 OpenAI API 时出错: {e}"
        print(error_msg)
        return False, error_msg


def parse_long_decode_output(output: str) -> Tuple[str, str]:
    """
    Parse model output in long_decode mode to extract answer and evidence

    Expected format:
    答案: xxx
    支撑材料: xxx

    Returns:
        Tuple[str, str]: (answer, evidence)
    """
    answer = ""
    evidence = ""

    lines = output.strip().split('\n')
    current_section = None

    for line in lines:
        line_stripped = line.strip()

        # 检测答案部分
        if line_stripped.startswith('答案:') or line_stripped.startswith('答案：'):
            current_section = 'answer'
            answer = line_stripped.split(':', 1)[-1].split('：', 1)[-1].strip()
            continue
        elif line_stripped.lower().startswith('answer:'):
            current_section = 'answer'
            answer = line_stripped.split(':', 1)[-1].strip()
            continue

        # 检测支撑材料部分
        if line_stripped.startswith('支撑材料:') or line_stripped.startswith('支撑材料：'):
            current_section = 'evidence'
            evidence = line_stripped.split(':', 1)[-1].split('：', 1)[-1].strip()
            continue
        elif line_stripped.lower().startswith('evidence:') or line_stripped.lower().startswith('supporting evidence:'):
            current_section = 'evidence'
            evidence = line_stripped.split(':', 1)[-1].strip()
            continue

        # 追加到当前部分
        if current_section == 'answer' and not evidence:
            answer += ' ' + line_stripped
        elif current_section == 'evidence':
            evidence += ' ' + line_stripped

    # 清理
    answer = answer.strip()
    evidence = evidence.strip()

    # 如果没有解析到格式化的输出，把整个输出作为答案
    if not answer and not evidence:
        answer = output.strip()

    return answer, evidence


def main(
    model_type='qwen',
    model_path='/mnt/data/models/Qwen2.5-7B-Instruct',
    draft_model_path='/mnt/data/models/Qwen2.5-3B-Instruct',  # Draft model path for DraftModel method
    data_path='/mnt/data/ktransformers-dev/result_reflect.json',
    cache_path='/mnt/data3/reflect/',
    result_path=None,  # Path to save results (CSV, TXT). If None, uses cache_path
    model_name='Qwen2.5-7B-Instruct',
    dataset_name='2wikimqa',  # 数据集名称，用于结果分文件夹存储
    max_cache_len=32768,
    rate=0.2,
    topk=10,
    preprocess=True,
    use_random_recall=False,  # 已废弃，保留用于向后兼容。请使用 recall_method_str
    recall_method_str='bge',  # 召回方法: 'bge', 'random', 'repeat_self', 'fixed_doc', 'no_preprocess_with_bias'
    random_seed=42,  # 随机种子（当 recall_method=random 时生效）
    fixed_doc_idx=0,  # 固定文档索引（当 recall_method=fixed_doc 时生效）
    kv_stats_path=None,  # 分布统计文件路径（当 recall_method=no_preprocess_with_bias 时必需）
    steering_alpha=1.0,  # Steering vector 强度系数（当 recall_method=no_preprocess_with_bias 时生效）
    steering_key_layers='all',  # 应用 key steering vector 的层（格式: "all", "0-10", "0,5,10"）
    steering_value_layers='all',  # 应用 value steering vector 的层（格式: "all", "0-10", "0,5,10"）
    use_per_head_steering=False,  # 是否使用per-head steering vectors（如果文件中有）
    preprocess_scope=PreprocessScope.GLOBAL,
    reprocess_method='FusionRAG',
    use_entropy_selection=False,  # 是否使用熵选层 (用于 QueryAttention 消融实验)
    entropy_top_k=4,  # 熵选层选择的层数
    draft_layer_selection='entropy',  # DraftModel 选层方式: 'entropy' (熵选层), 'last' (最后一层), 'fixed' (固定层), 或 'middle' (中间层)
    draft_fixed_layer=3,  # 固定层选择时使用的层号 (当 draft_layer_selection='fixed' 时生效)
    draft_threshold_factor=0.5,  # smart_query_selection 阈值因子 (默认 0.5，较小值会选择更多位置进入连通分量分析)
    bge_model_path='/mnt/data/models/bge-m3-FP16',
    revert_rope=True,
    device="cuda:0",
    use_multi_gpu=False,
    openai_api_key=None,
    openai_base_url="https://api.openai.com/v1",
    openai_model="gpt-4",
    max_samples=None,
    vattention_topk_ratio=0.5,  # vAttention/OracleDynamic: top-k 占总 budget 的比例 (默认 0.5 = 各占 50%)
    # OracleDynamic 动态 budget 参数
    epsilon=0.1,     # 误差容忍度 (如 0.1 = 10% 相对误差)
    delta=0.05,      # 置信度 (如 0.05 = 95% 置信度)
    min_rate=0.05,   # 动态 budget 的最小比例
    max_rate=0.5,    # 动态 budget 的最大比例
    # DraftModelLayerwise 参数
    layerwise_decay='linear',  # 'linear', 'exponential', 'cosine', 'step'
    layerwise_final_rate=0.05,  # 最后一层的 rate
    # DraftModel 相似度重排序改进
    use_similarity_rerank=False,  # 使用 query-doc 相似度重排序改进 DraftModel 选择
    rerank_multiplier=2.0,  # 重排序时先选择多少倍候选
    # Long decode 模式参数
    long_decode=False,  # 是否启用长 decode 模式（要求输出答案和支撑材料）
    long_decode_max_tokens=1000,  # long_decode 模式下的最大生成 token 数
    # 消融实验：文档重复参数
    repeat_k_times=1,  # 文档重复次数（默认1表示不重复，>1时开启消融模式）
):
    """
    Main function for FusionRAG testing on result_reflect.json

    Args:
        model_type: Type of model ('qwen', 'qwen2', 'qwen3', 'mistral', 'llama', 'pangu')
        model_path: Path to the language model
        data_path: Path to result_reflect.json
        cache_path: Path to save KV cache
        model_name: Model name for logging
        max_cache_len: Maximum cache length
        rate: Compression rate (0=no compression, 1=full recompute). 对于 OracleDynamic 方法会被忽略
        topk: Top-k similar documents to fuse in preprocess
        preprocess: Whether to use FusionRAG preprocess
        use_random_recall: If True, use random document sampling; if False, use BGE similarity (default: False)
        random_seed: Random seed for reproducibility when use_random_recall=True (default: 42)
        preprocess_scope: Scope for document retrieval (GLOBAL, PER_EXAMPLE, SKIP_UNTESTED)
        reprocess_method: Method name ('FusionRAG', 'Oracle', 'OracleAdaptive', 'vAttention', 'OracleDynamic', 'DraftModel', etc.)
        bge_model_path: Path to BGE model for computing similarity
        revert_rope: Whether to revert rope in preprocessing
        device: Device to use (for single GPU)
        use_multi_gpu: Whether to use multi-GPU with device_map='auto'
        openai_api_key: OpenAI API key for judging answers
        openai_base_url: OpenAI API base URL
        openai_model: OpenAI model for judging
        max_samples: Maximum number of main questions to test (None = all)
        vattention_topk_ratio: For vAttention/OracleDynamic, ratio of top-k vs random sampling (default: 0.5)
        epsilon: For OracleDynamic, error tolerance (e.g., 0.1 = 10% relative error)
        delta: For OracleDynamic, confidence level (e.g., 0.05 = 95% confidence)
        min_rate: For OracleDynamic, minimum recompute ratio
        max_rate: For OracleDynamic, maximum recompute ratio
    """

    # Create cache directories with model-specific and dataset-specific subdirectories
    # Different preprocess_scope uses different preprocess cache directories
    # 按数据集区分缓存目录，避免不同数据集的缓存冲突
    model_cache_root = os.path.join(cache_path, model_name, dataset_name)
    save_path =cache_path# "/mnt/data3/tmp/fusionrag_new/Qwen2.5-7B-Instruct/musique/preprocess_kv_cache_global_topk10_repeat_self" #os.path.join(model_cache_root, 'kv_cache')

    # Separate preprocess cache for different configurations
    # Cache naming includes all parameters that affect KV cache content:
    # - scope: global/per_example/skip_untested
    # - topk: number of documents to fuse
    # - recall_method: bge/random/repeat_self/fixed_doc

    # Convert recall_method_str to RecallMethod enum (with backward compatibility)
    if use_random_recall and recall_method_str == 'bge':
        # Backward compatibility: use_random_recall=True overrides recall_method_str
        recall_method_str = 'random'

    # Map string to enum
    recall_method_map = {
        'bge': RecallMethod.BGE,
        'random': RecallMethod.RANDOM,
        'repeat_self': RecallMethod.REPEAT_SELF,
        'fixed_doc': RecallMethod.FIXED_DOC,
        'random_text': RecallMethod.RANDOM_TEXT,
        'bge_shuffled': RecallMethod.BGE_SHUFFLED,
        'random_docs': RecallMethod.RANDOM_DOCS,
        'no_preprocess_with_bias': RecallMethod.NO_PREPROCESS_WITH_BIAS,
        'online_lazy': RecallMethod.ONLINE_LAZY,
        'self_supervised': RecallMethod.SELF_SUPERVISED,
    }
    recall_method_enum = recall_method_map.get(recall_method_str.lower(), RecallMethod.BGE)

    # Load KV distribution statistics for no_preprocess_with_bias method
    kv_distribution_stats = None
    if recall_method_enum == RecallMethod.NO_PREPROCESS_WITH_BIAS:
        if kv_stats_path is None:
            raise ValueError("--kv_stats_path must be provided when using no_preprocess_with_bias method")
        kv_distribution_stats = load_kv_distribution_stats(kv_stats_path, model_name=model_name, dataset_name=dataset_name)

        # Check format and print appropriate message
        if 'key_steering' in kv_distribution_stats:
            num_layers = len(kv_distribution_stats['key_steering'])
            print(f"Loaded steering vectors for {num_layers} layers (manifold steering format)")
        elif 'key_stats' in kv_distribution_stats:
            num_layers = len(kv_distribution_stats['key_stats'])
            print(f"Loaded distribution statistics for {num_layers} layers (BatchNorm format)")
        else:
            raise ValueError(f"Unknown format in KV stats file: {kv_stats_path}")

    # Format: preprocess_kv_cache_{scope}_topk{topk}_{recall_method}
    if preprocess_scope == PreprocessScope.GLOBAL:
        scope_str = "global"
    elif preprocess_scope == PreprocessScope.PER_EXAMPLE:
        scope_str = "per_example"
    elif preprocess_scope == PreprocessScope.SKIP_UNTESTED:
        scope_str = "skip_untested"
    else:
        scope_str = "default"

    cache_dir_name = f"preprocess_kv_cache_{scope_str}_topk{topk}_{recall_method_enum.value}"
    preprocess_save_path = os.path.join(model_cache_root, cache_dir_name)

    # 结果目录：使用配置特定的子目录结构
    # Format: {method}_{scope}_topk{topk}_{recall_method} or "nopreprocess"
    if result_path is not None:
        result_root = os.path.join(result_path, model_name, dataset_name)
    else:
        result_root = os.path.join(model_cache_root, 'results_root')

    # Create configuration-specific subdirectory
    if preprocess:
        # Format: Method_scope_topk{topk}_{recall_method}
        config_dir_name = f"{reprocess_method}_{preprocess_scope.value}_topk{topk}_{recall_method_enum.value}"
    else:
        # No preprocess: use simple "nopreprocess" directory
        config_dir_name = "nopreprocess"

    csv_path = os.path.join(result_root, config_dir_name)

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(preprocess_save_path, exist_ok=True)
    os.makedirs(csv_path, exist_ok=True)

    # 初始化评判缓存
    _load_judge_cache(csv_path)

    recall_method_display = {
        RecallMethod.BGE: "BGE Similarity",
        RecallMethod.RANDOM: "Random Recall",
        RecallMethod.REPEAT_SELF: "Repeat Self",
        RecallMethod.FIXED_DOC: f"Fixed Doc (idx={fixed_doc_idx})",
        RecallMethod.RANDOM_TEXT: "Random Text (BGE lengths)",
        RecallMethod.BGE_SHUFFLED: "BGE Shuffled (KV positions)",
        RecallMethod.RANDOM_DOCS: "Random Docs (original lengths)",
        RecallMethod.NO_PREPROCESS_WITH_BIAS: "No Preprocess + Distribution Bias"
    }
    print(f"Cache directories created under: {model_cache_root}")
    print(f"  - KV cache: {save_path}")
    print(f"  - Preprocess cache:")
    print(f"      Scope: {preprocess_scope.value}")
    print(f"      TopK: {topk}")
    print(f"      Recall: {recall_method_display.get(recall_method_enum, 'Unknown')}")
    print(f"      Path: {cache_dir_name}")
    print(f"  - Results directory: {config_dir_name}")
    print(f"      Full path: {csv_path}")

    # Load model and tokenizer
    print(f"Loading tokenizer and config from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config._attn_implementation = "sdpa"

    print(f"Loading {model_type} model...")
    if use_multi_gpu:
        print("Using multi-GPU with device_map='auto'")
    model, device_map = load_model(model_type, model_path, config, device, use_multi_gpu)

    # Load draft model if using DraftModel, DynamicDraftModel, DraftModelDynamic, or DraftModelLayerwise method
    # Note: Oracle method uses the main model itself, no need to load draft model
    draft_model = None
    if reprocess_method in ('DraftModel', 'DynamicDraftModel', 'DraftModelDynamic', 'DraftModelLayerwise'):
        if draft_model_path is None:
            raise ValueError("draft_model_path must be provided when using DraftModel/DynamicDraftModel/DraftModelDynamic/DraftModelLayerwise method")
        print(f"\nLoading draft model from {draft_model_path}...")
        draft_config = AutoConfig.from_pretrained(draft_model_path, trust_remote_code=True)
        draft_config._attn_implementation = "sdpa"
        # Draft model always on single GPU
        draft_model, _ = load_model('qwen', draft_model_path, draft_config, device, use_multi_gpu=False)
        draft_model.eval()
        print(f"Draft model loaded: {draft_model.config.num_hidden_layers} layers")
    elif reprocess_method == 'Oracle':
        print(f"\nOracle method: Using main model for attention computation (no draft model needed)")

    # Prepare data organized by main questions
    print("Preparing data organized by main questions...")
    questions_data, system_tensor, context_rank, corpus_lens = prepare_reflect_data(
        data_path, tokenizer, bge_model_path, model_type, topk, max_samples, preprocess,
        recall_method_enum, random_seed, fixed_doc_idx, preprocess_scope, dataset_name
    )
    # Initialize OpenAI client
    if openai_api_key is None:
        openai_api_key = os.environ.get("OPENAI_API_KEY")
    openai_client = OpenAI(api_key=openai_api_key, base_url=openai_base_url)

    # 创建线程池用于异步判断（max_workers=4 允许同时发起 4 个 API 请求）
    judge_executor = ThreadPoolExecutor(max_workers=4)

    # CSV file for results (simplified filenames since config is in directory name)
    rope_suffix = "_revert_rope" if revert_rope else ""
    long_decode_suffix = "_long_decode" if long_decode else ""

    # Extract draft model name for DraftModel methods
    # Note: rate=1 baseline files don't include draft model name (since draft model doesn't affect rate=1)
    draft_model_suffix = ""
    if reprocess_method in ('DraftModel', 'DynamicDraftModel', 'DraftModelDynamic', 'DraftModelLayerwise') and draft_model_path:
        draft_model_name = os.path.basename(draft_model_path.rstrip('/'))
        draft_model_suffix = f"_draft_{draft_model_name}"

    # Simplified filenames: config info is already in directory name
    # Format: rate_{rate}{suffixes}.csv
    csv_file = f"{csv_path}/rate_{rate}{draft_model_suffix}{rope_suffix}{long_decode_suffix}.csv"
    result_file = f"{csv_path}/rate_{rate}{draft_model_suffix}{rope_suffix}{long_decode_suffix}.txt"
    # rate=1 baseline (no draft model suffix)
    rate1_csv_file = f"{csv_path}/rate_1{rope_suffix}{long_decode_suffix}.csv"

    # Load rate=1 results for comparison if rate != 1
    rate1_results = {}
    if rate != 1:
        rate1_file_found = False
        if not os.path.exists(rate1_csv_file):
            # 尝试查找其他 rate=1 文件
            import glob
            rate1_pattern = f"{csv_path}/rate_1*.csv"
            rate1_files = glob.glob(rate1_pattern)
            if rate1_files:
                # 优先选择匹配当前模式的文件
                # long_decode 模式优先选择包含 _long_decode 的文件
                if long_decode:
                    long_decode_files = [f for f in rate1_files if '_long_decode' in f]
                    if long_decode_files:
                        rate1_csv_file = long_decode_files[0]
                    else:
                        rate1_csv_file = rate1_files[0]
                else:
                    # 非 long_decode 模式优先选择不包含 _long_decode 的文件
                    non_long_decode_files = [f for f in rate1_files if '_long_decode' not in f]
                    if non_long_decode_files:
                        rate1_csv_file = non_long_decode_files[0]
                    else:
                        rate1_csv_file = rate1_files[0]
                print(f"[INFO] Using alternative rate=1 file: {rate1_csv_file}")
                rate1_file_found = True
            else:
                print(f"[WARNING] Rate=1 results file not found: {rate1_csv_file}")
                print(f"[WARNING] No rate=1 files found in {csv_path}")
                print(f"[WARNING] Continuing without rate=1 baseline comparison...")
        else:
            rate1_file_found = True

        if rate1_file_found:
            print(f"\nLoading rate=1 baseline results from {rate1_csv_file}...")
            try:
                with open(rate1_csv_file, mode='r', newline='', encoding='utf-8') as file:
                    reader = csv.DictReader(file)
                    for row in reader:
                        key = (row['Main Question'], row['Sub Question'])
                        rate1_results[key] = {
                            'predicted': row['Predicted'],
                            'correct': row['Correct'],
                            'f1': row['F1'],
                            'em': row['EM'],
                            'reason': row['Reason'],
                            # long_decode 模式额外读取 evidence 相关字段
                            'evidence': row.get('Evidence', 'N/A'),
                            'evidence_matched': row.get('Evidence_Matched', 'N/A'),
                            'evidence_reason': row.get('Evidence_Reason', 'N/A'),
                        }
                print(f"Loaded {len(rate1_results)} rate=1 results for comparison")
            except Exception as e:
                print(f"[WARNING] Failed to load rate=1 results: {e}")
                print(f"[WARNING] Continuing without rate=1 baseline comparison...")
                rate1_results = {}

    # Write CSV header
    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if long_decode:
            # long_decode 模式：额外添加 Evidence, Evidence_Matched, Evidence_Reason, Gold_Docs 列
            if rate != 1:
                if reprocess_method in ('OracleDynamic', 'OracleAdaptive', 'DynamicDraftModel', 'DraftModelDynamic', 'DraftModelLayerwise'):
                    writer.writerow([
                        'Main Question', 'Sub Question', 'Ground Truth', 'Gold_Docs',
                        'Predicted', 'Correct', 'F1', 'EM', 'Reason',
                        'Evidence', 'Evidence_Matched', 'Evidence_Reason',
                        'Dynamic_Rate', 'Dispersion_Score',
                        'Rate1_Predicted', 'Rate1_Correct', 'Rate1_F1', 'Rate1_EM', 'Rate1_Reason',
                        'Rate1_Evidence', 'Rate1_Evidence_Matched', 'Rate1_Evidence_Reason'
                    ])
                else:
                    writer.writerow([
                        'Main Question', 'Sub Question', 'Ground Truth', 'Gold_Docs',
                        'Predicted', 'Correct', 'F1', 'EM', 'Reason',
                        'Evidence', 'Evidence_Matched', 'Evidence_Reason',
                        'Rate1_Predicted', 'Rate1_Correct', 'Rate1_F1', 'Rate1_EM', 'Rate1_Reason',
                        'Rate1_Evidence', 'Rate1_Evidence_Matched', 'Rate1_Evidence_Reason'
                    ])
            else:
                writer.writerow([
                    'Main Question', 'Sub Question', 'Ground Truth', 'Gold_Docs',
                    'Predicted', 'Correct', 'F1', 'EM', 'Reason',
                    'Evidence', 'Evidence_Matched', 'Evidence_Reason'
                ])
        else:
            # 原有逻辑
            if rate != 1:
                # 对于动态 rate 方法，额外添加 dynamic_rate 列
                if reprocess_method in ('OracleDynamic', 'OracleAdaptive', 'DynamicDraftModel', 'DraftModelDynamic', 'DraftModelLayerwise'):
                    writer.writerow([
                        'Main Question', 'Sub Question', 'Ground Truth',
                        'Predicted', 'Correct', 'F1', 'EM', 'Reason',
                        'Dynamic_Rate', 'Dispersion_Score',
                        'Rate1_Predicted', 'Rate1_Correct', 'Rate1_F1', 'Rate1_EM', 'Rate1_Reason'
                    ])
                else:
                    writer.writerow([
                        'Main Question', 'Sub Question', 'Ground Truth',
                        'Predicted', 'Correct', 'F1', 'EM', 'Reason',
                        'Rate1_Predicted', 'Rate1_Correct', 'Rate1_F1', 'Rate1_EM', 'Rate1_Reason'
                    ])
            else:
                writer.writerow(['Main Question', 'Sub Question', 'Ground Truth', 'Predicted', 'Correct', 'F1', 'EM', 'Reason'])

    # Initialize static cache
    # For multi-GPU, pass device_map; for single GPU, pass device string
    cache_device = device_map if use_multi_gpu else device
    past_key_values = StaticCache(
        config=model.config,
        max_batch_size=1,
        max_cache_len=max_cache_len,
        device=cache_device,
        dtype=model.dtype,
        passage_len=32768
    )

    # Answer sub-questions with on-demand KV cache generation
    print(f"\n{'='*80}")
    print("Answering sub-questions with on-demand KV cache generation")
    print(f"{'='*80}")

    system_len = system_tensor.shape[0]

    # Determine the device for input tensors
    # For multi-GPU, use the first device; for single GPU, use the specified device
    if use_multi_gpu:
        input_device = "cuda:0"  # First GPU for inputs
    else:
        input_device = device

    # Track results
    total_main_questions = 0
    correct_main_questions = 0
    total_sub_questions = 0
    correct_sub_questions = 0
    total_f1 = 0.0
    total_em = 0.0

    # ========== 统计总重算 token 数 ==========
    total_recompute_tokens = 0

    # long_decode 模式的 evidence 统计
    total_evidence = 0
    matched_evidence = 0

    # 收集 OracleDynamic 的动态 rate 信息
    dynamic_rate_stats = []  # List of (main_q_idx, sub_q_idx, dynamic_rate, cv, doc_len)

    # Define global system prompt ID
    SYSTEM_PROMPT_ID = -1

    # For self_supervised mode, always regenerate system prompt KV cache
    # because our system prompt is different from the original QA format
    # Also force reprocess_method to 'DraftModel' for self_supervised mode
    if recall_method_enum == RecallMethod.SELF_SUPERVISED:
        original_reprocess_method = reprocess_method
        reprocess_method = 'DraftModel'  # Force to use DraftModel to avoid pdb.set_trace()
        print(f"Self-supervised mode: forcing reprocess_method to 'DraftModel' (original: {original_reprocess_method})")

    if recall_method_enum == RecallMethod.SELF_SUPERVISED or rate != 1:
        system_cache_key = f"{save_path}/doc_{SYSTEM_PROMPT_ID}_key.pt"

        # Delete old system prompt KV cache if exists (for self_supervised mode)
        if recall_method_enum == RecallMethod.SELF_SUPERVISED and os.path.exists(system_cache_key):
            print(f"Deleting old system prompt KV cache for self_supervised mode...")
            os.remove(system_cache_key)
            system_cache_value = f"{save_path}/doc_{SYSTEM_PROMPT_ID}_value.pt"
            if os.path.exists(system_cache_value):
                os.remove(system_cache_value)

        if not os.path.exists(system_cache_key):
            print(f"\n{'='*80}")
            print(f"Generating global system prompt KV cache...")
            print(f"{'='*80}")
            prefill_and_save_kv_cache(
                model, tokenizer, past_key_values, system_tensor.unsqueeze(0).to(input_device),
                save_path=save_path,
                doc_id=SYSTEM_PROMPT_ID,
                system_len=system_len,
                passage_len=0,
                reprocess_method=reprocess_method,
                device=input_device,
                device_map=device_map
            )
            print(f"  System prompt KV cached globally at doc_{SYSTEM_PROMPT_ID}_key.pt")
        else:
            print(f"Global system prompt KV cache already exists (doc_{SYSTEM_PROMPT_ID}_key.pt)")
    else:
        print(f"Rate=1: Skipping system prompt KV cache generation")


    # Process each main question (on-demand cache generation)
    documents = scan_kv_cache_and_load_documents(save_path, dataset_name)

    # Apply max_samples limit
    if max_samples is not None and max_samples > 0:
        documents = documents[:max_samples]
        print(f"Limiting to first {max_samples} documents (out of {len(documents)} total)")

    total_docs = len(documents)
    total_f1 = 0.0
    total_em = 0.0
    results = []
    # for example_id, q_data in enumerate(questions_data):
    for example_id, doc_data in enumerate(documents):     

        doc_id = doc_data['doc_id']
        doc_text = doc_data['doc_text']

        print(f"\n{'='*80}")
        print(f"Document {example_id+1}/{len(documents)}: doc_id={doc_id}, length={len(doc_text)}")
        print(f"{'='*80}")

        # Step 1: Generate answer using document KV cache

        # Build tokens: system + docs + question
        # System prompt ends with <content>, so question text needs to close it
        # IMPORTANT: Must include 'Question: ' for load_kv_and_generate to work correctly
        if model_type == 'qwen3':
            question_text = f"<|im_end|>\n<|im_start|>user\n/no_think\nQuestion: Please repeat the given document content below exactly, word for word, without any modification.<|im_end|>\n<|im_start|>assistant\nAnswer: "
        else:
            question_text = f"<|im_end|>\n<|im_start|>user\nQuestion: Please repeat the given document content below exactly, word for word, without any modification.<|im_end|>\n<|im_start|>assistant\nAnswer: "
        question_tokens = tokenizer.encode(question_text, add_special_tokens=False)
        question_tensor = torch.tensor(question_tokens, dtype=torch.long)

        # question_text = f"</content>\n\nEND<|im_end|>\n<|im_start|>user\nQuestion: Please repeat the content above exactly.<|im_end|>\n<|im_start|>assistant\nAnswer: "
        # question_tokens = tokenizer.encode(question_text, add_special_tokens=False)
        # question_tensor = torch.tensor(question_tokens, dtype=torch.long)

        # 构造文档 prompt: 使用和原来相同的格式
        doc_prompt = f"Document: {doc_text}\n"
        doc_tokens = tokenizer.encode(doc_prompt, add_special_tokens=False)
        doc_tensor = torch.tensor(doc_tokens, dtype=torch.long)

        # Debug: print token lengths
        print(f"  System tokens: {system_tensor.shape[0]}")
        print(f"  Document tokens: {doc_tensor.shape[0]}")
        print(f"  Question tokens: {question_tensor.shape[0]}")
        print(f"  Total passages tokens: {system_tensor.shape[0] + doc_tensor.shape[0] + question_tensor.shape[0]}")

        passages = [system_tensor, doc_tensor, question_tensor]

        kv_doc_ids = [SYSTEM_PROMPT_ID, doc_id]

        current_max_new_tokens = long_decode_max_tokens if long_decode else 500

        # Load preprocessed KV cache and generate (FusionRAG, QueryAttention, DraftModel, Oracle, vAttention, OracleDynamic, etc.)
        load_path = preprocess_save_path if preprocess else save_path
        generated_tokens, _, extra_info = load_kv_and_generate(
            model, tokenizer, past_key_values, passages, load_path,
            doc_ids=kv_doc_ids,  # Use global doc IDs instead of chunk_ids
            max_new_tokens=current_max_new_tokens, revert_rope=revert_rope,
            reprocess_method=reprocess_method, rate=rate,
            draft_model=draft_model,  # DraftModel/DraftModelLayerwise 方法会用到
            use_entropy_selection=use_entropy_selection,
            entropy_top_k=entropy_top_k,
            draft_layer_selection=draft_layer_selection,  # DraftModel/Oracle/vAttention/OracleDynamic/OracleAdaptive/DraftModelLayerwise 选层方式
            draft_fixed_layer=draft_fixed_layer,  # 固定层选择时使用的层号
            draft_threshold_factor=draft_threshold_factor,  # smart_query_selection 阈值因子
            use_similarity_rerank=use_similarity_rerank,  # DraftModel 相似度重排序
            rerank_multiplier=rerank_multiplier,
            preprocess=preprocess, device=input_device, device_map=device_map,
            vattention_topk_ratio=vattention_topk_ratio,  # vAttention/OracleDynamic/OracleAdaptive: top-k 比例
            # OracleDynamic/OracleAdaptive 参数
            epsilon=epsilon,
            delta=delta,
            min_rate=min_rate,
            max_rate=max_rate,
            # DraftModelLayerwise 参数
            layerwise_decay=layerwise_decay,
            layerwise_final_rate=layerwise_final_rate,
            # 文本块1用原始KV cache (prefix cache hit)
            original_kv_path=save_path if preprocess else None,
            # 消融实验：文档重复
            repeat_k_times=repeat_k_times
        )

        # ========== 累加重算 token 统计 ==========
        if extra_info.get('recompute_token_count') is not None:
            total_recompute_tokens += extra_info['recompute_token_count']

        # Decode answer
        raw_output = tokenizer.decode(torch.tensor(generated_tokens[:-1]), skip_special_tokens=True)
        raw_output = raw_output.strip() if raw_output else ""

        # long_decode 模式：解析输出，提取答案和支撑材料

        answer = raw_output
        evidence = ""
        if not answer:
            answer = "[EMPTY]"
            print(f"Predicted: {answer} (WARNING: empty answer)")
        else:
            print(f"Predicted: {answer}")

        # Compute F1 and EM (比较原文和生成文本)
        try:
            # 清理生成文本
            generated_clean = answer.strip()
            if '</content>' in generated_clean:
                generated_clean = generated_clean.split('</content>')[0].strip()

            # 和原文比较
            f1_score = compute_f1(doc_text, generated_clean, tokenizer)
            em_score = 1.0 if _exact_match_score(doc_text, generated_clean) else 0.0
        except Exception as e:
            print(f"Warning: F1/EM computation failed: {e}")
            f1_score = 0.0
            em_score = 0.0
        print(f"F1: {f1_score:.4f}, EM: {em_score:.4f}")

        # 清理 GPU 缓存
        torch.cuda.empty_cache()

        # 收集结果
        total_f1 += f1_score
        total_em += em_score

        results.append({
            'doc_id': doc_id,
            'doc_length': len(doc_text),
            'generated_length': len(answer),
            'f1_score': f1_score,
            'em_score': em_score,
            'original_text': doc_text,
            'generated_text': answer,
        })

    # 输出统计
    avg_f1 = total_f1 / total_docs if total_docs > 0 else 0
    avg_em = total_em / total_docs if total_docs > 0 else 0

    print(f"\n{'='*80}")
    print(f"SELF-SUPERVISED RESULTS (rate={rate})")
    print(f"{'='*80}")
    print(f"Total Documents: {total_docs}")
    print(f"Average F1: {avg_f1:.4f}")
    print(f"Average EM: {avg_em:.4f}")
    print(f"{'='*80}")

    # 保存结果 - 文件名包含 rate
    if result_path is None:
        result_path = cache_path
    os.makedirs(result_path, exist_ok=True)

    # 将 rate 转换为文件名友好的格式 (0.5 -> "0-5", 1.0 -> "1-0")
    rate_str = str(rate).replace('.', '_')
    result_file = os.path.join(result_path, f'self_supervised_results_rate_{rate_str}.json')

    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump({
            'rate': rate,
            'reprocess_method': reprocess_method,
            'total_docs': total_docs,
            'avg_f1': avg_f1,
            'avg_em': avg_em,
            'results': results,
        }, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {result_file}")

    return  # 结束，不继续执行后续代码



if __name__ == '__main__':

  
    import argparse

    parser = argparse.ArgumentParser(description='FusionRAG Testing Script')

    # 模型配置
    parser.add_argument('--model_type', type=str, default='qwen',
                        choices=['qwen', 'qwen2', 'qwen3', 'mistral', 'llama', 'pangu'],
                        help='Model type')
    parser.add_argument('--model_path', type=str, default='/mnt/data/models/Qwen2.5-7B-Instruct',
                        help='Path to the main model')
    parser.add_argument('--model_name', type=str, default='Qwen2.5-7B-Instruct',
                        help='Model name for logging')
    parser.add_argument('--draft_model_path', type=str, default=None,
                        help='Path to draft model (for DraftModel method)')
    parser.add_argument('--bge_model_path', type=str, default='/mnt/data/models/bge-m3-FP16',
                        help='Path to BGE embedding model')

    # 数据配置
    parser.add_argument('--data_path', type=str, default='./data/result_reflect.json',
                        help='Path to dataset JSON file')
    parser.add_argument('--dataset_name', type=str, default='musique',
                        help='Dataset name for organizing results')
    parser.add_argument('--cache_path', type=str, default='/mnt/data/reflect/',
                        help='Path to save KV cache')
    parser.add_argument('--result_path', type=str, default=None,
                        help='Path to save results (CSV, TXT). If None, uses cache_path')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to test (None = all)')

    # FusionRAG 参数
    parser.add_argument('--rate', type=float, default=0.3,
                        help='Token recompute ratio (0-1)')
    parser.add_argument('--topk', type=int, default=10,
                        help='Top-k similar documents to fuse in preprocessing')
    parser.add_argument('--preprocess', type=lambda x: x.lower() == 'true', default=True,
                        help='Enable preprocessing (True/False)')
    parser.add_argument('--use_random_recall', type=lambda x: x.lower() == 'true', default=False,
                        help='[DEPRECATED] Use random document sampling. Please use --recall_method instead')
    parser.add_argument('--recall_method', type=str, default='bge',
                        # choices=['bge', 'random', 'repeat_self', 'fixed_doc', 'no_preprocess_with_bias'],
                        help='Document recall method: bge (similarity), random, repeat_self, fixed_doc, or no_preprocess_with_bias')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility (when recall_method=random)')
    parser.add_argument('--fixed_doc_idx', type=int, default=0,
                        help='Fixed document index to use (when recall_method=fixed_doc)')
    parser.add_argument('--kv_stats_path', type=str, default=None,
                        help='Path to KV distribution statistics file (.pt) for no_preprocess_with_bias method')
    parser.add_argument('--steering_alpha', type=float, default=1.0,
                        help='Steering vector strength (alpha) for no_preprocess_with_bias method (default: 1.0)')
    parser.add_argument('--steering_key_layers', type=str, default='all',
                        help='Layers to apply key steering vector. Format: "all", "0-10", "0,5,10", or "0-10,15,20-25" (default: all)')
    parser.add_argument('--steering_value_layers', type=str, default='all',
                        help='Layers to apply value steering vector. Format: "all", "0-10", "0,5,10", or "0-10,15,20-25" (default: all)')
    parser.add_argument('--use_per_head_steering', action='store_true', default=False,
                        help='Use per-head steering vectors if available in the stats file (default: False, use layer-level steering)')
    parser.add_argument('--reprocess_method', type=str, default='FusionRAG',
                        choices=['FusionRAG', 'Oracle', 'OracleAdaptive', 'OracleDynamic',
                                'vAttention', 'DraftModel', 'QueryAttention', 'DraftModelLayerwise'],
                        help='Token recompute method')
    parser.add_argument('--revert_rope', type=lambda x: x.lower() == 'true', default=True,
                        help='Revert RoPE during preprocessing (True/False)')
    parser.add_argument('--preprocess_scope', type=str, default='global',
                        choices=['global', 'per_example', 'skip_untested'],
                        help='Preprocessing scope')

    # 特殊方法参数
    parser.add_argument('--use_entropy_selection', type=lambda x: x.lower() == 'true', default=False,
                        help='Use entropy-based layer selection (True/False)')
    parser.add_argument('--entropy_top_k', type=int, default=4,
                        help='Number of layers to select based on entropy')
    parser.add_argument('--draft_layer_selection', type=str, default='entropy',
                        choices=['entropy', 'last', 'fixed', 'middle'],
                        help='Layer selection method for DraftModel/Oracle')
    parser.add_argument('--draft_fixed_layer', type=int, default=3,
                        help='Fixed layer index (when draft_layer_selection=fixed)')
    parser.add_argument('--draft_threshold_factor', type=float, default=0.5,
                        help='Threshold factor for smart_query_selection')

    # vAttention 参数
    parser.add_argument('--vattention_topk_ratio', type=float, default=0.5,
                        help='vAttention top-k vs random sampling ratio')

    # OracleDynamic 参数
    parser.add_argument('--epsilon', type=float, default=0.1,
                        help='Error tolerance for OracleDynamic')
    parser.add_argument('--delta', type=float, default=0.05,
                        help='Confidence level for OracleDynamic')
    parser.add_argument('--min_rate', type=float, default=0.05,
                        help='Minimum recompute ratio for OracleDynamic')
    parser.add_argument('--max_rate', type=float, default=0.5,
                        help='Maximum recompute ratio for OracleDynamic')

    # DraftModelLayerwise 参数
    parser.add_argument('--layerwise_decay', type=str, default='linear',
                        choices=['linear', 'exponential', 'cosine', 'step'],
                        help='Layerwise decay strategy')
    parser.add_argument('--layerwise_final_rate', type=float, default=0.05,
                        help='Final layer rate for layerwise methods')

    # DraftModel 相似度重排序
    parser.add_argument('--use_similarity_rerank', type=lambda x: x.lower() == 'true', default=False,
                        help='Use query-doc similarity reranking (True/False)')
    parser.add_argument('--rerank_multiplier', type=float, default=2.0,
                        help='Rerank candidate multiplier')

    # Long decode 参数
    parser.add_argument('--long_decode', type=lambda x: x.lower() == 'true', default=False,
                        help='Enable long decode mode (True/False)')
    parser.add_argument('--long_decode_max_tokens', type=int, default=1000,
                        help='Max tokens for long decode mode')

    # 消融实验：文档重复参数
    parser.add_argument('--repeat_k_times', type=int, default=1,
                        help='Repeat passages K times for ablation study (default=1, no repeat)')

    # GPU 配置
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use for single GPU')
    parser.add_argument('--use_multi_gpu', type=lambda x: x.lower() == 'true', default=False,
                        help='Use multi-GPU with device_map=auto (True/False)')

    # OpenAI 评判配置
    parser.add_argument('--openai_base_url', type=str, default='https://api.deepseek.com/v1',
                        help='OpenAI API base URL')
    parser.add_argument('--openai_api_key', type=str, default='sk-519d391217894b6e91e7c2ebf2a9f4df',
                        help='OpenAI API key')
    parser.add_argument('--openai_model', type=str, default='deepseek-chat',
                        help='OpenAI model for judging')

    # 其他
    parser.add_argument('--max_cache_len', type=int, default=32768,
                        help='Maximum cache length')

    args = parser.parse_args()

    # 解析 preprocess_scope
    scope_map = {
        'global': PreprocessScope.GLOBAL,
        'per_example': PreprocessScope.PER_EXAMPLE,
        'skip_untested': PreprocessScope.SKIP_UNTESTED
    }
    preprocess_scope = scope_map[args.preprocess_scope]

    # 调用 main 函数
    main(
        model_type=args.model_type,
        model_path=args.model_path,
        draft_model_path='/mnt/data/models/Qwen2.5-3B-Instruct',
        data_path=args.data_path,
        cache_path=args.cache_path,
        result_path=args.result_path,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        max_cache_len=args.max_cache_len,
        rate=args.rate,
        topk=args.topk,
        preprocess=args.preprocess,
        use_random_recall=args.use_random_recall,
        recall_method_str=args.recall_method,
        random_seed=args.random_seed,
        fixed_doc_idx=args.fixed_doc_idx,
        kv_stats_path=args.kv_stats_path,
        steering_alpha=args.steering_alpha,
        steering_key_layers=args.steering_key_layers,
        steering_value_layers=args.steering_value_layers,
        use_per_head_steering=args.use_per_head_steering,
        preprocess_scope=preprocess_scope,
        reprocess_method=args.reprocess_method,
        use_entropy_selection=args.use_entropy_selection,
        entropy_top_k=args.entropy_top_k,
        draft_layer_selection=args.draft_layer_selection,
        draft_fixed_layer=args.draft_fixed_layer,
        draft_threshold_factor=args.draft_threshold_factor,
        bge_model_path=args.bge_model_path,
        revert_rope=args.revert_rope,
        device=args.device,
        use_multi_gpu=args.use_multi_gpu,
        openai_api_key=args.openai_api_key,
        openai_base_url=args.openai_base_url,
        openai_model=args.openai_model,
        max_samples=args.max_samples,
        vattention_topk_ratio=args.vattention_topk_ratio,
        epsilon=args.epsilon,
        delta=args.delta,
        min_rate=args.min_rate,
        max_rate=args.max_rate,
        layerwise_decay=args.layerwise_decay,
        layerwise_final_rate=args.layerwise_final_rate,
        use_similarity_rerank=args.use_similarity_rerank,
        rerank_multiplier=args.rerank_multiplier,
        long_decode=args.long_decode,
        long_decode_max_tokens=args.long_decode_max_tokens,
        repeat_k_times=args.repeat_k_times,
    )
    # for rate in [0.2]:
    #     main(
    #         model_type='qwen',
    #         model_path='/mnt/data/models/Qwen2.5-7B-Instruct',
    #         draft_model_path='/mnt/data/models/Qwen2.5-3B-Instruct',  # Draft model for guidance
    #         data_path='./data/2wikimqa_reflect.json',
    #         cache_path='/mnt/data/reflect/',
    #         model_name='Qwen2.5-7B-Instruct',
    #         dataset_name='2wikimqa',
    #         rate=rate,  # 30% token selection
    #         topk=10,
    #         preprocess=True,
    #         use_entropy_selection=True,
    #         reprocess_method='DraftModel',  # 使用 Draft Model 指导的方法
    #         draft_layer_selection='entropy',  # 'entropy' (熵选层) 或 'last' (最后一层)
    #         preprocess_scope=PreprocessScope.GLOBAL,
    #         bge_model_path='/mnt/data/models/bge-m3-FP16',
    #         revert_rope=True,
    #         device="cuda:0",
    #         use_multi_gpu=True,
    #         openai_base_url="https://api.deepseek.com/v1",
    #         openai_api_key="sk-519d391217894b6e91e7c2ebf2a9f4df",
    #         openai_model="deepseek-chat",
    #         max_samples=200
    #     )
    # for rate in [1, 0.3]:
    #     main(
    #         model_type='qwen3',
    #         model_path='/mnt/data/models/Qwen3-32B',
    #         draft_model_path='/mnt/data/models/Qwen2.5-3B-Instruct',  # Draft model for guidance
    #         data_path='./data/result_locomo_category_2.json',
    #         cache_path='/mnt/data/reflect/',
    #         model_name='Qwen3-32B',
    #         dataset_name='locomo_temporal',
    #         rate=rate,  # 30% token selection
    #         topk=10,
    #         preprocess=True,
    #         use_entropy_selection=False,
    #         reprocess_method='DraftModel',  # 使用 Draft Model 指导的方法
    #         draft_layer_selection='entropy',  # 'entropy' (熵选层) 或 'last' (最后一层)
    #         preprocess_scope=PreprocessScope.GLOBAL,
    #         bge_model_path='/mnt/data/models/bge-m3-FP16',
    #         revert_rope=True,
    #         device="cuda:0",
    #         use_multi_gpu=True,
    #         openai_base_url="https://api.deepseek.com/v1",
    #         openai_api_key="sk-519d391217894b6e91e7c2ebf2a9f4df",
    #         openai_model="deepseek-chat",
    #         max_samples=None
    #     )

    # 收集 DraftModel 方法下每个问题的最小正确重算比例和 attention 分布特征
    # collect_optimal_rate(
    #     model_type='qwen',
    #     model_path='/mnt/data/models/Qwen2.5-7B-Instruct',
    #     draft_model_path='/mnt/data/models/Qwen2.5-3B-Instruct',
    #     data_path='./data/result_reflect.json',
    #     cache_path='/mnt/data/reflect/',
    #     model_name='Qwen2.5-7B-Instruct',
    #     dataset_name='musique',
    #     topk=10,
    #     preprocess=True,
    #     preprocess_scope=PreprocessScope.GLOBAL,
    #     bge_model_path='/mnt/data/models/bge-m3-FP16',
    #     revert_rope=True,
    #     device="cuda:0",
    #     use_multi_gpu=False,  # 单 GPU 运行
    #     max_samples=None,  # 处理所有样本
    #     output_dir="/mnt/data/wjh/FusionRAG/optimal_rate_search",
    # )
    # # QueryAttention 方法
    # main(
    #     model_type='qwen3',
    #     model_path='/mnt/data/models/Qwen3-32B',
    #     data_path='./result_reflect.json',
    #     cache_path='/mnt/data/reflect/',
    #     model_name='Qwen3-32B',
    #     rate=1,  # 30% token selection
    #     topk=10,
    #     preprocess=True,  # 开启预处理
    #     reprocess_method='QueryAttention',  # 测试 QueryAttention 方法
    #     preprocess_scope=PreprocessScope.GLOBAL,
    #     bge_model_path='/mnt/data/models/bge-m3-FP16',
    #     revert_rope=True,
    #     device="cuda:0",
    #     use_multi_gpu=True,  # Multi-GPU mode
    #     openai_base_url="https://api.deepseek.com/v1",
    #     openai_api_key="sk-519d391217894b6e91e7c2ebf2a9f4df",
    #     openai_model="deepseek-chat",
    #     max_samples=200  # Test samples
    #     )
    # main(
    #     model_type='qwen3',
    #     model_path='/mnt/data/models/Qwen3-32B',
    #     data_path='./result.json',
    #     cache_path='/mnt/data/junshi/',
    #     model_name='Qwen3-32B',
    #     rate=0.3,
    #     topk=10,
    #     preprocess=True,
    #     reprocess_method='FusionRAG',
    #     preprocess_scope=PreprocessScope.GLOBAL,
    #     bge_model_path='/mnt/data/models/bge-m3-FP16',
    #     revert_rope=True,
    #     device="cuda:0",
    #     use_multi_gpu=True,  # Set to True for multi-GPU (e.g., Qwen3-32B)
    #     openai_base_url="https://api.deepseek.com/v1",
    #     openai_api_key="sk-519d391217894b6e91e7c2ebf2a9f4df",
    #     openai_model="deepseek-chat",
    #     max_samples=200  # Test first 2 MAIN questions
    # )