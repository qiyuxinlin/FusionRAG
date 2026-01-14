#!/usr/bin/env python3


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
import json
import torch
import numpy as np
import random
from typing import List, Tuple
from tqdm import tqdm

def prepare_reflect_data_ramdom(
    data_path: str,
    tokenizer,
    bge_model_path: str,
    model_type: str = 'qwen2',
    topk: int = 10,
    max_main_questions: int = None,
    preprocess: bool = True,
    random_recall: bool =  True,  # 控制是否随机randon
    preprocess_scope: PreprocessScope = PreprocessScope.GLOBAL
) -> Tuple[List, torch.Tensor, List, List]:

    print(f"Loading dataset from {data_path}...")
    with open(data_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    if max_main_questions:
        dataset = dataset[:max_main_questions]
        print(f"Limited to first {max_main_questions} main questions")

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
    system_prompt = load_system_prompt(model_family, "2wikimqa")
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

        question_docs = []  # Documents for THIS question only
        doc_to_idx = {}  # Local doc -> chunk_id mapping for this question
        sub_questions_info = []

        # Check if this main question should be tested
        # Skip if main question's llm_judge is False
        should_test_main_question = True
        # if data_item.get('llm_judge', True) is False:
        #     should_test_main_question = False

        for sub_q_idx, sub_q in enumerate(intermediate_context):
            docs = sub_q.get("retrieve docs", [])
            doc_chunk_ids = []  # chunk_ids for this sub-question (local to this question)

            for doc in docs:
                if doc not in doc_to_idx:
                    # New document for this question
                    question_docs.append(doc)
                    chunk_id = len(question_docs)  # chunk_id starts from 1
                    doc_to_idx[doc] = chunk_id
                    doc_chunk_ids.append(chunk_id)
                else:
                    # Document already seen in this question
                    doc_chunk_ids.append(doc_to_idx[doc])

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
                'chunk_ids': doc_chunk_ids,  # chunk_ids for docs used by this sub-question
            })

        print(f"  Main question {main_q_idx + 1}: {len(question_docs)} unique documents, {len(sub_questions_info)} sub-questions")

        # Tokenize documents for this main question
        doc_tensors = []
        for doc in question_docs:
            doc_text = f"Document: {doc}\n"
            doc_tokens = tokenizer.encode(doc_text, add_special_tokens=False)
            doc_tensor = torch.tensor(doc_tokens, dtype=torch.long)
            doc_tensors.append(doc_tensor)

        # Add this question's docs to global corpus based on scope
        # For SKIP_UNTESTED, only add docs if should_test is True
        if preprocess_scope == PreprocessScope.SKIP_UNTESTED:
            if should_test_main_question:
                global_corpus.extend(question_docs)
                corpus_lens.append(len(question_docs))
            else:
                corpus_lens.append(0)  # No docs added for this question
        else:
            # GLOBAL and PER_EXAMPLE: add all docs
            global_corpus.extend(question_docs)
            corpus_lens.append(len(question_docs))

        # 获取 gold_docs（用于 long_decode 模式的支撑材料评估）
        gold_docs = data_item.get('gold_docs', [])

        questions_data.append({
            'main_question': main_question,
            'main_answer': main_answer,
            'sub_questions': sub_questions_info,
            'docs': question_docs,
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

    total_docs = sum(len(q['docs']) for q in questions_data)


    # STEP 2: Build FAISS index and compute context_rank based on scope
    context_rank = []
    if preprocess and len(global_corpus) > 0:
        import numpy as np
        import random

        print("\n" + "="*80)
        print(f"Randomly selecting context_rank (Scope: {preprocess_scope.value})...")
        print("="*80)

        total_docs_count = sum(corpus_lens)
        all_global_indices = list(range(total_docs_count))

        for q_idx, q_data in enumerate(questions_data):
            n_docs = len(q_data['docs'])
            if n_docs == 0: continue

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
                    sampled = random.choices(others, k=topk) # 允许重复
                else:
                    sampled = random.sample(others, k=topk)  # 不重复抽样
                
                q_context_rank.append(sampled)
            context_rank.append(np.array(q_context_rank)) 

        if len(context_rank) > 0:
            context_rank = np.vstack(context_rank)
            print(f"Random context_rank shape: {context_rank.shape}")

    return questions_data, system_tensor, context_rank, corpus_lens

def prepare_reflect_data(
    data_path: str,
    tokenizer,
    bge_model_path: str,
    model_type: str = 'qwen2',
    topk: int = 10,
    max_main_questions: int = None,
    preprocess: bool = True,
    use_random_recall: bool = False,  # 控制是否使用随机召回（True=随机, False=BGE相似度）
    random_seed: int = 42,  # 随机种子（当 use_random_recall=True 时生效）
    preprocess_scope: PreprocessScope = PreprocessScope.GLOBAL
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
    system_prompt = load_system_prompt(model_family, "2wikimqa")
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

        question_docs = []  # Documents for THIS question only
        doc_to_idx = {}  # Local doc -> chunk_id mapping for this question
        sub_questions_info = []

        # Check if this main question should be tested
        # Skip if main question's llm_judge is False
        should_test_main_question = True
        # if data_item.get('llm_judge', True) is False:
        #     should_test_main_question = False

        for sub_q_idx, sub_q in enumerate(intermediate_context):
            docs = sub_q.get("retrieve docs", [])
            doc_chunk_ids = []  # chunk_ids for this sub-question (local to this question)

            for doc in docs:
                if doc not in doc_to_idx:
                    # New document for this question
                    question_docs.append(doc)
                    chunk_id = len(question_docs)  # chunk_id starts from 1
                    doc_to_idx[doc] = chunk_id
                    doc_chunk_ids.append(chunk_id)
                else:
                    # Document already seen in this question
                    doc_chunk_ids.append(doc_to_idx[doc])

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
                'chunk_ids': doc_chunk_ids,  # chunk_ids for docs used by this sub-question
            })

        print(f"  Main question {main_q_idx + 1}: {len(question_docs)} unique documents, {len(sub_questions_info)} sub-questions")

        # Tokenize documents for this main question
        doc_tensors = []
        for doc in question_docs:
            doc_text = f"Document: {doc}\n"
            doc_tokens = tokenizer.encode(doc_text, add_special_tokens=False)
            doc_tensor = torch.tensor(doc_tokens, dtype=torch.long)
            doc_tensors.append(doc_tensor)

        # Add this question's docs to global corpus based on scope
        # For SKIP_UNTESTED, only add docs if should_test is True
        if preprocess_scope == PreprocessScope.SKIP_UNTESTED:
            if should_test_main_question:
                global_corpus.extend(question_docs)
                corpus_lens.append(len(question_docs))
            else:
                corpus_lens.append(0)  # No docs added for this question
        else:
            # GLOBAL and PER_EXAMPLE: add all docs
            global_corpus.extend(question_docs)
            corpus_lens.append(len(question_docs))

        # 获取 gold_docs（用于 long_decode 模式的支撑材料评估）
        gold_docs = data_item.get('gold_docs', [])

        questions_data.append({
            'main_question': main_question,
            'main_answer': main_answer,
            'sub_questions': sub_questions_info,
            'docs': question_docs,
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

    total_docs = sum(len(q['docs']) for q in questions_data)



    # STEP 2: Build FAISS index and compute context_rank based on scope
    context_rank = []
    if preprocess and len(global_corpus) > 0:
        if use_random_recall:
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

        else:
            # ========== BGE Similarity Mode ==========
            print("\n" + "="*80)
            print(f"Computing document similarity with BGE + FAISS (scope: {preprocess_scope.value})...")
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
    draft_model_path=None,  # Draft model path for DraftModel method
    data_path='/mnt/data/ktransformers-dev/result_reflect.json',
    cache_path='/mnt/data3/reflect/',
    result_path=None,  # Path to save results (CSV, TXT). If None, uses cache_path
    model_name='Qwen2.5-7B-Instruct',
    dataset_name='2wikimqa',  # 数据集名称，用于结果分文件夹存储
    max_cache_len=32768,
    rate=0.2,
    topk=10,
    preprocess=True,
    use_random_recall=False,  # 是否使用随机召回 (True=随机, False=BGE相似度)
    random_seed=42,  # 随机种子
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
    save_path = os.path.join(model_cache_root, 'kv_cache')

    # Separate preprocess cache for different configurations
    # Cache naming includes all parameters that affect KV cache content:
    # - scope: global/per_example/skip_untested
    # - topk: number of documents to fuse
    # - recall_method: random/bge (extensible for future methods)
    recall_method = "random" if use_random_recall else "bge"

    # Format: preprocess_kv_cache_{scope}_topk{topk}_{recall_method}
    if preprocess_scope == PreprocessScope.GLOBAL:
        scope_str = "global"
    elif preprocess_scope == PreprocessScope.PER_EXAMPLE:
        scope_str = "per_example"
    elif preprocess_scope == PreprocessScope.SKIP_UNTESTED:
        scope_str = "skip_untested"
    else:
        scope_str = "default"

    cache_dir_name = f"preprocess_kv_cache_{scope_str}_topk{topk}_{recall_method}"
    preprocess_save_path = os.path.join(model_cache_root, cache_dir_name)

    # 结果目录：如果指定了 result_path，使用它；否则使用 cache_path
    if result_path is not None:
        result_root = os.path.join(result_path, model_name, dataset_name)
        csv_path = os.path.join(result_root, 'results')
    else:
        csv_path = os.path.join(model_cache_root, 'results')
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(preprocess_save_path, exist_ok=True)
    os.makedirs(csv_path, exist_ok=True)

    # 初始化评判缓存
    _load_judge_cache(csv_path)

    recall_method_str = "Random Recall" if use_random_recall else "BGE Similarity"
    print(f"Cache directories created under: {model_cache_root}")
    print(f"  - KV cache: {save_path}")
    print(f"  - Preprocess cache:")
    print(f"      Scope: {preprocess_scope.value}")
    print(f"      TopK: {topk}")
    print(f"      Recall: {recall_method_str}")
    print(f"      Path: {cache_dir_name}")
    print(f"  - Results: {csv_path}")

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
        use_random_recall, random_seed, preprocess_scope
    )
    # Initialize OpenAI client
    if openai_api_key is None:
        openai_api_key = os.environ.get("OPENAI_API_KEY")
    openai_client = OpenAI(api_key=openai_api_key, base_url=openai_base_url)

    # 创建线程池用于异步判断（max_workers=4 允许同时发起 4 个 API 请求）
    judge_executor = ThreadPoolExecutor(max_workers=4)

    # CSV file for results (include preprocess_scope and revert_rope in filename)
    rope_suffix = "_revert_rope" if revert_rope else ""
    long_decode_suffix = "_long_decode" if long_decode else ""

    # Extract draft model name for DraftModel methods
    # Note: rate=1 baseline files don't include draft model name (since draft model doesn't affect rate=1)
    draft_model_suffix = ""
    if reprocess_method in ('DraftModel', 'DynamicDraftModel', 'DraftModelDynamic', 'DraftModelLayerwise') and draft_model_path:
        draft_model_name = os.path.basename(draft_model_path.rstrip('/'))
        draft_model_suffix = f"_draft_{draft_model_name}"

    if preprocess:
        csv_file = f"{csv_path}/{reprocess_method}_{preprocess_scope.value}_topk_{topk}_rate_{rate}{draft_model_suffix}{rope_suffix}{long_decode_suffix}.csv"
        result_file = f"{csv_path}/{reprocess_method}_{preprocess_scope.value}_topk_{topk}_rate_{rate}{draft_model_suffix}{rope_suffix}{long_decode_suffix}.txt"
        # rate=1 baseline: no draft model suffix (draft model doesn't matter when recomputing all tokens)
        # long_decode baseline 需要单独的 rate=1 文件
        rate1_csv_file = f"{csv_path}/{reprocess_method}_{preprocess_scope.value}_topk_{topk}_rate_1{rope_suffix}{long_decode_suffix}.csv"
    else:
        csv_file = f"{csv_path}/{reprocess_method}_rate_{rate}{draft_model_suffix}{rope_suffix}{long_decode_suffix}.csv"
        result_file = f"{csv_path}/{reprocess_method}_rate_{rate}{draft_model_suffix}{rope_suffix}{long_decode_suffix}.txt"
        rate1_csv_file = f"{csv_path}/{reprocess_method}_rate_1{rope_suffix}{long_decode_suffix}.csv"

    # Load rate=1 results for comparison if rate != 1
    rate1_results = {}
    if rate != 1:
        rate1_file_found = False
        if not os.path.exists(rate1_csv_file):
            # 尝试查找其他 rate=1 文件
            import glob
            rate1_pattern = f"{csv_path}/*_rate_1*.csv"
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

    # long_decode 模式的 evidence 统计
    total_evidence = 0
    matched_evidence = 0

    # 收集 OracleDynamic 的动态 rate 信息
    dynamic_rate_stats = []  # List of (main_q_idx, sub_q_idx, dynamic_rate, cv, doc_len)

    # Process each main question (on-demand cache generation)
    for example_id, q_data in enumerate(questions_data):
        print(f"\n{'='*80}")
        print(f"Main Question {example_id+1}/{len(questions_data)}: {q_data['main_question']}")
        print(f"{'='*80}")

        # Skip main questions that should not be tested
        if not q_data.get('should_test', True):
            print("⊘ SKIPPED (llm_judge=False or contains problematic answers)")
            print("  Note: Documents still included in global corpus for similarity computation")
            continue

        doc_tensors = q_data['doc_tensors']


        # Step 1: Generate KV cache for THIS main question's documents
        if rate != 1:  # Skip if full recompute
            # Generate system KV cache (chunk_id=0)
            system_cache_path = f'{save_path}/{example_id}_0_key.pt'
            if not os.path.exists(system_cache_path):
                print(f"Generating system KV cache...")
                input_tensor = system_tensor.unsqueeze(0)
                prefill_and_save_kv_cache(
                    model, tokenizer, past_key_values, input_tensor.to(input_device),
                    save_path=save_path, example_id=example_id, chunk_id=0,
                    system_len=system_len, passage_len=system_len,
                    reprocess_method=reprocess_method, device=input_device, device_map=device_map
                )

            # Generate KV cache for each document in THIS main question
            for doc_idx, doc_tensor in enumerate(doc_tensors):
                chunk_id = doc_idx + 1
                cache_key_path = f'{save_path}/{example_id}_{chunk_id}_key.pt'

                if not os.path.exists(cache_key_path):
                    passage_len = doc_tensor.shape[0]
                    input_tensor = torch.cat((system_tensor, doc_tensor)).unsqueeze(0)

                    prefill_and_save_kv_cache(
                        model, tokenizer, past_key_values, input_tensor.to(input_device),
                        save_path=save_path, example_id=example_id, chunk_id=chunk_id,
                        system_len=system_len, passage_len=passage_len,
                        reprocess_method=reprocess_method, device=input_device, device_map=device_map
                    )
                    print(f"  Generated KV cache for document {chunk_id}/{len(doc_tensors)}")

        # Step 2: FusionRAG preprocess (if enabled)
        if preprocess and rate != 1:
            # Copy system cache (chunk_id=0)
            system_preprocess_key = f"{preprocess_save_path}/{example_id}_0_key.pt"
            if not os.path.exists(system_preprocess_key):
                shutil.copy(f'{save_path}/{example_id}_0_key.pt', system_preprocess_key)
                shutil.copy(f'{save_path}/{example_id}_0_value.pt', f"{preprocess_save_path}/{example_id}_0_value.pt")

            # Preprocess each document
            for doc_idx in range(len(doc_tensors)):
                chunk_id = doc_idx + 1
                preprocess_key_path = f"{preprocess_save_path}/{example_id}_{chunk_id}_key.pt"

                if os.path.exists(preprocess_key_path):
                    continue

                print(f"  Preprocessing document {chunk_id}/{len(doc_tensors)} with FusionRAG...")

                # Show retrieved similar documents
                if len(context_rank) > 0:
                    global_doc_idx = sum(corpus_lens[:example_id]) + doc_idx
                    if global_doc_idx < len(context_rank):
                        similar_docs_info = []
                        for similar_global_idx in context_rank[global_doc_idx][:topk]:
                            # Skip invalid indices (from padding in PER_EXAMPLE mode)
                            if similar_global_idx < 0:
                                continue
                            if similar_global_idx == global_doc_idx:
                                continue
                            corpus_i, c_id = find_group_and_index(corpus_lens, similar_global_idx)
                            similar_chunk_id = c_id + 1
                            similar_cache_key_path = f"{save_path}/{corpus_i}_{similar_chunk_id}_key.pt"
                            cache_exists = os.path.exists(similar_cache_key_path)
                            status = "✓ cached" if cache_exists else "✗ need generate"
                            similar_docs_info.append(f"Q{corpus_i+1}-Doc{similar_chunk_id} ({status})")

                        if similar_docs_info:
                            print(f"    Retrieved similar docs: {', '.join(similar_docs_info)}")


                # STEP 1: Check and generate all required similar documents' cache FIRST
                # (to avoid past_key_values corruption during on-demand generation)
                if len(context_rank) > 0:
                    global_doc_idx = sum(corpus_lens[:example_id]) + doc_idx

                    if global_doc_idx < len(context_rank):
                        for similar_global_idx in context_rank[global_doc_idx][:topk]:
                            # Skip invalid indices (from padding in PER_EXAMPLE mode)
                            if similar_global_idx < 0:
                                continue
                            if similar_global_idx == global_doc_idx:
                                continue

                            corpus_i, c_id = find_group_and_index(corpus_lens, similar_global_idx)
                            similar_chunk_id = c_id + 1

                            # Check and generate if needed
                            similar_cache_key_path = f"{save_path}/{corpus_i}_{similar_chunk_id}_key.pt"
                            if not os.path.exists(similar_cache_key_path):
                                print(f"      → On-demand: Generating cache for Q{corpus_i+1}-Doc{similar_chunk_id}...")

                                # Generate system cache for that question if needed
                                other_system_cache_path = f'{save_path}/{corpus_i}_0_key.pt'
                                if not os.path.exists(other_system_cache_path):
                                    other_input = system_tensor.unsqueeze(0)
                                    prefill_and_save_kv_cache(
                                        model, tokenizer, past_key_values, other_input.to(input_device),
                                        save_path=save_path, example_id=corpus_i, chunk_id=0,
                                        system_len=system_len, passage_len=system_len,
                                        reprocess_method=reprocess_method, device=input_device, device_map=device_map
                                    )

                                # Generate the document cache
                                similar_doc_tensor = questions_data[corpus_i]['doc_tensors'][c_id]
                                other_passage_len = similar_doc_tensor.shape[0]
                                other_input = torch.cat((system_tensor, similar_doc_tensor)).unsqueeze(0)

                                prefill_and_save_kv_cache(
                                    model, tokenizer, past_key_values, other_input.to(input_device),
                                    save_path=save_path, example_id=corpus_i, chunk_id=similar_chunk_id,
                                    system_len=system_len, passage_len=other_passage_len,
                                    reprocess_method=reprocess_method, device=input_device, device_map=device_map
                                )

                # STEP 2: Now load all required cache into past_key_values
                # Reset cache
                past_len = 0
                for layer_idx in range(len(past_key_values.key_cache)):
                    past_key_values.past_tokens[layer_idx] = 0

                # Load system KV cache
                corpus_passages = [system_tensor]
                system_key_cache = torch.load(f"{save_path}/{example_id}_0_key.pt", weights_only=True)
                system_value_cache = torch.load(f"{save_path}/{example_id}_0_value.pt", weights_only=True)

                for layer_idx in range(len(past_key_values.key_cache)):
                    past_key_values.key_cache[layer_idx].narrow(2, 0, system_len).copy_(system_key_cache[layer_idx])
                    past_key_values.value_cache[layer_idx].narrow(2, 0, system_len).copy_(system_value_cache[layer_idx])
                    past_key_values.past_tokens[layer_idx] += system_len
                past_len += system_len

                # Load topk similar documents' KV cache
                if len(context_rank) > 0:
                    global_doc_idx = sum(corpus_lens[:example_id]) + doc_idx

                    if global_doc_idx < len(context_rank):
                        for similar_global_idx in context_rank[global_doc_idx][:topk]:
                            # Skip invalid indices (from padding in PER_EXAMPLE mode)
                            if similar_global_idx < 0:
                                continue
                            if similar_global_idx == global_doc_idx:
                                continue

                            corpus_i, c_id = find_group_and_index(corpus_lens, similar_global_idx)
                            similar_chunk_id = c_id + 1

                            # Load the similar document's cache (now guaranteed to exist)
                            similar_doc_tensor = questions_data[corpus_i]['doc_tensors'][c_id]
                            corpus_len = similar_doc_tensor.shape[0]
                            corpus_passages.append(similar_doc_tensor)

                            chunk_key_cache = torch.load(f"{save_path}/{corpus_i}_{similar_chunk_id}_key.pt", weights_only=True)
                            chunk_value_cache = torch.load(f"{save_path}/{corpus_i}_{similar_chunk_id}_value.pt", weights_only=True)

                            # Copy to past_key_values
                            for layer_idx in range(len(past_key_values.key_cache)):
                                past_key_values.key_cache[layer_idx].narrow(2, past_len, corpus_len).copy_(chunk_key_cache[layer_idx])
                                past_key_values.value_cache[layer_idx].narrow(2, past_len, corpus_len).copy_(chunk_value_cache[layer_idx])
                                past_key_values.past_tokens[layer_idx] += corpus_len
                            past_len += corpus_len

                # Add current document
                corpus_passages.append(doc_tensors[doc_idx])

                # Preprocess with fused KV cache
                prefill_with_cache_and_save_preprocess(
                    model, tokenizer, past_key_values, corpus_passages,
                    preprocess_save_path, example_id, chunk_id,
                    system_len=system_len, revert_rope=revert_rope,
                    reprocess_method=reprocess_method, device=input_device, device_map=device_map
                )

        # Step 3: Answer sub-questions
        # 使用线程池异步判断，主线程继续生成下一个答案
        judge_futures = []  # 存放判断任务的 Future 对象
        sub_q_results = []  # 存放每个 sub-question 的结果（用于后续统计和 CSV 写入）

        for sub_q_idx, sub_q_info in enumerate(q_data['sub_questions']):
            print(f"\nSub-question {sub_q_idx+1}/{len(q_data['sub_questions'])}")
            print(f"Question: {sub_q_info['query']}")
            print(f"Ground Truth: {sub_q_info['answer']}")

            # Build tokens: system + docs + question
            # Add /no_think for Qwen3 models to disable chain-of-thought
            if long_decode:
                # long_decode 模式：要求输出答案和支撑材料
                long_decode_format = """请按照以下格式回答问题：
                答案: [你的答案]
                支撑材料: [从文档中找到支持答案的关键句子或段落]

                """
                if model_type == 'qwen3':
                    question_text = f"<|im_end|>\n<|im_start|>user\n/no_think\n{long_decode_format}Question: {sub_q_info['query']}<|im_end|>\n<|im_start|>assistant\n"
                else:
                    question_text = f"<|im_end|>\n<|im_start|>user\n{long_decode_format}Question: {sub_q_info['query']}<|im_end|>\n<|im_start|>assistant\n"
            else:
                if model_type == 'qwen3':
                    question_text = f"<|im_end|>\n<|im_start|>user\n/no_think\nQuestion: {sub_q_info['query']}<|im_end|>\n<|im_start|>assistant\nAnswer: "
                else:
                    question_text = f"<|im_end|>\n<|im_start|>user\nQuestion: {sub_q_info['query']}<|im_end|>\n<|im_start|>assistant\nAnswer: "
            question_tokens = tokenizer.encode(question_text, add_special_tokens=False)
            question_tensor = torch.tensor(question_tokens, dtype=torch.long)

            # Get documents for this sub-question using chunk_ids
            doc_chunk_ids = sub_q_info['chunk_ids']  # List of chunk_ids (1-indexed) for documents
            sub_q_doc_tensors = [doc_tensors[chunk_id - 1] for chunk_id in doc_chunk_ids]  # Convert to 0-indexed

            iter_tokens = [system_tensor] + sub_q_doc_tensors + [question_tensor]

            # Prepare chunk_ids for load_kv_and_generate: [0, chunk_id1, chunk_id2, ...]
            # chunk_id 0 is system, then the actual document chunk_ids
            kv_chunk_ids = [0] + doc_chunk_ids

            # Generate answer using this main question's KV cache
            # long_decode 模式使用更多 tokens
            current_max_new_tokens = long_decode_max_tokens if long_decode else 500

            if rate == 1:
                # Full recompute
                inputs = torch.cat(iter_tokens).to(input_device).unsqueeze(0)
                from ktransformers.util.utils import prefill_and_generate
                generated_tokens, _, _ = prefill_and_generate(
                    model, tokenizer, inputs, max_new_tokens=current_max_new_tokens, device=input_device, device_map=device_map
                )
            else:
                # Load preprocessed KV cache and generate (FusionRAG, QueryAttention, DraftModel, Oracle, vAttention, OracleDynamic, etc.)
                load_path = preprocess_save_path if preprocess else save_path
                generated_tokens, _, extra_info = load_kv_and_generate(
                    model, tokenizer, past_key_values, iter_tokens, load_path, example_id,
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
                    preprocess=preprocess, device=input_device, chunk_ids=kv_chunk_ids, device_map=device_map,
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
                    original_kv_path=save_path if preprocess else None
                )

                # 收集 OracleDynamic/OracleAdaptive/DynamicDraftModel/DraftModelDynamic/DraftModelLayerwise 的动态 rate 信息
                if reprocess_method in ('OracleDynamic', 'OracleAdaptive', 'DynamicDraftModel', 'DraftModelDynamic', 'DraftModelLayerwise') and extra_info.get('dynamic_rate') is not None:
                    dynamic_rate_stats.append({
                        'main_q_idx': example_id + 1,
                        'sub_q_idx': sub_q_idx + 1,
                        'question': sub_q_info['query'][:50] + '...',
                        'dynamic_rate': extra_info['dynamic_rate'],
                        # 兼容不同方法的 coverage 字段名
                        'topk_coverage': extra_info.get('topk_coverage') or extra_info.get('coverage_50_ratio', 0),
                        'topk_count_for_coverage': extra_info.get('topk_count_for_coverage', 0),
                        'normalized_entropy': extra_info.get('normalized_entropy', 0),
                        'doc_len': extra_info.get('doc_len', 0),
                        'total_budget': extra_info.get('total_budget', 0)
                    })

            # Decode answer
            raw_output = tokenizer.decode(torch.tensor(generated_tokens[:-1]), skip_special_tokens=True)
            raw_output = raw_output.strip() if raw_output else ""

            # long_decode 模式：解析输出，提取答案和支撑材料
            if long_decode:
                answer, evidence = parse_long_decode_output(raw_output)
                if not answer:
                    answer = "[EMPTY]"
                    print(f"Predicted Answer: {answer} (WARNING: empty answer)")
                else:
                    print(f"Predicted Answer: {answer}")
                print(f"Extracted Evidence: {evidence[:200]}..." if len(evidence) > 200 else f"Extracted Evidence: {evidence}")
            else:
                answer = raw_output
                evidence = ""
                if not answer:
                    answer = "[EMPTY]"
                    print(f"Predicted: {answer} (WARNING: empty answer)")
                else:
                    print(f"Predicted: {answer}")

            # Compute F1 and EM (这个很快，同步计算)
            try:
                f1_score = compute_f1(answer, sub_q_info['answer'], tokenizer)
                em_score = 1.0 if _exact_match_score(answer, sub_q_info['answer']) else 0.0
            except Exception as e:
                print(f"Warning: F1/EM computation failed: {e}")
                f1_score = 0.0
                em_score = 0.0
            print(f"F1: {f1_score:.4f}, EM: {em_score:.4f}")

            # 提交判断任务到线程池（异步执行，不阻塞主线程）
            future = judge_executor.submit(
                judge_answer_with_openai,
                openai_client, openai_model,
                sub_q_info['query'], answer, sub_q_info['answer']
            )

            # long_decode 模式：提交支撑材料判断任务
            evidence_future = None
            if long_decode and evidence:
                # 获取当前子问题使用的检索文档
                retrieve_docs_for_subq = [q_data['docs'][chunk_id - 1] for chunk_id in doc_chunk_ids]
                evidence_future = judge_executor.submit(
                    judge_evidence_with_openai,
                    openai_client, openai_model,
                    sub_q_info['query'], evidence, q_data['gold_docs'],
                    retrieve_docs_for_subq  # 传入完整检索上下文
                )
            judge_futures.append(future)

            # 保存结果信息，等待判断完成后更新
            sub_q_results.append({
                'sub_q_idx': sub_q_idx,
                'sub_q_info': sub_q_info,
                'answer': answer,
                'f1_score': f1_score,
                'em_score': em_score,
                'future': future,
                'dynamic_rate': extra_info.get('dynamic_rate') if rate != 1 else None,
                'dispersion_score': extra_info.get('dispersion_score') if rate != 1 else None,
                # long_decode 模式额外字段
                'evidence': evidence if long_decode else None,
                'evidence_future': evidence_future if long_decode else None,
            })

            torch.cuda.empty_cache()

        # 等待该 main question 的所有判断任务完成
        print(f"\n⏳ Waiting for {len(judge_futures)} judgment(s) to complete...")
        all_sub_correct = True

        for result in sub_q_results:
            future = result['future']
            is_correct, judge_reason = future.result()  # 阻塞等待结果

            sub_q_info = result['sub_q_info']
            answer = result['answer']
            f1_score = result['f1_score']
            em_score = result['em_score']
            print(f"Sub-Q {result['sub_q_idx']+1}: {'✓ CORRECT' if is_correct else '✗ INCORRECT'} - {sub_q_info['query'][:50]}...")

            total_sub_questions += 1
            total_f1 += f1_score
            total_em += em_score
            if is_correct:
                correct_sub_questions += 1
            else:
                all_sub_correct = False

            # long_decode 模式：获取 evidence 判断结果
            evidence_matched = False
            evidence_reason = ""
            if long_decode:
                evidence = result.get('evidence', '')
                evidence_future = result.get('evidence_future')
                if evidence_future:
                    evidence_matched, evidence_reason = evidence_future.result()
                    print(f"  Evidence: {'✓ MATCHED' if evidence_matched else '✗ NOT MATCHED'}")
                    total_evidence += 1
                    if evidence_matched:
                        matched_evidence += 1
                else:
                    evidence_reason = "No evidence extracted"

            # Save to CSV
            with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if long_decode:
                    # long_decode 模式的 CSV 写入
                    evidence = result.get('evidence', '')
                    # 将 gold_docs 列表转换为字符串（用 ||| 分隔）
                    gold_docs_str = ' ||| '.join(q_data.get('gold_docs', []))
                    if rate != 1:
                        key = (q_data['main_question'], sub_q_info['query'])
                        rate1_data = rate1_results.get(key, {
                            'predicted': 'N/A',
                            'correct': 'N/A',
                            'f1': 'N/A',
                            'em': 'N/A',
                            'reason': 'N/A',
                            'evidence': 'N/A',
                            'evidence_matched': 'N/A',
                            'evidence_reason': 'N/A'
                        })
                        if reprocess_method in ('OracleDynamic', 'OracleAdaptive', 'DynamicDraftModel', 'DraftModelDynamic', 'DraftModelLayerwise'):
                            dynamic_rate = result.get('dynamic_rate', 'N/A')
                            dispersion_score = result.get('dispersion_score', 'N/A')
                            writer.writerow([
                                q_data['main_question'], sub_q_info['query'],
                                sub_q_info['answer'], gold_docs_str,
                                answer, is_correct, f1_score, em_score, judge_reason,
                                evidence, evidence_matched, evidence_reason,
                                dynamic_rate, dispersion_score,
                                rate1_data['predicted'], rate1_data['correct'],
                                rate1_data['f1'], rate1_data['em'], rate1_data['reason'],
                                rate1_data.get('evidence', 'N/A'),
                                rate1_data.get('evidence_matched', 'N/A'),
                                rate1_data.get('evidence_reason', 'N/A')
                            ])
                        else:
                            writer.writerow([
                                q_data['main_question'], sub_q_info['query'],
                                sub_q_info['answer'], gold_docs_str,
                                answer, is_correct, f1_score, em_score, judge_reason,
                                evidence, evidence_matched, evidence_reason,
                                rate1_data['predicted'], rate1_data['correct'],
                                rate1_data['f1'], rate1_data['em'], rate1_data['reason'],
                                rate1_data.get('evidence', 'N/A'),
                                rate1_data.get('evidence_matched', 'N/A'),
                                rate1_data.get('evidence_reason', 'N/A')
                            ])
                    else:
                        writer.writerow([
                            q_data['main_question'], sub_q_info['query'],
                            sub_q_info['answer'], gold_docs_str,
                            answer, is_correct, f1_score, em_score, judge_reason,
                            evidence, evidence_matched, evidence_reason
                        ])
                else:
                    # 原有逻辑
                    if rate != 1:
                        # Get rate=1 results for comparison
                        key = (q_data['main_question'], sub_q_info['query'])
                        rate1_data = rate1_results.get(key, {
                            'predicted': 'N/A',
                            'correct': 'N/A',
                            'f1': 'N/A',
                            'em': 'N/A',
                            'reason': 'N/A'
                        })
                        # 对于动态 rate 方法，额外保存 dynamic_rate 和 dispersion_score
                        if reprocess_method in ('OracleDynamic', 'OracleAdaptive', 'DynamicDraftModel', 'DraftModelDynamic', 'DraftModelLayerwise'):
                            dynamic_rate = result.get('dynamic_rate', 'N/A')
                            dispersion_score = result.get('dispersion_score', 'N/A')
                            writer.writerow([
                                q_data['main_question'], sub_q_info['query'],
                                sub_q_info['answer'], answer, is_correct, f1_score, em_score, judge_reason,
                                dynamic_rate, dispersion_score,
                                rate1_data['predicted'], rate1_data['correct'],
                                rate1_data['f1'], rate1_data['em'], rate1_data['reason']
                            ])
                        else:
                            writer.writerow([
                                q_data['main_question'], sub_q_info['query'],
                                sub_q_info['answer'], answer, is_correct, f1_score, em_score, judge_reason,
                                rate1_data['predicted'], rate1_data['correct'],
                                rate1_data['f1'], rate1_data['em'], rate1_data['reason']
                            ])
                    else:
                        writer.writerow([
                            q_data['main_question'], sub_q_info['query'],
                            sub_q_info['answer'], answer, is_correct, f1_score, em_score, judge_reason
                        ])

        # Main question result
        total_main_questions += 1
        if all_sub_correct:
            correct_main_questions += 1
            print(f"\n✓ Main question {example_id+1}: ALL {len(q_data['sub_questions'])} sub-questions CORRECT")
        else:
            print(f"\n✗ Main question {example_id+1}: Some sub-questions INCORRECT")

    # Final results
    main_q_acc = correct_main_questions / total_main_questions if total_main_questions > 0 else 0
    sub_q_acc = correct_sub_questions / total_sub_questions if total_sub_questions > 0 else 0
    avg_f1 = total_f1 / total_sub_questions if total_sub_questions > 0 else 0
    avg_em = total_em / total_sub_questions if total_sub_questions > 0 else 0

    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}")
    print(f"Main Questions: {correct_main_questions}/{total_main_questions} ({main_q_acc:.2%})")
    print(f"Sub Questions: {correct_sub_questions}/{total_sub_questions} ({sub_q_acc:.2%})")
    print(f"Average F1: {avg_f1:.4f}")
    print(f"Average EM: {avg_em:.4f}")

    # long_decode 模式：输出 evidence 统计
    if long_decode and total_evidence > 0:
        evidence_acc = matched_evidence / total_evidence
        print(f"Evidence Matched: {matched_evidence}/{total_evidence} ({evidence_acc:.2%})")

    # Show comparison with rate=1 if applicable
    if rate != 1 and len(rate1_results) > 0:
        # Calculate rate=1 statistics
        rate1_correct = sum(1 for v in rate1_results.values() if v['correct'].lower() == 'true')
        rate1_total = len(rate1_results)
        rate1_acc = rate1_correct / rate1_total if rate1_total > 0 else 0
        rate1_avg_f1 = sum(float(v['f1']) for v in rate1_results.values()) / rate1_total if rate1_total > 0 else 0
        rate1_avg_em = sum(float(v['em']) for v in rate1_results.values()) / rate1_total if rate1_total > 0 else 0

        print(f"\n{'='*80}")
        print("COMPARISON WITH RATE=1 BASELINE")
        print(f"{'='*80}")
        print(f"Current (rate={rate}):")
        print(f"  Sub Questions Accuracy: {sub_q_acc:.2%}, F1: {avg_f1:.4f}, EM: {avg_em:.4f}")
        print(f"Baseline (rate=1):")
        print(f"  Sub Questions Accuracy: {rate1_acc:.2%}, F1: {rate1_avg_f1:.4f}, EM: {rate1_avg_em:.4f}")
        print(f"Delta:")
        print(f"  Accuracy: {sub_q_acc - rate1_acc:+.2%}, F1: {avg_f1 - rate1_avg_f1:+.4f}, EM: {avg_em - rate1_avg_em:+.4f}")

        # long_decode 模式：输出 evidence 对比
        if long_decode and total_evidence > 0:
            # 计算 rate=1 的 evidence 准确率
            rate1_evidence_matched = sum(1 for v in rate1_results.values() if str(v.get('evidence_matched', '')).lower() == 'true')
            rate1_evidence_total = sum(1 for v in rate1_results.values() if v.get('evidence_matched', 'N/A') != 'N/A')
            rate1_evidence_acc = rate1_evidence_matched / rate1_evidence_total if rate1_evidence_total > 0 else 0
            evidence_acc = matched_evidence / total_evidence

            print(f"\nEvidence Comparison:")
            print(f"  Current (rate={rate}): {matched_evidence}/{total_evidence} ({evidence_acc:.2%})")
            print(f"  Baseline (rate=1): {rate1_evidence_matched}/{rate1_evidence_total} ({rate1_evidence_acc:.2%})")
            print(f"  Evidence Delta: {evidence_acc - rate1_evidence_acc:+.2%}")

        print(f"{'='*80}")

    print(f"{'='*80}")

    # 打印 OracleDynamic/OracleAdaptive/DynamicDraftModel 动态 rate 统计
    if reprocess_method in ('OracleDynamic', 'OracleAdaptive', 'DynamicDraftModel') and len(dynamic_rate_stats) > 0:
        print(f"\n{'='*80}")
        print(f"{reprocess_method.upper()}: 动态重算比例统计")
        print(f"{'='*80}")

        rates = [s['dynamic_rate'] for s in dynamic_rate_stats]
        entropies = [s['normalized_entropy'] for s in dynamic_rate_stats]
        topk_coverages = [s['topk_coverage'] for s in dynamic_rate_stats]

        print(f"\n总计 {len(dynamic_rate_stats)} 个子问题:")
        print(f"  重算比例 - 平均: {np.mean(rates):.2%}, 最小: {np.min(rates):.2%}, 最大: {np.max(rates):.2%}, 标准差: {np.std(rates):.2%}")
        print(f"  归一化熵 - 平均: {np.mean(entropies):.4f}, 最小: {np.min(entropies):.4f}, 最大: {np.max(entropies):.4f}")
        print(f"  Top-k覆盖 - 平均: {np.mean(topk_coverages):.2%}, 最小: {np.min(topk_coverages):.2%}, 最大: {np.max(topk_coverages):.2%}")

        # 打印每个问题的详细信息
        print(f"\n详细列表:")
        print(f"{'Main Q':<8} {'Sub Q':<8} {'Rate':<10} {'Entropy':<10} {'TopK Cov':<10} {'Budget':<10} {'Doc Len':<10} Question")
        print("-" * 120)
        for stat in dynamic_rate_stats:
            print(f"{stat['main_q_idx']:<8} {stat['sub_q_idx']:<8} {stat['dynamic_rate']:.2%}     {stat['normalized_entropy']:<10.4f} {stat['topk_coverage']:.2%}     {stat['total_budget']:<10} {stat['doc_len']:<10} {stat['question']}")

        print(f"{'='*80}")

    with open(result_file, 'w') as f:
        # 保存所有配置参数
        f.write("=" * 80 + "\n")
        f.write("CONFIGURATION\n")
        f.write("=" * 80 + "\n")
        f.write(f"model_type: {model_type}\n")
        f.write(f"model_path: {model_path}\n")
        f.write(f"draft_model_path: {draft_model_path}\n")
        f.write(f"data_path: {data_path}\n")
        f.write(f"cache_path: {cache_path}\n")
        f.write(f"model_name: {model_name}\n")
        f.write(f"dataset_name: {dataset_name}\n")
        f.write(f"max_cache_len: {max_cache_len}\n")
        f.write(f"rate: {rate}\n")
        f.write(f"topk: {topk}\n")
        f.write(f"preprocess: {preprocess}\n")
        f.write(f"preprocess_scope: {preprocess_scope}\n")
        f.write(f"reprocess_method: {reprocess_method}\n")
        f.write(f"use_entropy_selection: {use_entropy_selection}\n")
        f.write(f"entropy_top_k: {entropy_top_k}\n")
        f.write(f"draft_layer_selection: {draft_layer_selection}\n")
        f.write(f"draft_fixed_layer: {draft_fixed_layer}\n")
        f.write(f"draft_threshold_factor: {draft_threshold_factor}\n")
        f.write(f"bge_model_path: {bge_model_path}\n")
        f.write(f"revert_rope: {revert_rope}\n")
        f.write(f"device: {device}\n")
        f.write(f"use_multi_gpu: {use_multi_gpu}\n")
        f.write(f"openai_base_url: {openai_base_url}\n")
        f.write(f"openai_model: {openai_model}\n")
        f.write(f"max_samples: {max_samples}\n")
        f.write(f"vattention_topk_ratio: {vattention_topk_ratio}\n")
        f.write(f"epsilon: {epsilon}\n")
        f.write(f"delta: {delta}\n")
        f.write(f"min_rate: {min_rate}\n")
        f.write(f"max_rate: {max_rate}\n")
        f.write(f"layerwise_decay: {layerwise_decay}\n")
        f.write(f"layerwise_final_rate: {layerwise_final_rate}\n")
        f.write(f"use_similarity_rerank: {use_similarity_rerank}\n")
        f.write(f"rerank_multiplier: {rerank_multiplier}\n")
        f.write(f"long_decode: {long_decode}\n")
        f.write(f"long_decode_max_tokens: {long_decode_max_tokens}\n")
        f.write("\n")

        # 保存结果
        f.write("=" * 80 + "\n")
        f.write("RESULTS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Main Questions Accuracy: {correct_main_questions}/{total_main_questions} ({main_q_acc:.4f})\n")
        f.write(f"Sub Questions Accuracy: {correct_sub_questions}/{total_sub_questions} ({sub_q_acc:.4f})\n")
        f.write(f"Average F1 Score: {avg_f1:.4f}\n")
        f.write(f"Average EM Score: {avg_em:.4f}\n")

        # long_decode 模式的 evidence 统计
        if long_decode and total_evidence > 0:
            evidence_acc = matched_evidence / total_evidence
            f.write(f"Evidence Matched: {matched_evidence}/{total_evidence} ({evidence_acc:.4f})\n")

        # 保存 OracleDynamic/OracleAdaptive/DynamicDraftModel 统计到文件
        if reprocess_method in ('OracleDynamic', 'OracleAdaptive', 'DynamicDraftModel') and len(dynamic_rate_stats) > 0:
            rates = [s['dynamic_rate'] for s in dynamic_rate_stats]
            entropies = [s['normalized_entropy'] for s in dynamic_rate_stats]
            topk_coverages = [s['topk_coverage'] for s in dynamic_rate_stats]
            f.write(f"\n--- {reprocess_method} 动态重算比例统计 ---\n")
            f.write(f"子问题数量: {len(dynamic_rate_stats)}\n")
            f.write(f"重算比例 - 平均: {np.mean(rates):.4f}, 最小: {np.min(rates):.4f}, 最大: {np.max(rates):.4f}, 标准差: {np.std(rates):.4f}\n")
            f.write(f"归一化熵 - 平均: {np.mean(entropies):.4f}, 最小: {np.min(entropies):.4f}, 最大: {np.max(entropies):.4f}\n")
            f.write(f"Top-k覆盖 - 平均: {np.mean(topk_coverages):.4f}, 最小: {np.min(topk_coverages):.4f}, 最大: {np.max(topk_coverages):.4f}\n")

        # Add rate=1 comparison to file
        if rate != 1 and len(rate1_results) > 0:
            rate1_correct = sum(1 for v in rate1_results.values() if v['correct'].lower() == 'true')
            rate1_total = len(rate1_results)
            rate1_acc = rate1_correct / rate1_total if rate1_total > 0 else 0
            rate1_avg_f1 = sum(float(v['f1']) for v in rate1_results.values()) / rate1_total if rate1_total > 0 else 0
            rate1_avg_em = sum(float(v['em']) for v in rate1_results.values()) / rate1_total if rate1_total > 0 else 0

            f.write(f"\n--- Comparison with Rate=1 Baseline ---\n")
            f.write(f"Rate=1 Sub Questions Accuracy: {rate1_acc:.4f}\n")
            f.write(f"Rate=1 Average F1 Score: {rate1_avg_f1:.4f}\n")
            f.write(f"Rate=1 Average EM Score: {rate1_avg_em:.4f}\n")
            f.write(f"Accuracy Delta: {sub_q_acc - rate1_acc:+.4f}\n")
            f.write(f"F1 Delta: {avg_f1 - rate1_avg_f1:+.4f}\n")
            f.write(f"EM Delta: {avg_em - rate1_avg_em:+.4f}\n")

            # long_decode 模式：添加 evidence 对比到文件
            if long_decode and total_evidence > 0:
                rate1_evidence_matched = sum(1 for v in rate1_results.values() if str(v.get('evidence_matched', '')).lower() == 'true')
                rate1_evidence_total = sum(1 for v in rate1_results.values() if v.get('evidence_matched', 'N/A') != 'N/A')
                rate1_evidence_acc = rate1_evidence_matched / rate1_evidence_total if rate1_evidence_total > 0 else 0
                evidence_acc = matched_evidence / total_evidence

                f.write(f"\n--- Evidence Comparison ---\n")
                f.write(f"Current Evidence Matched: {matched_evidence}/{total_evidence} ({evidence_acc:.4f})\n")
                f.write(f"Rate=1 Evidence Matched: {rate1_evidence_matched}/{rate1_evidence_total} ({rate1_evidence_acc:.4f})\n")
                f.write(f"Evidence Delta: {evidence_acc - rate1_evidence_acc:+.4f}\n")

    # 关闭线程池
    judge_executor.shutdown(wait=True)

    print(f"\nResults saved to {csv_path}")


def collect_optimal_rate(
    model_type='qwen',
    model_path='/mnt/data/models/Qwen2.5-7B-Instruct',
    draft_model_path='/mnt/data/models/Qwen2.5-3B-Instruct',
    data_path='/mnt/data/wjh/FusionRAG/data/result_reflect.json',
    cache_path='/mnt/data/reflect/',
    model_name='Qwen2.5-7B-Instruct',
    dataset_name='musique',
    max_cache_len=32768,
    topk=10,
    preprocess=True,
    preprocess_scope=PreprocessScope.GLOBAL,
    revert_rope=True,
    bge_model_path='/mnt/data/models/bge-m3-FP16',
    device="cuda:0",
    use_multi_gpu=False,
    openai_api_key="sk-519d391217894b6e91e7c2ebf2a9f4df",
    openai_base_url="https://api.deepseek.com/v1",
    openai_model="deepseek-chat",
    max_samples=None,
    output_dir="/mnt/data/wjh/FusionRAG/optimal_rate_search",
):
    """
    收集每个问题在 DraftModel 方法下的最小正确重算比例和 attention 分布特征

    使用早停策略：从低到高尝试 rates，一旦答对就停止搜索
    """
    import pickle
    from ktransformers.util.utils import (
        compute_draft_model_attention,
        entropy_layer_selection,
        prefill_and_generate,
    )

    # 打印配置
    print("="*80)
    print("Configuration - Collect Optimal Rate Data")
    print("="*80)
    print(f"Main Model: {model_path}")
    print(f"Draft Model: {draft_model_path}")
    print(f"Data Path: {data_path}")
    print(f"Output Dir: {output_dir}")
    print(f"Reprocess Method: DraftModel")
    print(f"Draft Layer Selection: entropy")
    print(f"Preprocess: {preprocess}")
    print(f"Revert RoPE: {revert_rope}")
    print(f"TopK: {topk}")
    print(f"Judgment: DeepSeek API ({openai_model})")
    print("="*80)
    print()

    os.makedirs(output_dir, exist_ok=True)

    # 要尝试的 rates: 0, 0.05, 0.10, ..., 1.0
    rates_to_try = [round(r * 0.05, 2) for r in range(21)]
    print(f"Rates to try: {rates_to_try}")

    # 缓存路径
    model_cache_root = os.path.join(cache_path, model_name, dataset_name)
    save_path = os.path.join(model_cache_root, 'kv_cache')
    preprocess_save_path = os.path.join(model_cache_root, 'preprocess_kv_cache_global')

    # 加载模型
    print("\n加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config._attn_implementation = "sdpa"

    model, device_map = load_model(model_type, model_path, config, device, use_multi_gpu)
    model.eval()

    print(f"加载 Draft 模型: {draft_model_path}")
    draft_config = AutoConfig.from_pretrained(draft_model_path, trust_remote_code=True)
    draft_config._attn_implementation = "sdpa"
    draft_model, _ = load_model('qwen', draft_model_path, draft_config, device, use_multi_gpu=False)
    draft_model.eval()

    # 准备数据
    print("\n准备数据...")
    questions_data, system_tensor, context_rank, corpus_lens = prepare_reflect_data(
        data_path, tokenizer, bge_model_path, model_type, topk, max_samples, preprocess, preprocess_scope
    )
    system_len = system_tensor.shape[0]

    # 初始化 KV cache
    past_key_values = StaticCache(
        config=config, max_batch_size=1, max_cache_len=max_cache_len,
        device=device, dtype=config.torch_dtype, passage_len=max_cache_len
    )

    # 初始化 OpenAI client
    openai_client = OpenAI(api_key=openai_api_key, base_url=openai_base_url)

    # 收集结果
    all_results = []
    total_questions = sum(len(q['sub_questions']) for q in questions_data)
    processed = 0

    input_device = device

    print(f"\n开始处理 {total_questions} 个子问题...")
    print("="*80)

    for example_id, q_data in enumerate(questions_data):
        doc_tensors = q_data['doc_tensors']

        for sub_q_idx, sub_q_info in enumerate(q_data['sub_questions']):
            processed += 1
            ground_truth = sub_q_info['answer']
            query = sub_q_info['query']
            chunk_ids = sub_q_info['chunk_ids']

            question_id = f"Q{example_id+1}_Sub{sub_q_idx+1}"

            print(f"\n{'='*60}")
            print(f"[{processed}/{total_questions}] {question_id}")
            print(f"Query: {query}")
            print(f"Ground Truth: {ground_truth}")
            print(f"{'='*60}")

            # 构建输入
            sub_q_doc_tensors = [doc_tensors[cid - 1] for cid in chunk_ids]
            question_text = f"<|im_end|>\n<|im_start|>user\nQuestion: {query}<|im_end|>\n<|im_start|>assistant\nAnswer: "
            question_tokens = tokenizer.encode(question_text, add_special_tokens=False)
            question_tensor = torch.tensor(question_tokens, dtype=torch.long)

            passages = [system_tensor] + sub_q_doc_tensors + [question_tensor]
            passages_len = [p.shape[0] for p in passages]
            kv_chunk_ids = [0] + chunk_ids

            # Step 0: 检查必要的缓存是否存在
            cache_missing = False
            # 检查预处理缓存
            for chunk_id in kv_chunk_ids:
                key_cache_path = f"{preprocess_save_path}/{example_id}_{chunk_id}_key.pt"
                if not os.path.exists(key_cache_path):
                    print(f"  跳过: 预处理缓存不存在 {key_cache_path}")
                    cache_missing = True
                    break
            # 检查原始 KV cache (chunk_id=1 用原始缓存)
            if not cache_missing and len(chunk_ids) > 0:
                orig_key_path = f"{save_path}/{example_id}_{chunk_ids[0]}_key.pt"
                if not os.path.exists(orig_key_path):
                    print(f"  跳过: 原始KV缓存不存在 {orig_key_path}")
                    cache_missing = True
            if cache_missing:
                continue

            # Step 1: 计算 Draft Model Attention 并提取特征
            query_start = sum(passages_len[:-1])
            full_input = torch.cat(passages).unsqueeze(0).to(device)

            try:
                draft_attention = compute_draft_model_attention(draft_model, full_input, query_start, device)
            except Exception as e:
                print(f"  计算 attention 失败: {e}")
                continue

            # 提取 attention 特征
            attention_features = extract_attention_features_for_rate(
                draft_attention, query_start, system_len, passages_len, device
            )
            if attention_features is None:
                print(f"  提取特征失败")
                continue

            # Step 2: 从低到高测试 rates，找到第一个正确的就停止
            rate_results = []
            min_correct_rate = None

            for rate in rates_to_try:
                # 重置 KV cache
                for layer_idx in range(len(past_key_values.key_cache)):
                    past_key_values.past_tokens[layer_idx] = 0

                try:
                    if rate == 1.0:
                        inputs = torch.cat(passages).to(input_device).unsqueeze(0)
                        generated_tokens, _, _ = prefill_and_generate(
                            model, tokenizer, inputs, max_new_tokens=100,
                            device=input_device, device_map=device_map
                        )
                    else:
                        generated_tokens, _, _ = load_kv_and_generate(
                            model, tokenizer, past_key_values, passages,
                            preprocess_save_path, example_id,
                            max_new_tokens=100, revert_rope=revert_rope,
                            reprocess_method='DraftModel', rate=max(rate, 0.001),
                            draft_model=draft_model,
                            draft_layer_selection='entropy',
                            preprocess=preprocess,
                            chunk_ids=kv_chunk_ids, device=input_device, device_map=device_map,
                            original_kv_path=save_path
                        )

                    answer = tokenizer.decode(torch.tensor(generated_tokens[:-1]), skip_special_tokens=True).strip()
                    if not answer:
                        answer = "[EMPTY]"

                    # 使用 DeepSeek API 判断答案正确性
                    is_correct, reason = judge_answer_with_openai(
                        openai_client, openai_model, query, answer, ground_truth
                    )

                    result = {
                        'rate': rate,
                        'answer': answer,
                        'correct': is_correct,
                        'reason': reason,
                        'error': None
                    }

                except Exception as e:
                    import traceback
                    result = {
                        'rate': rate,
                        'answer': None,
                        'correct': False,
                        'reason': str(e),
                        'error': traceback.format_exc()
                    }
                    is_correct = False
                    answer = f"[ERROR: {str(e)[:50]}]"

                rate_results.append(result)

                # 打印结果（完整答案，不截断）
                status = "✓ CORRECT" if is_correct else "✗ wrong"
                print(f"  rate={rate:.2f}: {status}")
                print(f"    Answer: {answer}")

                # 早停：答对就停止
                if is_correct:
                    min_correct_rate = rate
                    print(f"  >>> 找到 min_correct_rate = {rate:.2f}, 停止搜索")
                    break

            if min_correct_rate is None:
                print(f"  >>> 所有 rate 都答错")

            # 保存结果
            question_result = {
                'question_id': question_id,
                'example_id': example_id,
                'sub_q_idx': sub_q_idx,
                'query': query,
                'ground_truth': ground_truth,
                'attention_features': attention_features,
                'rate_results': rate_results,
                'min_correct_rate': min_correct_rate,
            }
            all_results.append(question_result)

            # 定期保存 checkpoint
            if len(all_results) % 20 == 0:
                checkpoint_path = os.path.join(output_dir, 'checkpoint.pkl')
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump(all_results, f)
                print(f"  [Checkpoint saved: {len(all_results)} questions]")

            torch.cuda.empty_cache()

    # 保存最终结果
    print("\n" + "="*80)
    print("保存结果")
    print("="*80)

    output_path = os.path.join(output_dir, 'full_results.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"完整数据已保存到: {output_path}")

    # 统计
    min_rates = [r['min_correct_rate'] for r in all_results if r['min_correct_rate'] is not None]
    failed = [r for r in all_results if r['min_correct_rate'] is None]

    print(f"\n总问题数: {len(all_results)}")
    print(f"有正确答案的问题数: {len(min_rates)}")
    print(f"所有 rate 都答错的问题数: {len(failed)}")

    if min_rates:
        print(f"\n最小正确 rate 分布:")
        print(f"  平均值: {np.mean(min_rates):.3f}")
        print(f"  中位数: {np.median(min_rates):.3f}")
        print(f"  最小值: {np.min(min_rates):.3f}")
        print(f"  最大值: {np.max(min_rates):.3f}")

    return all_results


def extract_attention_features_for_rate(draft_attention, query_start, system_len, passages_len, device="cuda:0"):
    """
    从 draft model attention 中提取特征用于分析
    """
    from ktransformers.util.utils import entropy_layer_selection

    text_block1_len = passages_len[1] if len(passages_len) > 1 else 0
    selection_start = system_len + text_block1_len
    doc_len = sum(passages_len[2:-1]) if len(passages_len) > 2 else 0

    if doc_len == 0:
        return None

    # 收集各层的 query→doc attention
    layer_attention_dict = {}
    for layer_idx, layer_attn in draft_attention.items():
        query_to_doc = layer_attn[:, :, selection_start:selection_start + doc_len]
        doc_attention_avg = query_to_doc.mean(axis=(0, 1))
        layer_attention_dict[layer_idx] = torch.tensor(doc_attention_avg, device=device)

    # 熵选层
    active_layers, layer_entropy = entropy_layer_selection(
        layer_attention_dict, top_k=4, return_entropy=True
    )
    layer_attentions = [layer_attention_dict[idx] for idx in active_layers]
    aggregated_attn = torch.stack(layer_attentions).mean(dim=0).cpu().numpy()

    # 计算特征
    features = {}

    # 基本统计
    features['peak_strength'] = float(aggregated_attn.max())
    features['attention_mean'] = float(aggregated_attn.mean())
    features['attention_std'] = float(aggregated_attn.std())

    # Top-k concentration
    sorted_attn = np.sort(aggregated_attn)[::-1]
    total = sorted_attn.sum()

    for k in [5, 10, 20, 50]:
        features[f'top{k}_concentration'] = float(sorted_attn[:k].sum() / total) if total > 0 else 0

    # Coverage ratios
    cumsum = np.cumsum(sorted_attn)
    for coverage in [0.5, 0.7, 0.8, 0.9, 0.95]:
        coverage_idx = np.where(cumsum >= coverage * total)[0]
        tokens_needed = coverage_idx[0] + 1 if len(coverage_idx) > 0 else len(aggregated_attn)
        features[f'coverage_{int(coverage*100)}_ratio'] = float(tokens_needed / len(aggregated_attn))

    # Normalized entropy
    p = aggregated_attn / (aggregated_attn.sum() + 1e-10)
    p = np.clip(p, 1e-10, 1.0)
    entropy = -(p * np.log(p)).sum()
    max_entropy = np.log(len(aggregated_attn))
    features['normalized_entropy'] = float(entropy / max_entropy) if max_entropy > 0 else 0

    # Gini coefficient
    sorted_p = np.sort(p)
    n = len(sorted_p)
    gini = (2 * np.sum(np.arange(1, n+1) * sorted_p) / (n * sorted_p.sum()) - (n + 1) / n)
    features['gini'] = float(gini)

    # 文档长度
    features['doc_len'] = doc_len
    features['log_doc_len'] = float(np.log(doc_len + 1))
    features['active_layers'] = active_layers

    # 保存原始分布供后续分析
    features['attention_distribution'] = aggregated_attn.tolist()

    return features


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
                        help='Use random document sampling instead of BGE similarity (True/False)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility (when use_random_recall=True)')
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
        draft_model_path=args.draft_model_path,
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
        random_seed=args.random_seed,
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