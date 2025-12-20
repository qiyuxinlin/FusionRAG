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
    find_group_and_index
)
from ktransformers.models.custom_cache import StaticCache


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


def prepare_reflect_data(
    data_path: str,
    tokenizer,
    bge_model_path: str,
    model_type: str = 'qwen2',
    topk: int = 10,
    max_main_questions: int = None,
    preprocess: bool = True
) -> Tuple[List, torch.Tensor, List, List]:
    """
    Prepare data from result_reflect.json with GLOBAL document corpus for preprocessing

    Similar to prepare_data in unified_process_cache.py:
    - All documents from all questions form a global corpus
    - BGE model computes similarity across the entire corpus
    - Each document can reference similar documents from ANY question

    Args:
        model_type: Type of model to determine system prompt
        topk: Top-k similar documents for each document
        preprocess: Whether to compute global context_rank

    Returns:
        questions_data: List of dicts for each main question
        system_tensor: Tokenized system prompt
        context_rank: [total_docs x topk] array of similar document indices (global)
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

    # STEP 1: Build GLOBAL document corpus across all questions
    print("\n" + "="*80)
    print("Building global document corpus...")
    print("="*80)

    global_corpus = []  # All documents from all questions (in order)
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
        if data_item.get('llm_judge', True) is False:
            should_test_main_question = False

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
            if "No relevant information found" in answer:
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

        # Add this question's docs to global corpus
        global_corpus.extend(question_docs)
        corpus_lens.append(len(question_docs))

        questions_data.append({
            'main_question': main_question,
            'main_answer': main_answer,
            'sub_questions': sub_questions_info,
            'docs': question_docs,
            'doc_tensors': doc_tensors,
            'should_test': should_test_main_question,  # Whether to test this main question
        })

    # Statistics
    total_main_q = len(questions_data)
    testable_main_q = sum(1 for q in questions_data if q['should_test'])
    skipped_main_q = total_main_q - testable_main_q

    total_sub_q = sum(len(q['sub_questions']) for q in questions_data)
    testable_sub_q = sum(len(q['sub_questions']) for q in questions_data if q['should_test'])
    skipped_sub_q = total_sub_q - testable_sub_q

    total_docs = sum(len(q['docs']) for q in questions_data)

    print(f"\n{'='*80}")
    print("DATASET STATISTICS")
    print(f"{'='*80}")
    print(f"Total main questions: {total_main_q}")
    print(f"  - Testable: {testable_main_q}")
    print(f"  - Skipped (llm_judge=False or problematic answers): {skipped_main_q}")
    print(f"\nTotal sub-questions: {total_sub_q}")
    print(f"  - Testable: {testable_sub_q}")
    print(f"  - Skipped: {skipped_sub_q}")
    print(f"\nTotal documents (across all questions): {total_docs}")
    print(f"{'='*80}")

    # STEP 2: Build FAISS index and compute global context_rank
    context_rank = []
    if preprocess and len(global_corpus) > 0:
        print("\n" + "="*80)
        print("Computing global document similarity with BGE + FAISS...")
        print("="*80)

        import faiss
        from FlagEmbedding import FlagModel

        # Load BGE model
        print(f"Loading BGE model from {bge_model_path}...")
        bgem3 = FlagModel(bge_model_path, use_fp16=True)

        # Encode global corpus for FAISS index
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

        # Search for similar documents globally
        print(f"Searching for top-{topk} similar documents for each document...")
        corpus_embeddings_query = bgem3.encode_queries(global_corpus)
        corpus_embeddings_query = corpus_embeddings_query.astype(np.float32)
        score, idx = index.search(corpus_embeddings_query, k=topk)
        context_rank = idx  # Shape: [total_docs, topk]

        print(f"Context rank computed: {context_rank.shape}")
        bgem3 = None  # Free memory

    return questions_data, system_tensor, context_rank, corpus_lens


def judge_answer_with_openai(
    openai_client: OpenAI,
    openai_model: str,
    question: str,
    predicted_answer: str,
    ground_truth_answer: str
) -> bool:
    """
    Use OpenAI API to judge if the predicted answer is correct
    """
    judge_prompt = f"""You are an answer evaluator. Your task is to determine if a predicted answer is correct based on the ground truth answer.

Question: {question}

Ground Truth Answer: {ground_truth_answer}

Predicted Answer: {predicted_answer}

Does the predicted answer correctly answer the question? Consider the answer correct if:
1. The predicted answer contains the key information from the ground truth
2. The predicted answer is semantically equivalent to the ground truth
3. Minor wording differences are acceptable as long as the meaning is preserved

Respond with only "YES" if correct or "NO" if incorrect."""

    try:
        response = openai_client.chat.completions.create(
            model=openai_model,
            messages=[
                {"role": "system", "content": "You are an answer evaluator."},
                {"role": "user", "content": judge_prompt}
            ],
            temperature=0,
            max_tokens=10
        )

        judgment = response.choices[0].message.content.strip().upper()
        return "YES" in judgment

    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return False


def main(
    model_type='qwen',
    model_path='/mnt/data/models/Qwen2.5-7B-Instruct',
    data_path='/mnt/data/ktransformers-dev/result_reflect.json',
    cache_path='/mnt/data3/reflect/',
    model_name='Qwen2.5-7B-Instruct',
    max_cache_len=32768,
    rate=0.2,
    topk=10,
    preprocess=True,
    reprocess_method='FusionRAG',
    bge_model_path='/mnt/data/models/bge-m3-FP16',
    revert_rope=True,
    device="cuda:0",
    use_multi_gpu=False,
    openai_api_key=None,
    openai_base_url="https://api.openai.com/v1",
    openai_model="gpt-4",
    max_samples=None
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
        rate: Compression rate (0=no compression, 1=full recompute)
        topk: Top-k similar documents to fuse in preprocess
        preprocess: Whether to use FusionRAG preprocess
        reprocess_method: Method name ('FusionRAG')
        bge_model_path: Path to BGE model for computing similarity
        revert_rope: Whether to revert rope in preprocessing
        device: Device to use (for single GPU)
        use_multi_gpu: Whether to use multi-GPU with device_map='auto'
        openai_api_key: OpenAI API key for judging answers
        openai_base_url: OpenAI API base URL
        openai_model: OpenAI model for judging
        max_samples: Maximum number of main questions to test (None = all)
    """

    # Create cache directories with model-specific subdirectories
    model_cache_root = os.path.join(cache_path, model_name)
    save_path = os.path.join(model_cache_root, 'kv_cache')
    preprocess_save_path = os.path.join(model_cache_root, 'preprocess_kv_cache')
    csv_path = os.path.join(model_cache_root, 'results')
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(preprocess_save_path, exist_ok=True)
    os.makedirs(csv_path, exist_ok=True)

    print(f"Cache directories created under: {model_cache_root}")

    # Load model and tokenizer
    print(f"Loading tokenizer and config from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config._attn_implementation = "sdpa"

    print(f"Loading {model_type} model...")
    if use_multi_gpu:
        print("Using multi-GPU with device_map='auto'")
    model, device_map = load_model(model_type, model_path, config, device, use_multi_gpu)

    # Prepare data organized by main questions
    print("Preparing data organized by main questions...")
    questions_data, system_tensor, context_rank, corpus_lens = prepare_reflect_data(
        data_path, tokenizer, bge_model_path, model_type, topk, max_samples, preprocess
    )

    # Initialize OpenAI client
    if openai_api_key is None:
        openai_api_key = os.environ.get("OPENAI_API_KEY")
    openai_client = OpenAI(api_key=openai_api_key, base_url=openai_base_url)

    # CSV file for results
    if preprocess:
        csv_file = f"{csv_path}/fusionrag_topk_{topk}_rate_{rate}.csv"
        result_file = f"{csv_path}/fusionrag_topk_{topk}_rate_{rate}.txt"
    else:
        csv_file = f"{csv_path}/baseline_rate_{rate}.csv"
        result_file = f"{csv_path}/baseline_rate_{rate}.txt"

    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Main Question', 'Sub Question', 'Ground Truth', 'Predicted', 'Correct'])

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
        all_sub_correct = True

        for sub_q_idx, sub_q_info in enumerate(q_data['sub_questions']):
            print(f"\nSub-question {sub_q_idx+1}/{len(q_data['sub_questions'])}")
            print(f"Question: {sub_q_info['query']}")
            print(f"Ground Truth: {sub_q_info['answer']}")

            # Build tokens: system + docs + question
            # Add /no_think for Qwen3 models to disable chain-of-thought
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
            if rate == 1:
                # Full recompute
                inputs = torch.cat(iter_tokens).to(input_device).unsqueeze(0)
                from ktransformers.util.utils import prefill_and_generate
                generated_tokens, _, _ = prefill_and_generate(
                    model, tokenizer, inputs, max_new_tokens=50, device=input_device, device_map=device_map
                )
            else:
                # Load preprocessed KV cache and generate
                load_path = preprocess_save_path if preprocess else save_path
                generated_tokens, _ = load_kv_and_generate(
                    model, tokenizer, past_key_values, iter_tokens, load_path, example_id,
                    max_new_tokens=50, revert_rope=revert_rope,
                    reprocess_method=reprocess_method, rate=rate,
                    preprocess=preprocess, device=input_device, chunk_ids=kv_chunk_ids, device_map=device_map
                )

            # Decode answer
            answer = tokenizer.decode(torch.tensor(generated_tokens[:-1]), skip_special_tokens=True)
            print(f"Predicted: {answer}")

            # Judge
            is_correct = judge_answer_with_openai(
                openai_client, openai_model,
                sub_q_info['query'], answer, sub_q_info['answer']
            )

            print(f"Judgment: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")

            total_sub_questions += 1
            if is_correct:
                correct_sub_questions += 1
            else:
                all_sub_correct = False

            # Save to CSV
            with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    q_data['main_question'], sub_q_info['query'],
                    sub_q_info['answer'], answer, is_correct
                ])

            torch.cuda.empty_cache()

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

    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}")
    print(f"Main Questions: {correct_main_questions}/{total_main_questions} ({main_q_acc:.2%})")
    print(f"Sub Questions: {correct_sub_questions}/{total_sub_questions} ({sub_q_acc:.2%})")
    print(f"{'='*80}")

    with open(result_file, 'w') as f:
        f.write(f"Main Questions Accuracy: {correct_main_questions}/{total_main_questions} ({main_q_acc:.4f})\n")
        f.write(f"Sub Questions Accuracy: {correct_sub_questions}/{total_sub_questions} ({sub_q_acc:.4f})\n")

    print(f"\nResults saved to {csv_path}")


if __name__ == '__main__':
    main(
        model_type='qwen3',
        model_path='/mnt/data/models/Qwen3-32B',
        data_path='/mnt/data/ktransformers-dev/result_reflect.json',
        cache_path='/mnt/data3/reflect/',
        model_name='Qwen3-32B',
        rate=0.3,
        topk=10,
        preprocess=False,
        reprocess_method='FusionRAG',
        bge_model_path='/mnt/data/models/bge-m3-FP16',
        revert_rope=True,
        device="cuda:0",
        use_multi_gpu=True,  # Set to True for multi-GPU (e.g., Qwen3-32B)
        openai_base_url="https://api.deepseek.com/v1",
        openai_api_key="sk-519d391217894b6e91e7c2ebf2a9f4df",
        openai_model="deepseek-chat",
        max_samples=200  # Test first 2 MAIN questions
    )
