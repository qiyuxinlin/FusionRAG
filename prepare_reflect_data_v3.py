#!/usr/bin/env python3
"""
Modified prepare_reflect_data function that uses global document IDs
for cross-question document reuse.

Key changes:
1. Load optimized data where documents are represented as global doc IDs
2. Use DocumentPoolManager to resolve doc IDs to text
3. Return doc_id mapping for KV cache management
4. KV cache saved as doc_{doc_id}_key.pt instead of {example_id}_{chunk_id}_key.pt
"""

import json
import torch
from typing import List, Dict, Any, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.doc_pool_manager import DocumentPoolManager


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


def prepare_reflect_data_v3(
    data_path: str,
    doc_pool_path: str,
    tokenizer,
    bge_model_path: str,
    model_type: str = 'qwen2',
    topk: int = 10,
    max_main_questions: int = None,
    preprocess: bool = True,
    recall_method = None,  # RecallMethod enum
    random_seed: int = 42,
    fixed_doc_idx: int = 0,
    preprocess_scope = None  # PreprocessScope enum
) -> Tuple[List, torch.Tensor, List, List, DocumentPoolManager, Dict]:
    """
    Prepare data from optimized result_reflect.json with global document IDs.

    This version uses global doc IDs from the document pool, enabling cross-question
    document reuse for KV cache.

    Args:
        data_path: Path to optimized result_reflect.json (with doc IDs)
        doc_pool_path: Path to document pool JSON (e.g., musique_input.json)
        tokenizer: Tokenizer for encoding text
        bge_model_path: Path to BGE model
        model_type: Type of model to determine system prompt
        topk: Top-k similar documents for each document
        max_main_questions: Limit number of questions (None = all)
        preprocess: Whether to compute context_rank
        recall_method: RecallMethod enum
        random_seed: Random seed for reproducibility
        fixed_doc_idx: Fixed document index for ablation
        preprocess_scope: PreprocessScope enum

    Returns:
        questions_data: List of dicts for each main question
        system_tensor: Tokenized system prompt
        context_rank: [total_docs x topk] array of similar document indices
        corpus_lens: List of document counts per question
        doc_pool_manager: DocumentPoolManager instance
        doc_id_to_global_idx: Mapping from doc_id to global corpus index
    """
    print(f"Loading dataset from {data_path}...")
    print(f"Loading document pool from {doc_pool_path}...")

    # Initialize document pool manager
    doc_pool_manager = DocumentPoolManager(doc_pool_path)

    # Load optimized dataset
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

    # STEP 1: Build document corpus with global doc IDs
    print("\n" + "="*80)
    print(f"Building document corpus with global doc IDs")
    print("="*80)

    global_corpus_doc_ids = []  # List of global doc_ids in corpus order
    doc_id_to_global_idx = {}   # Mapping: doc_id -> global corpus index
    corpus_lens = []  # Number of unique docs per question
    questions_data = []

    # First pass: collect all unique document IDs and build question metadata
    for main_q_idx, data_item in enumerate(dataset):
        main_question = data_item["question"]
        main_answer = data_item["answer"]
        intermediate_context = data_item.get("intermediate_context", [])

        question_doc_ids = []  # Doc IDs for THIS question only
        doc_id_to_local_idx = {}  # Local doc_id -> index mapping for this question
        sub_questions_info = []

        # Check if this main question should be tested
        should_test_main_question = True

        for sub_q_idx, sub_q in enumerate(intermediate_context):
            # In optimized data, 'retrieve docs' contains doc IDs (integers)
            doc_ids = sub_q.get("retrieve docs", [])
            local_doc_indices = []  # Local indices for this sub-question

            for doc_id in doc_ids:
                # Skip if doc_id is a dict (error marker from optimization)
                if isinstance(doc_id, dict):
                    print(f"  Warning: Skipping invalid doc_id: {doc_id}")
                    continue

                if doc_id not in doc_id_to_local_idx:
                    # New document for this question
                    question_doc_ids.append(doc_id)
                    local_idx = len(question_doc_ids) - 1
                    doc_id_to_local_idx[doc_id] = local_idx
                    local_doc_indices.append(local_idx)

                    # Add to global corpus if not already present
                    if doc_id not in doc_id_to_global_idx:
                        global_idx = len(global_corpus_doc_ids)
                        global_corpus_doc_ids.append(doc_id)
                        doc_id_to_global_idx[doc_id] = global_idx
                else:
                    # Document already seen in this question
                    local_doc_indices.append(doc_id_to_local_idx[doc_id])

            # Remove "Intermediate queryXXX:" prefix from query
            query = sub_q['query']
            if query.startswith("Intermediate query"):
                colon_pos = query.find(":")
                if colon_pos != -1:
                    query = query[colon_pos + 1:].strip()

            # Remove "Intermediate answerXXX:" prefix from answer
            answer = sub_q['answer']
            if answer.startswith("Intermediate answer"):
                colon_pos = answer.find(":")
                if colon_pos != -1:
                    answer = answer[colon_pos + 1:].strip()

            # Check if any sub-question has problematic answer
            if "No relevant information found" in answer or "没有相关信息" in answer:
                should_test_main_question = False

            sub_questions_info.append({
                'query': query,
                'answer': answer,
                'local_doc_indices': local_doc_indices,  # Local indices within this question
            })

        print(f"  Main question {main_q_idx + 1}: {len(question_doc_ids)} unique documents, {len(sub_questions_info)} sub-questions")

        # Tokenize documents for this main question
        doc_tensors = []
        for doc_id in question_doc_ids:
            doc_text = doc_pool_manager.get_doc_text(doc_id)
            doc_text_formatted = f"Document: {doc_text}\n"
            doc_tokens = tokenizer.encode(doc_text_formatted, add_special_tokens=False)
            doc_tensor = torch.tensor(doc_tokens, dtype=torch.long)
            doc_tensors.append(doc_tensor)

        corpus_lens.append(len(question_doc_ids))

        # Get gold_docs (doc IDs in optimized data)
        gold_doc_ids = data_item.get('gold_docs', [])
        # Filter out any dict entries (errors from optimization)
        gold_doc_ids = [doc_id for doc_id in gold_doc_ids if isinstance(doc_id, int)]

        questions_data.append({
            'main_question': main_question,
            'main_answer': main_answer,
            'sub_questions': sub_questions_info,
            'doc_ids': question_doc_ids,  # Global doc IDs for this question
            'doc_tensors': doc_tensors,
            'should_test': should_test_main_question,
            'gold_doc_ids': gold_doc_ids,
        })

    print(f"\nTotal unique documents in corpus: {len(global_corpus_doc_ids)}")
    print(f"Total questions: {len(questions_data)}")

    # STEP 2: Compute context_rank using BGE (if preprocess=True)
    context_rank = []
    if preprocess:
        print("\n" + "="*80)
        print("Computing document similarity with BGE...")
        print("="*80)

        from FlagEmbedding import BGEM3FlagModel
        bge_model = BGEM3FlagModel(bge_model_path, use_fp16=True)

        # Get all unique document texts
        unique_doc_texts = [doc_pool_manager.get_doc_text(doc_id) for doc_id in global_corpus_doc_ids]

        # Encode all documents
        print(f"Encoding {len(unique_doc_texts)} unique documents...")
        embeddings = bge_model.encode(unique_doc_texts, batch_size=32, max_length=8192)['dense_vecs']

        # Compute similarity matrix and get top-k for each document
        print(f"Computing top-{topk} similar documents for each document...")
        similarity_matrix = embeddings @ embeddings.T

        # Get top-k indices (excluding self)
        context_rank = []
        for i in range(len(unique_doc_texts)):
            # Get similarity scores for document i
            scores = similarity_matrix[i]
            # Get top-(k+1) indices (including self)
            top_indices = scores.argsort()[::-1][:topk+1]
            # Remove self from top-k
            top_indices = [idx for idx in top_indices if idx != i][:topk]
            context_rank.append(top_indices)

        context_rank = torch.tensor(context_rank, dtype=torch.long)
        print(f"  Context rank shape: {context_rank.shape}")

    print("\n" + "="*80)
    print("Data preparation complete!")
    print("="*80)

    return questions_data, system_tensor, context_rank, corpus_lens, doc_pool_manager, doc_id_to_global_idx


if __name__ == "__main__":
    # Test the function
    import sys
    sys.path.insert(0, '/home/shm/document/exp/FusionRAG')
    from transformers import AutoTokenizer

    data_path = "/home/shm/document/exp/FusionRAG/data/result_reflect_optimized.json"
    doc_pool_path = "/home/shm/document/exp/FusionRAG/data/musique_input.json"
    model_path = "/mnt/data/models/Qwen2.5-7B-Instruct"
    bge_model_path = "/mnt/data/models/bge-m3-FP16"

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    questions_data, system_tensor, context_rank, corpus_lens, doc_pool_manager, doc_id_to_global_idx = prepare_reflect_data_v3(
        data_path=data_path,
        doc_pool_path=doc_pool_path,
        tokenizer=tokenizer,
        bge_model_path=bge_model_path,
        model_type='qwen2',
        topk=10,
        max_main_questions=2,
        preprocess=False
    )

    print(f"\n=== Test Results ===")
    print(f"Number of questions: {len(questions_data)}")
    print(f"System tensor shape: {system_tensor.shape}")
    print(f"Corpus lengths: {corpus_lens}")
    print(f"Total unique documents: {len(doc_id_to_global_idx)}")

    # Print first question details
    if questions_data:
        q = questions_data[0]
        print(f"\nFirst question:")
        print(f"  Question: {q['main_question'][:80]}...")
        print(f"  Doc IDs: {q['doc_ids']}")
        print(f"  Number of sub-questions: {len(q['sub_questions'])}")
