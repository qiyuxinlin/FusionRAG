#!/usr/bin/env python3
"""
Simple Online RAG System - Lazy KV Cache Generation

核心逻辑：
1. 第一次遇到文档 → 生成KV并保存到磁盘
2. 后续遇到相同文档 → 直接从磁盘加载KV

无需复杂的缓存管理，只需要 Check-or-Generate 模式
"""

import os
import json
import torch
import hashlib
import time
from typing import List, Dict, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from FlagEmbedding import FlagModel
import numpy as np
import faiss


def get_doc_id(doc_text: str) -> str:
    """根据文档内容生成唯一ID"""
    return hashlib.md5(doc_text.encode('utf-8')).hexdigest()


def load_or_generate_kv(
    doc_text: str,
    cache_dir: str,
    model,
    tokenizer,
    system_tokens: torch.Tensor,
    device: str = "cuda:0",
    device_map: Optional[dict] = None,
    verbose: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    加载或生成文档的KV Cache

    Args:
        doc_text: 文档文本
        cache_dir: 缓存目录
        model: 语言模型
        tokenizer: Tokenizer
        system_tokens: System prompt tokens
        device: 设备
        device_map: 多GPU映射
        verbose: 是否打印日志

    Returns:
        (key_cache, value_cache, doc_tokens)
    """
    from ktransformers.util.utils import prefill_and_save_kv_cache
    from ktransformers.models.custom_cache import StaticCache

    # 1. 生成文档ID
    doc_id = get_doc_id(doc_text)
    key_path = os.path.join(cache_dir, f"{doc_id}_key.pt")
    value_path = os.path.join(cache_dir, f"{doc_id}_value.pt")

    # 2. 检查是否已存在
    if os.path.exists(key_path) and os.path.exists(value_path):
        # Cache Hit - 直接加载
        if verbose:
            print(f"  ✓ Cache HIT: {doc_id[:8]}...")

        key_cache = torch.load(key_path, weights_only=True, map_location='cpu')
        value_cache = torch.load(value_path, weights_only=True, map_location='cpu')

        # 重新tokenize获取doc_tokens (或者也可以缓存)
        doc_tokens = torch.tensor(
            tokenizer.encode(doc_text, add_special_tokens=False),
            dtype=torch.int
        )

        return key_cache, value_cache, doc_tokens

    # 3. Cache Miss - 生成KV
    if verbose:
        print(f"  ✗ Cache MISS: {doc_id[:8]}... - Generating...", end=" ", flush=True)

    gen_start = time.time()

    # Tokenize
    doc_tokens = torch.tensor(
        tokenizer.encode(doc_text, add_special_tokens=False),
        dtype=torch.int
    )

    # 准备输入 (system + doc)
    input_tokens = torch.cat([system_tokens, doc_tokens])
    system_len = system_tokens.shape[0]
    doc_len = doc_tokens.shape[0]

    # 创建StaticCache
    input_device = "cuda:0" if device_map is not None else device
    max_cache_len = input_tokens.shape[0] + 1000

    past_key_values = StaticCache(
        model.config,
        batch_size=1,
        max_cache_len=max_cache_len,
        device=input_device,
        dtype=model.config.torch_dtype
    )

    # Forward pass生成KV
    input_tensor = input_tokens.unsqueeze(0).to(input_device)

    with torch.no_grad():
        inputs_embeds = model.model.embed_tokens(input_tensor).to(input_device)
        cache_position = torch.arange(0, input_tokens.shape[0], device=input_device)

        _ = model(
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            past_key_values=past_key_values,
            return_dict=False,
            use_cache=True
        )

    # 提取文档部分的KV (跳过system部分)
    key_cache = torch.stack([
        cache.cpu() for cache in past_key_values.key_cache
    ])[:, :, :, system_len:system_len + doc_len, :]

    value_cache = torch.stack([
        cache.cpu() for cache in past_key_values.value_cache
    ])[:, :, :, system_len:system_len + doc_len, :]

    # 4. 保存到磁盘
    os.makedirs(cache_dir, exist_ok=True)
    torch.save(key_cache, key_path)
    torch.save(value_cache, value_path)

    gen_time = time.time() - gen_start
    if verbose:
        print(f"Done ({gen_time:.2f}s)")

    return key_cache, value_cache, doc_tokens


def retrieve_documents_bge(
    query: str,
    document_pool: List[str],
    bge_model: FlagModel,
    topk: int = 10
) -> List[int]:
    """
    使用BGE召回相关文档

    Args:
        query: 查询
        document_pool: 文档池
        bge_model: BGE模型
        topk: 召回数量

    Returns:
        召回的文档索引
    """
    # 编码
    doc_embeddings = bge_model.encode(document_pool)
    query_embedding = bge_model.encode_queries([query])

    # FAISS搜索
    dim = doc_embeddings.shape[-1]
    index = faiss.index_factory(dim, 'Flat', faiss.METRIC_INNER_PRODUCT)
    doc_embeddings = doc_embeddings.astype(np.float32)
    index.train(doc_embeddings)
    index.add(doc_embeddings)

    query_embedding = query_embedding.astype(np.float32)
    _, indices = index.search(query_embedding, k=topk)

    return indices[0].tolist()


def process_query_simple(
    query: str,
    document_pool: List[str],
    cache_dir: str,
    model,
    tokenizer,
    bge_model: FlagModel,
    system_tokens: torch.Tensor,
    topk: int = 10,
    device: str = "cuda:0",
    device_map: Optional[dict] = None,
    verbose: bool = True
) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    简单的查询处理流程

    Args:
        query: 用户查询
        document_pool: 文档池
        cache_dir: KV缓存目录
        model: 语言模型
        tokenizer: Tokenizer
        bge_model: BGE模型
        system_tokens: System tokens
        topk: 召回文档数
        device: 设备
        device_map: 多GPU映射
        verbose: 详细日志

    Returns:
        (query_tokens, key_cache_list, value_cache_list, doc_tokens_list)
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")

    # 1. 文档召回
    if verbose:
        print(f"\n[Step 1] Retrieving documents...")

    retrieved_indices = retrieve_documents_bge(query, document_pool, bge_model, topk)
    retrieved_docs = [document_pool[idx] for idx in retrieved_indices]

    if verbose:
        print(f"  Retrieved {len(retrieved_docs)} documents: {retrieved_indices}")

    # 2. 加载或生成KV Cache
    if verbose:
        print(f"\n[Step 2] Loading/Generating KV Caches:")

    key_cache_list = []
    value_cache_list = []
    doc_tokens_list = []

    cache_hits = 0
    cache_misses = 0

    for i, doc_text in enumerate(retrieved_docs):
        # 检查缓存文件是否存在
        doc_id = get_doc_id(doc_text)
        key_path = os.path.join(cache_dir, f"{doc_id}_key.pt")

        is_hit = os.path.exists(key_path)
        if is_hit:
            cache_hits += 1
        else:
            cache_misses += 1

        # 加载或生成
        key_cache, value_cache, doc_tokens = load_or_generate_kv(
            doc_text=doc_text,
            cache_dir=cache_dir,
            model=model,
            tokenizer=tokenizer,
            system_tokens=system_tokens,
            device=device,
            device_map=device_map,
            verbose=verbose
        )

        key_cache_list.append(key_cache)
        value_cache_list.append(value_cache)
        doc_tokens_list.append(doc_tokens)

    # 3. 准备查询tokens
    query_tokens = torch.tensor(
        tokenizer.encode(query, add_special_tokens=False),
        dtype=torch.int
    )

    if verbose:
        print(f"\n[Step 3] Summary:")
        print(f"  Cache hits: {cache_hits}/{len(retrieved_docs)}")
        print(f"  Cache misses: {cache_misses}/{len(retrieved_docs)}")
        hit_rate = cache_hits / len(retrieved_docs) if len(retrieved_docs) > 0 else 0
        print(f"  Hit rate: {hit_rate:.1%}")

    return query_tokens, key_cache_list, value_cache_list, doc_tokens_list


def main_test():
    """测试脚本"""
    import argparse

    parser = argparse.ArgumentParser(description="Simple Online RAG Test")
    parser.add_argument("--model_path", type=str,
                       default="/mnt/data/models/Qwen2.5-7B-Instruct")
    parser.add_argument("--data_path", type=str,
                       default="/home/shm/document/exp/FusionRAG/data/result_reflect.json")
    parser.add_argument("--bge_model_path", type=str,
                       default="/mnt/data/models/bge-m3-FP16")
    parser.add_argument("--cache_dir", type=str,
                       default="/mnt/data3/tmp/simple_kv_cache")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--max_samples", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()

    print("="*80)
    print("Simple Online RAG Test")
    print("="*80)

    # 1. 加载模型
    print("\n[1] Loading model...")
    config = AutoConfig.from_pretrained(args.model_path)
    config.torch_dtype = torch.float16
    config._attn_implementation = "sdpa"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype=torch.float16,
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # System prompt
    system_prompt = "You are a helpful assistant."
    system_tokens = torch.tensor(
        tokenizer.encode(system_prompt, add_special_tokens=False),
        dtype=torch.int
    )

    # 2. 加载BGE模型
    print("\n[2] Loading BGE model...")
    bge_model = FlagModel(args.bge_model_path, use_fp16=True)

    # 3. 加载数据
    print("\n[3] Loading data...")
    with open(args.data_path, 'r', encoding='utf-8') as f:
        musique_data = json.load(f)

    if args.max_samples:
        musique_data = musique_data[:args.max_samples]

    # 构建文档池
    doc_set = set()
    document_pool = []
    for item in musique_data:
        for doc_dict in item.get('paragraphs', []):
            doc_text = doc_dict.get('paragraph_text', '')
            if doc_text and doc_text not in doc_set:
                doc_set.add(doc_text)
                document_pool.append(doc_text)

    print(f"  Total unique documents: {len(document_pool)}")

    # 4. 处理查询
    print("\n[4] Processing queries...")
    print("="*80)

    os.makedirs(args.cache_dir, exist_ok=True)

    total_hits = 0
    total_misses = 0

    for i, item in enumerate(musique_data):
        question = item.get('question', '')

        print(f"\n{'='*80}")
        print(f"Sample {i+1}/{len(musique_data)}")

        # 处理查询
        query_tokens, key_list, value_list, doc_tokens = process_query_simple(
            query=question,
            document_pool=document_pool,
            cache_dir=args.cache_dir,
            model=model,
            tokenizer=tokenizer,
            bge_model=bge_model,
            system_tokens=system_tokens,
            topk=args.topk,
            device=args.device,
            verbose=True
        )

        # 统计
        doc_ids = [get_doc_id(document_pool[idx]) for idx in
                   retrieve_documents_bge(question, document_pool, bge_model, args.topk)]

        for doc_id in doc_ids:
            key_path = os.path.join(args.cache_dir, f"{doc_id}_key.pt")
            # 注意：这里第一次生成后，后面就会命中了

    # 5. 最终统计
    print("\n" + "="*80)
    print("Final Statistics")
    print("="*80)

    # 统计缓存文件数
    cache_files = [f for f in os.listdir(args.cache_dir) if f.endswith('_key.pt')]
    print(f"Total cached documents: {len(cache_files)}")
    print(f"Cache directory: {args.cache_dir}")

    # 计算缓存大小
    total_size = 0
    for f in os.listdir(args.cache_dir):
        total_size += os.path.getsize(os.path.join(args.cache_dir, f))
    print(f"Cache size: {total_size / 1024**3:.2f} GB")


if __name__ == "__main__":
    main_test()
