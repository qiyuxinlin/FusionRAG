#!/usr/bin/env python3
"""
Online RAG System测试脚本

演示在Musique数据集上使用Online KV Cache系统
"""

import json
import argparse
import torch
import time
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from online_kv_cache_manager import OnlineKVCacheManager, KVCacheStore


def load_musique_data(data_path: str, max_samples: int = None) -> List[Dict]:
    """加载Musique数据集"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if max_samples:
        data = data[:max_samples]

    return data


def prepare_document_pool(musique_data: List[Dict]) -> List[str]:
    """
    准备文档池（所有文档的去重集合）

    Args:
        musique_data: Musique数据

    Returns:
        去重后的文档列表
    """
    doc_set = set()
    doc_list = []

    for item in musique_data:
        for doc_dict in item.get('paragraphs', []):
            doc_text = doc_dict.get('paragraph_text', '')
            if doc_text and doc_text not in doc_set:
                doc_set.add(doc_text)
                doc_list.append(doc_text)

    print(f"Total unique documents: {len(doc_list)}")
    return doc_list


def format_question_with_docs(question: str, documents: List[str]) -> str:
    """格式化问题和文档"""
    doc_text = "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(documents)])
    return f"{doc_text}\n\nQuestion: {question}\nAnswer:"


def test_online_rag(
    model_path: str,
    data_path: str,
    bge_model_path: str,
    topk: int = 10,
    max_samples: int = 10,
    max_memory_gb: float = 8.0,
    disk_cache_dir: str = "/tmp/online_kv_cache",
    max_new_tokens: int = 50
):
    """
    测试Online RAG系统

    Args:
        model_path: 模型路径
        data_path: Musique数据路径
        bge_model_path: BGE模型路径
        topk: 召回文档数
        max_samples: 最大测试样本数
        max_memory_gb: KV缓存最大内存
        disk_cache_dir: 磁盘缓存目录
        max_new_tokens: 最大生成tokens
    """
    print("="*80)
    print("Online RAG System Test")
    print("="*80)

    # 1. 加载模型
    print("\n[1] Loading model...")
    config = AutoConfig.from_pretrained(model_path)
    config.torch_dtype = torch.float16
    config._attn_implementation = "sdpa"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.float16,
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # 2. 加载数据
    print("\n[2] Loading Musique data...")
    musique_data = load_musique_data(data_path, max_samples)
    document_pool = prepare_document_pool(musique_data)

    # 3. 创建KV Cache管理器
    print("\n[3] Creating Online KV Cache Manager...")
    kv_store = KVCacheStore(
        max_memory_gb=max_memory_gb,
        disk_cache_dir=disk_cache_dir,
        enable_lru=True
    )

    manager = OnlineKVCacheManager(
        model=model,
        tokenizer=tokenizer,
        model_type='qwen2',
        bge_model_path=bge_model_path,
        kv_store=kv_store,
        device="cuda:0",
        system_prompt="You are a helpful assistant. Answer the question based on the provided documents."
    )

    # 4. 处理查询
    print("\n[4] Processing queries...")
    print("="*80)

    results = []
    total_time = 0

    for i, item in enumerate(musique_data):
        question = item.get('question', '')
        answer = item.get('answer', '')

        print(f"\n{'='*80}")
        print(f"Sample {i+1}/{len(musique_data)}")
        print(f"{'='*80}")
        print(f"Question: {question}")
        print(f"Ground Truth: {answer}")

        # 处理查询（召回 + 缓存检查 + 生成/加载KV）
        start_time = time.time()

        query_tokens, key_list, value_list, doc_tokens_list = manager.process_query(
            query=question,
            document_pool=document_pool,
            topk=topk,
            use_fusion=False,  # 可改为True使用融合
            verbose=True
        )

        process_time = time.time() - start_time
        total_time += process_time

        # 生成答案（简化版 - 实际需要完整实现）
        # predicted_answer = manager.generate_answer(
        #     query_tokens=query_tokens,
        #     key_cache_list=key_list,
        #     value_cache_list=value_list,
        #     max_new_tokens=max_new_tokens
        # )

        # 暂时使用占位符
        predicted_answer = "[Answer generation placeholder]"

        print(f"\nPredicted: {predicted_answer}")
        print(f"Processing time: {process_time:.2f}s")

        results.append({
            'question': question,
            'ground_truth': answer,
            'predicted': predicted_answer,
            'time': process_time
        })

        # 显示缓存统计
        stats = kv_store.get_stats()
        print(f"\nCache Stats:")
        print(f"  Hit Rate: {stats['hit_rate']:.2%}")
        print(f"  Memory Usage: {stats['memory_usage_gb']:.2f} GB")
        print(f"  Cached Docs: {stats['num_cached']}")

    # 5. 总结
    print("\n" + "="*80)
    print("Summary")
    print("="*80)

    final_stats = kv_store.get_stats()
    avg_time = total_time / len(musique_data)

    print(f"Total samples: {len(musique_data)}")
    print(f"Average time per query: {avg_time:.2f}s")
    print(f"\nFinal Cache Statistics:")
    print(f"  Total hits: {final_stats['hits']}")
    print(f"  Total misses: {final_stats['misses']}")
    print(f"  Hit rate: {final_stats['hit_rate']:.2%}")
    print(f"  Evictions: {final_stats['evictions']}")
    print(f"  Disk saves: {final_stats['disk_saves']}")
    print(f"  Disk loads: {final_stats['disk_loads']}")
    print(f"  Memory usage: {final_stats['memory_usage_gb']:.2f} GB")
    print(f"  Cached documents: {final_stats['num_cached']}")

    return results, final_stats


def main():
    parser = argparse.ArgumentParser(description="Test Online RAG System")

    parser.add_argument(
        "--model_path",
        type=str,
        default="/mnt/data/models/Qwen2.5-7B-Instruct",
        help="Path to the language model"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/shm/document/exp/FusionRAG/data/result_reflect.json",
        help="Path to Musique data"
    )
    parser.add_argument(
        "--bge_model_path",
        type=str,
        default="/mnt/data/models/bge-m3-FP16",
        help="Path to BGE model"
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="Number of documents to retrieve"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=20,
        help="Maximum number of samples to test"
    )
    parser.add_argument(
        "--max_memory_gb",
        type=float,
        default=8.0,
        help="Maximum memory for KV cache (GB)"
    )
    parser.add_argument(
        "--disk_cache_dir",
        type=str,
        default="/mnt/data3/tmp/online_kv_cache",
        help="Directory for disk cache"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=50,
        help="Maximum tokens to generate"
    )

    args = parser.parse_args()

    results, stats = test_online_rag(
        model_path=args.model_path,
        data_path=args.data_path,
        bge_model_path=args.bge_model_path,
        topk=args.topk,
        max_samples=args.max_samples,
        max_memory_gb=args.max_memory_gb,
        disk_cache_dir=args.disk_cache_dir,
        max_new_tokens=args.max_new_tokens
    )


if __name__ == "__main__":
    main()
