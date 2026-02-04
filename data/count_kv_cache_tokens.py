#!/usr/bin/env python3
"""
统计本地 KV cache 文件的总 token 数量
"""
import os
import torch
from pathlib import Path
from collections import defaultdict

def count_tokens_in_cache(cache_dir, pattern="*_key.pt"):
    """
    统计指定目录下 KV cache 的 token 数量

    Args:
        cache_dir: KV cache 存储目录
        pattern: 文件匹配模式

    Returns:
        dict: 统计信息 {
            'total_files': 文件总数,
            'total_tokens': 总 token 数,
            'by_example': {example_id: token_count},
            'by_chunk_type': {'system': tokens, 'document': tokens}
        }
    """
    cache_path = Path(cache_dir)

    if not cache_path.exists():
        print(f"警告：目录不存在 {cache_dir}")
        return None

    stats = {
        'total_files': 0,
        'total_tokens': 0,
        'by_example': defaultdict(int),
        'by_chunk_type': {'system': 0, 'document': 0},
        'files': []
    }

    # 遍历所有 key cache 文件
    key_files = list(cache_path.glob(pattern))

    if not key_files:
        print(f"警告：在 {cache_dir} 中没有找到匹配 {pattern} 的文件")
        return stats

    print(f"正在处理 {len(key_files)} 个 cache 文件...")

    for cache_file in key_files:
        try:
            # 加载 cache tensor
            cache = torch.load(cache_file, weights_only=True)

            # cache shape 通常是: [num_layers, batch_size, num_heads, seq_len, head_dim]
            # 提取序列长度（倒数第二维，即 -2）
            if isinstance(cache, torch.Tensor):
                if cache.dim() >= 2:
                    seq_len = cache.shape[-2]  # 倒数第二维是 seq_len
                else:
                    print(f"警告：{cache_file} 的 tensor shape 异常: {cache.shape}")
                    continue
            elif isinstance(cache, list):
                # list of tensors per layer
                if len(cache) > 0 and cache[0].dim() >= 2:
                    seq_len = cache[0].shape[-2]
                else:
                    print(f"警告：{cache_file} 的 cache list 结构异常")
                    continue
            else:
                print(f"警告：{cache_file} 的格式不支持: {type(cache)}")
                continue

            # 解析文件名获取 example_id 和 chunk_id
            # 格式: {example_id}_{chunk_id}_key.pt
            stem = cache_file.stem  # 去掉 .pt
            if stem.endswith('_key'):
                stem = stem[:-4]  # 去掉 _key

            parts = stem.split('_')
            if len(parts) >= 2:
                example_id = parts[0]
                chunk_id = parts[1]

                try:
                    chunk_id_int = int(chunk_id)
                except ValueError:
                    chunk_id_int = -1

                # 累计统计
                stats['total_files'] += 1
                stats['total_tokens'] += seq_len
                stats['by_example'][example_id] += seq_len
                stats['files'].append({
                    'file': str(cache_file),
                    'example_id': example_id,
                    'chunk_id': chunk_id_int,
                    'tokens': seq_len
                })

                # 按类型分类
                if chunk_id_int == 0:
                    stats['by_chunk_type']['system'] += seq_len
                else:
                    stats['by_chunk_type']['document'] += seq_len

        except Exception as e:
            print(f"错误：处理 {cache_file} 时出错: {e}")

    return stats


def print_statistics(stats, cache_name="Cache"):
    """打印统计信息"""
    if stats is None or stats['total_files'] == 0:
        print(f"\n{cache_name}: 无数据")
        return

    print(f"\n{'='*80}")
    print(f"{cache_name} Token 统计")
    print(f"{'='*80}")
    print(f"总文件数: {stats['total_files']:,}")
    print(f"总 Tokens: {stats['total_tokens']:,}")
    print(f"平均每个文件: {stats['total_tokens'] / stats['total_files']:,.2f} tokens")
    print(f"\n按类型分类:")
    print(f"  System (chunk_id=0): {stats['by_chunk_type']['system']:,} tokens")
    print(f"  Document (chunk_id>0): {stats['by_chunk_type']['document']:,} tokens")
    print(f"\n按问题分类 (前10个):")
    sorted_examples = sorted(stats['by_example'].items(), key=lambda x: x[1], reverse=True)[:10]
    for example_id, tokens in sorted_examples:
        print(f"  Question {example_id}: {tokens:,} tokens")


def main():
    # 配置路径
    base_path = "/mnt/data3/tmp/fusionrag"
    model_name = "Qwen2.5-7B-Instruct"
    dataset_name = "musique_v2"

    # 1. 主问题文档的 KV cache
    main_cache_dir = f"{base_path}/{model_name}/{dataset_name}/kv_cache"

    # 2. BGE 召回文档的预处理 KV cache
    preprocess_cache_dir = f"{base_path}/{model_name}/{dataset_name}/preprocess_kv_cache_global_topk10_bge"

    print("="*80)
    print("FusionRAG KV Cache Token 统计工具")
    print("="*80)
    print(f"模型: {model_name}")
    print(f"数据集: {dataset_name}")
    print()

    # 统计主问题文档 KV cache
    print("正在统计主问题文档 KV cache...")
    main_stats = count_tokens_in_cache(main_cache_dir, "*_0_key.pt")  # 只统计 system cache
    if main_stats:
        # 需要统计所有文件，不只是 _0
        main_stats_all = count_tokens_in_cache(main_cache_dir, "*_key.pt")
        print_statistics(main_stats_all, "Stage 1: 主问题文档 KV Cache")

    # 统计预处理 KV cache
    print("\n正在统计预处理 BGE 召回文档 KV cache...")
    preprocess_stats = count_tokens_in_cache(preprocess_cache_dir, "*_key.pt")
    if preprocess_stats:
        print_statistics(preprocess_stats, "Stage 2: BGE 召回文档预处理 KV Cache")

    # 汇总
    print(f"\n{'='*80}")
    print("总体统计")
    print(f"{'='*80}")

    total_stage1 = main_stats_all['total_tokens'] if main_stats else 0
    total_stage2 = preprocess_stats['total_tokens'] if preprocess_stats else 0
    total = total_stage1 + total_stage2

    print(f"Stage 1 (主问题文档): {total_stage1:,} tokens ({total_stage1/total*100:.2f}%)" if total > 0 else "Stage 1: 0 tokens")
    print(f"Stage 2 (BGE 召回文档): {total_stage2:,} tokens ({total_stage2/total*100:.2f}%)" if total > 0 else "Stage 2: 0 tokens")
    print(f"总计: {total:,} tokens")

    print(f"\n说明:")
    print(f"  - Stage 1: 每个主问题自己的文档生成的 KV cache")
    print(f"  - Stage 2: FusionRAG 预处理阶段，为 BGE 召回的文档生成的 KV cache")
    print(f"  - 每个 token 代表通过所有 transformer 层的一个位置")


if __name__ == "__main__":
    main()
