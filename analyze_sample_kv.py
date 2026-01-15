#!/usr/bin/env python3
"""
KV Cache 和样本差异分析工具

功能：
1. 从不同方法的结果中找到符合条件的样本（BGE对、no_preprocess错、Random对）
2. 加载并对比这些样本的KV cache统计特性
3. 计算KV cache相似度矩阵
4. 生成详细的分析报告

使用方法：
    python analyze_sample_kv.py --bge_csv <path> --random_csv <path> --no_prep_csv <path>
"""

import argparse
import pandas as pd
import numpy as np
import torch
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

#####################################################################
# 配置区域 - 可以直接修改这里的路径
#####################################################################

# 默认结果文件路径
DEFAULT_BGE_CSV = "/home/shm/document/exp/FusionRAG/result/Qwen2.5-7B-Instruct/musique/results/FusionRAG_global_topk_10_rate_0.1_revert_rope.csv"
DEFAULT_RANDOM_CSV = "/home/shm/document/exp/FusionRAG/result/preprocess_ramdom3/Qwen2.5-7B-Instruct/musique/FusionRAG_global_topk10_random/rate_0.1_revert_rope.csv"
DEFAULT_NO_PREP_CSV = "/home/shm/document/exp/FusionRAG/result/no_preprocess/Qwen2.5-7B-Instruct/musique/results/FusionRAG_rate_0.1_revert_rope.csv"

# KV Cache 路径
BGE_CACHE_DIR = "/mnt/data3/tmp/fusionrag/Qwen2.5-7B-Instruct/musique/preprocess_kv_cache_global_bge"
RANDOM_CACHE_DIR = "/mnt/data3/tmp/fusionrag/Qwen2.5-7B-Instruct/musique/preprocess_kv_cache_global_topk10_random"
NO_PREP_CACHE_DIR = "/mnt/data3/tmp/fusionrag/Qwen2.5-7B-Instruct/musique/kv_cache"
REPEAT_SELF_CACHE_DIR = "/mnt/data3/tmp/fusionrag/Qwen2.5-7B-Instruct/musique/preprocess_kv_cache_global_topk10_repeat_self"

# 输出目录
OUTPUT_DIR = "/home/shm/document/exp/FusionRAG/kv_analysis_output"

#####################################################################
# 工具函数
#####################################################################

def find_interesting_samples(bge_df: pd.DataFrame,
                            random_df: pd.DataFrame,
                            no_prep_df: pd.DataFrame,
                            condition: str = "bge_correct_noprep_wrong_random_correct") -> List[int]:
    """
    找到符合条件的样本索引

    条件类型：
    - bge_correct_noprep_wrong_random_correct: BGE对、no_preprocess错、Random对
    - bge_correct_random_wrong: BGE对、Random错（测试语义相似度重要性）
    - all_methods_correct: 三种方法都对（baseline）
    - all_methods_wrong: 三种方法都错（难例）
    """

    # 确保三个DataFrame长度相同
    min_len = min(len(bge_df), len(random_df), len(no_prep_df))
    bge_df = bge_df.iloc[:min_len]
    random_df = random_df.iloc[:min_len]
    no_prep_df = no_prep_df.iloc[:min_len]

    # 获取主问题的正确性（因为CSV是sub question和main question交替）
    # 主问题通常是奇数行（索引0, 2, 4, ...）
    # 但实际上CSV中Main Question列非空的是主问题行
    bge_correct = bge_df['Correct'].values
    random_correct = random_df['Correct'].values
    no_prep_correct = no_prep_df['Correct'].values

    interesting_indices = []

    if condition == "bge_correct_noprep_wrong_random_correct":
        for i in range(len(bge_correct)):
            if bge_correct[i] and (not no_prep_correct[i]) and random_correct[i]:
                interesting_indices.append(i)

    elif condition == "bge_correct_random_wrong":
        for i in range(len(bge_correct)):
            if bge_correct[i] and (not random_correct[i]):
                interesting_indices.append(i)

    elif condition == "all_methods_correct":
        for i in range(len(bge_correct)):
            if bge_correct[i] and random_correct[i] and no_prep_correct[i]:
                interesting_indices.append(i)

    elif condition == "all_methods_wrong":
        for i in range(len(bge_correct)):
            if (not bge_correct[i]) and (not random_correct[i]) and (not no_prep_correct[i]):
                interesting_indices.append(i)

    return interesting_indices


def load_kv_cache(cache_dir: str, example_id: str, chunk_id: int,
                 device: str = 'cpu') -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """
    加载特定样本的KV cache

    返回: (key_cache, value_cache) 或 None
    """
    key_path = os.path.join(cache_dir, f"{example_id}_{chunk_id}_key.pt")
    value_path = os.path.join(cache_dir, f"{example_id}_{chunk_id}_value.pt")

    if not os.path.exists(key_path) or not os.path.exists(value_path):
        print(f"⚠ Cache not found: {key_path}")
        return None

    try:
        key_cache = torch.load(key_path, map_location=device)
        value_cache = torch.load(value_path, map_location=device)

        # 转换为float32以避免BFloat16绘图问题
        if key_cache.dtype == torch.bfloat16:
            key_cache = key_cache.float()
        if value_cache.dtype == torch.bfloat16:
            value_cache = value_cache.float()

        return key_cache, value_cache
    except Exception as e:
        print(f"✗ Error loading cache: {e}")
        return None


def compute_kv_statistics(key_cache: torch.Tensor, value_cache: torch.Tensor,
                          method_name: str, layer_idx: Optional[int] = None) -> Dict:
    """
    计算KV cache的统计特性
    """
    stats = {
        'method': method_name,
        'layer': layer_idx if layer_idx is not None else 'all',

        # Key 统计
        'key_shape': tuple(key_cache.shape),
        'key_norm_mean': torch.norm(key_cache, dim=-1).mean().item(),
        'key_norm_std': torch.norm(key_cache, dim=-1).std().item(),
        'key_mean': key_cache.mean().item(),
        'key_std': key_cache.std().item(),
        'key_min': key_cache.min().item(),
        'key_max': key_cache.max().item(),
        'key_sparsity': (key_cache.abs() < 0.01).float().mean().item(),

        # Value 统计
        'value_shape': tuple(value_cache.shape),
        'value_norm_mean': torch.norm(value_cache, dim=-1).mean().item(),
        'value_norm_std': torch.norm(value_cache, dim=-1).std().item(),
        'value_mean': value_cache.mean().item(),
        'value_std': value_cache.std().item(),
        'value_min': value_cache.min().item(),
        'value_max': value_cache.max().item(),
        'value_sparsity': (value_cache.abs() < 0.01).float().mean().item(),
    }

    return stats


def compute_kv_similarity(kv1: Tuple[torch.Tensor, torch.Tensor],
                         kv2: Tuple[torch.Tensor, torch.Tensor],
                         name1: str, name2: str) -> Dict:
    """
    计算两组KV cache之间的相似度
    """
    key1, value1 = kv1
    key2, value2 = kv2

    # Flatten
    key1_flat = key1.flatten()
    key2_flat = key2.flatten()
    value1_flat = value1.flatten()
    value2_flat = value2.flatten()

    # 余弦相似度
    key_cosine = F.cosine_similarity(key1_flat.unsqueeze(0), key2_flat.unsqueeze(0), dim=1).item()
    value_cosine = F.cosine_similarity(value1_flat.unsqueeze(0), value2_flat.unsqueeze(0), dim=1).item()

    # L2 距离
    key_l2 = torch.norm(key1_flat - key2_flat, p=2).item()
    value_l2 = torch.norm(value1_flat - value2_flat, p=2).item()

    # 均方误差
    key_mse = F.mse_loss(key1, key2).item()
    value_mse = F.mse_loss(value1, value2).item()

    return {
        'pair': f"{name1} vs {name2}",
        'key_cosine_similarity': key_cosine,
        'value_cosine_similarity': value_cosine,
        'key_l2_distance': key_l2,
        'value_l2_distance': value_l2,
        'key_mse': key_mse,
        'value_mse': value_mse,
    }


def analyze_sample(sample_idx: int, bge_df: pd.DataFrame, random_df: pd.DataFrame,
                  no_prep_df: pd.DataFrame, output_dir: str):
    """
    深入分析单个样本的KV cache差异
    """
    print("\n" + "="*80)
    print(f"深入分析样本 #{sample_idx}")
    print("="*80)

    # 获取样本信息
    bge_row = bge_df.iloc[sample_idx]
    random_row = random_df.iloc[sample_idx]
    no_prep_row = no_prep_df.iloc[sample_idx]

    print("\n问题信息:")
    print(f"  Main Question: {bge_row.get('Main Question', 'N/A')}")
    print(f"  Sub Question: {bge_row.get('Sub Question', 'N/A')}")
    print(f"  Ground Truth: {bge_row.get('Ground Truth', 'N/A')}")

    print("\n三种方法的预测:")
    print(f"  BGE:         {bge_row['Predicted']} (正确: {bge_row['Correct']})")
    print(f"  Random:      {random_row['Predicted']} (正确: {random_row['Correct']})")
    print(f"  No Preprocess: {no_prep_row['Predicted']} (正确: {no_prep_row['Correct']})")

    # 推测example_id
    # CSV中多个行可能对应同一个主问题（多个子问题）
    # 我们使用一个简单的启发式方法：尝试多个可能的example_id
    possible_example_ids = [
        str(sample_idx // 2),  # 假设平均2个子问题
        str(sample_idx // 3),  # 假设平均3个子问题
        str(sample_idx),       # 1:1映射
    ]

    # 尝试找到存在的cache文件
    found_example_id = None
    chunk_id = 1  # 分析第一个文档

    for example_id in possible_example_ids:
        test_path = os.path.join(BGE_CACHE_DIR, f"{example_id}_{chunk_id}_key.pt")
        if os.path.exists(test_path):
            found_example_id = example_id
            break

    if found_example_id is None:
        # 如果都不存在，尝试列出附近的文件
        print(f"\n⚠ 未找到对应的KV cache，尝试过的example_ids: {possible_example_ids}")
        print(f"  建议：检查 {BGE_CACHE_DIR} 中的实际文件")
        return

    example_id = found_example_id
    print(f"\n加载KV Cache (example_id={example_id}, chunk_id={chunk_id})...")

    # 加载三种方法的KV cache
    kv_bge = load_kv_cache(BGE_CACHE_DIR, example_id, chunk_id)
    kv_random = load_kv_cache(RANDOM_CACHE_DIR, example_id, chunk_id)
    kv_no_prep = load_kv_cache(NO_PREP_CACHE_DIR, example_id, chunk_id)

    if kv_bge is None or kv_random is None or kv_no_prep is None:
        print("✗ 无法加载完整的KV cache，跳过此样本")
        return

    # 计算统计特性
    print("\n计算KV Cache统计特性...")
    stats_bge = compute_kv_statistics(*kv_bge, "BGE")
    stats_random = compute_kv_statistics(*kv_random, "Random")
    stats_no_prep = compute_kv_statistics(*kv_no_prep, "No Preprocess")

    # 打印统计对比
    print("\n" + "-"*80)
    print("KV Cache 统计对比")
    print("-"*80)
    print(f"{'指标':<25} {'BGE':>15} {'Random':>15} {'No Preprocess':>15}")
    print("-"*80)

    for key in ['key_norm_mean', 'key_std', 'key_sparsity',
                'value_norm_mean', 'value_std', 'value_sparsity']:
        print(f"{key:<25} {stats_bge[key]:>15.6f} {stats_random[key]:>15.6f} {stats_no_prep[key]:>15.6f}")

    # 计算相似度矩阵
    print("\n计算KV Cache相似度矩阵...")
    sim_bge_random = compute_kv_similarity(kv_bge, kv_random, "BGE", "Random")
    sim_bge_noprep = compute_kv_similarity(kv_bge, kv_no_prep, "BGE", "No Preprocess")
    sim_random_noprep = compute_kv_similarity(kv_random, kv_no_prep, "Random", "No Preprocess")

    print("\n" + "-"*80)
    print("KV Cache 相似度矩阵")
    print("-"*80)
    print(f"{'对比对':<30} {'Key Cosine':>15} {'Value Cosine':>15} {'Key MSE':>15}")
    print("-"*80)

    for sim in [sim_bge_random, sim_bge_noprep, sim_random_noprep]:
        print(f"{sim['pair']:<30} {sim['key_cosine_similarity']:>15.6f} "
              f"{sim['value_cosine_similarity']:>15.6f} {sim['key_mse']:>15.6e}")

    # 生成可视化
    os.makedirs(output_dir, exist_ok=True)
    visualize_kv_comparison(
        kv_bge, kv_random, kv_no_prep,
        stats_bge, stats_random, stats_no_prep,
        sim_bge_random, sim_bge_noprep, sim_random_noprep,
        output_dir, sample_idx
    )

    # 保存详细报告
    report = {
        'sample_idx': sample_idx,
        'question': bge_row.get('Main Question', bge_row.get('Sub Question', 'N/A')),
        'ground_truth': bge_row.get('Ground Truth', 'N/A'),
        'predictions': {
            'bge': {'predicted': bge_row['Predicted'], 'correct': bool(bge_row['Correct'])},
            'random': {'predicted': random_row['Predicted'], 'correct': bool(random_row['Correct'])},
            'no_prep': {'predicted': no_prep_row['Predicted'], 'correct': bool(no_prep_row['Correct'])},
        },
        'kv_statistics': {
            'bge': stats_bge,
            'random': stats_random,
            'no_prep': stats_no_prep,
        },
        'kv_similarity': {
            'bge_vs_random': sim_bge_random,
            'bge_vs_no_prep': sim_bge_noprep,
            'random_vs_no_prep': sim_random_noprep,
        }
    }

    report_path = os.path.join(output_dir, f'sample_{sample_idx}_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n✓ 详细报告已保存: {report_path}")


def visualize_kv_comparison(kv_bge, kv_random, kv_no_prep,
                           stats_bge, stats_random, stats_no_prep,
                           sim_bge_random, sim_bge_noprep, sim_random_noprep,
                           output_dir, sample_idx):
    """
    生成KV cache对比的可视化图表
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'KV Cache Analysis - Sample #{sample_idx}', fontsize=16, fontweight='bold')

    # 1. Key 范数分布对比
    ax = axes[0, 0]
    key_bge, _ = kv_bge
    key_random, _ = kv_random
    key_no_prep, _ = kv_no_prep

    key_norms_bge = torch.norm(key_bge, dim=-1).flatten().cpu().numpy()
    key_norms_random = torch.norm(key_random, dim=-1).flatten().cpu().numpy()
    key_norms_no_prep = torch.norm(key_no_prep, dim=-1).flatten().cpu().numpy()

    ax.hist(key_norms_bge, bins=50, alpha=0.5, label='BGE', color='blue')
    ax.hist(key_norms_random, bins=50, alpha=0.5, label='Random', color='orange')
    ax.hist(key_norms_no_prep, bins=50, alpha=0.5, label='No Preprocess', color='green')
    ax.set_xlabel('Key Norm')
    ax.set_ylabel('Frequency')
    ax.set_title('Key Norm Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Value 范数分布对比
    ax = axes[0, 1]
    _, value_bge = kv_bge
    _, value_random = kv_random
    _, value_no_prep = kv_no_prep

    value_norms_bge = torch.norm(value_bge, dim=-1).flatten().cpu().numpy()
    value_norms_random = torch.norm(value_random, dim=-1).flatten().cpu().numpy()
    value_norms_no_prep = torch.norm(value_no_prep, dim=-1).flatten().cpu().numpy()

    ax.hist(value_norms_bge, bins=50, alpha=0.5, label='BGE', color='blue')
    ax.hist(value_norms_random, bins=50, alpha=0.5, label='Random', color='orange')
    ax.hist(value_norms_no_prep, bins=50, alpha=0.5, label='No Preprocess', color='green')
    ax.set_xlabel('Value Norm')
    ax.set_ylabel('Frequency')
    ax.set_title('Value Norm Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. 相似度矩阵热图
    ax = axes[0, 2]
    similarity_matrix = np.array([
        [1.0, sim_bge_random['key_cosine_similarity'], sim_bge_noprep['key_cosine_similarity']],
        [sim_bge_random['key_cosine_similarity'], 1.0, sim_random_noprep['key_cosine_similarity']],
        [sim_bge_noprep['key_cosine_similarity'], sim_random_noprep['key_cosine_similarity'], 1.0]
    ])

    sns.heatmap(similarity_matrix, annot=True, fmt='.4f', cmap='RdYlGn',
                xticklabels=['BGE', 'Random', 'No Prep'],
                yticklabels=['BGE', 'Random', 'No Prep'],
                vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Cosine Similarity'})
    ax.set_title('Key Cache Cosine Similarity Matrix')

    # 4. 统计特性对比柱状图 - Key
    ax = axes[1, 0]
    metrics = ['key_norm_mean', 'key_std', 'key_sparsity']
    x = np.arange(len(metrics))
    width = 0.25

    bge_vals = [stats_bge[m] for m in metrics]
    random_vals = [stats_random[m] for m in metrics]
    no_prep_vals = [stats_no_prep[m] for m in metrics]

    ax.bar(x - width, bge_vals, width, label='BGE', color='blue', alpha=0.7)
    ax.bar(x, random_vals, width, label='Random', color='orange', alpha=0.7)
    ax.bar(x + width, no_prep_vals, width, label='No Preprocess', color='green', alpha=0.7)

    ax.set_ylabel('Value')
    ax.set_title('Key Statistics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('key_', '') for m in metrics], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 5. 统计特性对比柱状图 - Value
    ax = axes[1, 1]
    metrics = ['value_norm_mean', 'value_std', 'value_sparsity']

    bge_vals = [stats_bge[m] for m in metrics]
    random_vals = [stats_random[m] for m in metrics]
    no_prep_vals = [stats_no_prep[m] for m in metrics]

    ax.bar(x - width, bge_vals, width, label='BGE', color='blue', alpha=0.7)
    ax.bar(x, random_vals, width, label='Random', color='orange', alpha=0.7)
    ax.bar(x + width, no_prep_vals, width, label='No Preprocess', color='green', alpha=0.7)

    ax.set_ylabel('Value')
    ax.set_title('Value Statistics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('value_', '') for m in metrics], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 6. MSE 对比
    ax = axes[1, 2]
    pairs = ['BGE vs Random', 'BGE vs No Prep', 'Random vs No Prep']
    key_mse = [sim_bge_random['key_mse'], sim_bge_noprep['key_mse'], sim_random_noprep['key_mse']]
    value_mse = [sim_bge_random['value_mse'], sim_bge_noprep['value_mse'], sim_random_noprep['value_mse']]

    x = np.arange(len(pairs))
    width = 0.35

    ax.bar(x - width/2, key_mse, width, label='Key MSE', color='steelblue')
    ax.bar(x + width/2, value_mse, width, label='Value MSE', color='coral')

    ax.set_ylabel('MSE')
    ax.set_title('Mean Squared Error Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(pairs, rotation=15)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')

    plt.tight_layout()

    output_path = os.path.join(output_dir, f'sample_{sample_idx}_kv_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ 可视化图表已保存: {output_path}")


#####################################################################
# 主函数
#####################################################################

def main():
    parser = argparse.ArgumentParser(description='KV Cache Analysis Tool')
    parser.add_argument('--bge_csv', type=str, default=DEFAULT_BGE_CSV,
                       help='BGE结果CSV文件路径')
    parser.add_argument('--random_csv', type=str, default=DEFAULT_RANDOM_CSV,
                       help='Random结果CSV文件路径')
    parser.add_argument('--no_prep_csv', type=str, default=DEFAULT_NO_PREP_CSV,
                       help='No Preprocess结果CSV文件路径')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                       help='输出目录')
    parser.add_argument('--condition', type=str,
                       default='bge_correct_noprep_wrong_random_correct',
                       choices=['bge_correct_noprep_wrong_random_correct',
                               'bge_correct_random_wrong',
                               'all_methods_correct',
                               'all_methods_wrong'],
                       help='样本筛选条件')
    parser.add_argument('--max_samples', type=int, default=5,
                       help='分析的最大样本数')

    args = parser.parse_args()

    print("="*80)
    print("KV Cache 分析工具")
    print("="*80)

    # 读取结果文件
    print("\n读取结果文件...")
    bge_df = pd.read_csv(args.bge_csv)
    random_df = pd.read_csv(args.random_csv)
    no_prep_df = pd.read_csv(args.no_prep_csv)

    print(f"  BGE:         {len(bge_df)} 行")
    print(f"  Random:      {len(random_df)} 行")
    print(f"  No Preprocess: {len(no_prep_df)} 行")

    # 找到符合条件的样本
    print(f"\n筛选样本（条件: {args.condition}）...")
    interesting_indices = find_interesting_samples(bge_df, random_df, no_prep_df, args.condition)

    print(f"✓ 找到 {len(interesting_indices)} 个符合条件的样本")

    if len(interesting_indices) == 0:
        print("✗ 没有找到符合条件的样本，退出")
        return

    # 显示前几个样本
    print("\n符合条件的样本:")
    for i, idx in enumerate(interesting_indices[:10]):
        row = bge_df.iloc[idx]
        question = row.get('Main Question', row.get('Sub Question', f'Sample {idx}'))
        print(f"  [{i}] Sample #{idx}: {question[:80]}...")

    # 分析样本
    num_to_analyze = min(args.max_samples, len(interesting_indices))
    print(f"\n将分析前 {num_to_analyze} 个样本...")

    for i, sample_idx in enumerate(interesting_indices[:num_to_analyze]):
        try:
            analyze_sample(sample_idx, bge_df, random_df, no_prep_df, args.output_dir)
        except Exception as e:
            print(f"✗ 分析样本 #{sample_idx} 时出错: {e}")
            continue

    print("\n" + "="*80)
    print("分析完成！")
    print(f"结果保存在: {args.output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
