#!/usr/bin/env python3
"""
筛选样本的KV Cache L2距离分析

只分析 no_preprocess 答错但 random 答对的样本
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
from collections import defaultdict

#####################################################################
# 配置
#####################################################################

# 结果CSV路径
BGE_CSV = "/home/shm/document/exp/FusionRAG/result/Qwen2.5-7B-Instruct/musique/results/FusionRAG_global_topk_10_rate_0.1_revert_rope.csv"
RANDOM_CSV = "/home/shm/document/exp/FusionRAG/result/preprocess_ramdom3/Qwen2.5-7B-Instruct/musique/FusionRAG_global_topk10_random/rate_0.1_revert_rope.csv"
NO_PREP_CSV = "/home/shm/document/exp/FusionRAG/result/no_preprocess/Qwen2.5-7B-Instruct/musique/results/FusionRAG_rate_0.1_revert_rope.csv"

# KV Cache 路径
BGE_CACHE_DIR = "/mnt/data3/tmp/fusionrag/Qwen2.5-7B-Instruct/musique/preprocess_kv_cache_global_bge"
RANDOM_CACHE_DIR = "/mnt/data3/tmp/fusionrag/Qwen2.5-7B-Instruct/musique/preprocess_kv_cache_global_topk10_random"
NO_PREP_CACHE_DIR = "/mnt/data3/tmp/fusionrag/Qwen2.5-7B-Instruct/musique/kv_cache"

# 输出目录
OUTPUT_DIR = "/home/shm/document/exp/FusionRAG/kv_analysis_output"

#####################################################################
# 函数
#####################################################################

def load_kv_cache(cache_dir, example_id, chunk_id, device='cpu'):
    """加载KV cache"""
    key_path = os.path.join(cache_dir, f"{example_id}_{chunk_id}_key.pt")
    value_path = os.path.join(cache_dir, f"{example_id}_{chunk_id}_value.pt")

    if not os.path.exists(key_path) or not os.path.exists(value_path):
        return None

    try:
        key_cache = torch.load(key_path, map_location=device)
        value_cache = torch.load(value_path, map_location=device)

        # 转换为float32
        if key_cache.dtype == torch.bfloat16:
            key_cache = key_cache.float()
        if value_cache.dtype == torch.bfloat16:
            value_cache = value_cache.float()

        return key_cache, value_cache
    except Exception as e:
        return None


def compute_l2_distance(tensor1, tensor2):
    """计算两个tensor之间的L2距离"""
    flat1 = tensor1.flatten()
    flat2 = tensor2.flatten()

    if flat1.shape != flat2.shape:
        return None

    l2_dist = torch.norm(flat1 - flat2, p=2).item()
    return l2_dist


def find_filtered_samples(bge_df, random_df, no_prep_df):
    """
    找到 no_preprocess 答错但 random 答对的样本

    返回: List[int] 样本索引列表
    """
    min_len = min(len(bge_df), len(random_df), len(no_prep_df))

    random_correct = random_df['Correct'].values[:min_len]
    no_prep_correct = no_prep_df['Correct'].values[:min_len]

    filtered_indices = []

    for i in range(min_len):
        if random_correct[i] and (not no_prep_correct[i]):
            filtered_indices.append(i)

    return filtered_indices


def analyze_filtered_kv_files(filtered_indices):
    """分析筛选后样本的KV文件L2距离"""

    print("="*80)
    print("筛选样本 KV Cache L2 距离分析")
    print("条件：no_preprocess 答错 且 random 答对")
    print("="*80)

    print(f"\n找到 {len(filtered_indices)} 个符合条件的样本")

    # 收集L2距离数据
    results = {
        'bge_vs_random': {'key_l2': [], 'value_l2': []},
        'bge_vs_no_prep': {'key_l2': [], 'value_l2': []},
        'random_vs_no_prep': {'key_l2': [], 'value_l2': []},
    }

    # 记录样本信息
    sample_info = []

    failed_count = 0
    success_count = 0

    print("\n计算L2距离...")
    for sample_idx in tqdm(filtered_indices, desc="Processing"):
        # 尝试多个可能的example_id映射
        possible_example_ids = [
            str(sample_idx // 2),
            str(sample_idx // 3),
            str(sample_idx),
        ]

        found = False
        for example_id in possible_example_ids:
            chunk_id = 1  # 分析第一个文档

            # 加载三种方法的KV cache
            kv_bge = load_kv_cache(BGE_CACHE_DIR, example_id, chunk_id)
            kv_random = load_kv_cache(RANDOM_CACHE_DIR, example_id, chunk_id)
            kv_no_prep = load_kv_cache(NO_PREP_CACHE_DIR, example_id, chunk_id)

            if kv_bge is None or kv_random is None or kv_no_prep is None:
                continue

            found = True
            key_bge, value_bge = kv_bge
            key_random, value_random = kv_random
            key_no_prep, value_no_prep = kv_no_prep

            # BGE vs Random
            key_l2 = compute_l2_distance(key_bge, key_random)
            value_l2 = compute_l2_distance(value_bge, value_random)
            if key_l2 is not None and value_l2 is not None:
                results['bge_vs_random']['key_l2'].append(key_l2)
                results['bge_vs_random']['value_l2'].append(value_l2)

            # BGE vs No_prep
            key_l2 = compute_l2_distance(key_bge, key_no_prep)
            value_l2 = compute_l2_distance(value_bge, value_no_prep)
            if key_l2 is not None and value_l2 is not None:
                results['bge_vs_no_prep']['key_l2'].append(key_l2)
                results['bge_vs_no_prep']['value_l2'].append(value_l2)

            # Random vs No_prep
            key_l2 = compute_l2_distance(key_random, key_no_prep)
            value_l2 = compute_l2_distance(value_random, value_no_prep)
            if key_l2 is not None and value_l2 is not None:
                results['random_vs_no_prep']['key_l2'].append(key_l2)
                results['random_vs_no_prep']['value_l2'].append(value_l2)

            sample_info.append({
                'sample_idx': sample_idx,
                'example_id': example_id,
                'chunk_id': chunk_id,
            })

            success_count += 1
            break

        if not found:
            failed_count += 1

    print(f"\n成功分析: {success_count} 个样本")
    print(f"失败: {failed_count} 个样本")

    return results, sample_info


def plot_l2_distributions(results, output_dir, condition_name="filtered"):
    """绘制L2距离分布的柱状图"""

    os.makedirs(output_dir, exist_ok=True)

    # 准备数据
    pairs = ['BGE vs Random', 'BGE vs No_Prep', 'Random vs No_Prep']

    # Key L2
    key_l2_means = [
        np.mean(results['bge_vs_random']['key_l2']),
        np.mean(results['bge_vs_no_prep']['key_l2']),
        np.mean(results['random_vs_no_prep']['key_l2']),
    ]
    key_l2_stds = [
        np.std(results['bge_vs_random']['key_l2']),
        np.std(results['bge_vs_no_prep']['key_l2']),
        np.std(results['random_vs_no_prep']['key_l2']),
    ]

    # Value L2
    value_l2_means = [
        np.mean(results['bge_vs_random']['value_l2']),
        np.mean(results['bge_vs_no_prep']['value_l2']),
        np.mean(results['random_vs_no_prep']['value_l2']),
    ]
    value_l2_stds = [
        np.std(results['bge_vs_random']['value_l2']),
        np.std(results['bge_vs_no_prep']['value_l2']),
        np.std(results['random_vs_no_prep']['value_l2']),
    ]

    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('KV Cache L2 Distance Analysis\n(No_Prep Wrong & Random Correct Samples)',
                 fontsize=16, fontweight='bold')

    # 1. Key L2 均值柱状图
    ax = axes[0, 0]
    x = np.arange(len(pairs))
    bars = ax.bar(x, key_l2_means, yerr=key_l2_stds, capsize=5,
                   color=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.8, edgecolor='black')
    ax.set_xlabel('Comparison Pairs', fontsize=12)
    ax.set_ylabel('L2 Distance', fontsize=12)
    ax.set_title('Key Cache L2 Distance (Mean ± Std)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(pairs, rotation=15, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for i, (bar, mean, std) in enumerate(zip(bars, key_l2_means, key_l2_stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std,
                f'{mean:.2f}±{std:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 2. Value L2 均值柱状图
    ax = axes[0, 1]
    bars = ax.bar(x, value_l2_means, yerr=value_l2_stds, capsize=5,
                   color=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.8, edgecolor='black')
    ax.set_xlabel('Comparison Pairs', fontsize=12)
    ax.set_ylabel('L2 Distance', fontsize=12)
    ax.set_title('Value Cache L2 Distance (Mean ± Std)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(pairs, rotation=15, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    for i, (bar, mean, std) in enumerate(zip(bars, value_l2_means, value_l2_stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std,
                f'{mean:.2f}±{std:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 3. Key L2 分布直方图
    ax = axes[1, 0]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    labels = ['BGE vs Random', 'BGE vs No_Prep', 'Random vs No_Prep']

    for i, (key, label, color) in enumerate(zip(['bge_vs_random', 'bge_vs_no_prep', 'random_vs_no_prep'],
                                                  labels, colors)):
        data = results[key]['key_l2']
        ax.hist(data, bins=30, alpha=0.6, label=label, color=color, edgecolor='black')

    ax.set_xlabel('Key L2 Distance', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Key Cache L2 Distance Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # 4. Value L2 分布直方图
    ax = axes[1, 1]

    for i, (key, label, color) in enumerate(zip(['bge_vs_random', 'bge_vs_no_prep', 'random_vs_no_prep'],
                                                  labels, colors)):
        data = results[key]['value_l2']
        ax.hist(data, bins=30, alpha=0.6, label=label, color=color, edgecolor='black')

    ax.set_xlabel('Value L2 Distance', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Value Cache L2 Distance Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    output_path = os.path.join(output_dir, f'kv_l2_distance_{condition_name}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✓ 图表已保存: {output_path}")


def save_statistics(results, sample_info, output_dir, condition_name="filtered"):
    """保存统计信息"""

    stats = {
        'condition': 'no_preprocess wrong AND random correct',
        'sample_count': len(sample_info),
        'statistics': {}
    }

    for pair_name, data in results.items():
        key_l2 = data['key_l2']
        value_l2 = data['value_l2']

        stats['statistics'][pair_name] = {
            'key_l2': {
                'mean': float(np.mean(key_l2)),
                'std': float(np.std(key_l2)),
                'median': float(np.median(key_l2)),
                'min': float(np.min(key_l2)),
                'max': float(np.max(key_l2)),
                'count': len(key_l2),
            },
            'value_l2': {
                'mean': float(np.mean(value_l2)),
                'std': float(np.std(value_l2)),
                'median': float(np.median(value_l2)),
                'min': float(np.min(value_l2)),
                'max': float(np.max(value_l2)),
                'count': len(value_l2),
            }
        }

    output_path = os.path.join(output_dir, f'kv_l2_statistics_{condition_name}.json')
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"✓ 统计数据已保存: {output_path}")

    # 保存样本列表
    sample_list_path = os.path.join(output_dir, f'sample_list_{condition_name}.json')
    with open(sample_list_path, 'w') as f:
        json.dump(sample_info, f, indent=2)
    print(f"✓ 样本列表已保存: {sample_list_path}")

    # 打印统计摘要
    print("\n" + "="*80)
    print("统计摘要")
    print("="*80)
    print(f"筛选条件: no_preprocess 答错 且 random 答对")
    print(f"样本数量: {len(sample_info)}")

    for pair_name, pair_stats in stats['statistics'].items():
        print(f"\n{pair_name.upper().replace('_', ' ')}:")
        print(f"  Key L2:   {pair_stats['key_l2']['mean']:.2f} ± {pair_stats['key_l2']['std']:.2f}")
        print(f"            [min: {pair_stats['key_l2']['min']:.2f}, max: {pair_stats['key_l2']['max']:.2f}]")
        print(f"  Value L2: {pair_stats['value_l2']['mean']:.2f} ± {pair_stats['value_l2']['std']:.2f}")
        print(f"            [min: {pair_stats['value_l2']['min']:.2f}, max: {pair_stats['value_l2']['max']:.2f}]")


def main():
    print("开始筛选样本并分析KV Cache L2距离...\n")

    # 读取结果CSV
    print("读取结果CSV文件...")
    bge_df = pd.read_csv(BGE_CSV)
    random_df = pd.read_csv(RANDOM_CSV)
    no_prep_df = pd.read_csv(NO_PREP_CSV)

    print(f"  BGE:         {len(bge_df)} 行")
    print(f"  Random:      {len(random_df)} 行")
    print(f"  No Preprocess: {len(no_prep_df)} 行")

    # 筛选样本
    print("\n筛选符合条件的样本...")
    filtered_indices = find_filtered_samples(bge_df, random_df, no_prep_df)

    print(f"✓ 找到 {len(filtered_indices)} 个符合条件的样本")

    if len(filtered_indices) == 0:
        print("✗ 没有符合条件的样本，退出")
        return

    # 显示前几个样本
    print("\n前10个样本:")
    for i, idx in enumerate(filtered_indices[:10]):
        row = random_df.iloc[idx]
        question = row.get('Main Question', row.get('Sub Question', f'Sample {idx}'))
        print(f"  [{i}] Sample #{idx}: {question[:70]}...")

    # 分析KV文件
    results, sample_info = analyze_filtered_kv_files(filtered_indices)

    if len(sample_info) == 0:
        print("✗ 没有成功加载KV cache，退出")
        return

    # 绘制图表
    print("\n绘制图表...")
    plot_l2_distributions(results, OUTPUT_DIR, "no_prep_wrong_random_correct")

    # 保存统计信息
    print("\n保存统计信息...")
    save_statistics(results, sample_info, OUTPUT_DIR, "no_prep_wrong_random_correct")

    print("\n" + "="*80)
    print("分析完成！")
    print("="*80)


if __name__ == "__main__":
    main()
