#!/usr/bin/env python3
"""
全数据集KV Cache L2距离分析

分析整个数据集上所有KV文件，计算BGE、Random、No_preprocess之间的L2距离
并绘制柱状图展示分布
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import json
from collections import defaultdict

#####################################################################
# 配置
#####################################################################

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
        print(f"Error loading {key_path}: {e}")
        return None


def compute_l2_distance(tensor1, tensor2):
    """计算两个tensor之间的L2距离"""
    # Flatten
    flat1 = tensor1.flatten()
    flat2 = tensor2.flatten()

    # 确保形状相同
    if flat1.shape != flat2.shape:
        return None

    # L2距离
    l2_dist = torch.norm(flat1 - flat2, p=2).item()

    return l2_dist


def get_all_cache_files(cache_dir):
    """获取目录中所有的cache文件（example_id, chunk_id）对"""
    files = os.listdir(cache_dir)
    cache_pairs = set()

    for f in files:
        if f.endswith('_key.pt'):
            # 提取 example_id 和 chunk_id
            parts = f.replace('_key.pt', '').split('_')
            if len(parts) == 2:
                example_id, chunk_id = parts
                cache_pairs.add((example_id, chunk_id))

    return sorted(list(cache_pairs))


def analyze_all_kv_files():
    """分析所有KV文件的L2距离"""

    print("="*80)
    print("全数据集 KV Cache L2 距离分析")
    print("="*80)

    # 获取所有cache文件
    print("\n扫描cache文件...")
    bge_pairs = get_all_cache_files(BGE_CACHE_DIR)
    random_pairs = get_all_cache_files(RANDOM_CACHE_DIR)
    no_prep_pairs = get_all_cache_files(NO_PREP_CACHE_DIR)

    print(f"  BGE cache: {len(bge_pairs)} 文件")
    print(f"  Random cache: {len(random_pairs)} 文件")
    print(f"  No_prep cache: {len(no_prep_pairs)} 文件")

    # 找到三个目录都有的文件（交集）
    common_pairs = set(bge_pairs) & set(random_pairs) & set(no_prep_pairs)
    print(f"\n三个方法共有的cache文件: {len(common_pairs)} 个")

    if len(common_pairs) == 0:
        print("✗ 没有共同的cache文件，退出")
        return

    # 收集L2距离数据
    results = {
        'bge_vs_random': {'key_l2': [], 'value_l2': []},
        'bge_vs_no_prep': {'key_l2': [], 'value_l2': []},
        'random_vs_no_prep': {'key_l2': [], 'value_l2': []},
    }

    failed_count = 0

    print("\n计算L2距离...")
    for example_id, chunk_id in tqdm(sorted(common_pairs), desc="Processing"):
        # 加载三种方法的KV cache
        kv_bge = load_kv_cache(BGE_CACHE_DIR, example_id, chunk_id)
        kv_random = load_kv_cache(RANDOM_CACHE_DIR, example_id, chunk_id)
        kv_no_prep = load_kv_cache(NO_PREP_CACHE_DIR, example_id, chunk_id)

        if kv_bge is None or kv_random is None or kv_no_prep is None:
            failed_count += 1
            continue

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

    print(f"\n成功分析: {len(results['bge_vs_random']['key_l2'])} 个文件")
    print(f"失败: {failed_count} 个文件")

    return results


def plot_l2_distributions(results, output_dir):
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
    fig.suptitle('KV Cache L2 Distance Analysis (Full Dataset)', fontsize=16, fontweight='bold')

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
        ax.hist(data, bins=50, alpha=0.6, label=label, color=color, edgecolor='black')

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
        ax.hist(data, bins=50, alpha=0.6, label=label, color=color, edgecolor='black')

    ax.set_xlabel('Value L2 Distance', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Value Cache L2 Distance Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    output_path = os.path.join(output_dir, 'kv_l2_distance_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✓ 图表已保存: {output_path}")


def save_statistics(results, output_dir):
    """保存统计信息"""

    stats = {}

    for pair_name, data in results.items():
        key_l2 = data['key_l2']
        value_l2 = data['value_l2']

        stats[pair_name] = {
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

    output_path = os.path.join(output_dir, 'kv_l2_statistics.json')
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"✓ 统计数据已保存: {output_path}")

    # 打印统计摘要
    print("\n" + "="*80)
    print("统计摘要")
    print("="*80)

    for pair_name, pair_stats in stats.items():
        print(f"\n{pair_name.upper().replace('_', ' ')}:")
        print(f"  Key L2:   {pair_stats['key_l2']['mean']:.2f} ± {pair_stats['key_l2']['std']:.2f}")
        print(f"            [min: {pair_stats['key_l2']['min']:.2f}, max: {pair_stats['key_l2']['max']:.2f}]")
        print(f"  Value L2: {pair_stats['value_l2']['mean']:.2f} ± {pair_stats['value_l2']['std']:.2f}")
        print(f"            [min: {pair_stats['value_l2']['min']:.2f}, max: {pair_stats['value_l2']['max']:.2f}]")


def main():
    print("开始分析全数据集的KV Cache L2距离...\n")

    # 分析所有KV文件
    results = analyze_all_kv_files()

    if results is None:
        return

    # 绘制图表
    print("\n绘制图表...")
    plot_l2_distributions(results, OUTPUT_DIR)

    # 保存统计信息
    print("\n保存统计信息...")
    save_statistics(results, OUTPUT_DIR)

    print("\n" + "="*80)
    print("分析完成！")
    print("="*80)


if __name__ == "__main__":
    main()
