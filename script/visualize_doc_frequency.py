#!/home/shm/anaconda3/envs/fusionrag/bin/python
"""
可视化脚本：统计和可视化每个文档idx在 retrieved_results 中被使用的频率

输入文件: /home/shm/document/exp/FusionRAG/data/musique_merge_reflect_optimized.json
输出:
  - 统计结果 JSON
  - 柱状图（前20个最常用文档）
  - 分布直方图
  - 词云（按频率）
"""

import json
import os
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import numpy as np

# 使用英文字体
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_data(file_path):
    """加载 JSON 数据"""
    print(f"[1/5] Loading data from: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} records")
    return data

def extract_doc_ids(data):
    """提取所有 retrieved_results 中的文档 ID"""
    print("\n[2/5] Extracting document IDs...")

    doc_ids = []
    error_count = 0
    total_retrieved = 0

    for idx, record in enumerate(data):
        retrieved = record.get('retrieved_results', [])
        total_retrieved += len(retrieved)

        for item in retrieved:
            if isinstance(item, int):
                # 正常的文档 ID
                doc_ids.append(item)
            elif isinstance(item, dict):
                # 错误条目（如 'not_found_in_pool'）
                if item.get('idx') == -1:
                    error_count += 1

    print(f"  Total retrievals: {total_retrieved}")
    print(f"  Successful retrievals: {len(doc_ids)}")
    print(f"  Error retrievals: {error_count}")
    print(f"  Error rate: {error_count/total_retrieved*100:.2f}%")

    return doc_ids

def compute_statistics(doc_ids):
    """计算统计信息"""
    print("\n[3/5] Computing statistics...")

    counter = Counter(doc_ids)

    # 基本统计
    unique_docs = len(counter)
    total_usages = len(doc_ids)
    avg_usage = total_usages / unique_docs if unique_docs > 0 else 0

    # 排序
    most_common = counter.most_common()
    frequencies = list(counter.values())

    print(f"  Unique documents: {unique_docs}")
    print(f"  Total usages: {total_usages}")
    print(f"  Average usage: {avg_usage:.2f}")
    print(f"  Max frequency: {max(frequencies) if frequencies else 0}")
    print(f"  Min frequency: {min(frequencies) if frequencies else 0}")
    print(f"  Median: {np.median(frequencies) if frequencies else 0:.2f}")

    return counter, frequencies, most_common

def save_statistics(counter, frequencies, most_common, output_dir):
    """保存统计结果到 JSON"""
    print("\n[4/5] Saving statistics...")

    stats = {
        'unique_documents': len(counter),
        'total_usages': sum(counter.values()),
        'average_usage': sum(counter.values()) / len(counter) if counter else 0,
        'max_frequency': max(frequencies) if frequencies else 0,
        'min_frequency': min(frequencies) if frequencies else 0,
        'median_frequency': float(np.median(frequencies)) if frequencies else 0,
        'std_frequency': float(np.std(frequencies)) if frequencies else 0,
        'top_20_documents': [
            {'doc_id': int(doc_id), 'frequency': freq}
            for doc_id, freq in most_common[:20]
        ],
        'bottom_20_documents': [
            {'doc_id': int(doc_id), 'frequency': freq}
            for doc_id, freq in most_common[-20:]
        ]
    }

    output_path = os.path.join(output_dir, 'doc_frequency_stats.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"  Statistics saved to: {output_path}")

    return stats

def create_visualizations(counter, frequencies, most_common, stats, output_dir):
    """创建可视化图表"""
    print("\n[5/5] 创建可视化图表...")

    os.makedirs(output_dir, exist_ok=True)

    # 0. 使用频率饼图
    print("  生成: frequency_pie_chart.png")
    fig, ax = plt.subplots(figsize=(12, 9))

    # 统计各个频率的文档数量
    freq_dist = Counter(frequencies)

    # 准备饼图数据：按频率分组
    freq_labels = []
    freq_sizes = []
    colors = []

    # 从频率1到最高频率
    for freq in range(1, max(frequencies) + 1):
        if freq in freq_dist:
            count = freq_dist[freq]
            freq_labels.append(f'{freq} time{"s" if freq > 1 else ""}')
            freq_sizes.append(count)

    # 定义颜色渐变
    cmap = plt.cm.viridis
    colors = [cmap(i / len(freq_sizes)) for i in range(len(freq_sizes))]

    # 创建饼图
    wedges, texts, autotexts = ax.pie(
        freq_sizes,
        labels=freq_labels,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        textprops={'fontsize': 11}
    )

    # 美化百分比标签
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    ax.set_title('Document Usage Frequency Distribution',
                 fontsize=15, fontweight='bold', pad=20)

    # 添加图例说明
    legend_labels = [f'{label}: {size} docs ({size/len(counter)*100:.1f}%)'
                     for label, size in zip(freq_labels, freq_sizes)]
    ax.legend(wedges, legend_labels,
              title='Frequency Groups',
              loc='center left',
              bbox_to_anchor=(1, 0, 0.5, 1),
              fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'frequency_pie_chart.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # 1. 前20个最常用文档柱状图
    print("  生成: top_20_bar.png")
    fig, ax = plt.subplots(figsize=(14, 7))
    top_20 = most_common[:20]
    doc_ids = [str(doc_id) for doc_id, _ in top_20]
    freqs = [freq for _, freq in top_20]

    bars = ax.bar(range(len(doc_ids)), freqs, color='steelblue', edgecolor='navy', alpha=0.7)
    ax.set_xlabel('Document ID', fontsize=12, fontweight='bold')
    ax.set_ylabel('Usage Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Top 20 Most Frequently Used Documents', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(doc_ids)))
    ax.set_xticklabels(doc_ids, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_20_bar.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. 使用频率分布直方图
    print("  生成: frequency_distribution.png")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # 线性刻度
    ax1.hist(frequencies, bins=50, color='steelblue', edgecolor='white', alpha=0.7)
    ax1.set_xlabel('Usage Frequency', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Documents', fontsize=12, fontweight='bold')
    ax1.set_title('Document Usage Frequency Distribution (Linear Scale)',
                  fontsize=13, fontweight='bold')
    ax1.axvline(stats['average_usage'], color='red', linestyle='--',
                label=f"Mean: {stats['average_usage']:.2f}", linewidth=2)
    ax1.axvline(stats['median_frequency'], color='green', linestyle='--',
                label=f"Median: {stats['median_frequency']:.2f}", linewidth=2)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # 对数刻度
    ax2.hist(frequencies, bins=50, color='coral', edgecolor='white', alpha=0.7)
    ax2.set_xlabel('Usage Frequency', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Documents', fontsize=12, fontweight='bold')
    ax2.set_title('Document Usage Frequency Distribution (Log Scale)',
                  fontsize=13, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'frequency_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. 累计分布图 (CDF)
    print("  生成: cumulative_distribution.png")
    fig, ax = plt.subplots(figsize=(12, 6))

    sorted_freqs = sorted(frequencies, reverse=True)
    cumulative = np.cumsum(sorted_freqs)
    cumulative_pct = cumulative / cumulative[-1] * 100
    doc_rank = np.arange(1, len(sorted_freqs) + 1)

    ax.plot(doc_rank, cumulative_pct, linewidth=2, color='steelblue')
    ax.axhline(y=80, color='red', linestyle='--', alpha=0.5,
               label='80% of total usage')
    ax.axhline(y=50, color='orange', linestyle='--', alpha=0.5,
               label='50% of total usage')

    # 标记关键点
    idx_80 = np.argmax(cumulative_pct >= 80)
    idx_50 = np.argmax(cumulative_pct >= 50)
    ax.scatter([idx_80 + 1], [cumulative_pct[idx_80]], color='red', s=100, zorder=5)
    ax.scatter([idx_50 + 1], [cumulative_pct[idx_50]], color='orange', s=100, zorder=5)
    ax.text(idx_80 + 1, cumulative_pct[idx_80] + 3,
            f'Top {idx_80 + 1} docs\n({idx_80 + 1}/{len(sorted_freqs)} = {idx_80/len(sorted_freqs)*100:.1f}%)',
            ha='center', fontsize=9)
    ax.text(idx_50 + 1, cumulative_pct[idx_50] - 5,
            f'Top {idx_50 + 1} docs\n({idx_50 + 1}/{len(sorted_freqs)} = {idx_50/len(sorted_freqs)*100:.1f}%)',
            ha='center', fontsize=9)

    ax.set_xlabel('Document Rank (by usage frequency)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Usage Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Cumulative Distribution of Document Usage (Pareto Analysis)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, len(doc_rank) * 1.05)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cumulative_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 4. 长尾效应图
    print("  生成: long_tail.png")
    fig, ax = plt.subplots(figsize=(14, 6))

    # 绘制前100个文档的频率
    plot_limit = min(100, len(most_common))
    x = range(1, plot_limit + 1)
    y = [freq for _, freq in most_common[:plot_limit]]

    ax.bar(x, y, color='steelblue', alpha=0.7, width=0.8)
    ax.set_xlabel('Document Rank (by usage frequency)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Usage Frequency', fontsize=12, fontweight='bold')
    ax.set_title(f'Long Tail Effect - Top {plot_limit} Documents by Usage Frequency',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # 添加平均线
    ax.axhline(y=stats['average_usage'], color='red', linestyle='--',
               label=f"Mean: {stats['average_usage']:.2f}", linewidth=2)
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'long_tail.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n  All charts saved to: {output_dir}")

def main():
    # 配置
    input_file = "/home/shm/document/exp/FusionRAG/data/musique_merge_reflect_optimized.json"
    output_dir = "/home/shm/document/exp/FusionRAG/visualizations/doc_frequency"

    print("="*80)
    print("Document Usage Frequency Visualization Analysis")
    print("="*80)
    print(f"Input file: {input_file}")
    print(f"Output directory: {output_dir}")
    print()

    # 1. 加载数据
    data = load_data(input_file)

    # 2. 提取文档 ID
    doc_ids = extract_doc_ids(data)

    # 3. 计算统计
    counter, frequencies, most_common = compute_statistics(doc_ids)

    # 4. 保存统计
    os.makedirs(output_dir, exist_ok=True)
    stats = save_statistics(counter, frequencies, most_common, output_dir)

    # 5. 创建可视化
    create_visualizations(counter, frequencies, most_common, stats, output_dir)

    print("\n" + "="*80)
    print("Completed!")
    print("="*80)

    # 打印关键统计
    print(f"\nKey Findings:")
    print(f"  - Top 20 docs account for {sum(freq for _, freq in most_common[:20])/sum(frequencies)*100:.1f}% of total usage")
    print(f"  - Top 50 docs account for {sum(freq for _, freq in most_common[:50])/sum(frequencies)*100:.1f}% of total usage")
    print(f"  - Top 100 docs account for {sum(freq for _, freq in most_common[:100])/sum(frequencies)*100:.1f}% of total usage")
    print(f"  - {len([f for f in frequencies if f == 1])} docs used only once ({len([f for f in frequencies if f == 1])/len(frequencies)*100:.1f}%)")

if __name__ == "__main__":
    main()
