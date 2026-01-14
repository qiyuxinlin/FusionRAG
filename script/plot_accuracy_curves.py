#!/usr/bin/env python3
"""
FusionRAG 正确率曲线绘制脚本

直接修改下面的配置区域，然后运行：
    python plot_accuracy_curves.py
"""

import os
import re
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 无GUI环境

#############################################################################
# 配置区域 - 在这里修改输入输出路径
#############################################################################

# 输入：结果文件所在目录
INPUT_DIR = "/home/shm/document/exp/FusionRAG/result/no_preprocess/Qwen2.5-7B-Instruct/musique/results"

# 输出：保存图片的路径
OUTPUT_FILE = "/home/shm/document/exp/FusionRAG/result/fig/no_preprocess_accuracy_curves.png"

# 图表标题（可选，留空则使用默认标题）
TITLE = "FusionRAG_global_normal_topk10"

# 是否显示网格
SHOW_GRID = True

# 图片DPI（分辨率）
DPI = 300

#############################################################################
# 以下是脚本逻辑，通常不需要修改
#############################################################################

def extract_rate_from_filename(filename):
    """从文件名提取 rate 值"""
    match = re.search(r'rate[_-]([0-9.]+)', filename)
    if match:
        return float(match.group(1))
    return None

def extract_rate_from_content(content):
    """从文件内容提取 rate 值"""
    for line in content.split('\n'):
        if line.startswith('rate:'):
            return float(line.split(':')[1].strip())
    return None

def extract_accuracy(content):
    """
    从文件内容提取准确率

    Returns:
        dict: {'main': float, 'sub': float, 'f1': float, 'em': float}
    """
    results = {}

    for line in content.split('\n'):
        # Main Questions Accuracy: 69/135 (0.5111)
        if 'Main Questions Accuracy:' in line:
            match = re.search(r'\(([0-9.]+)\)', line)
            if match:
                results['main'] = float(match.group(1))

        # Sub Questions Accuracy: 167/250 (0.6680)
        if 'Sub Questions Accuracy:' in line:
            match = re.search(r'\(([0-9.]+)\)', line)
            if match:
                results['sub'] = float(match.group(1))

        # Average F1 Score: 0.4484
        if 'Average F1 Score:' in line:
            match = re.search(r':\s*([0-9.]+)', line)
            if match:
                results['f1'] = float(match.group(1))

        # Average EM Score: 0.1880
        if 'Average EM Score:' in line:
            match = re.search(r':\s*([0-9.]+)', line)
            if match:
                results['em'] = float(match.group(1))

    return results

def parse_txt_files(directory):
    """
    解析目录下所有 txt 文件

    Returns:
        list: [(rate, accuracies), ...] 其中 accuracies 是 dict
    """
    data_points = []

    txt_files = glob.glob(os.path.join(directory, "*.txt"))

    if not txt_files:
        print(f"❌ 错误: 在 {directory} 中没有找到 .txt 文件")
        return data_points

    print(f"✓ 找到 {len(txt_files)} 个 txt 文件\n")

    for txt_file in txt_files:
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # 提取 rate（优先从内容，备用从文件名）
            rate = extract_rate_from_content(content)
            if rate is None:
                rate = extract_rate_from_filename(os.path.basename(txt_file))

            if rate is None:
                print(f"⚠ 警告: 无法从 {os.path.basename(txt_file)} 提取 rate，跳过")
                continue

            # 提取准确率
            accuracies = extract_accuracy(content)

            if not accuracies:
                print(f"⚠ 警告: 无法从 {os.path.basename(txt_file)} 提取准确率，跳过")
                continue

            data_points.append((rate, accuracies))

            main_acc = accuracies.get('main', 0)
            sub_acc = accuracies.get('sub', 0)
            print(f"  Rate={rate:.1f}: Main={main_acc:.4f}, Sub={sub_acc:.4f}")

        except Exception as e:
            print(f"❌ 错误: 处理 {os.path.basename(txt_file)} 时出错: {e}")
            continue

    # 按 rate 排序
    data_points.sort(key=lambda x: x[0])

    return data_points

def plot_main_and_sub_curves(data_points, output_file, title=None, show_grid=True, dpi=300):
    """
    绘制主问题和子问题的准确率曲线（在同一张图上）

    Args:
        data_points: [(rate, accuracies), ...]
        output_file: 输出文件路径
        title: 图表标题
        show_grid: 是否显示网格
        dpi: 图片分辨率
    """
    if not data_points:
        print("❌ 错误: 没有数据点可以绘制")
        return

    rates = [x[0] for x in data_points]

    # 提取主问题和子问题的准确率
    main_accuracies = [x[1].get('main', 0) for x in data_points]
    sub_accuracies = [x[1].get('sub', 0) for x in data_points]

    # 创建图表
    plt.figure(figsize=(10, 6))

    # 绘制主问题准确率
    plt.plot(rates, main_accuracies,
             marker='o',
             linewidth=2.5,
             markersize=10,
             label='Main Questions Accuracy',
             color='#2E86AB',  # 深蓝色
             markerfacecolor='#2E86AB',
             markeredgecolor='white',
             markeredgewidth=1.5)

    # 绘制子问题准确率
    plt.plot(rates, sub_accuracies,
             marker='s',
             linewidth=2.5,
             markersize=9,
             label='Sub Questions Accuracy',
             color='#A23B72',  # 紫红色
             markerfacecolor='#A23B72',
             markeredgecolor='white',
             markeredgewidth=1.5)

    # 设置坐标轴标签
    plt.xlabel('Rate', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=14, fontweight='bold')

    # 设置标题
    if title:
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
    else:
        plt.title('FusionRAG: Main & Sub Questions Accuracy vs Rate',
                 fontsize=16, fontweight='bold', pad=20)

    # 设置网格
    if show_grid:
        plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

    # 设置图例
    plt.legend(fontsize=12, loc='best', framealpha=0.9, edgecolor='gray')

    # 设置坐标轴范围和刻度
    plt.xlim(min(rates) - 0.05, max(rates) + 0.05)
    y_min = min(min(main_accuracies), min(sub_accuracies))
    y_max = max(max(main_accuracies), max(sub_accuracies))
    y_range = y_max - y_min
    plt.ylim(max(0, y_min - 0.05 * y_range), min(1, y_max + 0.05 * y_range))

    # 设置刻度字体大小
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)

    # 紧凑布局
    plt.tight_layout()

    # 保存图片
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight')
    print(f"\n✓ 图片已保存到: {output_file}")
    print(f"  分辨率: {dpi} DPI")
    print(f"  包含曲线: 主问题准确率, 子问题准确率")

    plt.close()

    # 打印统计信息
    print("\n" + "="*60)
    print("数据统计:")
    print("="*60)
    print(f"{'Rate':<10} {'Main Acc':<12} {'Sub Acc':<12} {'Difference':<12}")
    print("-"*60)
    for rate, main_acc, sub_acc in zip(rates, main_accuracies, sub_accuracies):
        diff = sub_acc - main_acc
        print(f"{rate:<10.2f} {main_acc:<12.4f} {sub_acc:<12.4f} {diff:+12.4f}")
    print("="*60)

def main():
    print("="*60)
    print("FusionRAG 正确率曲线绘制")
    print("="*60)
    print(f"输入目录: {INPUT_DIR}")
    print(f"输出文件: {OUTPUT_FILE}")
    print("="*60)
    print()

    # 检查输入目录是否存在
    if not os.path.isdir(INPUT_DIR):
        print(f"❌ 错误: 输入目录不存在: {INPUT_DIR}")
        print("\n请在脚本开头的配置区域修改 INPUT_DIR 变量")
        return

    # 解析 txt 文件
    data_points = parse_txt_files(INPUT_DIR)

    if not data_points:
        print("\n❌ 错误: 没有找到有效的数据点")
        print("请确保目录中包含正确格式的 .txt 结果文件")
        return

    print()

    # 绘制曲线
    plot_main_and_sub_curves(
        data_points,
        output_file=OUTPUT_FILE,
        title=TITLE,
        show_grid=SHOW_GRID,
        dpi=DPI
    )

    print("\n✅ 完成！")

if __name__ == '__main__':
    main()
