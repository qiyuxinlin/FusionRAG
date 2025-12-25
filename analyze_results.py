#!/usr/bin/env python
"""
FusionRAG 结果分析工具
快速分析和可视化Process Cache实验的结果

使用方式:
    # 分析特定结果目录
    python analyze_results.py --csv_file output/cache/musique-200.jsonl/Mistral-7B/reprocess_method_processCache_rate_0.15.csv

    # 交互式选择
    python analyze_results.py --interactive
"""

import os
import sys
import csv
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import statistics


def print_header(text, char="=", width=80):
    """打印格式化的标题"""
    print("\n" + char * width)
    print(text.center(width))
    print(char * width + "\n")


def print_section(text, char="-", width=80):
    """打印格式化的小标题"""
    print("\n" + char * width)
    print(text)
    print(char * width + "\n")


def find_csv_files(root_dir: str) -> List[str]:
    """递归查找所有CSV结果文件"""
    csv_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.csv') and 'reprocess_method_' in file:
                csv_files.append(os.path.join(root, file))
    return sorted(csv_files)


def parse_csv_filename(filename: str) -> Dict[str, str]:
    """从CSV文件名解析参数"""
    basename = os.path.basename(filename)
    # 格式: reprocess_method_{method}_rate_{rate}_revert_rope_{rope}[_topk_{topk}][_full_recompute].csv

    params = {}
    parts = basename.replace('.csv', '').split('_')

    i = 0
    while i < len(parts):
        if i + 1 < len(parts):
            key = parts[i]
            if i + 2 < len(parts) and parts[i + 1].replace('.', '').replace('-', '').isdigit():
                # 数值参数
                params[key] = parts[i + 1]
                i += 2
            elif i + 2 < len(parts) and parts[i + 1] in ['True', 'False']:
                # 布尔参数
                params[key] = parts[i + 1]
                i += 2
            else:
                i += 1
        else:
            i += 1

    return params


def read_csv_results(csv_file: str) -> Tuple[List[Dict], int]:
    """读取CSV结果文件"""
    results = []
    total = 0

    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                results.append(row)
                total += 1
    except Exception as e:
        print(f"❌ 错误读取文件 {csv_file}: {str(e)}")
        return [], 0

    return results, total


def compute_metrics(results: List[Dict]) -> Dict[str, float]:
    """计算评估指标"""
    if not results:
        return {}

    exact_matches = 0
    rouge_scores = []

    for result in results:
        # 精确匹配
        if 'Pred Answer' in result and 'Real Answer' in result:
            pred = result['Pred Answer'].strip().lower()
            real = result['Real Answer'].strip().lower()
            if pred == real:
                exact_matches += 1

        # ROUGE分数(简化版)
        if 'ROUGE' in result:
            try:
                rouge_scores.append(float(result['ROUGE']))
            except:
                pass

    metrics = {
        'total_samples': len(results),
        'exact_matches': exact_matches,
        'em_score': exact_matches / len(results) if results else 0,
        'avg_rouge': statistics.mean(rouge_scores) if rouge_scores else 0,
    }

    return metrics


def analyze_results(csv_file: str) -> None:
    """分析单个结果文件"""
    if not os.path.exists(csv_file):
        print(f"❌ 文件不存在: {csv_file}")
        return

    print_header("结果分析", "=")
    print(f"文件: {csv_file}\n")

    # 解析文件名参数
    params = parse_csv_filename(csv_file)
    print("实验参数:")
    for key, value in params.items():
        print(f"  {key}: {value}")

    # 读取结果
    results, total = read_csv_results(csv_file)
    if total == 0:
        print(f"\n❌ 无法读取结果")
        return

    print(f"\n总样本数: {total}")

    # 计算指标
    metrics = compute_metrics(results)

    print_section("指标统计")
    print(f"精确匹配数: {metrics['exact_matches']} / {metrics['total_samples']}")
    print(f"EM Score: {metrics['em_score']:.4f} ({metrics['em_score']*100:.2f}%)")
    print(f"平均ROUGE分数: {metrics['avg_rouge']:.4f}")

    # 显示样本
    print_section("样本详情 (前10个)")
    print(f"{'序号':<5} {'问题':<40} {'答案匹配':<10}")
    print("-" * 60)

    for idx, result in enumerate(results[:10], 1):
        question = result.get('Question', '')[:35]
        pred = result.get('Pred Answer', '')
        real = result.get('Real Answer', '')
        match = '✓' if pred.lower().strip() == real.lower().strip() else '✗'

        print(f"{idx:<5} {question:<40} {match:<10}")

        if pred.lower().strip() != real.lower().strip():
            print(f"       Expected: {real[:50]}")
            print(f"       Got: {pred[:50]}")

    # 统计信息
    print_section("详细统计")

    # 按匹配情况分类
    matched = sum(1 for r in results if r.get('Pred Answer', '').lower().strip() == r.get('Real Answer', '').lower().strip())
    not_matched = total - matched

    print(f"完全匹配: {matched} ({matched/total*100:.2f}%)")
    print(f"不匹配: {not_matched} ({not_matched/total*100:.2f}%)")

    # 答案长度统计
    answer_lengths = []
    for result in results:
        if 'Pred Answer' in result:
            answer_lengths.append(len(result['Pred Answer']))

    if answer_lengths:
        print(f"\n答案长度统计:")
        print(f"  最小: {min(answer_lengths)} 字符")
        print(f"  最大: {max(answer_lengths)} 字符")
        print(f"  平均: {statistics.mean(answer_lengths):.1f} 字符")

    # 生成报告
    report_file = csv_file.replace('.csv', '_analysis.json')
    report = {
        'timestamp': datetime.now().isoformat(),
        'csv_file': csv_file,
        'parameters': params,
        'metrics': metrics,
        'detailed_stats': {
            'matched': matched,
            'not_matched': not_matched,
            'match_rate': matched / total if total > 0 else 0,
        }
    }

    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n✓ 详细报告已保存: {report_file}")


def interactive_select(root_dir: str = './output/cache/') -> str:
    """交互式选择CSV文件"""
    print_header("交互式结果选择")

    # 查找所有CSV文件
    csv_files = find_csv_files(root_dir)

    if not csv_files:
        print(f"❌ 在 {root_dir} 中未找到CSV结果文件")
        return None

    print(f"找到 {len(csv_files)} 个结果文件:\n")

    for idx, f in enumerate(csv_files, 1):
        # 获取相对路径
        rel_path = os.path.relpath(f, root_dir)
        print(f"{idx}. {rel_path}")

    while True:
        try:
            choice = input(f"\n请选择文件编号 (1-{len(csv_files)}) 或输入完整路径 ('q'退出): ").strip()

            if choice.lower() == 'q':
                return None

            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(csv_files):
                    return csv_files[idx]
                else:
                    print(f"❌ 无效的编号，请输入1-{len(csv_files)}")
            else:
                if os.path.exists(choice):
                    return choice
                else:
                    print(f"❌ 文件不存在: {choice}")

        except KeyboardInterrupt:
            return None
        except Exception as e:
            print(f"❌ 错误: {str(e)}")


def compare_results(csv_files: List[str]) -> None:
    """对比多个结果文件"""
    if not csv_files:
        return

    print_header("结果对比分析", "=")

    results_data = []

    for csv_file in csv_files:
        if not os.path.exists(csv_file):
            print(f"⚠️  文件不存在: {csv_file}")
            continue

        results, total = read_csv_results(csv_file)
        params = parse_csv_filename(csv_file)
        metrics = compute_metrics(results)

        results_data.append({
            'file': os.path.basename(csv_file),
            'params': params,
            'metrics': metrics,
        })

    if not results_data:
        print("❌ 无法加载任何结果")
        return

    # 显示对比表格
    print_section("性能对比")
    print(f"{'方法':<20} {'Rate':<8} {'EM':<8} {'ROUGE':<10} {'样本数':<8}")
    print("-" * 60)

    for data in results_data:
        method = data['params'].get('method', 'unknown')[:20]
        rate = data['params'].get('rate', '?')
        em = f"{data['metrics'].get('em_score', 0):.4f}"
        rouge = f"{data['metrics'].get('avg_rouge', 0):.4f}"
        samples = str(data['metrics'].get('total_samples', 0))

        print(f"{method:<20} {rate:<8} {em:<8} {rouge:<10} {samples:<8}")

    # 计算最佳结果
    best_em_idx = max(range(len(results_data)),
                      key=lambda i: results_data[i]['metrics'].get('em_score', 0))
    best_rouge_idx = max(range(len(results_data)),
                         key=lambda i: results_data[i]['metrics'].get('avg_rouge', 0))

    print("\n" + "="*60)
    print(f"EM最佳: {results_data[best_em_idx]['file']}")
    print(f"ROUGE最佳: {results_data[best_rouge_idx]['file']}")
    print("="*60)


def export_summary(root_dir: str = './output/cache/') -> None:
    """导出所有结果的总结"""
    print_header("导出结果总结")

    csv_files = find_csv_files(root_dir)

    if not csv_files:
        print(f"❌ 未找到结果文件")
        return

    summary_file = os.path.join(root_dir, 'results_summary.csv')

    with open(summary_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['File', 'Method', 'Rate', 'RopeRevert', 'EM Score', 'ROUGE', 'Total Samples'])

        for csv_file in csv_files:
            results, total = read_csv_results(csv_file)
            if not results:
                continue

            params = parse_csv_filename(csv_file)
            metrics = compute_metrics(results)

            filename = os.path.basename(csv_file)
            method = params.get('method', 'unknown')
            rate = params.get('rate', '?')
            rope = params.get('rope', 'False')
            em = f"{metrics.get('em_score', 0):.4f}"
            rouge = f"{metrics.get('avg_rouge', 0):.4f}"

            writer.writerow([filename, method, rate, rope, em, rouge, total])

    print(f"✓ 总结已导出: {summary_file}")

    # 显示摘要
    print_section("结果摘要")
    with open(summary_file, 'r') as f:
        print(f.read())


def main():
    parser = argparse.ArgumentParser(
        description="FusionRAG 结果分析工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:

  # 分析特定文件
  python analyze_results.py --csv_file output/cache/musique-200.jsonl/Mistral-7B/reprocess_method_processCache_rate_0.15.csv

  # 交互式选择
  python analyze_results.py --interactive

  # 对比多个文件
  python analyze_results.py --compare \\
    output/cache/musique-200.jsonl/Mistral-7B/reprocess_method_processCache_rate_0.15.csv \\
    output/cache/musique-200.jsonl/Mistral-7B/reprocess_method_cacheBlend_rate_0.15.csv

  # 导出总结
  python analyze_results.py --export-summary --cache_dir output/cache/
        """
    )

    parser.add_argument('--csv_file', type=str, default=None,
                        help='要分析的CSV文件路径')
    parser.add_argument('--interactive', action='store_true',
                        help='交互式选择文件')
    parser.add_argument('--compare', nargs='+', type=str,
                        help='对比多个结果文件')
    parser.add_argument('--export-summary', action='store_true',
                        help='导出所有结果的总结')
    parser.add_argument('--cache_dir', type=str, default='./output/cache/',
                        help='缓存目录')

    args = parser.parse_args()

    print_header("FusionRAG 结果分析工具", "=")

    # 处理参数
    if args.export_summary:
        export_summary(args.cache_dir)
    elif args.compare:
        compare_results(args.compare)
    elif args.csv_file:
        analyze_results(args.csv_file)
    elif args.interactive:
        csv_file = interactive_select(args.cache_dir)
        if csv_file:
            analyze_results(csv_file)
    else:
        # 默认: 导出总结
        print("未指定参数，使用 --help 查看用法")
        print("\n建议使用:")
        print("  python analyze_results.py --interactive")


if __name__ == "__main__":
    main()
