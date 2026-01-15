#!/usr/bin/env python3
"""
比较两个FusionRAG result文件，分析样本答对/答错情况
"""

import pandas as pd
import argparse
from pathlib import Path
from typing import Tuple, Dict
import sys


def load_result_file(file_path: str) -> pd.DataFrame:
    """加载结果CSV文件"""
    try:
        df = pd.read_csv(file_path)
        print(f"成功加载文件: {file_path}")
        print(f"  - 样本数量: {len(df)}")
        return df
    except Exception as e:
        print(f"错误: 无法加载文件 {file_path}")
        print(f"  - 错误信息: {str(e)}")
        sys.exit(1)


def compare_results(df1: pd.DataFrame, df2: pd.DataFrame,
                   name1: str, name2: str) -> Dict:
    """比较两个结果文件"""

    # 确保两个文件的问题数量一致
    if len(df1) != len(df2):
        print(f"警告: 两个文件的样本数量不一致 ({len(df1)} vs {len(df2)})")

    # 分析结果
    both_correct = []      # 都答对
    only_1_correct = []    # 只有方案1答对
    only_2_correct = []    # 只有方案2答对
    both_wrong = []        # 都答错

    min_len = min(len(df1), len(df2))

    for i in range(min_len):
        row1 = df1.iloc[i]
        row2 = df2.iloc[i]

        # 获取正确性标志
        correct1 = row1['Correct'] if pd.notna(row1['Correct']) else False
        correct2 = row2['Correct'] if pd.notna(row2['Correct']) else False

        # 转换为布尔值（处理字符串 'True'/'False' 的情况）
        if isinstance(correct1, str):
            correct1 = correct1.strip().lower() == 'true'
        if isinstance(correct2, str):
            correct2 = correct2.strip().lower() == 'true'

        item = {
            'index': i,
            'main_question': row1['Main Question'],
            'sub_question': row1['Sub Question'],
            'ground_truth': row1['Ground Truth'],
            'predicted_1': row1['Predicted'],
            'predicted_2': row2['Predicted'],
            'correct_1': correct1,
            'correct_2': correct2,
            'f1_1': row1['F1'] if pd.notna(row1['F1']) else 0.0,
            'f1_2': row2['F1'] if pd.notna(row2['F1']) else 0.0,
            'em_1': row1['EM'] if pd.notna(row1['EM']) else 0.0,
            'em_2': row2['EM'] if pd.notna(row2['EM']) else 0.0,
        }

        if correct1 and correct2:
            both_correct.append(item)
        elif correct1 and not correct2:
            only_1_correct.append(item)
        elif not correct1 and correct2:
            only_2_correct.append(item)
        else:
            both_wrong.append(item)

    return {
        'both_correct': both_correct,
        'only_1_correct': only_1_correct,
        'only_2_correct': only_2_correct,
        'both_wrong': both_wrong,
        'total': min_len,
        'name1': name1,
        'name2': name2
    }


def print_statistics(results: Dict):
    """打印统计信息"""
    total = results['total']
    both_correct = len(results['both_correct'])
    only_1 = len(results['only_1_correct'])
    only_2 = len(results['only_2_correct'])
    both_wrong = len(results['both_wrong'])

    name1 = results['name1']
    name2 = results['name2']

    print("\n" + "="*80)
    print("统计摘要")
    print("="*80)
    print(f"总样本数: {total}")
    print(f"\n【都答对】: {both_correct} ({both_correct/total*100:.2f}%)")
    print(f"【只有 {name1} 答对】: {only_1} ({only_1/total*100:.2f}%)")
    print(f"【只有 {name2} 答对】: {only_2} ({only_2/total*100:.2f}%)")
    print(f"【都答错】: {both_wrong} ({both_wrong/total*100:.2f}%)")

    # 准确率
    correct_1 = both_correct + only_1
    correct_2 = both_correct + only_2
    print(f"\n{name1} 准确率: {correct_1}/{total} = {correct_1/total*100:.2f}%")
    print(f"{name2} 准确率: {correct_2}/{total} = {correct_2/total*100:.2f}%")
    print(f"准确率提升: {(correct_2-correct_1)/total*100:+.2f}%")
    print("="*80)


def save_detailed_comparison(results: Dict, output_file: str):
    """保存详细的对比结果到文件"""
    name1 = results['name1']
    name2 = results['name2']

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("详细对比结果\n")
        f.write("="*100 + "\n\n")

        # 统计信息
        total = results['total']
        both_correct = len(results['both_correct'])
        only_1 = len(results['only_1_correct'])
        only_2 = len(results['only_2_correct'])
        both_wrong = len(results['both_wrong'])

        f.write(f"总样本数: {total}\n")
        f.write(f"都答对: {both_correct} ({both_correct/total*100:.2f}%)\n")
        f.write(f"只有 {name1} 答对: {only_1} ({only_1/total*100:.2f}%)\n")
        f.write(f"只有 {name2} 答对: {only_2} ({only_2/total*100:.2f}%)\n")
        f.write(f"都答错: {both_wrong} ({both_wrong/total*100:.2f}%)\n\n")

        correct_1 = both_correct + only_1
        correct_2 = both_correct + only_2
        f.write(f"{name1} 准确率: {correct_1}/{total} = {correct_1/total*100:.2f}%\n")
        f.write(f"{name2} 准确率: {correct_2}/{total} = {correct_2/total*100:.2f}%\n")
        f.write(f"准确率提升: {(correct_2-correct_1)/total*100:+.2f}%\n")
        f.write("\n" + "="*100 + "\n\n")

        # 详细列表
        sections = [
            ("只有 {} 答对 (方案1优势)".format(name1), results['only_1_correct']),
            ("只有 {} 答对 (方案2优势)".format(name2), results['only_2_correct']),
            ("都答对", results['both_correct']),
            ("都答错", results['both_wrong'])
        ]

        for section_name, items in sections:
            f.write(f"\n{'='*100}\n")
            f.write(f"{section_name} ({len(items)} 个样本)\n")
            f.write(f"{'='*100}\n\n")

            for idx, item in enumerate(items, 1):
                f.write(f"[{idx}] 样本索引: {item['index']}\n")
                f.write(f"主问题: {item['main_question']}\n")
                f.write(f"子问题: {item['sub_question']}\n")
                f.write(f"标准答案: {item['ground_truth']}\n")
                f.write(f"\n{name1}:\n")
                f.write(f"  预测答案: {item['predicted_1']}\n")
                f.write(f"  正确性: {item['correct_1']}  |  F1: {item['f1_1']:.4f}  |  EM: {item['em_1']:.4f}\n")
                f.write(f"\n{name2}:\n")
                f.write(f"  预测答案: {item['predicted_2']}\n")
                f.write(f"  正确性: {item['correct_2']}  |  F1: {item['f1_2']:.4f}  |  EM: {item['em_2']:.4f}\n")
                f.write(f"\n{'-'*100}\n\n")

    print(f"\n详细对比结果已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='比较两个FusionRAG result文件，分析样本答对/答错情况',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本用法
  python compare_results.py \\
    --file1 /path/to/result1.csv \\
    --file2 /path/to/result2.csv

  # 指定输出文件和名称
  python compare_results.py \\
    --file1 result1.csv \\
    --file2 result2.csv \\
    --name1 "Baseline" \\
    --name2 "FusionRAG" \\
    --output comparison_detail.txt
        """
    )

    # parser.add_argument('--file1', type=str, required=True,
    #                    help='第一个result CSV文件路径')
    # parser.add_argument('--file2', type=str, required=True,
    #                    help='第二个result CSV文件路径')
    parser.add_argument('--name1', type=str, default='方案1',
                       help='第一个方案的名称 (默认: 方案1)')
    parser.add_argument('--name2', type=str, default='方案2',
                       help='第二个方案的名称 (默认: 方案2)')
    parser.add_argument('--output', type=str, default='comparison_detail.txt',
                       help='输出详细对比结果的文件路径 (默认: comparison_detail.txt)')

    args = parser.parse_args()

    # 加载文件
    print(f"\n加载结果文件...")
    df1 = load_result_file("/home/shm/document/exp/FusionRAG/result/Qwen2.5-7B-Instruct/musique/results/FusionRAG_global_topk_10_rate_0.0_revert_rope.csv")
    df2 = load_result_file("/home/shm/document/exp/FusionRAG/result/BGE_SHUFFLED /Qwen2.5-7B-Instruct/musique/FusionRAG_global_topk10_bge/rate_0.0_revert_rope.csv")

    # 比较结果
    print(f"\n开始比较...")
    results = compare_results(df1, df2, args.name1, args.name2)

    # 打印统计
    print_statistics(results)

    # 保存详细对比
    save_detailed_comparison(results, args.output)

    print(f"\n完成!")


if __name__ == '__main__':
    main()
