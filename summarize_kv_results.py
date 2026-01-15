#!/usr/bin/env python3
"""
快速汇总KV分析结果

从JSON报告中提取关键指标并生成汇总表格
"""

import json
import glob
import numpy as np
from pathlib import Path
import sys

def main():
    output_dir = "/home/shm/document/exp/FusionRAG/kv_analysis_output"

    # 找到所有报告文件
    report_files = sorted(glob.glob(f"{output_dir}/sample_*_report.json"))

    if not report_files:
        print("⚠ 没有找到分析报告，请先运行 analyze_sample_kv.py")
        sys.exit(1)

    print("="*80)
    print("KV Cache 相似度分析汇总")
    print("="*80)
    print(f"找到 {len(report_files)} 个样本的分析报告\n")

    # 收集所有相似度数据
    key_sims_bge_random = []
    value_sims_bge_random = []
    key_sims_bge_noprep = []
    value_sims_bge_noprep = []
    key_sims_random_noprep = []
    value_sims_random_noprep = []

    # 详细表格
    print("-"*80)
    print(f"{'Sample':<10} {'Question':<40} {'Key(B-R)':<12} {'Val(B-R)':<12} {'Key(R-N)':<12} {'Val(R-N)':<12}")
    print("-"*80)

    for report_file in report_files:
        with open(report_file, 'r') as f:
            data = json.load(f)

        sample_idx = data['sample_idx']
        question = data['question'][:37] + "..." if len(data['question']) > 40 else data['question']

        # 提取相似度
        bge_random = data['kv_similarity']['bge_vs_random']
        bge_noprep = data['kv_similarity']['bge_vs_no_prep']
        random_noprep = data['kv_similarity']['random_vs_no_prep']

        key_sim_br = bge_random['key_cosine_similarity']
        val_sim_br = bge_random['value_cosine_similarity']
        key_sim_rn = random_noprep['key_cosine_similarity']
        val_sim_rn = random_noprep['value_cosine_similarity']

        key_sims_bge_random.append(key_sim_br)
        value_sims_bge_random.append(val_sim_br)
        key_sims_bge_noprep.append(bge_noprep['key_cosine_similarity'])
        value_sims_bge_noprep.append(bge_noprep['value_cosine_similarity'])
        key_sims_random_noprep.append(key_sim_rn)
        value_sims_random_noprep.append(val_sim_rn)

        print(f"#{sample_idx:<9} {question:<40} {key_sim_br:>11.4f} {val_sim_br:>11.4f} {key_sim_rn:>11.4f} {val_sim_rn:>11.4f}")

    print("-"*80)

    # 统计汇总
    print("\n" + "="*80)
    print("统计汇总")
    print("="*80)

    def print_stats(name, values):
        mean = np.mean(values)
        std = np.std(values)
        median = np.median(values)
        min_val = np.min(values)
        max_val = np.max(values)
        print(f"{name:<30} 均值: {mean:.4f} ± {std:.4f} | 中位数: {median:.4f} | 范围: [{min_val:.4f}, {max_val:.4f}]")

    print("\nBGE vs Random:")
    print_stats("  Key Cosine Similarity", key_sims_bge_random)
    print_stats("  Value Cosine Similarity", value_sims_bge_random)

    print("\nBGE vs No Preprocess:")
    print_stats("  Key Cosine Similarity", key_sims_bge_noprep)
    print_stats("  Value Cosine Similarity", value_sims_bge_noprep)

    print("\nRandom vs No Preprocess:")
    print_stats("  Key Cosine Similarity", key_sims_random_noprep)
    print_stats("  Value Cosine Similarity", value_sims_random_noprep)

    # 关键结论
    print("\n" + "="*80)
    print("🎯 关键结论")
    print("="*80)

    avg_key_sim = np.mean(key_sims_bge_random)
    avg_val_sim = np.mean(value_sims_bge_random)

    print(f"\n1. BGE vs Random 的 Key Cache 平均相似度: {avg_key_sim:.4f}")
    if avg_key_sim > 0.95:
        print("   ✅ 极高相似（> 0.95）！**强烈支持位置适应假设**")
    elif avg_key_sim > 0.9:
        print("   ✓ 高度相似（> 0.9），支持位置适应假设")
    elif avg_key_sim > 0.8:
        print("   ~ 中等相似（> 0.8），假设有一定支撑")
    else:
        print("   ✗ 相似度较低（< 0.8），语义相似度可能重要")

    print(f"\n2. BGE vs Random 的 Value Cache 平均相似度: {avg_val_sim:.4f}")
    if avg_val_sim > 0.95:
        print("   ✅ 极高相似（> 0.95）")
    elif avg_val_sim > 0.85:
        print("   ✓ 高度相似（> 0.85）")
    elif avg_val_sim > 0.7:
        print("   ~ 中等相似（> 0.7）")
    else:
        print("   ⚠ 相似度中等偏低（< 0.7）")

    # 对比Random vs No_Preprocess
    avg_val_rn = np.mean(value_sims_random_noprep)
    print(f"\n3. Random vs No_Preprocess 的 Value Cache 平均相似度: {avg_val_rn:.4f}")
    if avg_val_rn > avg_val_sim:
        delta = avg_val_rn - avg_val_sim
        print(f"   ⚠ 比 BGE vs Random 更相似（+{delta:.4f}）")
        print("   这表明：Random和No_Preprocess的Value更接近，")
        print("   进一步支持\"召回方法对KV影响不大\"的假设")

    # 最终建议
    print("\n" + "="*80)
    print("📊 论文建议")
    print("="*80)

    if avg_key_sim > 0.95:
        print("""
✅ **强证据支持位置适应假设**

关键发现：
1. BGE 和 Random 的 Key Cache 几乎完全相同（相似度 > 0.95）
2. 这表明召回方法（BGE相似度 vs 随机）对KV cache的影响极小
3. FusionRAG 的性能提升主要来自位置适应（将KV从短文本空间适应到长文本空间）

论文可以包含：
- KV相似度表格（展示高相似度）
- 可视化图表（KV分布对比）
- 结论：预处理阶段的跨文档注意力不是主要因素
""")
    elif avg_key_sim > 0.85:
        print("""
✓ **中等证据支持位置适应假设**

建议：
- 继续分析更多样本
- 结合注意力可视化验证
- 可能需要额外实验（如repeat_self消融）
""")
    else:
        print("""
⚠ **需要重新审视假设**

发现：
- BGE 和 Random 的 KV Cache 差异较大
- 语义相似度可能确实影响了KV表示
- 建议进行注意力分析，看跨文档注意力占比
""")

    print("="*80)


if __name__ == "__main__":
    main()
