#!/usr/bin/env python3
"""
KV融合预处理的快速入门示例

演示如何使用独立的KV融合模块进行实验
"""

import os
import sys
import torch
import json
from pathlib import Path

# 添加项目路径
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)


def example1_basic_fusion():
    """
    示例1: 基础的KV融合
    对比BGE召回 vs 随机召回
    """
    print("\n" + "="*80)
    print("示例1: 基础KV融合 - BGE vs Random")
    print("="*80)

    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    from kv_fusion_preprocess import preprocess_documents_pipeline, RecallMethod

    # 配置
    model_path = "/path/to/your/model"  # 修改为你的模型路径
    bge_model_path = "/path/to/bge-m3"   # 修改为BGE模型路径
    output_dir = "./examples/example1_basic"

    print(f"\n加载模型: {model_path}")

    # 加载模型
    config = AutoConfig.from_pretrained(model_path)
    config.torch_dtype = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=config.torch_dtype,
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 准备示例文档
    documents = [
        {'text': 'Artificial intelligence is transforming healthcare through diagnostic tools.', 'id': 'doc_0'},
        {'text': 'Machine learning algorithms can predict patient outcomes with high accuracy.', 'id': 'doc_1'},
        {'text': 'Deep learning models are used in medical image analysis.', 'id': 'doc_2'},
        {'text': 'The weather today is sunny and warm.', 'id': 'doc_3'},  # 无关文档
        {'text': 'Natural language processing helps in clinical documentation.', 'id': 'doc_4'},
    ]

    system_prompt = "You are a helpful AI assistant specialized in medical AI."

    print(f"\n准备了 {len(documents)} 个文档")

    # 对比两种方法
    for method in [RecallMethod.BGE, RecallMethod.RANDOM]:
        print(f"\n处理方法: {method.value}")

        result = preprocess_documents_pipeline(
            model=model,
            tokenizer=tokenizer,
            model_type='qwen2',  # 根据你的模型调整
            documents=documents,
            system_prompt=system_prompt,
            output_dir=os.path.join(output_dir, method.value),
            bge_model_path=bge_model_path if method == RecallMethod.BGE else None,
            topk=2,  # 融合top-2文档
            recall_method=method,
            device='cuda:0',
            random_seed=42
        )

        print(f"✓ 完成! KV cache保存在: {result['preprocess_dir']}")
        print(f"  - 文档数量: {result['num_documents']}")
        print(f"  - System长度: {result['system_len']}")
        print(f"  - 文档长度: {result['doc_lengths']}")

    print("\n分析两种方法的差异...")

    from analyse_kv_preprocessing import KVComparisonAnalyzer

    analyzer = KVComparisonAnalyzer(output_dir)
    analyzer.add_method('no_preprocess', os.path.join(output_dir, 'bge/no_preprocess'))
    analyzer.add_method('bge', os.path.join(output_dir, 'bge/preprocess_topk2_bge'))
    analyzer.add_method('random', os.path.join(output_dir, 'random/preprocess_topk2_random'))

    for doc_idx in range(len(documents)):
        comparison = analyzer.compare_all_methods(f'doc_{doc_idx}', 'no_preprocess')
        analyzer.visualize_comparison(
            comparison,
            os.path.join(output_dir, f'analysis/doc_{doc_idx}')
        )

    print(f"\n✓ 分析完成! 结果保存在: {output_dir}/analysis/")


def example2_position_importance():
    """
    示例2: 研究位置信息的重要性
    对比正常BGE召回 vs 打乱KV位置
    """
    print("\n" + "="*80)
    print("示例2: 位置信息的重要性 - BGE vs BGE_SHUFFLED")
    print("="*80)

    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    from kv_fusion_preprocess import preprocess_documents_pipeline, RecallMethod

    model_path = "/path/to/your/model"
    bge_model_path = "/path/to/bge-m3"
    output_dir = "./examples/example2_position"

    print(f"\n加载模型: {model_path}")

    config = AutoConfig.from_pretrained(model_path)
    config.torch_dtype = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=config.torch_dtype,
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 准备文档 - 使用有明显顺序结构的文档
    documents = [
        {'text': 'Step 1: Initialize the model parameters.'},
        {'text': 'Step 2: Load the training data.'},
        {'text': 'Step 3: Train the model for multiple epochs.'},
        {'text': 'Step 4: Evaluate on validation set.'},
        {'text': 'Step 5: Save the best checkpoint.'},
    ]

    system_prompt = "You are explaining a machine learning pipeline."

    # 对比正常召回和打乱位置
    for method in [RecallMethod.BGE, RecallMethod.BGE_SHUFFLED]:
        print(f"\n处理方法: {method.value}")

        result = preprocess_documents_pipeline(
            model=model,
            tokenizer=tokenizer,
            model_type='qwen2',
            documents=documents,
            system_prompt=system_prompt,
            output_dir=os.path.join(output_dir, method.value),
            bge_model_path=bge_model_path,
            topk=3,
            recall_method=method,
            device='cuda:0',
            random_seed=42
        )

        print(f"✓ 完成! {result['preprocess_dir']}")

    print("\n分析位置打乱的影响...")

    from analyse_kv_preprocessing import KVComparisonAnalyzer

    analyzer = KVComparisonAnalyzer(output_dir)
    analyzer.add_method('bge', os.path.join(output_dir, 'bge/preprocess_topk3_bge'))
    analyzer.add_method('bge_shuffled', os.path.join(output_dir, 'bge_shuffled/preprocess_topk3_bge_shuffled'))

    comparison = analyzer.compare_all_methods('doc_2', 'bge')  # 以BGE为参考
    analyzer.visualize_comparison(comparison, os.path.join(output_dir, 'analysis'))

    # 输出关键指标
    print("\n关键发现:")
    for method_name in ['bge_shuffled']:
        if method_name in comparison['methods']:
            method_data = comparison['methods'][method_name]

            # 计算平均相对差异
            key_diffs = [
                layer_diff['relative']
                for layer_diff in method_data['difference_l2']['key_diff'].values()
            ]
            avg_key_diff = sum(key_diffs) / len(key_diffs)

            print(f"  {method_name}:")
            print(f"    平均Key相对差异: {avg_key_diff:.4f}")
            print(f"    最大影响层: Layer {key_diffs.index(max(key_diffs))}")

    print(f"\n✓ 结果保存在: {output_dir}/analysis/")


def example3_topk_analysis():
    """
    示例3: 研究topk数量的影响
    对比topk=1, 2, 3, 5的效果
    """
    print("\n" + "="*80)
    print("示例3: TopK数量的影响")
    print("="*80)

    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    from kv_fusion_preprocess import preprocess_documents_pipeline, RecallMethod
    from analyse_kv_preprocessing import KVComparisonAnalyzer

    model_path = "/path/to/your/model"
    bge_model_path = "/path/to/bge-m3"
    output_dir = "./examples/example3_topk"

    print(f"\n加载模型: {model_path}")

    config = AutoConfig.from_pretrained(model_path)
    config.torch_dtype = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=config.torch_dtype,
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 准备较多文档以支持较大的topk
    documents = [
        {'text': f'Document {i}: This is document number {i} about AI and machine learning.'}
        for i in range(10)
    ]

    system_prompt = "You are a helpful AI assistant."

    # 测试不同的topk值
    topk_values = [1, 2, 3, 5]

    for topk in topk_values:
        print(f"\n处理 topk={topk}")

        result = preprocess_documents_pipeline(
            model=model,
            tokenizer=tokenizer,
            model_type='qwen2',
            documents=documents,
            system_prompt=system_prompt,
            output_dir=os.path.join(output_dir, f'topk{topk}'),
            bge_model_path=bge_model_path,
            topk=topk,
            recall_method=RecallMethod.BGE,
            device='cuda:0',
            random_seed=42
        )

        print(f"✓ 完成! {result['preprocess_dir']}")

    print("\n分析不同topk的效果...")

    analyzer = KVComparisonAnalyzer(output_dir)
    analyzer.add_method('no_preprocess', os.path.join(output_dir, 'topk1/no_preprocess'))

    for topk in topk_values:
        analyzer.add_method(
            f'topk{topk}',
            os.path.join(output_dir, f'topk{topk}/preprocess_topk{topk}_bge')
        )

    comparison = analyzer.compare_all_methods('doc_0', 'no_preprocess')
    analyzer.visualize_comparison(comparison, os.path.join(output_dir, 'analysis'))

    # 分析topk趋势
    print("\nTopK趋势分析:")
    print(f"{'TopK':<8} {'平均Key差异':<15} {'平均Value差异':<15}")
    print("-" * 40)

    for topk in topk_values:
        method_name = f'topk{topk}'
        if method_name in comparison['methods']:
            method_data = comparison['methods'][method_name]

            key_diffs = [
                layer_diff['relative']
                for layer_diff in method_data['difference_l2']['key_diff'].values()
            ]
            value_diffs = [
                layer_diff['relative']
                for layer_diff in method_data['difference_l2']['value_diff'].values()
            ]

            avg_key = sum(key_diffs) / len(key_diffs)
            avg_value = sum(value_diffs) / len(value_diffs)

            print(f"{topk:<8} {avg_key:<15.4f} {avg_value:<15.4f}")

    print(f"\n✓ 结果保存在: {output_dir}/analysis/")


def example4_custom_analysis():
    """
    示例4: 自定义KV分析
    展示如何进行细粒度的KV cache分析
    """
    print("\n" + "="*80)
    print("示例4: 自定义KV分析")
    print("="*80)

    from analyse_kv_preprocessing import KVCacheAnalyzer
    import matplotlib.pyplot as plt
    import numpy as np

    # 假设已经生成了KV cache
    kv_cache_dir = "./examples/example1_basic/bge/preprocess_topk2_bge"

    if not os.path.exists(kv_cache_dir):
        print(f"错误: KV cache目录不存在: {kv_cache_dir}")
        print("请先运行 example1_basic_fusion()")
        return

    analyzer = KVCacheAnalyzer(kv_cache_dir)

    # 加载第一个文档的KV cache
    print("\n加载 doc_0 的KV cache...")
    key_cache, value_cache = analyzer.load_kv_cache('doc_0')

    print(f"  Key cache: {len(key_cache)} 层")
    print(f"  第0层 Key shape: {key_cache[0].shape}")
    print(f"  第0层 Value shape: {value_cache[0].shape}")

    # 计算统计信息
    print("\n计算统计信息...")
    stats = analyzer.compute_kv_statistics(key_cache, value_cache)

    print("\n各层统计 (前5层):")
    print(f"{'Layer':<8} {'Key Mean':<12} {'Key Std':<12} {'Key Norm':<12}")
    print("-" * 48)

    for layer_idx in range(min(5, stats['num_layers'])):
        layer_name = f'layer_{layer_idx}'
        key_stats = stats['key_stats'][layer_name]
        print(f"{layer_idx:<8} {key_stats['mean']:<12.4f} {key_stats['std']:<12.4f} {key_stats['l2_norm']:<12.2f}")

    # 自定义可视化: 绘制每层的Key/Value范数
    print("\n生成自定义可视化...")

    num_layers = stats['num_layers']
    key_norms = [stats['key_stats'][f'layer_{i}']['l2_norm'] for i in range(num_layers)]
    value_norms = [stats['value_stats'][f'layer_{i}']['l2_norm'] for i in range(num_layers)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(range(num_layers), key_norms, marker='o', color='blue', label='Key L2 Norm')
    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('L2 Norm')
    ax1.set_title('Key Cache: L2 Norm per Layer')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(range(num_layers), value_norms, marker='o', color='red', label='Value L2 Norm')
    ax2.set_xlabel('Layer Index')
    ax2.set_ylabel('L2 Norm')
    ax2.set_title('Value Cache: L2 Norm per Layer')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    output_path = "./examples/example4_custom/layer_norms.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"✓ 可视化保存在: {output_path}")

    # 自定义分析: 找出变化最大的层
    print("\n分析层间差异...")

    key_norm_diffs = []
    for i in range(1, num_layers):
        diff = abs(key_norms[i] - key_norms[i-1])
        key_norm_diffs.append(diff)

    max_diff_idx = key_norm_diffs.index(max(key_norm_diffs))
    print(f"  Key范数变化最大的层: Layer {max_diff_idx} -> Layer {max_diff_idx+1}")
    print(f"  差异: {max(key_norm_diffs):.2f}")


def main():
    """主函数"""
    print("\n" + "="*80)
    print("KV融合预处理 - 示例脚本")
    print("="*80)

    print("\n可用的示例:")
    print("  1. 基础KV融合 (BGE vs Random)")
    print("  2. 位置信息重要性 (BGE vs BGE_SHUFFLED)")
    print("  3. TopK数量影响分析")
    print("  4. 自定义KV分析")

    choice = input("\n请选择要运行的示例 (1-4, 或 'all' 运行全部): ")

    if choice == '1':
        example1_basic_fusion()
    elif choice == '2':
        example2_position_importance()
    elif choice == '3':
        example3_topk_analysis()
    elif choice == '4':
        example4_custom_analysis()
    elif choice.lower() == 'all':
        print("\n运行所有示例...")
        try:
            example1_basic_fusion()
        except Exception as e:
            print(f"示例1失败: {e}")

        try:
            example2_position_importance()
        except Exception as e:
            print(f"示例2失败: {e}")

        try:
            example3_topk_analysis()
        except Exception as e:
            print(f"示例3失败: {e}")

        try:
            example4_custom_analysis()
        except Exception as e:
            print(f"示例4失败: {e}")
    else:
        print("无效的选择!")
        return

    print("\n" + "="*80)
    print("所有示例完成!")
    print("="*80)


if __name__ == "__main__":
    # 快速测试模式: 不需要交互,直接运行示例4
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        example4_custom_analysis()
    else:
        main()
