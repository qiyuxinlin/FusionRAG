#!/usr/bin/env python3
"""
Test Dynamic Ratio Calculation

测试动态重算比例功能，对比固定比例和动态比例的效果
"""

import os
import sys

# 设置 CUDA 设备
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from per_head_generation import main_with_draft_model


def test_fixed_ratio():
    """测试固定比例 (30%)"""
    print("\n" + "="*100)
    print("TEST 1: Fixed Ratio (30%)")
    print("="*100 + "\n")

    results = main_with_draft_model(
        target_model_path='/mnt/data/models/Qwen2.5-7B-Instruct',
        draft_model_path='/mnt/data/models/Qwen2.5-3B-Instruct',
        data_path='./result_reflect.json',
        cache_path='/mnt/data/reflect/',
        model_name='Qwen2.5-7B-Instruct',
        bge_model_path='/mnt/data/models/bge-m3-FP16',
        example_idx=4,
        sub_question_idx=1,
        total_ratio=0.3,
        use_dynamic_ratio=False,  # 使用固定比例
        max_new_tokens=100,
        device='cuda:0',
        output_path='./results/fixed_ratio_30.json'
    )

    return results


def test_dynamic_ratio():
    """测试动态比例"""
    print("\n" + "="*100)
    print("TEST 2: Dynamic Ratio")
    print("="*100 + "\n")

    results = main_with_draft_model(
        target_model_path='/mnt/data/models/Qwen2.5-7B-Instruct',
        draft_model_path='/mnt/data/models/Qwen2.5-3B-Instruct',
        data_path='./result_reflect.json',
        cache_path='/mnt/data/reflect/',
        model_name='Qwen2.5-7B-Instruct',
        bge_model_path='/mnt/data/models/bge-m3-FP16',
        example_idx=4,
        sub_question_idx=1,
        total_ratio=0.3,  # 作为 base_ratio
        use_dynamic_ratio=True,  # 使用动态比例
        min_ratio=0.20,
        max_ratio=0.50,
        max_new_tokens=100,
        device='cuda:0',
        output_path='./results/dynamic_ratio.json'
    )

    return results


def compare_results(fixed_results, dynamic_results):
    """对比两种方法的结果"""
    print("\n" + "="*100)
    print("COMPARISON: Fixed vs Dynamic Ratio")
    print("="*100 + "\n")

    print("Configuration:")
    print(f"  Fixed ratio:    {fixed_results['ratio_config']['actual_ratio']:.2%}")
    print(f"  Dynamic ratio:  {dynamic_results['ratio_config']['actual_ratio']:.2%}")
    print()

    print("Token Selection:")
    print(f"  Fixed:   {fixed_results['selected_tokens']} tokens ({fixed_results['selection_ratio']:.2%})")
    print(f"  Dynamic: {dynamic_results['selected_tokens']} tokens ({dynamic_results['selection_ratio']:.2%})")
    print()

    print("Generated Answers:")
    print(f"  Question: {fixed_results['question']}")
    print(f"  Ground Truth: {fixed_results['ground_truth']}")
    print()
    print(f"  Fixed (30%):   {fixed_results['generated_answer']}")
    print(f"  Dynamic:       {dynamic_results['generated_answer']}")
    print()

    # 检查答案正确性
    def check_answer(answer, label):
        has_1216 = '1216' in answer
        has_1220 = '1220' in answer
        correct = has_1216 and has_1220

        print(f"{label}:")
        print(f"  Contains '1216': {'✓' if has_1216 else '✗'}")
        print(f"  Contains '1220': {'✓' if has_1220 else '✗'}")
        print(f"  CORRECT: {'✓ YES' if correct else '✗ NO'}")
        print()

        return correct

    fixed_correct = check_answer(fixed_results['generated_answer'], "Fixed (30%)")
    dynamic_correct = check_answer(dynamic_results['generated_answer'], "Dynamic")

    # Dynamic analysis details
    if dynamic_results['dynamic_analysis']:
        print("Dynamic Ratio Analysis Details:")
        analysis = dynamic_results['dynamic_analysis']

        print(f"  Coverage Analysis:")
        for threshold, ratio in analysis['coverage_analysis'].items():
            print(f"    {threshold} coverage: {ratio:.2%}")

        print(f"\n  Distribution Features:")
        print(f"    Connected components: {analysis['num_components']}")
        print(f"    Spread ratio: {analysis['spread_ratio']:.4f}")
        print(f"    Gini coefficient: {analysis['gini_coefficient']:.4f}")

        print(f"\n  Adjustments Applied:")
        for key, value in analysis['adjustments'].items():
            print(f"    {key}: {value:+.4f}")

        print(f"\n  Final Ratio: {analysis['dynamic_ratio']:.2%}")

    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)

    if fixed_correct and dynamic_correct:
        print("✓ Both methods produced CORRECT answers")
        if dynamic_results['selected_tokens'] < fixed_results['selected_tokens']:
            savings = (1 - dynamic_results['selection_ratio'] / fixed_results['selection_ratio']) * 100
            print(f"  Dynamic ratio saved {savings:.1f}% tokens while maintaining correctness!")
        elif dynamic_results['selected_tokens'] > fixed_results['selected_tokens']:
            increase = (dynamic_results['selection_ratio'] / fixed_results['selection_ratio'] - 1) * 100
            print(f"  Dynamic ratio used {increase:.1f}% more tokens")
    elif fixed_correct and not dynamic_correct:
        print("✗ Fixed ratio correct, but Dynamic ratio FAILED")
        print("  → Dynamic ratio may be too aggressive, consider increasing min_ratio")
    elif not fixed_correct and dynamic_correct:
        print("✓ Dynamic ratio succeeded where Fixed ratio failed!")
    else:
        print("✗ Both methods FAILED to produce correct answers")

    print("="*100 + "\n")


def main():
    """主测试函数"""
    print("\n" + "="*100)
    print("Dynamic Ratio Calculation - Test Suite")
    print("="*100)

    # 创建结果目录
    os.makedirs('./results', exist_ok=True)

    # 测试 1: 固定比例
    fixed_results = test_fixed_ratio()

    # 测试 2: 动态比例
    dynamic_results = test_dynamic_ratio()

    # 对比结果
    compare_results(fixed_results, dynamic_results)

    print("\nAll tests completed!")
    print(f"Results saved to:")
    print(f"  - ./results/fixed_ratio_30.json")
    print(f"  - ./results/dynamic_ratio.json")


if __name__ == '__main__':
    main()
