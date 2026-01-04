#!/usr/bin/env python3
"""
测试逐层动态重算比例策略

运行: CUDA_VISIBLE_DEVICES=0 timeout 1800 python test_layerwise_dynamic.py

对比：
1. 固定 rate=0.3 (baseline)
2. 固定 rate=0.175 (avg of layerwise linear)
3. 逐层递减 (linear: 0.3 → 0.05)
4. 逐层递减 (exponential: 0.3 → 0.05)
"""

import sys
sys.path.insert(0, '/mnt/data/wjh/FusionRAG')

from test_fusionrag_reflect import main


def test_fixed_rate_baseline():
    """测试固定 rate=0.3 作为 baseline"""
    print("\n" + "="*80)
    print("TEST 1: Fixed Rate = 0.30 (Baseline)")
    print("="*80 + "\n")

    main(
        model_type='qwen',
        model_path='/mnt/data/models/Qwen2.5-7B-Instruct',
        model_name='Qwen2.5-7B-Instruct',
        data_path='/mnt/data/wjh/FusionRAG/result_reflect.json',
        cache_path='/mnt/data/reflect/',
        device='cuda:0',
        max_samples=10,  # 只测试 10 个样本
        topk=10,
        rate=0.30,
        revert_rope=True,
        preprocess=True,
        reprocess_method='DraftModel',
        draft_model_path='/mnt/data/models/Qwen2.5-3B-Instruct',
        draft_layer_selection='entropy',
        bge_model_path='/mnt/data/models/bge-m3',
        openai_base_url="https://api.deepseek.com/v1",
        openai_api_key="sk-519d391217894b6e91e7c2ebf2a9f4df",
        openai_model="deepseek-chat",
    )


def test_layerwise_linear():
    """测试逐层线性递减"""
    print("\n" + "="*80)
    print("TEST 2: Layerwise Linear Decay (0.30 -> 0.05)")
    print("="*80 + "\n")

    main(
        model_type='qwen',
        model_path='/mnt/data/models/Qwen2.5-7B-Instruct',
        model_name='Qwen2.5-7B-Instruct',
        data_path='/mnt/data/wjh/FusionRAG/result_reflect.json',
        cache_path='/mnt/data/reflect/',
        device='cuda:0',
        max_samples=10,
        topk=10,
        rate=0.30,  # initial rate
        revert_rope=True,
        preprocess=True,
        reprocess_method='DraftModelLayerwise',  # 新方法
        draft_model_path='/mnt/data/models/Qwen2.5-3B-Instruct',
        draft_layer_selection='entropy',
        bge_model_path='/mnt/data/models/bge-m3',
        # 逐层递减参数
        layerwise_decay='linear',
        layerwise_final_rate=0.05,
        openai_base_url="https://api.deepseek.com/v1",
        openai_api_key="sk-519d391217894b6e91e7c2ebf2a9f4df",
        openai_model="deepseek-chat",
    )


def test_layerwise_exponential():
    """测试逐层指数递减"""
    print("\n" + "="*80)
    print("TEST 3: Layerwise Exponential Decay (0.30 -> 0.05)")
    print("="*80 + "\n")

    main(
        model_type='qwen',
        model_path='/mnt/data/models/Qwen2.5-7B-Instruct',
        model_name='Qwen2.5-7B-Instruct',
        data_path='/mnt/data/wjh/FusionRAG/result_reflect.json',
        cache_path='/mnt/data/reflect/',
        device='cuda:0',
        max_samples=10,
        topk=10,
        rate=0.30,  # initial rate
        revert_rope=True,
        preprocess=True,
        reprocess_method='DraftModelLayerwise',  # 新方法
        draft_model_path='/mnt/data/models/Qwen2.5-3B-Instruct',
        draft_layer_selection='entropy',
        bge_model_path='/mnt/data/models/bge-m3',
        # 逐层递减参数
        layerwise_decay='exponential',
        layerwise_final_rate=0.05,
        openai_base_url="https://api.deepseek.com/v1",
        openai_api_key="sk-519d391217894b6e91e7c2ebf2a9f4df",
        openai_model="deepseek-chat",
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', type=int, default=0, help='Test number (0=all, 1=baseline, 2=linear, 3=exp)')
    args = parser.parse_args()

    if args.test == 0 or args.test == 1:
        test_fixed_rate_baseline()

    if args.test == 0 or args.test == 2:
        test_layerwise_linear()

    if args.test == 0 or args.test == 3:
        test_layerwise_exponential()

    print("\n" + "="*80)
    print("ALL TESTS COMPLETED")
    print("="*80)
