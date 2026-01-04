#!/usr/bin/env python3
"""
测试 DraftModelDynamic 方法。
运行：CUDA_VISIBLE_DEVICES=0 python test_draft_dynamic.py

最终公式 (基于大量分析的结论):

  rate = coverage_90_ratio * scale_factor

其中:
  - coverage_90_ratio = 达到 90% attention 覆盖需要的 token 比例
  - scale_factor = 0.7 (经验缩放系数)

物理意义:
  "达到 90% attention 覆盖需要多少比例的 token，就用多少比例去重算"

分析结论:
  - 所有 attention 特征（gini, entropy, cross_doc）都无法可靠区分 Easy/Medium
  - Easy 和 Medium 的特征分布高度重叠（分离度 < 0.5）
  - 问题的"难度"并不反映在 attention 分布中
  - 因此采用简单公式，不做复杂的条件判断
"""

import sys
sys.path.insert(0, '/mnt/data/wjh/FusionRAG')

from test_fusionrag_reflect import main

if __name__ == "__main__":
    main(
        model_type='qwen',
        model_path='/mnt/data/models/Qwen2.5-7B-Instruct',
        model_name='Qwen2.5-7B-Instruct',
        data_path='/mnt/data/wjh/FusionRAG/result_reflect.json',
        cache_path='/mnt/data/reflect/',
        device='cuda:0',
        max_samples=None,  # 测试所有问题
        topk=10,
        rate=0.3,  # 作为参考
        revert_rope=True,
        preprocess=True,
        reprocess_method='DraftModelDynamic',  # 使用动态算法
        draft_model_path='/mnt/data/models/Qwen2.5-3B-Instruct',
        draft_layer_selection='entropy',
        bge_model_path='/mnt/data/models/bge-m3',
        # 动态 rate 参数
        min_rate=0.10,   # 最小 rate
        max_rate=0.35,   # 最大 rate (略高于固定 0.3，留有余地)
        # OpenAI API (使用DeepSeek)
        openai_base_url="https://api.deepseek.com/v1",
        openai_api_key="sk-519d391217894b6e91e7c2ebf2a9f4df",
        openai_model="deepseek-chat",
    )
