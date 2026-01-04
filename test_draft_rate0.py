#!/usr/bin/env python3
"""
测试 DraftModel 在 rate=0（不重算）时的效果。
运行：CUDA_VISIBLE_DEVICES=0 python test_draft_rate0.py
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
        rate=0.0,  # 不重算
        revert_rope=True,
        preprocess=True,
        reprocess_method='DraftModel',  # 使用 DraftModel 方法（但 rate=0 实际不会调用选择逻辑）
        draft_model_path='/mnt/data/models/Qwen2.5-3B-Instruct',
        draft_layer_selection='entropy',
        bge_model_path='/mnt/data/models/bge-m3',
        # OpenAI API (使用DeepSeek)
        openai_base_url="https://api.deepseek.com/v1",
        openai_api_key="sk-519d391217894b6e91e7c2ebf2a9f4df",
        openai_model="deepseek-chat",
    )
