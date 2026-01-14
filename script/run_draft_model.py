#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from test_fusionrag_reflect import main, PreprocessScope

if __name__ == '__main__':

    main(
        model_type='qwen',
        model_path='/mnt/data/models/Qwen2.5-7B-Instruct',
        draft_model_path='/mnt/data/models/Qwen2.5-3B-Instruct',
        data_path='./data/result_reflect.json',
        cache_path='/mnt/data/reflect/',
        model_name='Qwen2.5-7B-Instruct',
        dataset_name='musique',
        rate=0.2,  # base rate
        topk=10,
        preprocess=True,
        use_entropy_selection=False,
        reprocess_method='DynamicDraftModel',  # 使用新的 DynamicDraftModel 方法
        draft_layer_selection='entropy',
        preprocess_scope=PreprocessScope.GLOBAL,
        bge_model_path='/mnt/data/models/bge-m3-FP16',
        revert_rope=False,
        device="cuda:0",
        use_multi_gpu=True,
        openai_base_url="https://api.deepseek.com/v1",
        openai_api_key="sk-519d391217894b6e91e7c2ebf2a9f4df",
        openai_model="deepseek-chat",
        max_samples=200,
        # 动态 rate 范围
        min_rate=0.15,
        max_rate=0.35
    )
