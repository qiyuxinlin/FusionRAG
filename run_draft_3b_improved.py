#!/usr/bin/env python3
"""
使用改进的 DraftModel (相似度重排序) 运行测试
"""
import sys
sys.path.insert(0, '/mnt/data/wjh/FusionRAG')

from test_fusionrag_reflect import main, PreprocessScope

if __name__ == '__main__':
    main(
        model_type='qwen',
        model_path='/mnt/data/models/Qwen2.5-7B-Instruct',
        draft_model_path='/mnt/data/models/Qwen2.5-3B-Instruct',
        data_path='./result_reflect.json',
        cache_path='/mnt/data/reflect/',
        model_name='Qwen2.5-7B-Instruct',
        rate=0.2,  # 测试 20% 重算比例
        topk=10,
        preprocess=True,
        use_entropy_selection=True,
        reprocess_method='DraftModel',
        draft_layer_selection='entropy',
        preprocess_scope=PreprocessScope.GLOBAL,
        bge_model_path='/mnt/data/models/bge-m3-FP16',
        revert_rope=False,
        device="cuda:0",
        use_multi_gpu=False,
        openai_base_url="https://api.deepseek.com/v1",
        openai_api_key="sk-519d391217894b6e91e7c2ebf2a9f4df",
        openai_model="deepseek-chat",
        max_samples=200,
        # 新增：启用相似度重排序改进
        use_similarity_rerank=True,
        rerank_multiplier=2.0
    )
