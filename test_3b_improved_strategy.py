#!/usr/bin/env python3
"""
测试改进后的 3B DraftModel 策略

实验表明最佳参数组合：
- entropy_top_k=10 (选择更多层聚合，而非默认的 4 层)
- threshold_factor=0.3 (更低的阈值，使更多位置进入连通分量分析)

这样可以让 3B 选择的 tokens 更接近 7B (重叠率从 58.1% 提升到 62.8%)
"""
import sys
sys.path.insert(0, '/mnt/data/wjh/FusionRAG')

from test_fusionrag_reflect import main, PreprocessScope

if __name__ == '__main__':
    # 改进策略：更多熵选层 (10层) + 更低的阈值 (0.3)
    # 实验发现这个组合与 7B 选择重叠率最高 (62.79%)
    main(
        model_type='qwen',
        model_path='/mnt/data/models/Qwen2.5-7B-Instruct',
        draft_model_path='/mnt/data/models/Qwen2.5-3B-Instruct',
        data_path='./data/result_reflect.json',
        cache_path='/mnt/data/reflect/',
        model_name='Qwen2.5-7B-Instruct',
        dataset_name='2wikimqa',  # 数据集名称
        rate=0.2,
        topk=10,
        preprocess=True,
        use_entropy_selection=True,
        reprocess_method='DraftModel',
        draft_layer_selection='entropy',  # 熵选层
        entropy_top_k=2,  # 选择 2 层聚合 (测试更少层)
        draft_threshold_factor=0.3,  # 更低的阈值因子 (默认 0.5)
        preprocess_scope=PreprocessScope.GLOBAL,
        bge_model_path='/mnt/data/models/bge-m3-FP16',
        revert_rope=False,
        device="cuda:0",
        use_multi_gpu=False,
        openai_base_url="https://api.deepseek.com/v1",
        openai_api_key="sk-519d391217894b6e91e7c2ebf2a9f4df",
        openai_model="deepseek-chat",
        max_samples=200,
    )
