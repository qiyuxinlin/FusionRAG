#!/home/shm/anaconda3/envs/fusionrag/bin/python
"""
简单的 KV Cache 验证脚本

验证方法：检查 extra_info 中的统计信息，确认 KV cache 是否被使用
"""

import os
import sys
import json
import torch
from pathlib import Path

# 添加项目路径
sys.path.insert(0, '/home/shm/document/exp/FusionRAG')

from ktransformers.util.utils import load_kv_and_generate
from transformers import AutoTokenizer
from ktransformers.models.modeling_qwen2 import Qwen2ForCausalLM
from ktransformers.models.custom_cache import StaticCache

def main():
    # 配置
    KV_CACHE_DIR = "/mnt/data3/tmp/fusionrag_online_lazy/Qwen2.5-7B-Instruct/musique/kv_cache"
    DOC_POOL_PATH = "/home/shm/document/exp/FusionRAG/data/musique_input_rebuilt.json"
    MODEL_PATH = "/mnt/data/models/Qwen2.5-7B-Instruct"

    print("="*80)
    print("KV Cache 验证测试")
    print("="*80)

    # 1. 加载文档池
    print("\n[1/4] 加载文档池...")
    with open(DOC_POOL_PATH, 'r', encoding='utf-8') as f:
        doc_pool = json.load(f)
    doc_map = {doc['id']: doc['text'] for doc in doc_pool}
    print(f"  加载了 {len(doc_map)} 个文档")

    # 2. 扫描可用的 KV cache
    print("\n[2/4] 扫描 KV cache 文件...")
    cache_path = Path(KV_CACHE_DIR)
    key_files = list(cache_path.glob('doc_*_key.pt'))
    import re
    doc_ids = set()
    for f in key_files:
        match = re.match(r'doc_(\d+)_key\.pt', f.name)
        if match:
            doc_ids.add(int(match.group(1)))
    available_doc_ids = sorted(doc_ids)
    print(f"  找到 {len(available_doc_ids)} 个文档的 KV cache")

    # 3. 匹配文档
    print("\n[3/4] 匹配 KV cache 和文档池...")
    test_doc_ids = [doc_id for doc_id in available_doc_ids if doc_id in doc_map]
    print(f"  可测试的文档数: {len(test_doc_ids)}")
    if len(test_doc_ids) == 0:
        print("错误：没有可测试的文档！")
        return

    # 4. 加载模型
    print("\n[4/4] 加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    torch.set_default_dtype(torch.float16)
    with torch.no_grad():
        model = Qwen2ForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="cuda:0",
            trust_remote_code=True
        )
    model.eval()
    print("  模型加载完成")

    # 5. 测试单个文档
    print("\n" + "="*80)
    print("测试单个文档")
    print("="*80)

    doc_id = test_doc_ids[0]
    original_text = doc_map[doc_id]

    print(f"\n[Doc {doc_id}]")
    print(f"  原始文本长度: {len(original_text)} chars")
    print(f"  原始文本预览: {original_text[:100]}...")

    # 准备输入
    document_text_formatted = f"Document: {original_text}\n"
    document_tokens = tokenizer.encode(document_text_formatted, return_tensors='pt').squeeze(0)

    # 简单的 question：要求模型继续生成
    question = "Please summarize the above document in one sentence:"
    question_tokens = tokenizer.encode(question, return_tensors='pt').squeeze(0)

    passages = [document_tokens, question_tokens]

    print(f"  Document tokens: {len(document_tokens)}")
    print(f"  Question tokens: {len(question_tokens)}")

    # 初始化 KV cache
    past_key_values = StaticCache(
        config=model.config,
        max_batch_size=1,
        max_cache_len=32768,
        device='cuda',
        dtype=model.dtype,
        passage_len=32768
    )

    try:
        # 调用 load_kv_and_generate
        generated_tokens, _, extra_info = load_kv_and_generate(
            model, tokenizer, past_key_values, passages,
            doc_ids=[doc_id],
            load_path=KV_CACHE_DIR,
            max_new_tokens=50,
            device="cuda",
            revert_rope=True,
            reprocess_method='DraftModel',
            rate=0.0  # 全部使用 KV cache
        )

        print("\n" + "="*80)
        print("验证结果")
        print("="*80)

        # 检查 extra_info
        if extra_info and 'recompute_token_count' in extra_info:
            recompute_count = extra_info['recompute_token_count']
            print(f"\n✓ 成功！")
            print(f"  重算 token 数: {recompute_count}")
            print(f"  总 token 数: {len(document_tokens) + len(question_tokens)}")

            if recompute_count == len(question_tokens):
                print(f"\n✓ 验证通过：只有 question 被重算，Document 100% 使用了 KV cache！")
                print(f"  Document tokens: {len(document_tokens)} (来自 KV cache)")
                print(f"  Question tokens: {len(question_tokens)} (重新计算)")
            else:
                print(f"\n⚠ 警告：重算 token 数不符合预期")
                print(f"  预期重算: {len(question_tokens)} (只有 question)")
                print(f"  实际重算: {recompute_count}")
        else:
            print(f"\n⚠ 警告：extra_info 中没有 recompute_token_count 信息")
            print(f"  extra_info: {extra_info}")

        # 解码生成结果
        if isinstance(generated_tokens, list):
            generated_tokens = torch.tensor(generated_tokens)
        generated_text = tokenizer.decode(generated_tokens)
        print(f"\n生成的回答预览:")
        print(f"  {generated_text[:200]}...")

    except Exception as e:
        import traceback
        print(f"\n✗ 错误: {e}")
        print(f"\n[TRACEBACK]")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
