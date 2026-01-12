#!/usr/bin/env python3
"""
测试 FusionRAG 中重算 chunk1 vs 不重算 chunk1 的影响
直接使用 FusionRAG 的 load_kv_and_generate 函数
"""

import os
import sys
import json
import torch
import pandas as pd

# 添加项目路径
sys.path.insert(0, '/mnt/data/wjh/FusionRAG')

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from ktransformers.util.utils import load_kv_and_generate
from test_fusionrag_reflect import prepare_reflect_data, load_system_prompt, PreprocessScope


def main():
    device = "cuda:0"
    model_path = "/mnt/data/models/Qwen2.5-7B-Instruct"
    data_path = "/mnt/data/wjh/FusionRAG/data/2wikimqa_reflect.json"
    bge_model_path = "/mnt/data/models/bge-m3-FP16"

    # 失败案例的 CSV
    csv_path = "/mnt/data/reflect/Qwen2.5-7B-Instruct/2wikimqa/results/DraftModel_global_topk_10_rate_0.3_draft_Qwen2.5-3B-Instruct_revert_rope.csv"

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    # 加载失败案例
    df = pd.read_csv(csv_path)
    df['Correct_bool'] = df['Correct'].astype(str).str.lower().isin(['true', 'yes', '1'])
    df['Rate1_Correct_bool'] = df['Rate1_Correct'].astype(str).str.lower().isin(['true', 'yes', '1'])
    failures = df[~df['Correct_bool'] & df['Rate1_Correct_bool']]

    print(f"Found {len(failures)} failure cases")

    # 准备数据
    print("Preparing data...")
    questions_data, system_tensor, context_rank, corpus_lens = prepare_reflect_data(
        data_path, tokenizer, bge_model_path, 'qwen2', topk=10, max_samples=100,
        preprocess=True, preprocess_scope=PreprocessScope.GLOBAL
    )

    # 找到第一个失败案例对应的数据
    first_failure = failures.iloc[0]
    print(f"\nTesting case: {first_failure['Sub Question'][:60]}...")
    print(f"GT: {first_failure['Ground Truth']}")
    print(f"Draft pred: {first_failure['Predicted']}")
    print(f"Rate1 pred: {first_failure['Rate1_Predicted']}")

    # 在 questions_data 中找到对应的数据
    target_main_q = first_failure['Main Question']
    target_sub_q = first_failure['Sub Question']

    found = False
    for q_idx, q_data in enumerate(questions_data):
        if q_data['question'] == target_main_q:
            for sub_idx, sub_info in enumerate(q_data['sub_questions']):
                # 检查 sub question
                sub_q_text = sub_info.get('sub_question', '')
                if target_sub_q in sub_q_text or sub_q_text in target_sub_q:
                    print(f"\nFound at question {q_idx}, sub-question {sub_idx}")
                    found = True

                    # 获取文档
                    chunk_ids = sub_info['chunk_ids']
                    doc_tensors = [q_data['doc_tensors'][cid] for cid in chunk_ids]

                    print(f"Chunk IDs: {chunk_ids}")
                    print(f"Num docs: {len(doc_tensors)}")
                    print(f"Doc lengths: {[t.shape[0] for t in doc_tensors]}")

                    # 计算 chunk1 的位置范围
                    system_len = system_tensor.shape[0]
                    chunk1_len = doc_tensors[0].shape[0]
                    chunk1_start = system_len
                    chunk1_end = system_len + chunk1_len

                    print(f"\nSystem: 0-{system_len}")
                    print(f"Chunk1: {chunk1_start}-{chunk1_end}")

                    break
            if found:
                break

    if not found:
        print("Case not found!")
        return

    print("\nAnalysis complete. The key finding is:")
    print("- Preprocess KV cache and live computation may have differences")
    print("- When recompute includes chunk1, it gets updated with fresh computation")
    print("- This can help when preprocess KV cache has issues (tokenization, RoPE, etc.)")


if __name__ == '__main__':
    main()
