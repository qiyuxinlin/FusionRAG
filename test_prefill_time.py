#!/usr/bin/env python3
"""
测试 7B 和 3B Qwen 模型的 prefill 时间对比
使用 result_reflect.json 中的真实测试数据
"""

import torch
import time
import json
import numpy as np
from transformers import AutoTokenizer, AutoConfig

# 模型路径
MODEL_7B_PATH = '/mnt/data/models/Qwen2.5-7B-Instruct'
MODEL_3B_PATH = '/mnt/data/models/Qwen2.5-3B-Instruct'
DATA_PATH = './result_reflect.json'


def get_system_prompt():
    """获取系统提示词"""
    return """You are a helpful assistant. Answer the question based on the provided documents.
You should only provide the exact answer to the question without any explanation.
Answer the question using the same language as the question."""


def load_model(model_path, device="cuda:0"):
    """加载模型"""
    print(f"\n加载模型: {model_path}")

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.torch_dtype = torch.bfloat16

    from ktransformers.models.modeling_qwen2 import Qwen2ForCausalLM

    torch.set_default_dtype(config.torch_dtype)
    with torch.no_grad():
        model = Qwen2ForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=config.torch_dtype,
            device_map=device
        )

    model.eval()
    return model, config


def create_cache(config, max_batch_size, max_seq_len, device):
    """创建 StaticCache"""
    from ktransformers.models.custom_cache import StaticCache
    return StaticCache(
        config=config,
        max_batch_size=max_batch_size,
        max_cache_len=max_seq_len,
        device=device,
        dtype=config.torch_dtype
    )


def measure_prefill_time(model, config, input_ids, device, num_warmup=2, num_runs=5):
    """测量 prefill 时间"""
    input_ids = input_ids.to(device)
    batch_size, seq_len = input_ids.shape

    # Warmup
    print(f"  Warmup ({num_warmup} runs)...")
    for _ in range(num_warmup):
        cache = create_cache(config, batch_size, seq_len + 100, device)
        with torch.no_grad():
            _ = model(input_ids, past_key_values=cache, use_cache=True)
        torch.cuda.synchronize()
        del cache
        torch.cuda.empty_cache()

    # 正式测量
    print(f"  Measuring ({num_runs} runs)...")
    times = []
    for i in range(num_runs):
        cache = create_cache(config, batch_size, seq_len + 100, device)

        torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.no_grad():
            outputs = model(input_ids, past_key_values=cache, use_cache=True)

        torch.cuda.synchronize()
        end = time.perf_counter()

        times.append(end - start)
        print(f"    Run {i+1}: {times[-1]*1000:.2f} ms")
        del cache
        torch.cuda.empty_cache()

    return times


def prepare_test_input(tokenizer, example_idx=0, sub_q_idx=0):
    """
    从 result_reflect.json 准备测试输入
    """
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    example = data[example_idx]
    sub_q = example['sub_questions'][sub_q_idx]

    # 构建系统提示
    system_prompt = get_system_prompt()
    system_text = f"<|im_start|>system\n{system_prompt}"

    # 构建文档
    docs = example['docs']
    chunk_ids = sub_q['chunk_ids']  # 这个子问题使用的文档 ID

    doc_texts = []
    for chunk_id in chunk_ids:
        doc = docs[chunk_id - 1]  # chunk_id 从 1 开始
        doc_texts.append(f"\n{doc}")

    # 构建问题
    question_text = f"<|im_end|>\n<|im_start|>user\nQuestion: {sub_q['query']}<|im_end|>\n<|im_start|>assistant\nAnswer: "

    # 完整输入
    full_text = system_text + "".join(doc_texts) + question_text

    # Tokenize
    input_ids = tokenizer.encode(full_text, return_tensors='pt')

    return input_ids, sub_q['query'], len(chunk_ids)


def main():
    device = "cuda:0"

    # 加载 tokenizer (两个模型共用)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_7B_PATH, trust_remote_code=True)

    # 准备测试输入 (使用第一个 example 的第一个 sub_question)
    input_ids, query, num_docs = prepare_test_input(tokenizer, example_idx=0, sub_q_idx=0)
    seq_len = input_ids.shape[1]

    print(f"\n{'='*60}")
    print("测试配置")
    print(f"{'='*60}")
    print(f"  问题: {query[:80]}...")
    print(f"  文档数量: {num_docs}")
    print(f"  序列长度: {seq_len} tokens")
    print(f"{'='*60}")

    results = {}

    # 测试 3B 模型
    print("\n" + "="*60)
    print("测试 Qwen2.5-3B-Instruct")
    print("="*60)

    model_3b, config_3b = load_model(MODEL_3B_PATH, device)
    print(f"\n序列长度: {seq_len} tokens")
    times_3b = measure_prefill_time(model_3b, config_3b, input_ids, device)
    results['3B'] = times_3b

    # 释放 3B 模型内存
    del model_3b
    torch.cuda.empty_cache()

    # 测试 7B 模型
    print("\n" + "="*60)
    print("测试 Qwen2.5-7B-Instruct")
    print("="*60)

    model_7b, config_7b = load_model(MODEL_7B_PATH, device)
    print(f"\n序列长度: {seq_len} tokens")
    times_7b = measure_prefill_time(model_7b, config_7b, input_ids, device)
    results['7B'] = times_7b

    # 释放 7B 模型内存
    del model_7b
    torch.cuda.empty_cache()

    # 打印结果汇总
    print("\n" + "="*60)
    print("结果汇总")
    print("="*60)

    times_3b = results['3B']
    times_7b = results['7B']

    avg_3b = np.mean(times_3b) * 1000  # ms
    std_3b = np.std(times_3b) * 1000
    avg_7b = np.mean(times_7b) * 1000
    std_7b = np.std(times_7b) * 1000

    speedup = avg_7b / avg_3b

    print(f"\n序列长度: {seq_len} tokens")
    print(f"  Qwen2.5-3B: {avg_3b:.2f} ± {std_3b:.2f} ms")
    print(f"  Qwen2.5-7B: {avg_7b:.2f} ± {std_7b:.2f} ms")
    print(f"  7B/3B 比值: {speedup:.2f}x (3B 比 7B 快 {speedup:.2f} 倍)")

    # 计算吞吐量 (tokens/second)
    throughput_3b = seq_len / (avg_3b / 1000)
    throughput_7b = seq_len / (avg_7b / 1000)
    print(f"\n吞吐量:")
    print(f"  Qwen2.5-3B: {throughput_3b:.0f} tokens/s")
    print(f"  Qwen2.5-7B: {throughput_7b:.0f} tokens/s")


if __name__ == '__main__':
    main()
