#!/usr/bin/env python3
"""
验证脚本：测试模型能否在给定完整文档 + repeat prompt 的情况下重复文档

这是 baseline 测试，不涉及 KV cache 的保存/加载
目的：验证模型本身是否具备重复文档的能力
"""

import os
import sys
import json
import torch
import argparse
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

# 添加项目路径
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_dir)

from ktransformers.util.utils_v2 import compute_f1

import ast  # 用于处理单引号的字典格式
import re
def verify_repeat_capability(model, tokenizer, doc_text: str, repeat_prompt: str = None,
                            max_new_tokens: int = 512, device: str = "cuda:0"):
    """
    验证模型能否重复文档

    参数:
        model: 语言模型
        tokenizer: 分词器
        doc_text: 文档文本
        repeat_prompt: 重复指令（如果为 None，使用默认明确指令）
        max_new_tokens: 最大生成 token 数
        device: 设备

    返回:
        result: 包含结果的字典
    """
    # 使用默认的 prompt（策略 B：prompt 在前，文档在 <content> 标签内）
    # 改用 JSON 格式输出，避免复杂的标签提取
    if repeat_prompt is None:
        repeat_prompt = (
            "\n\nInstruction: Please repeat the text inside the <content> tags word-for-word. "
            "Output your response in JSON format with key 'repeated_text'.\n"
            "Repetition: <content>"
        )

    # Tokenize 文档
    doc_tokens = tokenizer.encode(doc_text, add_special_tokens=False)
    doc_len = len(doc_tokens)

    print(f"\n文档长度: {len(doc_text)} 字符, {doc_len} tokens")
    print(f"文档预览: {doc_text}")
    print(f"Prompt: '{repeat_prompt}'")

    # 构造完整输入：[prompt_tokens, <content>, doc_tokens, </content>]
    # 策略 B：prompt 在前，文档包裹在标签中
    prompt_tokens = tokenizer.encode(repeat_prompt, add_special_tokens=False)
    content_start_tag = tokenizer.encode("<content>", add_special_tokens=False)
    content_end_tag = tokenizer.encode("</content>", add_special_tokens=False)

    # 完整序列：[prompt, <content>, doc, </content>]
    full_input_tokens = prompt_tokens + content_start_tag + doc_tokens + content_end_tag
    full_input = torch.tensor(full_input_tokens).unsqueeze(0).to(device)

    print(f"完整输入: {len(full_input_tokens)} tokens")
    print(f"  - prompt: {len(prompt_tokens)} tokens")
    print(f"  - <content>: {len(content_start_tag)} tokens")
    print(f"  - doc: {doc_len} tokens")
    print(f"  - </content>: {len(content_end_tag)} tokens")

    attention_mask = torch.ones(full_input.shape, device=device)

    # 方法1：使用 model.generate (HuggingFace 内置方法)
    print("\n=== 使用 model.generate ===")
    with torch.no_grad():
        outputs_gen = model.generate(
            full_input,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # 解码（只包含生成的部分）
    generated_tokens_gen = outputs_gen[0][full_input.shape[1]:]
    generated_text_gen = tokenizer.decode(generated_tokens_gen, skip_special_tokens=True)

    print(f"原始生成内容预览: '{generated_text_gen[:400]}...'")

    # --- JSON 解析逻辑 ---
    clean_output = ""
    
    # 1. 预处理：定位最外层的 {}，并去除 Markdown 标记
    content_to_parse = generated_text_gen.strip()
    
    # 尝试找到 JSON 的起止位置
    start_idx = content_to_parse.find('{')
    end_idx = content_to_parse.rfind('}')

    if start_idx != -1 and end_idx != -1:
        json_str = content_to_parse[start_idx : end_idx + 1]
        
        # 2. 尝试解析
        try:
            # 优先尝试标准 JSON
            data = json.loads(json_str)
            clean_output = data.get("repeated_text", "")
        except json.JSONDecodeError:
            try:
                # 备选：尝试解析 Python 字典（LLM 经常用单引号）
                data = ast.literal_eval(json_str)
                if isinstance(data, dict):
                    clean_output = data.get("repeated_text", "")
            except (ValueError, SyntaxError):
                print("  [Warning] JSON/AST 解析失败，尝试正则提取...")
                pass
    
    # 3. 兜底策略：如果 JSON 解析失败，尝试用正则暴力提取 value
    # 这种情况常见于文本中包含未转义的引号，破坏了 JSON 结构
    if not clean_output:
        # 匹配 "repeated_text": "..." 或 'repeated_text': '...'
        # re.DOTALL 允许 . 匹配换行符
        pattern = r"['\"]repeated_text['\"]\s*:\s*['\"](.*?)['\"]\s*}"
        match = re.search(pattern, generated_text_gen, re.DOTALL)
        if match:
            clean_output = match.group(1)
        else:
            # 最后的退路：假设模型没听话，直接输出了文本，没有 JSON 格式
            # 只有当生成内容长度可观时才这么做，否则可能是空或者废话
            if len(generated_text_gen) > len(doc_text) * 0.5:
                print("  [Warning] 未检测到 JSON 格式，假设全部输出为复述内容。")
                clean_output = generated_text_gen.strip()

    # --- 结果统计 ---
    print(f"\n提取后长度: {len(clean_output)} 字符")
    print(f"原文长度: {len(doc_text)} 字符")
    print(f"生成内容预览: {clean_output[:200]}")

    # 计算 F1 (假设 compute_f1 函数在外部已定义)
    f1_gen = compute_f1(clean_output, doc_text, tokenizer)
    print(f"F1 分数: {f1_gen:.4f}")

    return {
        'doc_length': len(doc_text),
        'doc_tokens': doc_len,
        'original_text': doc_text,
        'generated_raw': generated_text_gen,
        'generated_clean': clean_output,
        'generate': {
            'text': clean_output,
            'length': len(clean_output),
            'f1': f1_gen
        }
    }


def main():
    parser = argparse.ArgumentParser(description='验证模型重复文档的能力')
    parser.add_argument('--model_path', type=str, default='/mnt/data/models/Qwen2.5-7B-Instruct')
    parser.add_argument('--data_path', type=str, default='./data/2wiki_input_rebuilt.json')
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--max_new_tokens', type=int, default=4096)
    parser.add_argument('--num_tests', type=int, default=5)
    parser.add_argument('--repeat_prompt', type=str, default=None,
                        help='Repeat prompt (默认: "Please repeat the above text word for word...")')

    args = parser.parse_args()

    print("="*80)
    print("验证模型重复文档能力")
    print("="*80)

    # 加载数据
    print(f"\n从 {args.data_path} 加载数据...")
    with open(args.data_path, 'r', encoding='utf-8') as f:
        docs = json.load(f)
    print(f"已加载 {len(docs)} 个文档")

    # 加载模型
    print(f"\n加载模型 {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map=args.device,
        trust_remote_code=True
    )
    model.eval()
    print("模型加载成功")

    # 测试多个文档
    results = []
    for i in range(min(args.num_tests, len(docs))):
        doc = docs[i]
        print(f"\n{'='*80}")
        print(f"测试 {i+1}/{args.num_tests}")
        print(f"{'='*80}")

        result = verify_repeat_capability(
            model=model,
            tokenizer=tokenizer,
            doc_text=doc['text'],
            repeat_prompt=args.repeat_prompt,
            max_new_tokens=args.max_new_tokens,
            device=args.device
        )

        result['doc_id'] = doc['id']
        results.append(result)

    # 汇总
    print("\n" + "="*80)
    print("汇总")
    print("="*80)

    if results:
        avg_f1 = sum(r['generate']['f1'] for r in results) / len(results)

        print(f"\n平均 F1 分数: {avg_f1:.4f}")

        print(f"\n各测试结果:")
        for r in results:
            print(f"  Doc {r['doc_id']}:")
            print(f"    原文: {r['doc_length']} 字符")
            print(f"    生成: {r['generate']['length']} 字符")
            print(f"    F1: {r['generate']['f1']:.4f}")

        # 显示一个示例
        if results and len(results[0]['generated_clean']) > 0:
            print(f"\n=== 示例 (Doc {results[0]['doc_id']}) ===")
            print(f"原文: {results[0]['original_text'][:200]}...")
            print(f"生成: {results[0]['generated_clean'][:200]}...")


if __name__ == '__main__':
    main()
