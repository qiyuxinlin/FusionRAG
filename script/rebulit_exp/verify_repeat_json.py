#!/usr/bin/env python3
"""
验证脚本：使用 JSON 格式测试模型重复文档能力
"""

import os
import sys
import json
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_dir)

from ktransformers.util.utils_v2 import compute_f1


def verify_with_json(model, tokenizer, doc_text: str, max_new_tokens: int = 512, device: str = "cuda:0"):
    """使用 JSON 格式验证重复能力"""

    # Tokenize 文档
    doc_tokens = tokenizer.encode(doc_text, add_special_tokens=False)
    doc_len = len(doc_tokens)

    print(f"\n文档长度: {len(doc_text)} 字符, {doc_len} tokens")
    print(f"文档内容: {doc_text}")

    # 构造输入：[prompt, <content>, doc, </content>]
    # 要求输出 JSON 格式
    prompt = (
        "\n\nInstruction: Repeat the text inside the <content> tags word-for-word. "
        "Output your response as JSON with key 'repeated_text'.\n"
        "Repetition: <content>"
    )

    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    content_start_tag = tokenizer.encode("<content>", add_special_tokens=False)
    content_end_tag = tokenizer.encode("</content>", add_special_tokens=False)

    full_input_tokens = prompt_tokens + content_start_tag + doc_tokens + content_end_tag
    full_input = torch.tensor(full_input_tokens).unsqueeze(0).to(device)

    print(f"完整输入: {len(full_input_tokens)} tokens")
    attention_mask = torch.ones(full_input.shape, device=device)

    # 生成
    print("\n=== 生成 ===")
    with torch.no_grad():
        outputs = model.generate(
            full_input,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # 解码生成的部分
    generated_tokens = outputs[0][full_input.shape[1]:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    print(f"原始生成: '{generated_text[:400]}...'")

    # 提取 JSON 中的 'repeated_text' 字段
    clean_output = ""
    try:
        # 查找 JSON 对象
        json_start = generated_text.find('{')
        if json_start >= 0:
            # 找到匹配的结束 }
            brace_count = 0
            json_end = -1
            for i in range(json_start, len(generated_text)):
                if generated_text[i] == '{':
                    brace_count += 1
                elif generated_text[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = i + 1
                        break

            if json_end > json_start:
                json_str = generated_text[json_start:json_end]
                print(f"[提取] JSON 字符串: {json_str[:200]}...")

                data = json.loads(json_str)

                if 'repeated_text' in data:
                    clean_output = data['repeated_text']
                    print(f"[成功] 从 JSON 提取 'repeated_text'")
                else:
                    print(f"[警告] JSON 中没有 'repeated_text'，可用字段: {list(data.keys())}")
                    # 尝试使用其他字段
                    if 'text' in data:
                        clean_output = data['text']
                    elif 'content' in data:
                        clean_output = data['content']
                    else:
                        clean_output = str(data)
            else:
                print(f"[错误] 未找到完整的 JSON 对象")
                clean_output = generated_text.strip()
        else:
            print(f"[错误] 未找到 JSON 对象开始")
            # 尝试直接找 repeated_text 字段的值
            if '"repeated_text"' in generated_text:
                start = generated_text.find('"repeated_text"') + len('"repeated_text"')
                # 跳过冒号和空格
                while start < len(generated_text) and generated_text[start] not in ['"', "'"]:
                    start += 1
                if start < len(generated_text):
                    end_quote = generated_text[start+1:].find(generated_text[start])
                    if end_quote > 0:
                        clean_output = generated_text[start+1:start+1+end_quote]
                        print(f"[备用] 直接解析 repeated_text 字段")
                    else:
                        clean_output = generated_text[start+1:].strip()[0:100]
                        print(f"[备用] 取 repeated_text 后的部分内容")
                else:
                    clean_output = generated_text.strip()
            else:
                clean_output = generated_text.strip()
                print(f"[回退] 使用原始生成")

    except json.JSONDecodeError as e:
        print(f"[错误] JSON 解析失败: {e}")
        # 回退到简单提取
        if "Assistant:" in generated_text:
            assistant_pos = generated_text.find("Assistant:") + len("Assistant:")
            clean_output = generated_text[assistant_pos:].strip()
        else:
            clean_output = generated_text.strip()

    print(f"\n提取后长度: {len(clean_output)} 字符")
    print(f"原文长度: {len(doc_text)} 字符")
    print(f"提取内容: {clean_output[:200]}...")

    # 计算 F1
    f1 = compute_f1(clean_output, doc_text, tokenizer)
    print(f"F1 分数: {f1:.4f}")

    return {
        'doc_length': len(doc_text),
        'original_text': doc_text,
        'generated_raw': generated_text,
        'generated_clean': clean_output,
        'f1': f1
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='/mnt/data/models/Qwen2.5-7B-Instruct')
    parser.add_argument('--data_path', default='./data/2wiki_input_rebuilt.json')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    parser.add_argument('--num_tests', type=int, default=5)

    args = parser.parse_args()

    print("="*80)
    print("验证模型重复文档能力 (JSON 格式)")
    print("="*80)

    # 加载数据
    print(f"\n从 {args.data_path} 加载数据...")
    with open(args.data_path, 'r', encoding='utf-8') as f:
        docs = json.load(f)
    print(f"已加载 {len(docs)} 个文档")

    # 加载模型
    print(f"\n加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map=args.device,
        trust_remote_code=True
    )
    model.eval()
    print("模型加载成功")

    # 测试
    results = []
    for i in range(min(args.num_tests, len(docs))):
        doc = docs[i]
        print(f"\n{'='*80}")
        print(f"测试 {i+1}/{args.num_tests}")
        print(f"{'='*80}")

        result = verify_with_json(
            model=model,
            tokenizer=tokenizer,
            doc_text=doc['text'],
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
        avg_f1 = sum(r['f1'] for r in results) / len(results)

        print(f"\n平均 F1 分数: {avg_f1:.4f}")

        print(f"\n各测试结果:")
        for r in results:
            print(f"  Doc {r['doc_id']}: F1={r['f1']:.4f}, " +
                  f"原文={r['doc_length']}字符, " +
                  f"生成={len(r['generated_clean'])}字符")


if __name__ == '__main__':
    main()
