#!/usr/bin/env python3
"""
测试 3B 模型直接回答 critical cases 的能力 (v2)

使用 test_fusionrag_reflect.py 的数据加载方式
"""

import json
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from test_fusionrag_reflect import prepare_reflect_data, load_system_prompt


def generate_answer(model, tokenizer, full_input_ids, max_new_tokens=100):
    """生成答案"""
    with torch.no_grad():
        outputs = model.generate(
            full_input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0][full_input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()


def simple_judge(predicted, ground_truth):
    """简单判断答案是否正确"""
    import re

    predicted_lower = predicted.lower()
    ground_truth_lower = ground_truth.lower()

    # 提取可能的人名/地名（首字母大写的词组）
    gt_entities = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', ground_truth))

    # 检查关键实体是否出现在预测中
    for entity in gt_entities:
        if len(entity) > 3 and entity.lower() in predicted_lower:
            return True

    # 检查关键词重叠
    gt_words = set(ground_truth_lower.split())
    pred_words = set(predicted_lower.split())
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'of', 'in', 'to', 'and', 'for', 'on', 'with', 'had', 'has', 'his', 'her'}
    gt_words = gt_words - stop_words
    pred_words = pred_words - stop_words

    if len(gt_words) > 0:
        overlap = gt_words & pred_words
        recall = len(overlap) / len(gt_words)
        if recall > 0.4:
            return True

    return False


def main():
    device = "cuda:0"

    # Load critical cases
    with open('/mnt/data/wjh/FusionRAG/critical_cases_3b_vs_7b.json', 'r', encoding='utf-8') as f:
        critical_cases = json.load(f)

    print(f"Loaded {len(critical_cases)} critical cases")
    print("These are cases where 7B draft model got correct but 3B draft model got wrong\n")

    # Load 3B model
    print("Loading 3B model...")
    config = AutoConfig.from_pretrained('/mnt/data/models/Qwen2.5-3B-Instruct', trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        '/mnt/data/models/Qwen2.5-3B-Instruct',
        config=config,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained('/mnt/data/models/Qwen2.5-3B-Instruct', trust_remote_code=True)

    # Load dataset using the same method as test_fusionrag_reflect.py
    print("\nLoading dataset...")
    questions_data, system_tensor, _, _ = prepare_reflect_data(
        './data/result_reflect.json',
        tokenizer,
        '/mnt/data/models/bge-m3-FP16',
        'qwen',
        topk=10,
        max_main_questions=200,
        preprocess=False
    )

    system_len = system_tensor.shape[0]

    # Test each critical case
    results = []
    correct_count = 0
    tested_count = 0

    print("\n" + "=" * 80)
    print("Testing 3B Model Direct Answers on Critical Cases")
    print("=" * 80)

    for idx, case in enumerate(critical_cases):
        print(f"\n--- Case {idx + 1}/{len(critical_cases)} ---")
        print(f"Question: {case['sub_question'][:80]}...")
        print(f"Ground Truth: {case['ground_truth'][:80]}...")
        print(f"3B Draft Predicted: {case['pred_3b'][:60]}...")
        print(f"7B Draft Predicted: {case['pred_7b'][:60]}...")

        # Find matching question in dataset
        found = False
        for q_data in questions_data:
            for sub_q_info in q_data['sub_questions']:
                if sub_q_info['query'] == case['sub_question']:
                    found = True

                    # Build input (system + docs + question)
                    doc_chunk_ids = sub_q_info['chunk_ids']
                    doc_tensors = q_data['doc_tensors']
                    sub_q_doc_tensors = [doc_tensors[chunk_id - 1] for chunk_id in doc_chunk_ids]

                    question_text = f"<|im_end|>\n<|im_start|>user\nQuestion: {sub_q_info['query']}<|im_end|>\n<|im_start|>assistant\nAnswer: "
                    question_tokens = tokenizer.encode(question_text, add_special_tokens=False)
                    question_tensor = torch.tensor(question_tokens, dtype=torch.long)

                    all_tokens = [system_tensor] + sub_q_doc_tensors + [question_tensor]
                    full_input = torch.cat(all_tokens).unsqueeze(0).to(device)

                    doc_len = sum(t.shape[0] for t in sub_q_doc_tensors)
                    print(f"  Input: system={system_len}, doc={doc_len}, query={len(question_tokens)}, total={full_input.shape[1]}")

                    # Generate answer with 3B (full prefill)
                    answer_3b_direct = generate_answer(model, tokenizer, full_input, max_new_tokens=100)
                    print(f"  3B Direct Answer: {answer_3b_direct[:80]}...")

                    # Judge
                    is_correct = simple_judge(answer_3b_direct, case['ground_truth'])
                    tested_count += 1
                    if is_correct:
                        correct_count += 1
                        print(f"  Result: ✓ CORRECT")
                    else:
                        print(f"  Result: ✗ WRONG")

                    results.append({
                        'question': case['sub_question'],
                        'ground_truth': case['ground_truth'],
                        'pred_3b_draft': case['pred_3b'],
                        'pred_7b_draft': case['pred_7b'],
                        'pred_3b_direct': answer_3b_direct,
                        'is_correct_direct': is_correct
                    })
                    break
            if found:
                break

        if not found:
            print("  WARNING: Could not find question in dataset")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total critical cases: {len(critical_cases)}")
    print(f"Successfully tested: {tested_count}")
    print(f"3B Direct (full prefill) correct: {correct_count}/{tested_count} ({correct_count/tested_count*100:.1f}%)" if tested_count > 0 else "No tests completed")

    print(f"\nInterpretation:")
    if tested_count > 0:
        accuracy = correct_count / tested_count
        if accuracy > 0.7:
            print("  → 3B model CAN answer most of these questions with full prefill")
            print("  → The problem is in TOKEN SELECTION, not model capability")
            print("  → Attention distillation should help!")
        elif accuracy > 0.4:
            print("  → 3B model can answer SOME questions with full prefill")
            print("  → Token selection is part of the problem, but model capability also matters")
        else:
            print("  → 3B model struggles even with full prefill")
            print("  → The problem is primarily in model capability, not just token selection")
            print("  → Attention distillation alone may not be sufficient")

    # Save results
    output_path = '/mnt/data/wjh/FusionRAG/test_3b_direct_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
