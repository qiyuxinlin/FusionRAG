#!/usr/bin/env python3
"""
测试 3B 模型直接回答 critical cases 的能力

问题：3B 作为 draft model 没答对，但 7B 能答对的问题，
      3B 模型本身（完整 prefill，不做 token selection）能答对吗？
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig


def load_model(model_path, device):
    """加载模型"""
    print(f"Loading model from {model_path}...")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer


def generate_answer(model, tokenizer, system_prompt, docs, question, max_new_tokens=100):
    """生成答案"""
    # 构建完整 prompt
    messages = [
        {"role": "system", "content": system_prompt},
    ]

    # 添加文档
    doc_content = "\n\n".join([f"Document: {doc}" for doc in docs])
    messages.append({"role": "user", "content": f"{doc_content}\n\nQuestion: {question}"})

    # Apply chat template
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Tokenize
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()


def simple_judge(predicted, ground_truth):
    """简单判断答案是否正确"""
    predicted_lower = predicted.lower()
    ground_truth_lower = ground_truth.lower()

    # 检查 ground truth 中的关键词是否在预测中
    # 提取 ground truth 中的关键实体
    gt_words = set(ground_truth_lower.split())
    pred_words = set(predicted_lower.split())

    # 移除常见词
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'of', 'in', 'to', 'and', 'for', 'on', 'with'}
    gt_words = gt_words - stop_words
    pred_words = pred_words - stop_words

    # 计算重叠
    overlap = gt_words & pred_words
    if len(gt_words) > 0:
        recall = len(overlap) / len(gt_words)
    else:
        recall = 0

    # 检查关键名词是否出现
    # 提取可能的人名/地名（首字母大写的词）
    import re
    gt_entities = set(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', ground_truth))
    pred_text = predicted

    entity_match = any(entity.lower() in pred_text.lower() for entity in gt_entities if len(entity) > 3)

    return recall > 0.3 or entity_match


def main():
    device = "cuda:0"

    # Load critical cases
    with open('/mnt/data/wjh/FusionRAG/critical_cases_3b_vs_7b.json', 'r', encoding='utf-8') as f:
        critical_cases = json.load(f)

    print(f"Loaded {len(critical_cases)} critical cases")
    print("These are cases where 7B draft model got correct but 3B draft model got wrong\n")

    # Load 3B model
    model_3b, tokenizer = load_model('/mnt/data/models/Qwen2.5-3B-Instruct', device)

    # Load dataset to get documents
    print("\nLoading dataset to get document content...")
    with open('./data/result_reflect.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    # Build question to docs mapping
    question_to_data = {}
    for item in dataset:
        for sub_q in item.get('sub_questions', []):
            # Try different field names
            question_key = sub_q.get('decomposed_question') or sub_q.get('question') or sub_q.get('query', '')
            docs = sub_q.get('paragraphs') or sub_q.get('ctxs', [])
            if isinstance(docs, list) and len(docs) > 0:
                if isinstance(docs[0], dict):
                    docs = [d.get('text', d.get('content', str(d))) for d in docs]
            question_to_data[question_key] = {
                'docs': docs,
                'answer': sub_q.get('decomposed_answer') or sub_q.get('answer', '')
            }
            # Also store by truncated question for fuzzy matching
            if len(question_key) > 50:
                question_to_data[question_key[:50]] = question_to_data[question_key]

    # System prompt
    system_prompt = """You are a helpful assistant. Answer the question based on the given documents.
Be concise and provide only the relevant information."""

    # Test each critical case
    results = []
    correct_count = 0

    print("\n" + "=" * 80)
    print("Testing 3B Model Direct Answers on Critical Cases")
    print("=" * 80)

    for idx, case in enumerate(critical_cases):
        print(f"\n--- Case {idx + 1}/{len(critical_cases)} ---")
        print(f"Question: {case['sub_question'][:80]}...")
        print(f"Ground Truth: {case['ground_truth'][:80]}...")
        print(f"3B Draft Predicted: {case['pred_3b'][:80]}...")
        print(f"7B Draft Predicted: {case['pred_7b'][:80]}...")

        # Get documents for this question
        q_data = question_to_data.get(case['sub_question'])
        if q_data is None:
            print("  WARNING: Could not find documents for this question")
            continue

        docs = q_data['docs']
        if not docs:
            print("  WARNING: No documents found")
            continue

        # Generate answer with 3B (full prefill, no token selection)
        print(f"\n  Generating 3B direct answer (full prefill)...")
        answer_3b_direct = generate_answer(
            model_3b, tokenizer, system_prompt, docs, case['sub_question']
        )

        print(f"  3B Direct Answer: {answer_3b_direct[:100]}...")

        # Judge
        is_correct = simple_judge(answer_3b_direct, case['ground_truth'])
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

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total critical cases: {len(critical_cases)}")
    print(f"3B Direct (full prefill) correct: {correct_count}/{len(results)} ({correct_count/len(results)*100:.1f}%)")
    print(f"\nInterpretation:")
    if correct_count / len(results) > 0.7:
        print("  → 3B model CAN answer most of these questions with full prefill")
        print("  → The problem is in TOKEN SELECTION, not model capability")
        print("  → Attention distillation is a promising direction!")
    else:
        print("  → 3B model struggles even with full prefill")
        print("  → The problem may be in model capability, not just token selection")

    # Save results
    output_path = '/mnt/data/wjh/FusionRAG/test_3b_direct_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
