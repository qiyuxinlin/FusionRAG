#!/usr/bin/env python3
"""
测试 LLM 评判一致性
检查答案中是否有不可见字符导致判断不一致
"""

import csv
from openai import OpenAI

# 初始化 API
client = OpenAI(
    api_key="sk-519d391217894b6e91e7c2ebf2a9f4df",
    base_url="https://api.deepseek.com/v1"
)

def show_hidden_chars(s):
    """显示字符串中的不可见字符"""
    result = []
    for c in s:
        if c == '\n':
            result.append('\\n')
        elif c == '\r':
            result.append('\\r')
        elif c == '\t':
            result.append('\\t')
        elif ord(c) < 32 or ord(c) == 127:
            result.append(f'\\x{ord(c):02x}')
        else:
            result.append(c)
    return ''.join(result)

def judge_answer(question, predicted, ground_truth):
    """调用 LLM 评判答案"""
    judge_prompt = f"""你是一个答案评估专家。你的任务是判断预测答案是否正确地回答了问题。

问题: {question}

标准答案: {ground_truth}

预测答案: {predicted}

请判断预测答案是否正确回答了问题。判断标准：
1. 预测答案包含了标准答案的关键信息
2. 预测答案与标准答案在语义上等价
3. 允许措辞上的细微差异，只要意思保持一致即可

请按照以下格式回答：
判断: [正确/错误]
原因: [详细说明为什么正确或错误，至少30字]"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是一个专业的答案评估专家。"},
            {"role": "user", "content": judge_prompt}
        ],
        temperature=0,
        max_tokens=300
    )

    result = response.choices[0].message.content.strip()

    # 解析结果
    is_correct = False
    for line in result.split('\n'):
        if '判断' in line:
            if '正确' in line:
                is_correct = True
            elif '错误' in line:
                is_correct = False
            break

    return is_correct, result

# 读取两个文件
rate0_file = '/mnt/data/reflect/Qwen2.5-7B-Instruct/2wikimqa/results/DraftModel_global_topk_10_rate_0_draft_Qwen2.5-3B-Instruct.csv'
rate1_file = '/mnt/data/reflect/Qwen2.5-7B-Instruct/2wikimqa/results/DraftModel_global_topk_10_rate_1.csv'

with open(rate0_file, 'r', encoding='utf-8') as f:
    rate0_data = list(csv.DictReader(f))

with open(rate1_file, 'r', encoding='utf-8') as f:
    rate1_data = list(csv.DictReader(f))

# 找出答案相同但判断不一致的案例
problem_cases = []
for r0 in rate0_data:
    for r1 in rate1_data:
        if r0.get('Sub Question') == r1.get('Sub Question'):
            r0_correct = r0.get('Correct', '').lower() == 'true'
            r1_correct = r1.get('Correct', '').lower() == 'true'

            r0_pred = r0.get('Predicted', '')
            r1_pred = r1.get('Predicted', '')

            # 答案相同但判断不一致
            if r0_pred == r1_pred and r0_correct != r1_correct:
                problem_cases.append({
                    'question': r0.get('Sub Question', ''),
                    'ground_truth': r0.get('Ground Truth', ''),
                    'rate0_pred': r0_pred,
                    'rate1_pred': r1_pred,
                    'rate0_correct': r0_correct,
                    'rate1_correct': r1_correct,
                })
            break

print(f"找到 {len(problem_cases)} 个答案相同但判断不一致的案例")
print("=" * 80)

# 对前 5 个案例进行详细分析
for i, case in enumerate(problem_cases[:5]):
    print(f"\n{'='*80}")
    print(f"案例 {i+1}")
    print(f"{'='*80}")

    print(f"\n问题: {case['question'][:100]}...")
    print(f"\n标准答案: {case['ground_truth'][:100]}")
    print(f"标准答案(显示隐藏字符): {show_hidden_chars(case['ground_truth'][:100])}")

    print(f"\nrate=0 预测: {case['rate0_pred'][:100]}")
    print(f"rate=0 预测(显示隐藏字符): {show_hidden_chars(case['rate0_pred'][:100])}")
    print(f"rate=0 原始判断: {'正确' if case['rate0_correct'] else '错误'}")

    print(f"\nrate=1 预测: {case['rate1_pred'][:100]}")
    print(f"rate=1 预测(显示隐藏字符): {show_hidden_chars(case['rate1_pred'][:100])}")
    print(f"rate=1 原始判断: {'正确' if case['rate1_correct'] else '错误'}")

    # 检查是否完全相等
    print(f"\n答案完全相等: {case['rate0_pred'] == case['rate1_pred']}")
    print(f"答案字节相等: {case['rate0_pred'].encode() == case['rate1_pred'].encode()}")

    # 重新评判
    print(f"\n--- 重新评判 ---")
    new_correct, new_reason = judge_answer(
        case['question'],
        case['rate0_pred'],
        case['ground_truth']
    )
    print(f"重新评判结果: {'正确' if new_correct else '错误'}")
    print(f"评判理由: {new_reason[:200]}...")

    # 再评判一次，看是否一致
    new_correct2, new_reason2 = judge_answer(
        case['question'],
        case['rate0_pred'],
        case['ground_truth']
    )
    print(f"\n第二次评判结果: {'正确' if new_correct2 else '错误'}")
    print(f"两次评判一致: {new_correct == new_correct2}")
