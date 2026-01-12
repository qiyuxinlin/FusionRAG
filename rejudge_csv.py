#!/usr/bin/env python3
"""
重新用 LLM 判分 CSV 文件中的答案
"""

import pandas as pd
from openai import OpenAI
from typing import Tuple
import time
from tqdm import tqdm

def judge_answer_with_openai(
    openai_client: OpenAI,
    openai_model: str,
    question: str,
    predicted_answer: str,
    ground_truth_answer: str
) -> Tuple[bool, str]:
    """
    Use OpenAI API to judge if the predicted answer is correct
    """
    question = question.strip()
    predicted_answer = predicted_answer.strip()
    ground_truth_answer = ground_truth_answer.strip()

    judge_prompt = f"""你是一个答案评估专家。你的任务是判断预测答案是否正确地回答了问题。

问题: {question}

标准答案: {ground_truth_answer}

预测答案: {predicted_answer}

请判断预测答案是否正确回答了问题。判断标准：
1. 预测答案包含了标准答案的关键信息
2. 预测答案与标准答案在语义上等价
3. 允许措辞上的细微差异，只要意思保持一致即可

请按照以下格式回答：
判断: [正确/错误]
原因: [详细说明为什么正确或错误，至少30字]"""

    try:
        response = openai_client.chat.completions.create(
            model=openai_model,
            messages=[
                {"role": "system", "content": "你是一个专业的答案评估专家。"},
                {"role": "user", "content": judge_prompt}
            ],
            temperature=0,
            max_tokens=300
        )

        result = response.choices[0].message.content.strip()

        is_correct = False
        reason = result

        # 只提取判断值，避免原因文本干扰
        lines = result.split('\n')
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            # 查找判断行：必须以"判断"开头，避免匹配到原因文本中的"判断标准"等词
            if (line_stripped.startswith('判断') or line_stripped.lower().startswith('judgment')) and (':' in line or '：' in line):
                judgment_value = line.split(':', 1)[-1].split('：', 1)[-1].strip()
                if '正确' in judgment_value or '对' in judgment_value:
                    is_correct = True
                elif '错误' in judgment_value or '错' in judgment_value:
                    is_correct = False
                continue
            if line_stripped.startswith('原因') or line_stripped.lower().startswith('reason'):
                if ':' in line or '：' in line:
                    reason_start = line.split(':', 1)[-1].split('：', 1)[-1].strip()
                    if len(lines) > i + 1 and not reason_start:
                        reason = '\n'.join(lines[i+1:]).strip()
                    else:
                        reason = reason_start + '\n' + '\n'.join(lines[i+1:]).strip()
                    reason = reason.strip()
                    break

        if not reason or len(reason) < 10:
            reason = result

        return is_correct, reason

    except Exception as e:
        error_msg = f"调用 OpenAI API 时出错: {e}"
        print(error_msg)
        return False, error_msg


def main():
    csv_path = '/mnt/data/reflect/Qwen2.5-7B-Instruct/2wikimqa/results/DraftModel_global_topk_10_rate_0.3_draft_Qwen2.5-3B-Instruct.csv'
    output_path = '/mnt/data/reflect/Qwen2.5-7B-Instruct/2wikimqa/results/DraftModel_global_topk_10_rate_0.3_draft_Qwen2.5-3B-Instruct_rejudged.csv'

    # 初始化 OpenAI client
    client = OpenAI(
        base_url="https://api.deepseek.com/v1",
        api_key="sk-519d391217894b6e91e7c2ebf2a9f4df"
    )
    model = "deepseek-chat"

    # 读取 CSV
    df = pd.read_csv(csv_path)
    print(f"读取 {len(df)} 行数据")

    # 新列
    new_correct = []
    new_reason = []

    # 重新判分
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="重新判分"):
        question = row['Sub Question']
        predicted = str(row['Predicted'])
        ground_truth = str(row['Ground Truth'])

        is_correct, reason = judge_answer_with_openai(
            client, model, question, predicted, ground_truth
        )

        new_correct.append(is_correct)
        new_reason.append(reason)

        # 每 10 个请求休息一下，避免 rate limit
        if (idx + 1) % 10 == 0:
            time.sleep(0.5)

    # 更新 DataFrame
    df['New_Correct'] = new_correct
    df['New_Reason'] = new_reason

    # 比较新旧判断
    df['Old_Correct'] = df['Correct'].astype(str).str.lower().isin(['true', 'yes', '1'])
    df['Changed'] = df['Old_Correct'] != df['New_Correct']

    # 保存结果
    df.to_csv(output_path, index=False)
    print(f"\n结果已保存到: {output_path}")

    # 统计
    print("\n=== 统计 ===")
    print(f"原始正确数: {df['Old_Correct'].sum()} / {len(df)} ({df['Old_Correct'].mean()*100:.2f}%)")
    print(f"新判正确数: {df['New_Correct'].sum()} / {len(df)} ({df['New_Correct'].mean()*100:.2f}%)")
    print(f"判断变化数: {df['Changed'].sum()}")

    # 显示变化的案例
    changed = df[df['Changed']]
    if len(changed) > 0:
        print(f"\n=== 判断发生变化的案例 ({len(changed)} 个) ===")
        for i, (_, row) in enumerate(changed.head(10).iterrows()):
            print(f"\n{i+1}. Q: {row['Sub Question'][:60]}...")
            print(f"   Predicted: {str(row['Predicted'])[:50]}")
            print(f"   Ground Truth: {str(row['Ground Truth'])[:50]}")
            print(f"   旧判断: {row['Old_Correct']} → 新判断: {row['New_Correct']}")


if __name__ == '__main__':
    main()
