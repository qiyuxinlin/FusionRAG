import copy
import json
import os
from typing import List, Dict, Any, Tuple
from openai import OpenAI
from run_question import FusionRAGModel


def prepare_reflect_data(
    data_path: str,
    max_main_questions=200,
):
    print(f"Loading dataset from {data_path}...")
    with open(data_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    if max_main_questions:
        dataset = dataset[:max_main_questions]
        print(f"Limited to first {max_main_questions} main questions")
    all_questions = []

    for main_q_idx, data_item in enumerate(dataset):
        main_question = data_item["question"]
        main_answer = data_item["answer"]
        intermediate_context = data_item.get("intermediate_context", [])

        question_docs = []  # Documents for THIS question only
        doc_to_idx = {}  # Local doc -> chunk_id mapping for this question
        sub_questions_info = []

        should_test_main_question = True
        if data_item.get('llm_judge', True) is False:
            should_test_main_question = False

        for sub_q_idx, sub_q in enumerate(intermediate_context):
            docs = sub_q.get("retrieve docs", [])
            query = sub_q['query']
            if query.startswith("Intermediate query"):
                # Find the colon and extract text after it
                colon_pos = query.find(":")
                if colon_pos != -1:
                    query = query[colon_pos + 1:].strip()

            # Remove "Intermediate answerXXX:" prefix from answer
            answer = sub_q['answer']
            if answer.startswith("Intermediate answer"):
                # Find the colon and extract text after it
                colon_pos = answer.find(":")
                if colon_pos != -1:
                    answer = answer[colon_pos + 1:].strip()

            if "No relevant information found".lower() in answer.lower() or "没有相关信息" in answer:
                ""
            else:
                sub_questions_info.append({
                    'query': query,
                    'answer': answer,
                    'gold_docs': docs,  # chunk_ids for docs used by this sub-question
                })
        if should_test_main_question:
            all_questions.extend(sub_questions_info)
    return all_questions

def judge_answer_with_openai(
    openai_client: OpenAI,
    openai_model: str,
    question: str,
    predicted_answer: str,
    ground_truth_answer: str
) -> Tuple[bool, str]:
    """
    Use OpenAI API to judge if the predicted answer is correct

    Returns:
        Tuple[bool, str]: (is_correct, reason)
    """
    judge_prompt = f"""你是一个答案评估专家。你的任务是判断预测答案是否正确地回答了问题。

问题: {question}

标准答案: {ground_truth_answer}

预测答案: {predicted_answer}

请判断预测答案是否正确回答了问题。判断标准：
1. 预测答案包含了标准答案的关键信息
2. 预测答案与标准答案在语义上等价
3. 允许措辞上的细微差异，只要意思保持一致即可

请按照以下格式回答：
{{
"reason": "",
"answer": "", ## output right or wrong
}}
"""

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

        # 解析返回结果
        is_correct = False
        reason = result
        print(result)

        res_json = json.loads(result)

        return res_json["answer"] == "right", res_json["reason"]

    except Exception as e:
        error_msg = f"调用 OpenAI API 时出错: {e}"
        print(error_msg)
        return False, error_msg


def test_question(fusion_rag_model, total_run=200, rate=0.3, reprocess_method="DraftModel"):
    keyword = f"model_{fusion_rag_model.model_name}_rate_{rate}_reprocess_method_{reprocess_method}_preprocess_{fusion_rag_model.preprocess_method}"
    all_result = []
    openai_client = OpenAI(api_key="sk-27b5e2809a7148aaba768b6ea0de76b5", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1/")
    all_questions = prepare_reflect_data(data_path=f"./result_reflect.json", max_main_questions=total_run)
    print(f"all_questions len={len(all_questions)}")
    revert_rope = True
    if fusion_rag_model.preprocess == True and fusion_rag_model.preprocess_method == "space":
        revert_rope = False
    print(f"revert_rope={revert_rope}")
    for question in all_questions:
        # if "Which university employed Ernst Mach?" not in question["query"]:
        #     continue
        system_len, doc_tensors_total_length, query_len, decode_len, answer, docs_lens = fusion_rag_model.run_one_question(
            query=f'{question["query"]}',
            retrieved_docs=question["gold_docs"],
            model_type='qwen3',
            rate=rate,
            reprocess_method=reprocess_method,
            revert_rope=revert_rope,
            max_new_tokens=300,
        )
        is_correct, judge_reason = judge_answer_with_openai(
            openai_client=openai_client,
            openai_model="deepseek-v3.2",
            question=question["query"],
            ground_truth_answer=question["answer"],
            predicted_answer=answer,
        )
        question_copy = copy.deepcopy(question)
        question_copy["llm_answer"] = answer
        question_copy["llm_judge"] = is_correct
        question_copy["llm_judge_reason"] = judge_reason

        print(f"Judgment: {'✓ CORRECT' if is_correct else '✗ INCORRECT'} ")
        print(f"Question: {question['query']}")
        print(f"Answer: {question['answer']}")
        print(f"Fusionrag answer: {answer}")
        print("="*80)
        all_result.append(question_copy)
        with open(f"./results/result_{keyword}.json", 'w', encoding='utf-8') as f:
            json.dump(all_result, f, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3,4"
    print(f"start testing run_question")


    fusion_rag_model = FusionRAGModel(
        model_path='/data2/qy_tmp/xumengyao/Qwen3-32B',
        use_multi_gpu=True,
        model_type="qwen3",
        model_name="Qwen3-32B",
        device="cuda:0",
        cache_path='/data2/qy_tmp/xumengyao/fusionrag/',
        draft_model_device="cuda:0",
        draft_model_path='/data2/qy_tmp/xumengyao/Qwen2.5-3B-Instruct',
        draft_model_type="qwen",
        file_input="/home/qy_tmp/xumengyao/all_data/musique_input.json",
        preprocess_model_path="/data2/qy_tmp/xumengyao/bge-m3",

        preprocess=True,
        preprocess_method="space"
    )

    test_question(fusion_rag_model,
                  total_run=200,
                  rate=0.2,
                  reprocess_method="DraftModel"
                  )