import copy
import json
import os
import time
from typing import List, Dict, Any, Tuple
from openai import OpenAI
from run_question import FusionRAGModel
import multiprocessing
from multiprocessing import Process, Lock, Manager
import threading


def prepare_reflect_data(
        data_path: str,
        max_main_questions=200,
        start_idx=0,
        end_idx=None
):
    """准备数据，可以指定起始和结束索引"""
    print(f"Loading dataset from {data_path}...")
    with open(data_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    if max_main_questions:
        dataset = dataset[:max_main_questions]
        print(f"Limited to first {max_main_questions} main questions")

    if end_idx is None or end_idx > len(dataset):
        end_idx = len(dataset)

    dataset = dataset[start_idx:end_idx]
    print(f"Processing questions {start_idx} to {end_idx}")

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


def write_result_to_shared_file(result, shared_file_path, file_lock):
    """将结果写入共享文件（线程安全）"""
    with file_lock:
        # 读取现有结果
        existing_results = []
        if os.path.exists(shared_file_path):
            try:
                with open(shared_file_path, 'r', encoding='utf-8') as f:
                    existing_results = json.load(f)
            except json.JSONDecodeError:
                # 如果文件为空或格式错误，重新开始
                existing_results = []

        # 添加新结果
        existing_results.append(result)

        # 写入文件
        with open(shared_file_path, 'w', encoding='utf-8') as f:
            json.dump(existing_results, f, ensure_ascii=False, indent=4)


def write_result_to_individual_file(result, individual_file_path):
    """将结果写入单个进程的独立文件"""
    # 读取现有结果
    existing_results = []
    if os.path.exists(individual_file_path):
        try:
            with open(individual_file_path, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
        except json.JSONDecodeError:
            # 如果文件为空或格式错误，重新开始
            existing_results = []

    # 添加新结果
    existing_results.append(result)

    # 写入文件
    with open(individual_file_path, 'w', encoding='utf-8') as f:
        json.dump(existing_results, f, ensure_ascii=False, indent=4)


def run_test_process(
        process_id,
        gpu_ids,
        start_idx,
        end_idx,
        total_run=200,
        rate=0.3,
        reprocess_method="",
        shared_file_path=None,
        file_lock=None,
        result_queue=None,
        preprocess_method="",
        questions_to_run=[]
):
    """单个测试进程的运行函数"""
    print(f"Process {process_id}: Starting with GPUs {gpu_ids}")

    # 设置当前进程可见的GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

    # 根据GPU分配情况设置设备
    if len(gpu_ids) > 0:
        main_device = f"cuda:0"
        draft_device = f"cuda:{min(1, len(gpu_ids) - 1)}" if len(gpu_ids) > 1 else "cuda:0"
    else:
        main_device = "cuda:0"
        draft_device = "cuda:0"

    fusion_rag_model = FusionRAGModel(
        model_path='/data2/qy_tmp/xumengyao/Qwen3-32B',
        use_multi_gpu=True,
        model_type="qwen3",
        model_name="Qwen3-32B",
        device=main_device,
        cache_path='/data2/qy_tmp/xumengyao/fusionrag/',
        draft_model_device=draft_device,
        draft_model_path='/data2/qy_tmp/xumengyao/Qwen2.5-3B-Instruct',
        draft_model_type="qwen",
        file_input="/home/qy_tmp/xumengyao/all_data/musique_input.json",
        preprocess_model_path="/data2/qy_tmp/xumengyao/bge-m3",
        preprocess=True,
        preprocess_method=preprocess_method
    )

    # 创建结果文件名
    keyword = f"model_{fusion_rag_model.model_name}_rate_{rate}_reprocess_method_{reprocess_method}_preprocess_{fusion_rag_model.preprocess_method}_process_{process_id}"
    individual_file_path = f"./results/result_{keyword}.json"

    openai_client = OpenAI(api_key="sk-27b5e2809a7148aaba768b6ea0de76b5",
                           base_url="https://dashscope.aliyuncs.com/compatible-mode/v1/")

    # 获取该进程需要处理的问题
    all_questions = prepare_reflect_data(
        data_path=f"./result_reflect.json",
        max_main_questions=total_run,
        start_idx=start_idx,
        end_idx=end_idx
    )

    print(f"Process {process_id}: Processing {len(all_questions)} questions (main questions {start_idx} to {end_idx})")

    revert_rope = True
    if fusion_rag_model.preprocess == True and fusion_rag_model.preprocess_method == "space":
        revert_rope = False
    print(f"Process {process_id}: revert_rope={revert_rope}")

    for question in all_questions:
        if question["query"] == "":
            continue
        ## debug
        if len(questions_to_run)>0:
            if question["query"] not in questions_to_run:
                continue
        system_len, doc_tensors_total_length, query_len, decode_len, answer, docs_lens = fusion_rag_model.run_one_question(
            query=f'{question["query"]}',
            retrieved_docs=[f"Document: {doc}" for doc in question["gold_docs"]],
            model_type='qwen3',
            rate=rate,
            reprocess_method=reprocess_method,
            revert_rope=revert_rope,
            max_new_tokens=300,
        )
        if "</think>" in answer:
            answer = answer.split("</think>")[1].replace("\n\n", "")

        is_correct, judge_reason = judge_answer_with_openai(
            openai_client=openai_client,
            openai_model="deepseek-v3.2",
            question=question["query"],
            ground_truth_answer=question["answer"],
            predicted_answer=answer,
        )
        if "调用 OpenAI API 时出错" in judge_reason:
            continue

        question_copy = copy.deepcopy(question)
        question_copy["llm_answer"] = answer
        question_copy["llm_judge"] = is_correct
        question_copy["llm_judge_reason"] = judge_reason
        question_copy["process_id"] = process_id
        question_copy["gpu_ids"] = gpu_ids
        question_copy["timestamp"] = time.time()

        print(f"Process {process_id} - Judgment: {'✓ CORRECT' if is_correct else '✗ INCORRECT'} ")
        print(f"Process {process_id} - Question: {question['query'][:100]}...")
        print(f"Process {process_id} - Answer: {question['answer'][:100]}...")
        print(f"Process {process_id} - Fusionrag answer: {answer[:100]}...")
        print("=" * 80)

        # 写入独立文件
        # write_result_to_individual_file(question_copy, individual_file_path)

        # 写入共享文件
        # if shared_file_path and file_lock:
        #     write_result_to_shared_file(question_copy, shared_file_path, file_lock)

        # 将结果放入队列（如果需要进一步处理）
        if result_queue:
            result_queue.put(question_copy)

    print(f"Process {process_id}: Completed processing all questions")

    # 将完成信号放入队列
    if result_queue:
        result_queue.put({"process_id": process_id, "status": "completed"})


def real_time_monitor(result_queue, total_processes, keyword_base):
    """实时监控队列并更新统计信息"""
    completed_processes = 0
    all_results = []

    while completed_processes < total_processes:
        try:
            result = result_queue.get(timeout=10)  # 10秒超时

            if isinstance(result, dict) and result.get("status") == "completed":
                completed_processes += 1
                print(
                    f"Monitor: Process {result['process_id']} completed. Total completed: {completed_processes}/{total_processes}")
            else:
                # 这是一个结果
                all_results.append(result)

                # 更新统计信息
                correct_count = sum(1 for item in all_results if item.get("llm_judge", False))
                total_count = len(all_results)
                accuracy = correct_count / total_count if total_count > 0 else 0

                print(f"\nMonitor: New result received from Process {result.get('process_id', 'unknown')}")
                print(f"Monitor: Current stats - Correct: {correct_count}/{total_count} ({accuracy:.2%})")

                # 每10个结果保存一次汇总文件
                if len(all_results) % 1 == 0:
                    summary_file = f"./results/summary_{keyword_base}_interim.json"
                    with open(summary_file, 'w', encoding='utf-8') as f:
                        json.dump(all_results, f, ensure_ascii=False, indent=4)
                    print(f"Monitor: Interim summary saved to {summary_file}")

        except Exception as e:
            # 超时或其他错误，继续等待
            continue

    # 所有进程完成，保存最终汇总
    print("\nMonitor: All processes completed!")

    # 最终统计
    correct_count = sum(1 for item in all_results if item.get("llm_judge", False))
    total_count = len(all_results)
    accuracy = correct_count / total_count if total_count > 0 else 0

    print(f"Final stats - Correct: {correct_count}/{total_count} ({accuracy:.2%})")

    # 保存最终汇总文件
    # final_summary_file = f"./results/summary_{keyword_base}_final.json"
    # with open(final_summary_file, 'w', encoding='utf-8') as f:
    #     json.dump(all_results, f, ensure_ascii=False, indent=4)
    #
    # print(f"Final summary saved to {final_summary_file}")

    return all_results


def test_question_multiprocess(total_run=200, rate=0.3, reprocess_method="", preprocess_method="", questions_to_run=[]):
    """多进程测试主函数"""
    print("Starting multiprocess testing")

    # 确保结果目录存在
    os.makedirs("./results", exist_ok=True)

    # 定义两个进程的GPU分配
    gpu_configs = [
        ([0, 1, 2, 3], 0),  # 进程1: 使用0-3号GPU，处理前半部分数据
        ([4, 5, 6, 7], 1),  # 进程2: 使用4-7号GPU，处理后半部分数据
    ]

    # 计算每个进程处理的数据范围
    half_run = total_run // 2
    data_ranges = [
        (0, half_run),  # 进程1处理0到half_run
        (half_run, total_run),  # 进程2处理half_run到total_run
    ]

    # 创建共享结果文件路径
    keyword_base = f"model_Qwen3-32B_rate_{rate}_reprocess_method_{reprocess_method}_preprocess_{preprocess_method}"
    shared_file_path = f"./results/result_shared_{keyword_base}.json"

    # 创建管理器和锁
    manager = Manager()
    file_lock = manager.Lock()
    result_queue = manager.Queue()

    # 创建进程列表
    processes = []

    # 启动所有进程
    for (gpu_ids, process_id), (start_idx, end_idx) in zip(gpu_configs, data_ranges):
        p = Process(
            target=run_test_process,
            args=(
                process_id,
                gpu_ids,
                start_idx,
                end_idx,
                total_run,
                rate,
                reprocess_method,
                shared_file_path,
                file_lock,
                result_queue,
                preprocess_method,
                questions_to_run
            )
        )
        processes.append(p)
        p.start()

    # 启动实时监控线程
    monitor_thread = threading.Thread(
        target=real_time_monitor,
        args=(result_queue, len(processes), keyword_base)
    )
    monitor_thread.start()

    # 等待所有进程完成
    for p in processes:
        p.join()

    # 等待监控线程完成
    monitor_thread.join()

    # 合并共享文件中的所有结果（作为备份）
    if os.path.exists(shared_file_path):
        with open(shared_file_path, 'r', encoding='utf-8') as f:
            all_results = json.load(f)

        merged_file = f"./results/result_merged_{keyword_base}.json"
        with open(merged_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)

        print(f"Merged results saved to {merged_file}")

        return all_results

    return []


if __name__ == '__main__':
    print(f"start testing run_question with multiprocess")
    questions_to_run = [
        "What is the Federation Cup in sports?"
    ]
    questions_to_run = []
    # 运行多进程测试
    all_results = test_question_multiprocess(
        total_run=200,
        rate=0.0,
        reprocess_method="DraftModel",
        preprocess_method="space",
        questions_to_run=questions_to_run
    )

    # all_results = test_question_multiprocess(
    #     total_run=200,
    #     rate=0.2,
    #     reprocess_method="DraftModel",
    #     preprocess_method="space",
    #     questions_to_run=[]
    # )

    print("All tests completed!")