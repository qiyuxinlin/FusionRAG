import csv
import json
import chardet


def csv_to_json(csv_file_path, json_file_path):
    with open(csv_file_path, 'rb') as f:
        raw_data = f.read(10000)
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        confidence = result['confidence']
        print(f"检测到编码: {encoding} (可信度: {confidence * 100:.1f}%)")
    # 读取CSV文件
    with open(csv_file_path, 'r', encoding='GB18030') as csv_file:
        # 使用csv.DictReader读取，第一行作为键
        csv_reader = csv.DictReader(csv_file)

        # 将每一行转换为字典，并存储在列表中
        data_list = []
        for row in csv_reader:
            data_list.append(row)

    # 将数据写入JSON文件
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data_list, json_file, ensure_ascii=False, indent=2)


# 使用示例
def compare(file1: str, file2: str):
    all_questions = []
    with open(file1, 'r', encoding='utf-8') as f:
        res1 = json.load(f)
        for result in res1:
            if "Sub Question" in result:
                all_questions.append(result[result["Sub Question"]])
    with open(file2, 'r', encoding='utf-8') as f:
        res2 = json.load(f)


csv_to_json("./DraftModel_global_topk_10_rate_0.3_draft_Qwen2.5-3B-Instruct_revert_rope_long_decode_new.csv", "./DraftModel_global_topk_10_rate_0.3_draft_Qwen2.5-3B-Instruct_revert_rope_long_decode_new.json")