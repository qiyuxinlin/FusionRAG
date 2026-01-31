#!/usr/bin/env python3
"""
从BGE数据集提取实际使用的文档，构建新的文档池，然后重新生成优化数据集。
"""
import json
from typing import Dict, List, Set, Any

def load_json(filepath: str) -> Any:
    """Load JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data: Any, filepath: str) -> None:
    """Save data to JSON file with proper formatting."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def extract_all_docs_from_bge(bge_data: List[Dict]) -> Set[str]:
    """从BGE数据集提取所有唯一文档文本。"""
    all_docs = set()

    for sample in bge_data:
        # 从 gold_docs 提取
        for doc in sample.get('gold_docs', []):
            if isinstance(doc, str) and doc.strip():
                all_docs.add(doc.strip())

        # 从 retrieved_results 提取
        for doc in sample.get('retrieved_results', []):
            if isinstance(doc, str) and doc.strip():
                all_docs.add(doc.strip())

        # 从 intermediate_context 提取
        for context in sample.get('intermediate_context', []):
            # retrieve docs
            for doc in context.get('retrieve docs', []):
                if isinstance(doc, str) and doc.strip():
                    all_docs.add(doc.strip())

            # retrieve docs supported
            for doc in context.get('retrieve docs supported', []):
                if isinstance(doc, str) and doc.strip():
                    all_docs.add(doc.strip())

    return all_docs

def build_new_doc_pool(doc_texts: Set[str]) -> List[Dict]:
    """构建新的文档池，为每个文档分配ID。"""
    # 排序以确保一致性
    sorted_docs = sorted(list(doc_texts))

    # 分配ID（从0或1开始）
    doc_pool = []
    for idx, text in enumerate(sorted_docs, start=1):
        doc_pool.append({
            'id': idx,
            'text': text
        })

    return doc_pool

def build_text_to_id_map(doc_pool: List[Dict]) -> Dict[str, int]:
    """构建文本到ID的映射。"""
    text_to_id = {}
    for doc in doc_pool:
        text = doc['text'].strip()
        doc_id = doc['id']
        text_to_id[text] = doc_id
    return text_to_id

def replace_docs_with_ids(bge_data: List[Dict], text_to_id: Dict[str, int]) -> List[Dict]:
    """将BGE数据集中的文档文本替换为doc_id。"""
    optimized_data = []

    not_found_count = 0
    total_count = 0

    for sample in bge_data:
        optimized = sample.copy()

        # 替换 gold_docs
        if 'gold_docs' in optimized and optimized['gold_docs']:
            new_gold_docs = []
            for doc in optimized['gold_docs']:
                total_count += 1
                text = doc.strip() if isinstance(doc, str) else ""
                if text in text_to_id:
                    new_gold_docs.append(text_to_id[text])
                else:
                    not_found_count += 1
                    new_gold_docs.append({
                        "idx": -1,
                        "error": "not_found_in_pool",
                        "text_preview": text[:100]
                    })
            optimized['gold_docs'] = new_gold_docs

        # 替换 retrieved_results
        if 'retrieved_results' in optimized and optimized['retrieved_results']:
            new_results = []
            for doc in optimized['retrieved_results']:
                total_count += 1
                text = doc.strip() if isinstance(doc, str) else ""
                if text in text_to_id:
                    new_results.append(text_to_id[text])
                else:
                    not_found_count += 1
                    new_results.append({
                        "idx": -1,
                        "error": "not_found_in_pool",
                        "text_preview": text[:100]
                    })
            optimized['retrieved_results'] = new_results

        # 替换 intermediate_context
        if 'intermediate_context' in optimized:
            new_contexts = []
            for context in optimized['intermediate_context']:
                new_context = context.copy()

                # 替换 retrieve docs
                if 'retrieve docs' in new_context and new_context['retrieve docs']:
                    new_retrieve_docs = []
                    for doc in new_context['retrieve docs']:
                        total_count += 1
                        text = doc.strip() if isinstance(doc, str) else ""
                        if text in text_to_id:
                            new_retrieve_docs.append(text_to_id[text])
                        else:
                            not_found_count += 1
                            new_retrieve_docs.append({
                                "idx": -1,
                                "error": "not_found_in_pool",
                                "text_preview": text[:100]
                            })
                    new_context['retrieve docs'] = new_retrieve_docs

                # 替换 retrieve docs supported
                if 'retrieve docs supported' in new_context and new_context['retrieve docs supported']:
                    new_supported_docs = []
                    for doc in new_context['retrieve docs supported']:
                        total_count += 1
                        text = doc.strip() if isinstance(doc, str) else ""
                        if text in text_to_id:
                            new_supported_docs.append(text_to_id[text])
                        else:
                            not_found_count += 1
                            new_supported_docs.append({
                                "idx": -1,
                                "error": "not_found_in_pool",
                                "text_preview": text[:100]
                            })
                    new_context['retrieve docs supported'] = new_supported_docs

                new_contexts.append(new_context)
            optimized['intermediate_context'] = new_contexts

        optimized_data.append(optimized)

    return optimized_data, not_found_count, total_count

def main():
    # 文件路径
    bge_path = '/home/shm/document/exp/FusionRAG/data/result_reflect_merged.json'
    new_pool_path = '/home/shm/document/exp/FusionRAG/data/musique_input_rebuilt.json'
    optimized_path = '/home/shm/document/exp/FusionRAG/data/musique_merge_reflect_optimized.json'
    # doc_pool_path = '/home/shm/document/exp/FusionRAG/data/musique_input.json'
    # input_path = '/home/shm/document/exp/FusionRAG/data/result_reflect_merged.json'
    # output_path = '/home/shm/document/exp/FusionRAG/data/result_musique_merge_reflect_optimized.json'
    print("=" * 80)
    print("步骤 1: 加载BGE数据集")
    print("=" * 80)
    bge_data = load_json(bge_path)
    print(f"加载了 {len(bge_data)} 个样本")

    print("\n" + "=" * 80)
    print("步骤 2: 提取所有唯一文档")
    print("=" * 80)
    all_docs = extract_all_docs_from_bge(bge_data)
    print(f"找到 {len(all_docs)} 个唯一文档")

    print("\n" + "=" * 80)
    print("步骤 3: 构建新文档池")
    print("=" * 80)
    doc_pool = build_new_doc_pool(all_docs)
    print(f"新文档池包含 {len(doc_pool)} 个文档")
    print(f"ID范围: 1 到 {doc_pool[-1]['id']}")

    # 保存新文档池
    save_json(doc_pool, new_pool_path)
    print(f"已保存到: {new_pool_path}")

    print("\n" + "=" * 80)
    print("步骤 4: 构建文本到ID的映射")
    print("=" * 80)
    text_to_id = build_text_to_id_map(doc_pool)
    print(f"映射包含 {len(text_to_id)} 个条目")

    print("\n" + "=" * 80)
    print("步骤 5: 替换文档文本为doc_id")
    print("=" * 80)
    optimized_data, not_found, total = replace_docs_with_ids(bge_data, text_to_id)
    print(f"总文档数: {total}")
    print(f"成功匹配: {total - not_found} ({(total-not_found)/total*100:.1f}%)")
    print(f"未匹配: {not_found} ({not_found/total*100:.1f}%)")

    if not_found > 0:
        print(f"\n⚠️  警告: 仍有 {not_found} 个文档未匹配！这不应该发生。")
    else:
        print(f"\n✓ 所有文档都成功匹配！")

    # 保存优化后的数据集
    save_json(optimized_data, optimized_path)
    print(f"已保存到: {optimized_path}")

    print("\n" + "=" * 80)
    print("完成！")
    print("=" * 80)
    print(f"\n生成的文件:")
    print(f"  1. 新文档池: {new_pool_path} ({len(doc_pool)} 个文档)")
    print(f"  2. 优化数据集: {optimized_path} ({len(optimized_data)} 个样本)")
    print(f"\n下一步:")
    print(f"  修改 test_fusionrag_reflect_v2.py 中的文档池路径:")
    print(f"    corpus_filename = '2wiki_input_rebuilt.json'")

if __name__ == "__main__":
    main()
