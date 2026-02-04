#!/home/shm/anaconda3/envs/fusionrag/bin/python
"""
KV Cache 单元测试 - 验证 KV cache 加载的正确性

测试方法：加载单个文档的 KV cache，让模型重复文档内容，对比生成结果与原始文档
"""

import os
import sys
import json
import re
import torch
import time
from pathlib import Path

# 添加项目路径
sys.path.insert(0, '/home/shm/document/exp/FusionRAG')

# 导入 utils.py 中的函数（使用正确的 doc_{doc_id}_key.pt 格式）
from ktransformers.util.utils import load_kv_and_generate

def load_document_pool(doc_pool_path):
    """加载文档池，创建 id -> text 映射"""
    with open(doc_pool_path, 'r', encoding='utf-8') as f:
        doc_pool = json.load(f)
    return {doc['id']: doc['text'] for doc in doc_pool}

def get_available_doc_ids(cache_dir):
    """扫描 KV cache 目录，提取所有可用的 doc_id"""
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        raise FileNotFoundError(f"KV cache 目录不存在: {cache_dir}")

    key_files = list(cache_path.glob('doc_*_key.pt'))
    doc_ids = set()

    for f in key_files:
        match = re.match(r'doc_(\d+)_key\.pt', f.name)
        if match:
            doc_ids.add(int(match.group(1)))

    return sorted(doc_ids)

def evaluate_repetition(original, generated):
    """
    评估重复准确性

    Args:
        original: 原始文档文本
        generated: 模型生成的文本

    Returns:
        dict: 包含各种准确性指标
    """
    # 尝试从生成文本中提取 JSON 格式的 repeated_text
    try:
        # 查找 JSON 对象
        json_match = re.search(r'\{[^{}]*"repeated_text"[^{}]*\}', generated, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            json_obj = json.loads(json_str)
            repeated = json_obj.get('repeated_text', '')
            # 如果 repeated_text 被引号包围，去除引号
            if repeated.startswith('"') and repeated.endswith('"'):
                repeated = repeated[1:-1]
        else:
            # 如果没找到 JSON，尝试直接提取文本
            repeated = generated
    except Exception as e:
        print(f"  [Warning] JSON 解析失败: {e}")
        repeated = generated

    # 清理文本
    original_clean = original.strip()
    repeated_clean = repeated.strip()

    # 计算完全匹配
    exact_match = (original_clean == repeated_clean)

    # 字符级准确率
    if len(original_clean) > 0:
        char_matches = sum(c1 == c2 for c1, c2 in zip(original_clean, repeated_clean))
        char_accuracy = char_matches / len(original_clean)
    else:
        char_accuracy = 0.0

    # 词级准确率
    original_words = original_clean.split()
    repeated_words = repeated_clean.split()

    if len(original_words) > 0:
        word_matches = sum(w1 == w2 for w1, w2 in zip(original_words, repeated_words))
        word_accuracy = word_matches / len(original_words)
    else:
        word_accuracy = 0.0

    return {
        'exact_match': exact_match,
        'char_accuracy': char_accuracy,
        'word_accuracy': word_accuracy,
        'repeated_text': repeated_clean[:200],  # 保存前 200 字符用于检查
        'original_length': len(original_clean),
        'generated_length': len(repeated_clean)
    }

def test_single_document(model, tokenizer, past_key_values, doc_id, original_text,
                          cache_dir, max_new_tokens=500):
    """
    测试单个文档的 KV cache 加载和生成

    Args:
        model: 语言模型
        tokenizer: 分词器
        past_key_values: KV cache 容器
        doc_id: 文档 ID
        original_text: 原始文档文本
        cache_dir: KV cache 目录
        max_new_tokens: 最大生成 token 数

    Returns:
        dict: 测试结果
    """
    # 重要：KV cache 生成时使用了 "Document: " 前缀
    # 参考 test_fusionrag_reflect_v2.py 第 585 行
    # doc_text_formatted = f"Document: {doc_text}\n"
    document_text_formatted = f"Document: {original_text}\n"
    document_tokens = tokenizer.encode(document_text_formatted, return_tensors='pt').squeeze(0)

    # passages[1]: 问题/Instruction（重新计算）
    instruction = """Instruction: Please repeat the text from the previous conversation exactly as it is, word for word, maintaining the original order and format. Output the result in JSON format, with a key-value pair where the key is 'repeated_text' and the value is the repeated text from the prior conversation:"""
    question_tokens = tokenizer.encode(instruction, return_tensors='pt').squeeze(0)

    print(f"\n[Doc {doc_id}] Testing...")
    print(f"  Original text length: {len(original_text)} chars")
    print(f"  Formatted text length: {len(document_text_formatted)} chars")
    print(f"  Document tokens: {len(document_tokens)}")
    print(f"  Instruction tokens: {len(question_tokens)}")

    try:
        # 加载 KV cache 并生成
        # passages[0] = document (从 KV cache 加载)
        # passages[1] = question (重新计算)
        passages = [document_tokens, question_tokens]

        # 使用 rate=0 表示全部使用 KV cache，不需要重计算
        # 使用 DraftModel 方法避开 pdb 断点
        generated_tokens, _, extra_info = load_kv_and_generate(
            model, tokenizer, past_key_values, passages,
            doc_ids=[doc_id],  # passages[:-1] 中文档的 doc_id 列表
            load_path=cache_dir,
            max_new_tokens=max_new_tokens,
            device="cuda",
            revert_rope=True,  # 需要 RoPE 位置调整
            reprocess_method='DraftModel',  # 避开 pdb 断点
            rate=0.0  # rate=0 表示全部使用 KV cache，不需要重计算
        )

        # 解码结果
        # generated_tokens 是一个列表，需要转换为 tensor
        if isinstance(generated_tokens, list):
            generated_tokens = torch.tensor(generated_tokens)
        generated_text = tokenizer.decode(generated_tokens)
        print(f"  Generated tokens: {len(generated_tokens)}")
        print(f"  Generated text length: {len(generated_text)} chars")

        # 评估：注意这里对比的是原始文本（不含前缀）
        result = evaluate_repetition(original_text, generated_text)
        result['doc_id'] = doc_id
        result['success'] = True
        result['error'] = None

        # 打印结果
        print(f"  Exact Match: {result['exact_match']}")
        print(f"  Char Accuracy: {result['char_accuracy']*100:.2f}%")
        print(f"  Word Accuracy: {result['word_accuracy']*100:.2f}%")

        # 如果不匹配，显示部分差异
        if not result['exact_match'] and result['char_accuracy'] < 0.95:
            print(f"  [Sample] Original: {original_text[:100]}...")
            print(f"  [Sample] Generated: {result['repeated_text'][:100]}...")

        return result

    except Exception as e:
        import traceback
        print(f"  [ERROR] {e}")
        print(f"  [TRACEBACK] {traceback.format_exc()}")
        return {
            'doc_id': doc_id,
            'success': False,
            'error': str(e),
            'exact_match': False,
            'char_accuracy': 0.0,
            'word_accuracy': 0.0
        }

def main():
    # 配置
    KV_CACHE_DIR = "/mnt/data3/tmp/fusionrag_online_lazy/Qwen2.5-7B-Instruct/musique/kv_cache"
    DOC_POOL_PATH = "/home/shm/document/exp/FusionRAG/data/musique_input_rebuilt.json"
    MODEL_PATH = "/mnt/data/models/Qwen2.5-7B-Instruct"
    NUM_TEST_DOCS = 10  # 测试文档数量

    print("="*80)
    print("KV Cache 单元测试")
    print("="*80)
    print(f"KV Cache 目录: {KV_CACHE_DIR}")
    print(f"文档池路径: {DOC_POOL_PATH}")
    print(f"模型路径: {MODEL_PATH}")
    print(f"测试文档数: {NUM_TEST_DOCS}")
    print()

    # 1. 加载文档池
    print("[1/5] 加载文档池...")
    doc_map = load_document_pool(DOC_POOL_PATH)
    print(f"  加载了 {len(doc_map)} 个文档")
    print()

    # 2. 扫描可用的 KV cache
    print("[2/5] 扫描 KV cache 文件...")
    available_doc_ids = get_available_doc_ids(KV_CACHE_DIR)
    print(f"  找到 {len(available_doc_ids)} 个文档的 KV cache")
    print()

    # 3. 过滤出在文档池中存在的文档
    print("[3/5] 匹配 KV cache 和文档池...")
    test_doc_ids = [doc_id for doc_id in available_doc_ids if doc_id in doc_map]
    print(f"  可测试的文档数: {len(test_doc_ids)}")
    print()

    if len(test_doc_ids) == 0:
        print("错误：没有可测试的文档！")
        return

    # 4. 加载模型和 tokenizer
    print("[4/5] 加载模型...")
    from transformers import AutoTokenizer
    from ktransformers.models.modeling_qwen2 import Qwen2ForCausalLM
    from ktransformers.models.custom_cache import StaticCache

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    # 使用 ktransformers 的模型类
    torch.set_default_dtype(torch.float16)
    with torch.no_grad():
        model = Qwen2ForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="cuda:0",
            trust_remote_code=True
        )
    model.eval()
    print("  模型加载完成")
    print()

    # 5. 运行测试
    print(f"[5/5] 开始测试前 {min(NUM_TEST_DOCS, len(test_doc_ids))} 个文档...")
    print("="*80)

    results = []
    for i, doc_id in enumerate(test_doc_ids[:NUM_TEST_DOCS]):
        original_text = doc_map[doc_id]

        # 初始化 past_key_values - 使用 StaticCache，参考 test_fusionrag_reflect.py
        max_cache_len = 32768
        past_key_values = StaticCache(
            config=model.config,
            max_batch_size=1,
            max_cache_len=max_cache_len,
            device='cuda',  # 使用默认 cuda (通常是 cuda:0)
            dtype=model.dtype,
            passage_len=32768
        )

        # 测试单个文档
        result = test_single_document(
            model, tokenizer, past_key_values, doc_id, original_text,
            KV_CACHE_DIR, max_new_tokens=200  # 减少到 200 以加快测试
        )
        results.append(result)

    # 6. 输出统计报告
    print("\n" + "="*80)
    print("测试报告")
    print("="*80)

    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    print(f"\n总体统计:")
    print(f"  测试总数: {len(results)}")
    print(f"  成功: {len(successful)}")
    print(f"  失败: {len(failed)}")

    if successful:
        exact_matches = sum(1 for r in successful if r['exact_match'])
        avg_char_acc = sum(r['char_accuracy'] for r in successful) / len(successful)
        avg_word_acc = sum(r['word_accuracy'] for r in successful) / len(successful)

        print(f"\n准确性统计:")
        print(f"  完全匹配: {exact_matches}/{len(successful)} ({exact_matches/len(successful)*100:.1f}%)")
        print(f"  平均字符准确率: {avg_char_acc*100:.2f}%")
        print(f"  平均词准确率: {avg_word_acc*100:.2f}%")

        # 显示成功率
        high_accuracy = sum(1 for r in successful if r['char_accuracy'] >= 0.95)
        print(f"  高准确率 (≥95%): {high_accuracy}/{len(successful)} ({high_accuracy/len(successful)*100:.1f}%)")

    if failed:
        print(f"\n失败案例:")
        for r in failed:
            print(f"  Doc {r['doc_id']}: {r['error']}")

    # 7. 保存详细结果
    output_path = "/home/shm/document/exp/FusionRAG/script/kv_regenerate/test_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'config': {
                'kv_cache_dir': KV_CACHE_DIR,
                'doc_pool_path': DOC_POOL_PATH,
                'num_tested': len(results),
            },
            'results': results
        }, f, ensure_ascii=False, indent=2)
    print(f"\n详细结果已保存至: {output_path}")

if __name__ == "__main__":
    main()
