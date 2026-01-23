#!/usr/bin/env python3
"""
Generate QA pairs using DeepSeek API from documents
实时逐行保存版本
"""
import json
import random
import os
import time
from typing import List, Dict
import requests

# ============== 配置区域 ==============
# DeepSeek API 配置
DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"  # 或者你自定义的URL
DEEPSEEK_API_KEY = "sk-519d391217894b6e91e7c2ebf2a9f4df"

# 文档配置
CORPUS_PATH = "/home/shm/document/exp/FusionRAG/data/musique_input_clean_one_use.json"
OUTPUT_PATH = "/home/shm/document/exp/FusionRAG/data/generated_qa_pairs.jsonl"

# 生成配置
NUM_DOCS = 100  # 要生成的文档数量
MODEL_NAME = "deepseek-chat"  # 或 "deepseek-coder"

# ============== API 调用函数 ==============
def call_deepseek_api(prompt: str, api_base: str, api_key: str, model: str = "deepseek-chat") -> str:

    url = f"{api_base}/chat/completions"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.7,
        "max_tokens": 2000
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        print(f"  ❌ API调用失败: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"  响应内容: {e.response.text}")
        raise

# ============== 问答生成函数 ==============
def generate_qa_for_document(document: Dict, api_base: str, api_key: str, model: str) -> Dict:

    doc_id = document['id']
    doc_text = document['text']
    
    # 截断过长的文档（保留前2000字符）
    if len(doc_text) > 2000:
        doc_text = doc_text[:2000] + "..."
    
    prompt = f"""Please read the following document and generate a question-answer pair based on its content.

Document:
{doc_text}

Requirements:
1. The question should be specific and answerable from the document
2. The answer should be accurate and complete, based ONLY on the provided document
3. Do not use external knowledge
4. Generate challenging questions that require understanding the document
5. The answer must be concise, no more than 10 words.

Please output in the following JSON format:
{{
    "question": "your generated question",
    "answer": "the answer to the question"
}}

Output only the JSON, no other text:"""

    print(f"  Generating QA for document {doc_id}...")
    
    try:
        response = call_deepseek_api(prompt, api_base, api_key, model)
        
        # 尝试解析JSON响应
        # 移除可能的markdown代码块标记
        response = response.strip()
        if response.startswith('```json'):
            response = response[7:]
        if response.startswith('```'):
            response = response[3:]
        if response.endswith('```'):
            response = response[:-3]
        response = response.strip()
        
        qa_pair = json.loads(response)
        
        return {
            "question": qa_pair["question"],
            "answer": qa_pair["answer"],
            "gold_docs_id": doc_id,
            "gold_docs":doc_text,
        }
        
    except json.JSONDecodeError as e:
        print(f"  ❌ JSON解析失败: {e}")
        print(f"  响应内容: {response[:500]}...")
        return None
    except Exception as e:
        print(f"  ❌ 生成失败: {e}")
        return None

# ============== 主函数 ==============
def main():
    print("="*80)
    print("DeepSeek QA Pair Generator (Real-time Saving)")
    print("="*80)
    
    # 检查API Key
    if not DEEPSEEK_API_KEY:
        print("\n❌ 错误: 请设置 DEEPSEEK_API_KEY")
        print("   在脚本开头的配置区域填入你的API Key")
        print("\n   或者使用环境变量:")
        print("   export DEEPSEEK_API_KEY='your-api-key'")
        return
    
    # 1. 读取语料库
    print(f"\n[Step 1] Reading corpus from {CORPUS_PATH}...")
    try:
        with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
            corpus = json.load(f)
        print(f"  ✅ Loaded {len(corpus)} documents")
    except FileNotFoundError:
        print(f"  ❌ File not found: {CORPUS_PATH}")
        return
    except Exception as e:
        print(f"  ❌ Error loading corpus: {e}")
        return
    
    # 2. 随机选择文档
    print(f"\n[Step 2] Randomly selecting {NUM_DOCS} documents...")
    if NUM_DOCS > len(corpus):
        print(f"  ⚠ Warning: NUM_DOCS ({NUM_DOCS}) > corpus size ({len(corpus)})")
        print(f"  Using all documents instead")
        selected_docs = corpus
    else:
        selected_docs = random.sample(corpus, NUM_DOCS)
    
    print(f"  ✅ Selected {len(selected_docs)} documents")
    print(f"  Selected doc IDs: {[doc['id'] for doc in selected_docs]}")
    
    # 3. 打开输出文件（JSONL格式）
    print(f"\n[Step 3] Opening output file: {OUTPUT_PATH}")
    print(f"  Mode: append (real-time saving)")
    
    # 如果文件已存在，询问是否覆盖
    if os.path.exists(OUTPUT_PATH):
        print(f"  ⚠ File already exists!")
        response = input(f"  Overwrite? (y/n): ").strip().lower()
        if response == 'y':
            open_mode = 'w'
            print(f"  → Will overwrite existing file")
        else:
            open_mode = 'a'
            print(f"  → Will append to existing file")
    else:
        open_mode = 'w'
    
    # 4. 为每个文档生成问答对并实时保存
    print(f"\n[Step 4] Generating QA pairs with real-time saving...")
    print(f"  Model: {MODEL_NAME}")
    print(f"  API Base: {DEEPSEEK_API_BASE}")
    print(f"  Output: {OUTPUT_PATH} (JSONL format)")
    
    success_count = 0
    fail_count = 0
    
    try:
        with open(OUTPUT_PATH, open_mode, encoding='utf-8') as f:
            for i, doc in enumerate(selected_docs, 1):
                print(f"\n  [{i}/{len(selected_docs)}] Document ID: {doc['id']}")
                
                try:
                    # 生成问答对
                    qa_pair = generate_qa_for_document(doc, DEEPSEEK_API_BASE, DEEPSEEK_API_KEY, MODEL_NAME)
                    
                    if qa_pair:
                        # 添加时间戳
                        # qa_pair['generated_at'] = time.strftime("%Y-%m-%d %H:%M:%S")
                        # qa_pair['model'] = MODEL_NAME
                        
                        # 实时保存到文件（JSONL格式：每行一个JSON对象）
                        f.write(json.dumps(qa_pair, ensure_ascii=False) + '\n')
                        f.flush()  # 强制刷新到磁盘
                        
                        success_count += 1
                        print(f"  ✅ Success!")
                        print(f"     Question: {qa_pair['question'][:80]}...")
                        print(f"     Answer: {qa_pair['answer'][:80]}...")
                        print(f"     💾 Saved to file immediately")
                    else:
                        fail_count += 1
                        print(f"  ❌ Failed to generate QA pair")
                    
                    # 避免API限流
                    if i < len(selected_docs):
                        sleep_time = 0.5
                        print(f"     ⏳ Waiting {sleep_time}s before next request...")
                        time.sleep(sleep_time)
                        
                except KeyboardInterrupt:
                    print(f"\n  ⚠ Interrupted by user")
                    print(f"  ✅ Generated pairs saved so far: {success_count}")
                    break
                except Exception as e:
                    fail_count += 1
                    print(f"  ❌ Error: {e}")
                    continue
    
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        return
    
    # 5. 统计信息
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nTotal documents processed: {len(selected_docs)}")
    print(f"Successful QA pairs: {success_count}")
    print(f"Failed: {fail_count}")
    if len(selected_docs) > 0:
        print(f"Success rate: {success_count/len(selected_docs)*100:.1f}%")
    
    print(f"\n💾 All saved pairs have been written to:")
    print(f"   {OUTPUT_PATH}")
    
    # 显示文件行数
    try:
        with open(OUTPUT_PATH, 'r') as f:
            line_count = sum(1 for _ in f)
        print(f"   Total lines in file: {line_count}")
    except:
        pass
    
    # 显示最后几个样本
    print(f"\n📝 Last saved sample:")
    try:
        with open(OUTPUT_PATH, 'r') as f:
            lines = f.readlines()
            if lines:
                last_sample = json.loads(lines[-1])
                print(f"  Doc ID: {last_sample['doc_id']}")
                print(f"  Question: {last_sample['question']}")
                print(f"  Answer: {last_sample['answer']}")
    except:
        print(f"  Unable to read sample")
    
    print(f"\n✅ Generation completed!")

if __name__ == "__main__":
    # 支持环境变量配置
    if not DEEPSEEK_API_KEY and os.environ.get('DEEPSEEK_API_KEY'):
        DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY')
    
    main()
