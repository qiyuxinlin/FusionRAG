#!/usr/bin/env python3
"""
Generate QA pairs using DeepSeek API via OpenAI SDK
"""
import json
import random
import os
import time
from typing import List, Dict
from openai import OpenAI

# ============== 配置区域 ==============
# DeepSeek API 配置
DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"  # 或者你自定义的URL
DEEPSEEK_API_KEY = ""  # 请填入你的API Key

# 文档配置
CORPUS_PATH = "/home/shm/document/exp/FusionRAG/data/musique_input_clean.json"
OUTPUT_PATH = "/home/shm/document/exp/FusionRAG/data/generated_qa_pairs.json"

# 生成配置
NUM_DOCS = 10  # 要生成的文档数量
MODEL_NAME = "deepseek-chat"  # 或 "deepseek-coder"

# ============== API 调用函数 ==============
def call_deepseek_api(prompt: str, client, model: str = "deepseek-chat") -> str:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"  ❌ API调用失败: {e}")
        raise

# ============== 问答生成函数 ==============
def generate_qa_for_document(document: Dict, client, model: str) -> Dict:

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

Please output in the following JSON format:
{{
    "question": "your generated question",
    "answer": "the answer to the question"
}}

Output only the JSON, no other text:"""

    print(f"  Generating QA for document {doc_id}...")
    
    try:
        response = call_deepseek_api(prompt, client, model)
        
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
            "doc_id": doc_id,
            "doc_text_preview": document['text'][:200] + "..." if len(document['text']) > 200 else document['text']
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
    print("DeepSeek QA Pair Generator (OpenAI SDK)")
    print("="*80)
    
    # 检查API Key
    if not DEEPSEEK_API_KEY:
        print("\n❌ 错误: 请设置 DEEPSEEK_API_KEY")
        print("   在脚本开头的配置区域填入你的API Key")
        print("\n   或者使用环境变量:")
        print("   export DEEPSEEK_API_KEY='your-api-key'")
        return
    
    # 初始化OpenAI客户端
    print(f"\n[Init] Initializing DeepSeek client...")
    print(f"  API Base: {DEEPSEEK_API_BASE}")
    print(f"  Model: {MODEL_NAME}")
    
    try:
        client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_API_BASE
        )
        print("  ✅ Client initialized")
    except Exception as e:
        print(f"  ❌ Failed to initialize client: {e}")
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
    
    # 3. 为每个文档生成问答对
    print(f"\n[Step 3] Generating QA pairs...")
    
    qa_pairs = []
    success_count = 0
    fail_count = 0
    
    for i, doc in enumerate(selected_docs, 1):
        print(f"\n  [{i}/{len(selected_docs)}] Document ID: {doc['id']}")
        
        try:
            qa_pair = generate_qa_for_document(doc, client, MODEL_NAME)
            
            if qa_pair:
                qa_pairs.append(qa_pair)
                success_count += 1
                print(f"  ✅ Success!")
                print(f"     Question: {qa_pair['question'][:80]}...")
                print(f"     Answer: {qa_pair['answer'][:80]}...")
            else:
                fail_count += 1
                print(f"  ❌ Failed to generate QA pair")
            
            # 避免API限流，稍微延迟
            if i < len(selected_docs):
                time.sleep(0.5)
                
        except Exception as e:
            fail_count += 1
            print(f"  ❌ Error: {e}")
            continue
    
    # 4. 保存结果
    print(f"\n[Step 4] Saving results to {OUTPUT_PATH}...")
    
    output_data = {
        "metadata": {
            "total_generated": len(qa_pairs),
            "success_count": success_count,
            "fail_count": fail_count,
            "model": MODEL_NAME,
            "api_base": DEEPSEEK_API_BASE,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "qa_pairs": qa_pairs
    }
    
    try:
        with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"  ✅ Saved {len(qa_pairs)} QA pairs")
    except Exception as e:
        print(f"  ❌ Error saving results: {e}")
        return
    
    # 5. 统计信息
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nTotal documents processed: {len(selected_docs)}")
    print(f"Successful QA pairs: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Success rate: {success_count/len(selected_docs)*100:.1f}%")
    
    if qa_pairs:
        print(f"\n📝 Sample QA pair:")
        sample = qa_pairs[0]
        print(f"  Doc ID: {sample['doc_id']}")
        print(f"  Question: {sample['question']}")
        print(f"  Answer: {sample['answer']}")
    
    print(f"\n✅ Results saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    # 支持环境变量配置
    if not DEEPSEEK_API_KEY and os.environ.get('DEEPSEEK_API_KEY'):
        DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY')
    
    main()
