# FusionRAG 文档编号与KV缓存机制详解

## 📚 概述

`test_fusionrag_reflect.py` 使用了一套**两级编号系统**来管理文档：
1. **局部编号（Local Index）**: 每个问题内部的文档索引
2. **全局编号（Global Index）**: 跨所有问题的统一文档索引

这个设计允许在预处理阶段进行**跨问题的文档检索和KV融合**。

---

## 🔢 文档编号系统

### 1. 三种索引类型

```python
# 对于问题 example_id 的第 doc_idx 个文档：

local_idx = doc_idx                           # 局部索引：0, 1, 2, ...
chunk_id = doc_idx + 1                        # Chunk ID：1, 2, 3, ...
global_doc_idx = sum(corpus_lens[:example_id]) + doc_idx  # 全局索引
```

| 索引类型 | 范围 | 用途 |
|---------|------|------|
| **local_idx** | 每个问题从0开始 | 遍历问题内的文档 |
| **chunk_id** | 每个问题从1开始 | KV缓存文件命名 |
| **global_doc_idx** | 全局从0开始连续 | 跨问题检索和融合 |

### 2. corpus_lens 数组

`corpus_lens` 记录每个问题的文档数量：

```python
corpus_lens = [5, 3, 7]  # 问题0有5个文档，问题1有3个，问题2有7个
```

**计算全局索引**：
```python
global_offset = sum(corpus_lens[:example_id])
global_doc_idx = global_offset + local_idx
```

### 3. 完整示例

假设有3个问题，文档数分别为 [5, 3, 7]：

| 问题ID | 局部索引 | Chunk ID | 全局索引 | KV文件名 |
|-------|---------|---------|---------|----------|
| **问题0** | | | | |
| 0 | 0 | 1 | 0 | `0_1_key.pt` |
| 0 | 1 | 2 | 1 | `0_2_key.pt` |
| 0 | 2 | 3 | 2 | `0_3_key.pt` |
| 0 | 3 | 4 | 3 | `0_4_key.pt` |
| 0 | 4 | 5 | 4 | `0_5_key.pt` |
| **问题1** | | | | |
| 1 | 0 | 1 | 5 | `1_1_key.pt` |
| 1 | 1 | 2 | 6 | `1_2_key.pt` |
| 1 | 2 | 3 | 7 | `1_3_key.pt` |
| **问题2** | | | | |
| 2 | 0 | 1 | 8 | `2_1_key.pt` |
| 2 | 1 | 2 | 9 | `2_2_key.pt` |
| 2 | 2 | 3 | 10 | `2_3_key.pt` |
| 2 | 3 | 4 | 11 | `2_4_key.pt` |
| 2 | 4 | 5 | 12 | `2_5_key.pt` |
| 2 | 5 | 6 | 13 | `2_6_key.pt` |
| 2 | 6 | 7 | 14 | `2_7_key.pt` |

---

## 💾 KV缓存文件结构

### 1. 文件命名规则

```
cache_directory/
├── 0_0_key.pt       # 问题0的system prompt KV
├── 0_0_value.pt
├── 0_1_key.pt       # 问题0的第1个文档 KV
├── 0_1_value.pt
├── 0_2_key.pt       # 问题0的第2个文档 KV
├── 0_2_value.pt
├── ...
├── 1_0_key.pt       # 问题1的system prompt KV
├── 1_0_value.pt
├── 1_1_key.pt       # 问题1的第1个文档 KV
├── 1_1_value.pt
└── ...
```

**命名格式**: `{example_id}_{chunk_id}_{key/value}.pt`

- `example_id`: 问题编号（从0开始）
- `chunk_id`: 文档编号（0表示system prompt，1+表示文档）
- `key/value`: KV缓存的类型

### 2. System Prompt 特殊处理

每个问题都有自己的 system prompt KV缓存：
- `chunk_id = 0` 专门用于 system prompt
- 文件名: `{example_id}_0_key.pt` 和 `{example_id}_0_value.pt`

---

## 🔄 KV缓存生成流程

### 阶段1: 数据准备（prepare_reflect_data）

```python
# test_fusionrag_reflect.py:407-499

questions_data = []
corpus_lens = []
global_corpus = []

for main_q_idx, data_item in enumerate(dataset):
    question_docs = []  # 当前问题的文档
    doc_to_idx = {}     # 文档去重映射

    # 从 intermediate_context 提取所有文档
    for sub_q in intermediate_context:
        docs = sub_q.get("retrieve docs", [])
        for doc in docs:
            if doc not in doc_to_idx:
                question_docs.append(doc)
                chunk_id = len(question_docs)  # chunk_id 从 1 开始
                doc_to_idx[doc] = chunk_id

    # Tokenize 文档
    doc_tensors = []
    for doc in question_docs:
        doc_text = f"Document: {doc}\n"
        doc_tensor = tokenizer.encode(doc_text)
        doc_tensors.append(doc_tensor)

    # 添加到全局语料库
    global_corpus.extend(question_docs)
    corpus_lens.append(len(question_docs))

    questions_data.append({
        'docs': question_docs,
        'doc_tensors': doc_tensors,
        'sub_questions': sub_questions_info,
        ...
    })
```

**关键变量**:
- `global_corpus`: 所有文档的列表（按问题顺序拼接）
- `corpus_lens`: `[5, 3, 7, ...]` 记录每个问题的文档数
- `questions_data[i]['doc_tensors']`: 问题i的所有文档tokens

### 阶段2: 计算相似度矩阵（context_rank）

```python
# test_fusionrag_reflect.py:846-927

if recall_method == RecallMethod.BGE:
    # 使用 BGE 模型计算文档相似度
    bge_model = BGEM3FlagModel(bge_model_path)
    embeddings = bge_model.encode(global_corpus)

    # 构建 FAISS 索引
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # 为每个文档检索 topk 个最相似文档
    similarities, idx = index.search(embeddings, topk + 1)

    # context_rank: [total_docs x topk] 全局索引矩阵
    context_rank = idx[:, 1:]  # 排除自己
```

**context_rank 结构**:
```python
context_rank[global_doc_idx] = [sim_doc_1, sim_doc_2, ..., sim_doc_topk]
# 每一行是一个文档的 topk 个最相似文档的全局索引
```

示例：
```
global_doc_idx=0: [3, 7, 12, 5, ...]   # 全局文档0最相似的是3, 7, 12...
global_doc_idx=1: [0, 9, 4, 11, ...]
...
```

### 阶段3: 生成无预处理的KV缓存

```python
# test_fusionrag_reflect.py:1845-1871

for example_id, q_data in enumerate(questions_data):
    # 1. 生成 system prompt 的 KV (chunk_id=0)
    system_cache_path = f'{save_path}/{example_id}_0_key.pt'
    if not os.path.exists(system_cache_path):
        input_tensor = system_tensor.unsqueeze(0)
        prefill_and_save_kv_cache(
            model, tokenizer, past_key_values, input_tensor,
            save_path=save_path,
            example_id=example_id,
            chunk_id=0,  # System prompt 使用 chunk_id=0
            system_len=system_len,
            passage_len=system_len,
            ...
        )

    # 2. 为每个文档生成 KV cache (chunk_id=1, 2, 3, ...)
    for doc_idx, doc_tensor in enumerate(q_data['doc_tensors']):
        chunk_id = doc_idx + 1
        cache_key_path = f'{save_path}/{example_id}_{chunk_id}_key.pt'

        if not os.path.exists(cache_key_path):
            # Input = system_prompt + document
            input_tensor = torch.cat((system_tensor, doc_tensor)).unsqueeze(0)

            prefill_and_save_kv_cache(
                model, tokenizer, past_key_values, input_tensor,
                save_path=save_path,
                example_id=example_id,
                chunk_id=chunk_id,
                system_len=system_len,
                passage_len=doc_tensor.shape[0],
                ...
            )
```

**生成的文件**:
```
问题0 (5个文档):
  0_0_key.pt, 0_0_value.pt     # System prompt
  0_1_key.pt, 0_1_value.pt     # 文档1
  0_2_key.pt, 0_2_value.pt     # 文档2
  0_3_key.pt, 0_3_value.pt     # 文档3
  0_4_key.pt, 0_4_value.pt     # 文档4
  0_5_key.pt, 0_5_value.pt     # 文档5
```

### 阶段4: FusionRAG 预处理（融合相似文档的KV）

```python
# test_fusionrag_reflect.py:1914-2200

for example_id, q_data in enumerate(questions_data):
    for doc_idx, doc_tensor in enumerate(q_data['doc_tensors']):
        chunk_id = doc_idx + 1

        # 计算当前文档的全局索引
        global_doc_idx = sum(corpus_lens[:example_id]) + doc_idx

        # 初始化：加载 system prompt KV
        load_kv_cache(f'{save_path}/{example_id}_0_key.pt', past_key_values)

        # 加载当前文档的相似文档的 KV
        if global_doc_idx < len(context_rank):
            for similar_global_idx in context_rank[global_doc_idx][:topk]:
                # 将全局索引转换回 (example_id, chunk_id)
                corpus_i, c_id = find_group_and_index(corpus_lens, similar_global_idx)
                similar_chunk_id = c_id + 1

                # 加载相似文档的 KV
                similar_cache_path = f'{save_path}/{corpus_i}_{similar_chunk_id}_key.pt'
                load_kv_cache(similar_cache_path, past_key_values)

        # 使用融合后的 KV 再次前向传播，保存预处理后的 KV
        prefill_with_cache_and_save_preprocess(
            model, tokenizer, past_key_values, doc_tensor,
            save_path=preprocess_save_path,
            example_id=example_id,
            chunk_id=chunk_id,
            ...
        )
```

**关键函数**:
```python
def find_group_and_index(corpus_lens, global_idx):
    """
    将全局索引转换为 (问题ID, 局部索引)

    Args:
        corpus_lens: [5, 3, 7]
        global_idx: 6

    Returns:
        corpus_i: 1      # 问题1
        local_idx: 1     # 局部索引1
    """
    cumsum = 0
    for i, length in enumerate(corpus_lens):
        if global_idx < cumsum + length:
            return i, global_idx - cumsum
        cumsum += length
    return -1, -1
```

**示例**:

假设 `global_doc_idx=6` (问题1的第2个文档)，`topk=3`：

```python
context_rank[6] = [2, 9, 13]  # 最相似的3个文档

# 融合过程：
# 1. 加载 1_0 (问题1的system prompt)
# 2. 加载 0_3 (全局索引2 → 问题0的第3个文档)
# 3. 加载 2_2 (全局索引9 → 问题2的第2个文档)
# 4. 加载 2_6 (全局索引13 → 问题2的第6个文档)
# 5. 前向传播当前文档，保存融合后的KV
```

生成的预处理KV文件：
```
preprocess_cache/1_2_key.pt
preprocess_cache/1_2_value.pt
# 包含了相似文档 [2, 9, 13] 的融合信息
```

---

## 🎯 跨问题检索的意义

### 为什么需要全局索引？

**原因**: FusionRAG 的核心是**跨问题的文档KV融合**

```
问题1的文档3 可能与 问题5的文档7 相似
                ↓
        使用全局索引进行检索
                ↓
    加载问题5文档7的KV，融合到问题1文档3的KV中
```

### PreprocessScope 的作用

| Scope | 检索范围 | 用途 |
|-------|---------|------|
| **GLOBAL** | 所有问题的所有文档 | 最大化KV融合机会（默认） |
| **PER_EXAMPLE** | 仅当前问题的文档 | 避免跨问题信息泄露 |
| **SKIP_UNTESTED** | 排除测试集问题的文档 | 防止训练/测试集污染 |

---

## 📊 完整流程图

```
┌─────────────────────────────────────────────────────────────┐
│  1. 数据加载 (prepare_reflect_data)                         │
│     - 读取 result_reflect.json                              │
│     - 提取所有文档到 global_corpus                          │
│     - 记录 corpus_lens = [5, 3, 7, ...]                     │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  2. 计算相似度矩阵 (context_rank)                           │
│     - BGE 编码所有文档                                      │
│     - FAISS 检索每个文档的 topk 相似文档                    │
│     - context_rank[global_idx] = [sim1, sim2, ..., simk]   │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  3. 生成原始 KV 缓存                                        │
│     for 问题 i:                                             │
│       - 保存 i_0 (system prompt KV)                         │
│       for 文档 j:                                           │
│         - 保存 i_j (文档 j 的 KV)                           │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  4. FusionRAG 预处理 (可选)                                 │
│     for 问题 i 的文档 j:                                    │
│       global_idx = sum(corpus_lens[:i]) + j                 │
│       similar_docs = context_rank[global_idx]               │
│       for sim_idx in similar_docs:                          │
│         (corpus_i, c_id) = find_group(corpus_lens, sim_idx) │
│         加载 corpus_i_{c_id+1} 的 KV                        │
│       融合所有相似文档的 KV                                 │
│       保存预处理后的 KV 到 preprocess_cache/i_j             │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│  5. 生成答案                                                │
│     for 子问题:                                             │
│       加载相关文档的 KV (原始或预处理)                      │
│       生成答案                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔍 关键代码位置

| 功能 | 文件位置 |
|------|---------|
| 数据准备 | `test_fusionrag_reflect.py:407-499` |
| 计算 context_rank | `test_fusionrag_reflect.py:735-927` |
| 生成原始 KV | `test_fusionrag_reflect.py:1845-1871` |
| FusionRAG 预处理 | `test_fusionrag_reflect.py:1914-2200` |
| 全局索引转换 | `ktransformers/util/utils.py:find_group_and_index()` |
| KV 缓存生成 | `ktransformers/util/utils.py:prefill_and_save_kv_cache()` |

---

## 💡 总结

1. **三级编号**: local_idx (0-based) → chunk_id (1-based) → global_doc_idx (0-based 全局)

2. **corpus_lens 是关键**: 记录每个问题的文档数，用于全局/局部索引转换

3. **KV 文件命名**: `{example_id}_{chunk_id}_{key/value}.pt`
   - `chunk_id=0`: system prompt
   - `chunk_id≥1`: 文档

4. **跨问题检索**: 通过 global_doc_idx 实现跨问题的文档相似度检索和KV融合

5. **FusionRAG 的核心**:
   - 使用全局索引找到相似文档（可能来自不同问题）
   - 加载这些文档的KV缓存
   - 融合成新的预处理KV缓存
   - 提升推理质量
