# FusionRAG 迁移指南：使用全局文档ID实现跨问题KV缓存复用

## 概述

本指南说明如何将 FusionRAG 从基于问题的KV缓存组织方式迁移到基于全局文档ID的方式，实现跨问题的文档KV缓存复用。

## 问题背景

### 当前实现的问题

**原有方式（按问题分组）:**
```python
# KV缓存路径: {example_id}_{chunk_id}_key.pt
cache_path = f'{save_path}/{example_id}_{chunk_id}_key.pt'

# 例如:
# - 问题1的文档1: 0_1_key.pt
# - 问题2的文档1: 1_1_key.pt  # 即使是相同文档，也会重复保存
```

**问题:**
1. 相同的文档在不同问题中会重复生成和保存KV缓存
2. 浪费存储空间
3. 浪费计算时间（重复计算）
4. 无法实现跨问题的文档复用

### 新实现的优势

**新方式（全局文档ID）:**
```python
# KV缓存路径: doc_{doc_id}_key.pt
cache_path = f'{save_path}/doc_{doc_id}_key.pt'

# 例如:
# - 文档ID=174: doc_174_key.pt  # 无论在哪个问题中，都使用同一个缓存
# - 文档ID=797: doc_797_key.pt
```

**优势:**
1. **跨问题复用**：相同文档只需生成一次KV缓存
2. **节省存储**：消除重复的KV缓存文件
3. **加速处理**：后续问题使用已缓存的文档时无需重新计算
4. **online lazy模式更高效**：首次遇到文档时生成并缓存，后续直接复用

## 核心变更

### 1. 数据格式变更

#### 原始数据格式 (`result_reflect.json`)
```json
{
    "question": "...",
    "gold_docs": [
        "National Cycle Route 57 is part of...",  // 完整文档文本
        "The National Cycle Network (NCN) is..."
    ],
    "intermediate_context": [{
        "retrieve docs": [
            "National Cycle Route 57 is part of...",  // 完整文档文本
            "The National Cycle Network..."
        ]
    }]
}
```

#### 优化后的数据格式 (`result_reflect_optimized.json`)
```json
{
    "question": "...",
    "gold_docs": [174, 797],  // 全局文档ID
    "intermediate_context": [{
        "retrieve docs": [174, 797, 231, ...]  // 全局文档ID列表
    }]
}
```

#### 文档池 (`musique_input.json`)
```json
[
    {"id": 0, "text": "Sant Martí d'Empúries is...", "metadata": {"lang": "en"}},
    {"id": 1, "text": "The Image Expedition was...", "metadata": {"lang": "en"}},
    ...
    {"id": 174, "text": "National Cycle Route 57 is...", "metadata": {"lang": "en"}},
    {"id": 797, "text": "The National Cycle Network...", "metadata": {"lang": "en"}}
]
```

### 2. 代码架构变更

#### 新增组件

1. **DocumentPoolManager** (`data/doc_pool_manager.py`)
   - 管理全局文档池
   - 提供doc_id到文档文本的映射

2. **prepare_reflect_data_v3** (`prepare_reflect_data_v3.py`)
   - 加载优化后的数据（包含doc_id）
   - 返回doc_id映射信息
   - 使用DocumentPoolManager解析文档

3. **KVCacheManagerV3** (`kv_cache_manager_v3.py`)
   - 管理基于doc_id的KV缓存
   - 追踪已缓存的文档
   - 提供统一的缓存路径接口

## 迁移步骤

### Step 1: 使用优化后的数据文件

**修改配置:**

```bash
# 在 run_online_lazy_sweep.sh 中
# 旧:
DATA_PATH="./data/result_reflect.json"

# 新:
DATA_PATH="./data/result_reflect_optimized.json"
DOC_POOL_PATH="./data/musique_input.json"
```

### Step 2: 修改数据准备函数

**在 `test_fusionrag_reflect_v2.py` 中:**

```python
# 旧的函数调用
questions_data, system_tensor, context_rank, corpus_lens = prepare_reflect_data(
    data_path, tokenizer, bge_model_path, model_type, topk, max_samples, preprocess
)

# 新的函数调用
from prepare_reflect_data_v3 import prepare_reflect_data_v3
from data.doc_pool_manager import DocumentPoolManager

questions_data, system_tensor, context_rank, corpus_lens, doc_pool_manager, doc_id_to_global_idx = prepare_reflect_data_v3(
    data_path=data_path,
    doc_pool_path=doc_pool_path,  # 新增参数
    tokenizer=tokenizer,
    bge_model_path=bge_model_path,
    model_type=model_type,
    topk=topk,
    max_main_questions=max_samples,
    preprocess=preprocess,
    recall_method=recall_method_enum,
    random_seed=random_seed,
    fixed_doc_idx=fixed_doc_idx,
    preprocess_scope=preprocess_scope
)
```

### Step 3: 修改KV缓存生成逻辑

**原有代码（line 1665-1678）:**
```python
# Generate KV cache for each document in THIS main question
for doc_idx, doc_tensor in enumerate(doc_tensors):
    chunk_id = doc_idx + 1
    cache_key_path = f'{save_path}/{example_id}_{chunk_id}_key.pt'

    if not os.path.exists(cache_key_path):
        passage_len = doc_tensor.shape[0]
        input_tensor = torch.cat((system_tensor, doc_tensor)).unsqueeze(0)

        prefill_and_save_kv_cache(
            model, tokenizer, past_key_values, input_tensor.to(input_device),
            save_path=save_path, example_id=example_id, chunk_id=chunk_id,
            system_len=system_len, passage_len=passage_len,
            reprocess_method=reprocess_method, device=input_device, device_map=device_map
        )
```

**新代码:**
```python
# 初始化KV缓存管理器（在main函数开始处）
from kv_cache_manager_v3 import KVCacheManagerV3
kv_cache_manager = KVCacheManagerV3(model_cache_root)

# Generate KV cache for each document using global doc IDs
for doc_id, doc_tensor in zip(q_data['doc_ids'], q_data['doc_tensors']):
    if not kv_cache_manager.is_cached(doc_id):
        print(f"  Generating KV cache for doc_id={doc_id}...")

        passage_len = doc_tensor.shape[0]
        input_tensor = torch.cat((system_tensor, doc_tensor)).unsqueeze(0)

        # Generate KV cache
        prefill_and_save_kv_cache(
            model, tokenizer, past_key_values, input_tensor.to(input_device),
            save_path=save_path, example_id=f'doc', chunk_id=doc_id,  # 使用doc_id
            system_len=system_len, passage_len=passage_len,
            reprocess_method=reprocess_method, device=input_device, device_map=device_map
        )

        # 或者使用新的保存方式（如果修改了prefill_and_save_kv_cache）
        # key_cache = past_key_values.key_cache_to_save
        # value_cache = past_key_values.value_cache_to_save
        # kv_cache_manager.save_kv_cache(doc_id, key_cache, value_cache)
    else:
        print(f"  Doc_id={doc_id} already cached, skipping generation")
```

### Step 4: 修改KV缓存加载逻辑

**原有代码（line 1799-1824）:**
```python
corpus_i, c_id = find_group_and_index(corpus_lens, similar_global_idx)
similar_chunk_id = c_id + 1

# Check and generate if needed
similar_cache_key_path = f"{save_path}/{corpus_i}_{similar_chunk_id}_key.pt"
if not os.path.exists(similar_cache_key_path):
    print(f"      → On-demand: Generating cache for Q{corpus_i+1}-Doc{similar_chunk_id}...")
    # ... generate cache
```

**新代码:**
```python
# 从global_idx获取doc_id
similar_doc_id = global_corpus_doc_ids[similar_global_idx]

# Check and generate if needed
if not kv_cache_manager.is_cached(similar_doc_id):
    print(f"      → On-demand: Generating cache for doc_id={similar_doc_id}...")

    # Get document tensor
    similar_doc_text = doc_pool_manager.get_doc_text(similar_doc_id)
    similar_doc_tokens = tokenizer.encode(f"Document: {similar_doc_text}\n", add_special_tokens=False)
    similar_doc_tensor = torch.tensor(similar_doc_tokens, dtype=torch.long)

    passage_len = similar_doc_tensor.shape[0]
    input_tensor = torch.cat((system_tensor, similar_doc_tensor)).unsqueeze(0)

    prefill_and_save_kv_cache(
        model, tokenizer, past_key_values, input_tensor.to(input_device),
        save_path=save_path, example_id=f'doc', chunk_id=similar_doc_id,
        system_len=system_len, passage_len=passage_len,
        reprocess_method=reprocess_method, device=input_device, device_map=device_map
    )
```

**加载缓存:**
```python
# 旧方式
similar_cache_key_path = f"{save_path}/{corpus_i}_{similar_chunk_id}_key.pt"
similar_key_cache = torch.load(similar_cache_key_path, weights_only=True)
similar_value_cache = torch.load(f"{save_path}/{corpus_i}_{similar_chunk_id}_value.pt", weights_only=True)

# 新方式
similar_key_cache, similar_value_cache = kv_cache_manager.load_kv_cache(similar_doc_id)

# 或者直接使用路径（如果不使用manager）
similar_cache_key_path = f"{save_path}/doc_{similar_doc_id}_key.pt"
similar_key_cache = torch.load(similar_cache_key_path, weights_only=True)
similar_value_cache = torch.load(f"{save_path}/doc_{similar_doc_id}_value.pt", weights_only=True)
```

### Step 5: 修改问题数据结构

在 `questions_data` 中添加 `doc_ids` 字段:

```python
# 旧结构
questions_data.append({
    'main_question': main_question,
    'main_answer': main_answer,
    'sub_questions': sub_questions_info,
    'doc_tensors': doc_tensors,
    # ...
})

# 新结构
questions_data.append({
    'main_question': main_question,
    'main_answer': main_answer,
    'sub_questions': sub_questions_info,
    'doc_ids': question_doc_ids,  # 新增：全局doc ID列表
    'doc_tensors': doc_tensors,
    # ...
})
```

## 完整的修改清单

### 必须修改的文件

1. **test_fusionrag_reflect_v2.py**
   - [ ] 导入新的模块
   - [ ] 修改 prepare_reflect_data 调用
   - [ ] 初始化 KVCacheManagerV3
   - [ ] 修改KV缓存生成循环（line ~1665）
   - [ ] 修改KV缓存加载循环（line ~1799）
   - [ ] 修改preprocess中的相似文档加载（line ~1844）

2. **run_online_lazy_sweep.sh**
   - [ ] 添加 DOC_POOL_PATH 变量
   - [ ] 修改 DATA_PATH 指向优化后的数据
   - [ ] 传递 --doc_pool_path 参数

3. **添加新文件**
   - [ ] `data/doc_pool_manager.py`
   - [ ] `prepare_reflect_data_v3.py`
   - [ ] `kv_cache_manager_v3.py`

### 兼容性注意事项

1. **旧缓存不兼容**: 使用新格式后，旧的 `{example_id}_{chunk_id}_key.pt` 缓存无法使用
2. **需要清理旧缓存**: 建议先清除旧的缓存目录
3. **数据文件版本**: 确保使用 `result_reflect_optimized.json`

## 测试和验证

### 运行测试脚本

```bash
# 测试DocumentPoolManager
cd /home/shm/document/exp/FusionRAG
python data/doc_pool_manager.py

# 测试prepare_reflect_data_v3
python prepare_reflect_data_v3.py

# 测试KVCacheManagerV3
python kv_cache_manager_v3.py
```

### 验证跨问题复用

运行两次相同的测试，第二次应该明显更快：

```bash
# 第一次运行：生成所有KV缓存
bash run_online_lazy_sweep.sh

# 第二次运行：应该跨问题复用缓存
bash run_online_lazy_sweep.sh
```

检查缓存使用情况：

```bash
# 查看缓存统计
ls /mnt/data3/tmp/fusionrag_online_lazy/Qwen2.5-7B-Instruct/musique/kv_cache_v3/ | wc -l

# 应该看到 doc_{id}_key.pt 格式的文件
ls /mnt/data3/tmp/fusionrag_online_lazy/Qwen2.5-7B-Instruct/musique/kv_cache_v3/ | head -10
```

## 性能预期

### 存储空间节省

- **原方式**: 如果M个问题包含N个唯一文档，但总共引用了R次（R > N），会保存R份KV缓存
- **新方式**: 只保存N份KV缓存

**示例**: 假设200个问题，平均每个问题10个文档，但只有1000个唯一文档：
- 原方式: 2000份缓存
- 新方式: 1000份缓存
- **节省: 50%**

### 计算时间节省

第二次及后续运行时：
- 遇到已缓存文档时，无需重新计算KV
- Online lazy模式下，跨问题的文档复用显著加速

## 常见问题

### Q1: 如何处理文档池中不存在的文档？

在数据优化过程中，所有文档都应该匹配到文档池中的ID。如果遇到未匹配的文档，会保存为error标记：

```json
{"idx": -1, "error": "not_found_in_pool", "text_preview": "..."}
```

迁移时应跳过这些错误标记。

### Q2: 是否需要重新计算context_rank？

不需要。context_rank计算的是文档之间的相似度，与文档ID无关。新的`prepare_reflect_data_v3`会自动处理。

### Q3: 如何回退到旧版本？

1. 恢复使用 `result_reflect.json`
2. 使用原来的 `prepare_reflect_data` 函数
3. 恢复原来的KV缓存路径格式

### Q4: 新旧缓存能否共存？

可以。建议使用不同的缓存目录：
- 旧缓存: `/path/to/cache/kv_cache/`
- 新缓存: `/path/to/cache/kv_cache_v3/`

## 总结

本次迁移的核心是将KV缓存的组织方式从**按问题分组**改为**按全局文档ID索引**，实现了：

✅ 跨问题的文档KV缓存复用
✅ 减少存储空间占用
✅ 加速Online Lazy模式
✅ 更清晰的缓存管理逻辑

迁移后，系统将更高效地处理包含重复文档的多问题场景。
