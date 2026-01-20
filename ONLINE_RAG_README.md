# Online RAG System 使用指南

## 概述

这是一个**动态在线KV Cache管理系统**，实现了真实RAG应用场景的推理流程。

### 核心特性

✅ **动态文档召回** - 根据Query实时召回相关文档
✅ **智能缓存管理** - 自动检测KV Cache是否存在
✅ **LRU内存管理** - 自动淘汰最少使用的缓存
✅ **磁盘持久化** - 支持缓存溢出到磁盘
✅ **高缓存命中率** - 相同文档无需重复计算
✅ **统计信息** - 实时监控缓存性能

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                     Online RAG Pipeline                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────┐
              │   1. Document Retrieval   │
              │   (BGE + FAISS)           │
              └───────────┬───────────────┘
                          │
                          ▼
              ┌───────────────────────────┐
              │   2. KV Cache Manager     │
              │   Check Memory/Disk       │
              └───────────┬───────────────┘
                          │
                ┌─────────┴─────────┐
                │                   │
                ▼                   ▼
      ┌─────────────────┐ ┌─────────────────┐
      │  Cache HIT      │ │  Cache MISS     │
      │  → Load KV      │ │  → Generate KV  │
      │  → Fast         │ │  → Save to Store│
      └────────┬────────┘ └────────┬────────┘
               │                   │
               └─────────┬─────────┘
                         │
                         ▼
              ┌───────────────────────────┐
              │   3. Answer Generation    │
              │   (Use Cached KV)         │
              └───────────────────────────┘
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install torch transformers faiss-gpu FlagEmbedding
```

### 2. 基本使用

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from online_kv_cache_manager import OnlineKVCacheManager, KVCacheStore
import torch

# 加载模型
model_path = "/path/to/Qwen2.5-7B-Instruct"
config = AutoConfig.from_pretrained(model_path)
config.torch_dtype = torch.float16

model = AutoModelForCausalLM.from_pretrained(
    model_path, config=config, torch_dtype=torch.float16, device_map='auto'
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 创建KV Cache存储（内存 + 磁盘）
kv_store = KVCacheStore(
    max_memory_gb=5.0,              # 最大内存5GB
    disk_cache_dir="/tmp/kv_cache", # 磁盘缓存路径
    enable_lru=True                 # 启用LRU淘汰
)

# 创建在线管理器
manager = OnlineKVCacheManager(
    model=model,
    tokenizer=tokenizer,
    bge_model_path="/path/to/bge-m3",
    kv_store=kv_store,
    device="cuda:0"
)

# 准备文档池
documents = [
    "The Eiffel Tower is in Paris, France.",
    "Python is a programming language.",
    "Tokyo is the capital of Japan.",
    # ... more documents
]

# 处理查询
query = "Where is the Eiffel Tower?"
query_tokens, key_list, value_list, doc_tokens = manager.process_query(
    query=query,
    document_pool=documents,
    topk=3,
    verbose=True  # 显示详细信息
)

# 查看缓存统计
stats = kv_store.get_stats()
print(f"Cache hit rate: {stats['hit_rate']:.2%}")
print(f"Memory usage: {stats['memory_usage_gb']:.2f} GB")
```

### 3. 在Musique数据集上测试

```bash
cd /home/shm/document/exp/FusionRAG

# 测试20个样本
python test_online_rag.py \
    --model_path /mnt/data/models/Qwen2.5-7B-Instruct \
    --data_path ./data/result_reflect.json \
    --bge_model_path /mnt/data/models/bge-m3-FP16 \
    --topk 10 \
    --max_samples 20 \
    --max_memory_gb 8.0 \
    --disk_cache_dir /mnt/data3/tmp/online_kv_cache
```

---

## 核心组件详解

### 1. KVCacheStore - 缓存存储管理器

负责KV Cache的存储、检索和淘汰。

#### 主要功能

- **内存缓存** - 使用OrderedDict实现LRU
- **磁盘持久化** - 内存溢出时自动保存到磁盘
- **快速查询** - 基于文档内容的MD5哈希

#### API

```python
# 创建存储
kv_store = KVCacheStore(
    max_memory_gb=10.0,              # 最大内存
    disk_cache_dir="/path/to/cache", # 磁盘路径（None=仅内存）
    enable_lru=True                  # LRU淘汰策略
)

# 获取缓存（自动查内存→磁盘）
entry = kv_store.get(doc_text)
if entry:
    print(f"Cache hit! Hit count: {entry.hit_count}")
    key_cache = entry.key_cache
    value_cache = entry.value_cache

# 存储缓存
kv_store.put(
    doc_text=document,
    key_cache=key_tensor,
    value_cache=value_tensor,
    doc_tokens=tokens
)

# 统计信息
stats = kv_store.get_stats()
# {
#   'hits': 150,
#   'misses': 50,
#   'hit_rate': 0.75,
#   'evictions': 10,
#   'memory_usage_gb': 4.2,
#   'num_cached': 100
# }
```

### 2. OnlineKVCacheManager - 在线管理器

整合文档召回、缓存管理和答案生成。

#### 主要方法

##### `retrieve_documents(query, document_pool, topk)`
根据Query召回相关文档

```python
indices = manager.retrieve_documents(
    query="Where is Paris?",
    document_pool=all_documents,
    topk=5
)
# Returns: [3, 7, 12, 45, 89]  # 文档索引
```

##### `process_query(query, document_pool, topk, verbose)`
完整的查询处理流程

```python
query_tokens, key_list, value_list, doc_tokens = manager.process_query(
    query="What is machine learning?",
    document_pool=documents,
    topk=10,
    use_fusion=False,    # 是否使用FusionRAG融合
    revert_rope=True,    # 是否revert RoPE
    verbose=True         # 打印详细日志
)

# 输出示例：
# [Step 1] Retrieved 10 documents:
#   1. Doc 42: Machine learning is a subset of AI...
#   2. Doc 87: Neural networks are used in ML...
#   ...
# [Step 2] Processing KV Caches:
#   ✓ Doc 1: Cache HIT (hit_count=3)
#   ✗ Doc 2: Cache MISS - Generating... Done (0.15s)
#   ✓ Doc 3: Cache HIT (hit_count=1)
#   ...
# [Step 3] Summary:
#   Cache hits: 7/10
#   Cache misses: 3/10
#   Total time: 0.52s
```

##### `generate_answer(query_tokens, key_list, value_list)`
使用缓存的KV生成答案

```python
answer = manager.generate_answer(
    query_tokens=query_tokens,
    key_cache_list=key_list,
    value_cache_list=value_list,
    max_new_tokens=100,
    temperature=0.7
)
```

---

## 性能优化

### 缓存命中率优化

1. **增大内存限制**
   ```python
   kv_store = KVCacheStore(max_memory_gb=20.0)  # 更多内存
   ```

2. **启用磁盘缓存**
   ```python
   kv_store = KVCacheStore(
       max_memory_gb=5.0,
       disk_cache_dir="/fast/ssd/cache"  # 使用SSD
   )
   ```

3. **预热缓存**
   ```python
   # 预先为常见文档生成KV
   for doc in frequent_documents:
       manager._generate_kv_cache(doc, save_to_store=True)
   ```

### 内存管理

**监控内存使用**
```python
stats = kv_store.get_stats()
if stats['memory_usage_gb'] > 8.0:
    print("Warning: High memory usage!")
```

**手动清理**
```python
kv_store.clear()  # 清空所有缓存
```

**调整LRU策略**
```python
# 禁用LRU（先进先出）
kv_store = KVCacheStore(enable_lru=False)
```

---

## 与Offline模式对比

| 特性 | Offline模式 | Online模式 |
|------|------------|-----------|
| **预处理** | 预先生成所有KV | 按需生成 |
| **首次查询** | 快（已缓存） | 慢（需生成） |
| **后续查询** | 快（已缓存） | 快（已缓存） |
| **内存占用** | 高（所有文档） | 低（仅活跃文档） |
| **磁盘占用** | 高（所有KV） | 中（淘汰策略） |
| **灵活性** | 低（固定文档） | 高（动态召回） |
| **适用场景** | 固定数据集评测 | 真实RAG应用 |

---

## 高级用法

### 1. 自定义召回策略

```python
class CustomRetriever(OnlineKVCacheManager):
    def retrieve_documents(self, query, document_pool, topk):
        # 实现自定义召回逻辑
        # 例如：混合BM25 + BGE
        bm25_scores = self.compute_bm25(query, document_pool)
        bge_scores = self.compute_bge_similarity(query, document_pool)

        combined_scores = 0.5 * bm25_scores + 0.5 * bge_scores
        top_indices = combined_scores.argsort()[-topk:]

        return top_indices.tolist()
```

### 2. 集成在线融合

```python
# TODO: 实现在线融合逻辑
query_tokens, key_list, value_list, doc_tokens = manager.process_query(
    query=query,
    document_pool=documents,
    topk=10,
    use_fusion=True,      # 启用FusionRAG融合
    revert_rope=True
)
```

### 3. 批量查询优化

```python
queries = ["Query 1", "Query 2", "Query 3", ...]

for query in queries:
    # 第一次查询会生成KV
    # 后续查询如果召回相同文档，会命中缓存
    results = manager.process_query(query, documents, topk=5)

    # 监控缓存命中率
    stats = kv_store.get_stats()
    print(f"Current hit rate: {stats['hit_rate']:.2%}")
```

---

## 实验对比

### 场景1: 固定文档池（200个文档）

```bash
# 测试50个查询
python test_online_rag.py \
    --max_samples 50 \
    --topk 10 \
    --max_memory_gb 5.0
```

**预期结果：**
- 前10个查询：缓存命中率 ~30%（部分文档重复）
- 后40个查询：缓存命中率 ~80%（大部分文档已缓存）
- 平均查询时间：首次 ~2s，后续 ~0.3s

### 场景2: 大规模文档池（10000个文档）

```bash
python test_online_rag.py \
    --max_samples 100 \
    --topk 10 \
    --max_memory_gb 10.0 \
    --disk_cache_dir /fast/ssd/cache
```

**预期结果：**
- 缓存命中率：50-70%（取决于文档重复度）
- 内存淘汰：~500次（LRU自动管理）
- 磁盘读写：~200次

---

## 故障排查

### 问题1: 内存溢出

**症状**：`RuntimeError: CUDA out of memory`

**解决方案**：
```python
# 1. 减小内存限制
kv_store = KVCacheStore(max_memory_gb=3.0)

# 2. 启用磁盘缓存
kv_store = KVCacheStore(
    max_memory_gb=3.0,
    disk_cache_dir="/tmp/cache"
)

# 3. 减小topk
manager.process_query(query, documents, topk=5)  # 原来是10
```

### 问题2: 缓存命中率低

**症状**：`hit_rate < 30%`

**原因分析**：
- 文档重复度低
- 内存限制太小导致频繁淘汰

**解决方案**：
```python
# 1. 增大内存
kv_store = KVCacheStore(max_memory_gb=15.0)

# 2. 检查文档重复度
doc_hashes = [kv_store.get_doc_id(doc) for doc in documents]
unique_rate = len(set(doc_hashes)) / len(doc_hashes)
print(f"Unique doc rate: {unique_rate:.2%}")

# 3. 预热常见文档
for doc in frequent_docs:
    manager._generate_kv_cache(doc)
```

### 问题3: 磁盘I/O慢

**症状**：`disk_loads` 时间长

**解决方案**：
```python
# 使用SSD或内存盘
kv_store = KVCacheStore(
    max_memory_gb=5.0,
    disk_cache_dir="/dev/shm/kv_cache"  # RAM disk
)
```

---

## TODO / 未来改进

- [ ] 实现在线融合逻辑
- [ ] 支持分布式缓存（Redis）
- [ ] 添加缓存预热策略
- [ ] 优化磁盘I/O（异步写入）
- [ ] 支持增量更新（文档变更时）
- [ ] 添加缓存压缩（减小存储）
- [ ] 实现多级缓存（L1内存 + L2磁盘 + L3远程）

---

## 引用

如果使用此代码，请引用：

```bibtex
@software{online_fusionrag_2024,
  title={Online KV Cache Management for Dynamic RAG Systems},
  author={Your Name},
  year={2024},
  url={https://github.com/...}
}
```

---

## License

MIT License
