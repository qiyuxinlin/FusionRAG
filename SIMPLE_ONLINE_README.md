# Simple Online RAG - 惰性KV Cache生成

## 核心思想

**取消Offline预处理阶段，改为在线惰性生成(Lazy Initialization)**

```
传统Offline模式：
  预处理阶段: 为所有文档生成KV → 保存
  推理阶段:   加载预生成的KV → 推理

Simple Online模式：
  推理阶段:   检查文档KV是否存在
              ├─ 存在 → 加载
              └─ 不存在 → 生成并保存 → 加载
```

---

## 工作流程

### 单次查询流程

```
Query → 召回文档[doc_i, doc_i+1, ..., doc_i+n]
         ↓
    遍历每个文档
         ├─ 检查: {cache_dir}/{doc_hash}_key.pt 是否存在？
         │
         ├─ YES (Cache Hit)
         │   └─→ 直接加载 key_cache, value_cache
         │
         └─ NO (Cache Miss)
             └─→ 生成 KV → 保存到磁盘 → 返回 KV
         ↓
    收集所有 KV → 组合 → 推理生成答案
```

### 核心代码逻辑

```python
def load_or_generate_kv(doc_text, cache_dir, model, tokenizer, ...):
    # 1. 生成文档唯一ID (MD5哈希)
    doc_id = hashlib.md5(doc_text.encode()).hexdigest()
    kv_path = f"{cache_dir}/{doc_id}_key.pt"

    # 2. 检查缓存
    if os.path.exists(kv_path):
        # Cache Hit - 加载
        return torch.load(kv_path)

    # 3. Cache Miss - 生成
    kv = model.forward(doc_text)  # 完整前向传播

    # 4. 保存到磁盘
    torch.save(kv, kv_path)

    return kv
```

---

## 快速使用

### 运行测试

```bash
cd /home/shm/document/exp/FusionRAG

# 测试20个样本，每次召回10个文档
bash run_simple_online.sh 20 10

# 输出示例：
# Sample 1/20
# [Step 1] Retrieving documents...
#   Retrieved 10 documents: [42, 87, 123, ...]
# [Step 2] Loading/Generating KV Caches:
#   ✗ Cache MISS: a3f2b8c1... - Generating... Done (0.18s)
#   ✗ Cache MISS: d4e9c7a2... - Generating... Done (0.15s)
#   ✓ Cache HIT: f1b3d8e4...
#   ...
# [Step 3] Summary:
#   Cache hits: 3/10
#   Cache misses: 7/10
#   Hit rate: 30.0%
```

### 第二次运行（高命中率）

```bash
# 再次运行相同的测试
bash run_simple_online.sh 20 10

# 输出示例：
# Sample 1/20
# [Step 2] Loading/Generating KV Caches:
#   ✓ Cache HIT: a3f2b8c1...
#   ✓ Cache HIT: d4e9c7a2...
#   ✓ Cache HIT: f1b3d8e4...
#   ...
# [Step 3] Summary:
#   Cache hits: 10/10  ← 全部命中！
#   Hit rate: 100.0%
```

---

## 性能特点

### 第一次运行（冷启动）

```
样本1: Cache hits 0/10  (0%)   - 耗时 ~2.0s
样本2: Cache hits 2/10  (20%)  - 耗时 ~1.6s
样本3: Cache hits 5/10  (50%)  - 耗时 ~1.0s
...
样本20: Cache hits 9/10 (90%)  - 耗时 ~0.4s

平均命中率: ~50%
平均耗时: ~1.0s/query
```

### 第二次运行（热启动）

```
样本1: Cache hits 10/10 (100%) - 耗时 ~0.3s
样本2: Cache hits 10/10 (100%) - 耗时 ~0.3s
...
样本20: Cache hits 10/10 (100%) - 耗时 ~0.3s

平均命中率: ~95%
平均耗时: ~0.3s/query
```

---

## 存储管理

### 缓存文件结构

```
/mnt/data3/tmp/simple_kv_cache/
├── a3f2b8c1f9e7d4a2_key.pt      # 文档1的Key cache
├── a3f2b8c1f9e7d4a2_value.pt    # 文档1的Value cache
├── d4e9c7a2b8f1e3d5_key.pt      # 文档2的Key cache
├── d4e9c7a2b8f1e3d5_value.pt    # 文档2的Value cache
└── ...
```

### 文档ID生成规则

```python
doc_id = MD5(doc_text)

# 示例：
doc_text = "The Eiffel Tower is in Paris."
doc_id = "a3f2b8c1f9e7d4a2..."  # MD5哈希（32字符）
```

**特点**：
- ✅ 相同文档内容 → 相同ID → 自动复用缓存
- ✅ 不同文档内容 → 不同ID → 独立缓存
- ✅ 跨数据集通用（只要文档内容相同）

### 缓存大小估算

```
单个文档KV Cache大小（Qwen2.5-7B-Instruct）:
  - 文档长度: 500 tokens
  - Num layers: 28
  - Num heads: 28
  - Head dim: 128

  Key cache: 28 × 28 × 500 × 128 × 2 bytes (fp16) ≈ 100 MB
  Value cache: 同上 ≈ 100 MB

  总计: ~200 MB/文档
```

**实际占用**：
- 200个唯一文档 → ~40 GB
- 1000个唯一文档 → ~200 GB

💡 **建议**：使用大容量SSD存储缓存

---

## 与Offline模式对比

| 特性 | Offline模式 | Simple Online模式 |
|------|------------|------------------|
| **预处理时间** | 需要（10-30分钟） | 不需要（0分钟） ✅ |
| **首次查询速度** | 快（已预生成） | 慢（按需生成） |
| **后续查询速度** | 快 | 快（已缓存） ✅ |
| **存储占用** | 全量（所有文档） | 按需（仅用过的） ✅ |
| **灵活性** | 低（需重新预处理） | 高（自动适应） ✅ |
| **内存管理** | 需要 | 不需要 ✅ |

### 适用场景

**Offline模式**：
- ✅ 固定数据集评测
- ✅ 重复运行相同实验
- ✅ 文档池不变

**Simple Online模式**：
- ✅ 动态文档池
- ✅ 新文档频繁加入
- ✅ 不想预处理
- ✅ 真实RAG应用

---

## 代码结构

### 核心函数

#### 1. `get_doc_id(doc_text)`
生成文档唯一ID

```python
doc_id = get_doc_id("The Eiffel Tower is in Paris.")
# Returns: "a3f2b8c1f9e7d4a2..."
```

#### 2. `load_or_generate_kv(doc_text, cache_dir, model, ...)`
检查缓存并加载或生成KV

```python
key, value, tokens = load_or_generate_kv(
    doc_text="Some document...",
    cache_dir="/tmp/cache",
    model=model,
    tokenizer=tokenizer,
    system_tokens=system_tokens,
    verbose=True
)
# 自动处理：检查 → 加载或生成 → 返回
```

#### 3. `retrieve_documents_bge(query, document_pool, bge_model, topk)`
BGE文档召回

```python
indices = retrieve_documents_bge(
    query="Where is Paris?",
    document_pool=all_docs,
    bge_model=bge_model,
    topk=10
)
# Returns: [42, 87, 123, ...]
```

#### 4. `process_query_simple(query, document_pool, cache_dir, ...)`
完整的查询处理流程

```python
query_tokens, key_list, value_list, doc_tokens = process_query_simple(
    query="What is machine learning?",
    document_pool=docs,
    cache_dir="/tmp/cache",
    model=model,
    tokenizer=tokenizer,
    bge_model=bge_model,
    system_tokens=system_tokens,
    topk=10,
    verbose=True
)
```

---

## 集成到现有代码

### 替换现有的offline预处理

```python
# 原来的代码（offline）:
# 1. 预处理阶段
for doc in all_documents:
    generate_and_save_kv(doc, preprocess_cache_dir)

# 2. 推理阶段
for query in queries:
    doc_indices = retrieve(query, topk=10)
    kvs = [load_kv(idx, preprocess_cache_dir) for idx in doc_indices]
    answer = generate(query, kvs)

# ========================================

# 新代码（simple online）:
# 只需要推理阶段
for query in queries:
    doc_indices = retrieve(query, topk=10)
    kvs = [
        load_or_generate_kv(docs[idx], cache_dir, model, tokenizer)
        for idx in doc_indices
    ]
    answer = generate(query, kvs)
```

### 修改 test_fusionrag_reflect.py

在 `test_fusionrag_reflect.py` 中集成：

```python
# 在主循环中，替换KV加载部分
for q_idx, q_data in enumerate(questions_data):
    retrieved_indices = context_rank[q_idx]  # 召回的文档索引

    # 使用 Simple Online 模式
    for doc_idx in retrieved_indices:
        doc_text = all_documents[doc_idx]

        # 自动检查缓存并加载或生成
        key_cache, value_cache, doc_tokens = load_or_generate_kv(
            doc_text=doc_text,
            cache_dir=simple_cache_dir,
            model=model,
            tokenizer=tokenizer,
            system_tokens=system_tensor,
            verbose=False
        )

        # 后续使用 key_cache, value_cache 进行推理
        ...
```

---

## 高级用法

### 1. 清空缓存

```bash
# 清空所有缓存，重新开始
rm -rf /mnt/data3/tmp/simple_kv_cache/*

# 或者只清空特定文档
rm /mnt/data3/tmp/simple_kv_cache/a3f2b8c1*
```

### 2. 预热常用文档

```python
# 如果知道哪些文档会频繁使用，可以预先生成
frequent_docs = [doc1, doc2, doc3, ...]

for doc in frequent_docs:
    load_or_generate_kv(
        doc_text=doc,
        cache_dir=cache_dir,
        model=model,
        tokenizer=tokenizer,
        system_tokens=system_tokens,
        verbose=True
    )
```

### 3. 查看缓存统计

```bash
cd /mnt/data3/tmp/simple_kv_cache

# 缓存的文档数
ls *_key.pt | wc -l

# 总大小
du -sh .

# 每个文件大小
du -h *_key.pt | head -10
```

---

## 故障排查

### 问题1: 磁盘空间不足

**症状**: `OSError: [Errno 28] No space left on device`

**解决**:
```bash
# 1. 检查磁盘空间
df -h /mnt/data3

# 2. 清理旧缓存
rm -rf /mnt/data3/tmp/simple_kv_cache/*

# 3. 或者使用其他磁盘
bash run_simple_online.sh 20 10  # 修改脚本中的 CACHE_DIR
```

### 问题2: 缓存文件损坏

**症状**: `RuntimeError: PytorchStreamReader failed reading zip archive`

**解决**:
```bash
# 删除损坏的文件
rm /mnt/data3/tmp/simple_kv_cache/xxx_key.pt
rm /mnt/data3/tmp/simple_kv_cache/xxx_value.pt

# 重新运行（会自动重新生成）
bash run_simple_online.sh 20 10
```

### 问题3: BGE模型加载失败

**症状**: `OSError: Can't load model for 'bge-m3'`

**解决**:
```bash
# 检查BGE模型路径
ls /mnt/data/models/bge-m3-FP16

# 或者不使用BGE（使用简单召回）
# 修改代码中的 retrieve_documents_bge 函数
```

---

## 总结

### 核心优势

✅ **零预处理时间** - 不需要offline阶段
✅ **按需存储** - 只缓存实际用过的文档
✅ **自动管理** - 无需LRU、淘汰等复杂逻辑
✅ **简单直观** - 代码量少，易于理解和修改
✅ **跨数据集复用** - 相同文档在不同数据集间共享缓存

### 与复杂Online系统的区别

| 特性 | Simple Online | 复杂Online (LRU) |
|------|--------------|-----------------|
| **内存管理** | 无（纯磁盘） | 有（内存+磁盘） |
| **缓存淘汰** | 无 | LRU淘汰策略 |
| **代码复杂度** | 低（~200行） | 高（~600行） |
| **存储限制** | 仅受磁盘限制 | 受内存限制 |
| **适用场景** | 大磁盘可用 | 内存受限 |

**推荐使用Simple Online**，除非：
- 磁盘空间非常有限（< 100GB）
- 需要极致的查询速度优化

---

## License

MIT License
