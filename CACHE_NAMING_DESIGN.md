# FusionRAG Cache 命名设计文档

## 设计原则

Preprocess KV Cache 的命名必须包含**所有影响 cache 内容的参数**，确保不同配置使用独立的 cache，避免错误复用。

---

## 命名格式

```
preprocess_kv_cache_{scope}_topk{topk}_{recall_method}
```

### 参数说明

| 参数 | 含义 | 可能的值 | 为什么重要 |
|------|------|----------|-----------|
| **scope** | 文档检索范围 | `global`, `per_example`, `skip_untested` | 决定了从哪个文档池中召回相似文档 |
| **topk** | 融合文档数量 | `5`, `10`, `20`, etc. | 直接影响融合的 KV cache 大小和内容 |
| **recall_method** | 召回方法 | `bge`, `random`, `bm25`, `tfidf`, ... | 决定如何选择相似文档 |

---

## 命名示例

### 当前支持的配置

```bash
# BGE 相似度召回，global scope，topk=10
preprocess_kv_cache_global_topk10_bge/

# 随机召回，global scope，topk=10
preprocess_kv_cache_global_topk10_random/

# BGE 相似度召回，global scope，topk=5
preprocess_kv_cache_global_topk5_bge/

# 随机召回，per-example scope，topk=10
preprocess_kv_cache_per_example_topk10_random/
```

### 未来可能的扩展

```bash
# BM25 召回
preprocess_kv_cache_global_topk10_bm25/

# TF-IDF 召回
preprocess_kv_cache_global_topk10_tfidf/

# 其他 embedding 模型（如 E5, GTE）
preprocess_kv_cache_global_topk10_e5/
preprocess_kv_cache_global_topk10_gte/

# 混合召回（如 BM25 + BGE）
preprocess_kv_cache_global_topk10_hybrid_bm25_bge/
```

---

## 代码实现位置

**文件**: `/home/shm/document/exp/FusionRAG/test_fusionrag_reflect.py`

**函数**: `main()`

**位置**: Lines 1054-1072

**核心代码**:
```python
# 确定召回方法字符串
recall_method = "random" if use_random_recall else "bge"

# 确定 scope 字符串
if preprocess_scope == PreprocessScope.GLOBAL:
    scope_str = "global"
elif preprocess_scope == PreprocessScope.PER_EXAMPLE:
    scope_str = "per_example"
elif preprocess_scope == PreprocessScope.SKIP_UNTESTED:
    scope_str = "skip_untested"
else:
    scope_str = "default"

# 生成 cache 目录名
cache_dir_name = f"preprocess_kv_cache_{scope_str}_topk{topk}_{recall_method}"
preprocess_save_path = os.path.join(model_cache_root, cache_dir_name)
```

---

## 如何添加新的召回方法

### 步骤 1: 添加命令行参数（如果需要）

**文件**: `test_fusionrag_reflect.py`

**位置**: 命令行参数解析部分（约 line 2354+）

```python
# 示例：添加 BM25 召回方法
parser.add_argument('--recall_method', type=str, default='bge',
                    choices=['bge', 'random', 'bm25', 'tfidf'],
                    help='Document recall method')
```

### 步骤 2: 修改 recall_method 字符串生成逻辑

**位置**: Lines 1059

**修改前**:
```python
recall_method = "random" if use_random_recall else "bge"
```

**修改后**（示例）:
```python
# 支持更多召回方法
if args.recall_method == 'random':
    recall_method = "random"
elif args.recall_method == 'bm25':
    recall_method = "bm25"
elif args.recall_method == 'tfidf':
    recall_method = "tfidf"
else:
    recall_method = "bge"  # 默认
```

### 步骤 3: 实现新的召回逻辑

**位置**: `prepare_reflect_data()` 函数中的文档召回部分

**参考**: Lines 520-565（Random 召回实现）和 Lines 456-519（BGE 召回实现）

**示例**（BM25）:
```python
if recall_method == 'bm25':
    # ========== BM25 Recall Mode ==========
    print("\n" + "="*80)
    print(f"Using BM25 recall (scope: {preprocess_scope.value})...")
    print("="*80)

    from rank_bm25 import BM25Okapi

    # 分词所有文档
    tokenized_corpus = [doc.split() for doc in all_documents]
    bm25 = BM25Okapi(tokenized_corpus)

    # 为每个文档计算 BM25 相似度
    for q_idx, q_data in enumerate(questions_data):
        n_docs = len(q_data['docs'])
        if n_docs == 0:
            continue

        global_offset = sum(corpus_lens[:q_idx])
        q_context_rank = []

        for i in range(n_docs):
            current_doc = q_data['docs'][i]
            tokenized_query = current_doc.split()

            # 计算 BM25 分数
            scores = bm25.get_scores(tokenized_query)

            # 获取 top-k
            top_indices = np.argsort(scores)[::-1][:topk+1]

            # 过滤掉自身
            current_global_idx = global_offset + i
            top_indices = [idx for idx in top_indices if idx != current_global_idx][:topk]

            q_context_rank.append(top_indices)

        context_rank.append(np.array(q_context_rank))
```

### 步骤 4: 更新 Shell 脚本

**文件**: `run_fusionrag.sh`, `run_fusionrag_sweep.sh`

**添加新参数**:
```bash
# 召回方法配置
RECALL_METHOD="bm25"  # 可选: bge, random, bm25, tfidf

# 传递给 Python
"--recall_method" "${RECALL_METHOD}"
```

---

## 实验对比示例

### 对比不同召回方法

```bash
# 测试 BGE 召回
USE_RANDOM_RECALL="false"
TOPK="10"
./run_fusionrag_sweep.sh

# 测试随机召回
USE_RANDOM_RECALL="true"
TOPK="10"
./run_fusionrag_sweep.sh

# 结果会保存在不同的 cache 目录：
# preprocess_kv_cache_global_topk10_bge/
# preprocess_kv_cache_global_topk10_random/
```

### 对比不同 TopK 值

```bash
# TopK=5
TOPK="5"
USE_RANDOM_RECALL="false"
./run_fusionrag_sweep.sh

# TopK=10
TOPK="10"
USE_RANDOM_RECALL="false"
./run_fusionrag_sweep.sh

# TopK=20
TOPK="20"
USE_RANDOM_RECALL="false"
./run_fusionrag_sweep.sh

# 结果会保存在不同的 cache 目录：
# preprocess_kv_cache_global_topk5_bge/
# preprocess_kv_cache_global_topk10_bge/
# preprocess_kv_cache_global_topk20_bge/
```

---

## Cache 自动管理

### Cache 检查逻辑

代码会自动检查 cache 是否已存在：

```python
preprocess_key_path = f"{preprocess_save_path}/{example_id}_{chunk_id}_key.pt"

if os.path.exists(preprocess_key_path):
    continue  # 使用已有 cache
else:
    # 生成新 cache
    prefill_with_cache_and_save_preprocess(...)
```

### Cache 目录结构

```
/mnt/data3/tmp/fusionrag/
└── Qwen2.5-7B-Instruct/
    └── musique/
        ├── kv_cache/                                    # 文档 KV cache
        ├── preprocess_kv_cache_global_topk10_bge/      # BGE, topk=10
        ├── preprocess_kv_cache_global_topk10_random/   # Random, topk=10
        ├── preprocess_kv_cache_global_topk5_bge/       # BGE, topk=5
        ├── preprocess_kv_cache_global_topk20_bge/      # BGE, topk=20
        └── results/                                     # 实验结果
```

### Cache 清理

```bash
# 清理特定配置的 cache
rm -rf /mnt/data3/tmp/fusionrag/Qwen2.5-7B-Instruct/musique/preprocess_kv_cache_global_topk10_random/

# 清理所有 preprocess cache（保留文档 KV cache）
rm -rf /mnt/data3/tmp/fusionrag/Qwen2.5-7B-Instruct/musique/preprocess_kv_cache_*/

# 清理全部 cache
rm -rf /mnt/data3/tmp/fusionrag/Qwen2.5-7B-Instruct/musique/
```

---

## 验证方法

### 1. 检查 Cache 目录是否正确创建

运行实验后：
```bash
ls -lh /mnt/data3/tmp/fusionrag/Qwen2.5-7B-Instruct/musique/
```

应该看到对应配置的 cache 目录。

### 2. 查看启动日志

运行时会打印详细的 cache 配置：
```
Cache directories created under: /mnt/data3/tmp/fusionrag/Qwen2.5-7B-Instruct/musique
  - KV cache: .../kv_cache
  - Preprocess cache:
      Scope: global
      TopK: 10
      Recall: Random Recall
      Path: preprocess_kv_cache_global_topk10_random
  - Results: .../results
```

### 3. 确认不同配置产生不同结果

对比不同配置的实验结果：
```bash
# 比较 topk=5 vs topk=10
diff result/.../FusionRAG_global_topk_5_rate_0.15_revert_rope.csv \
     result/.../FusionRAG_global_topk_10_rate_0.15_revert_rope.csv

# 比较 random vs bge
diff result/.../FusionRAG_global_topk_10_rate_0.15_revert_rope.csv \
     result/preprocess_ramdom2/.../FusionRAG_global_topk_10_rate_0.15_revert_rope.csv
```

---

## 常见问题

### Q1: 修改 topk 后是否需要删除旧 cache？
**A**: 不需要。新的 topk 会使用新的 cache 目录（如 `topk5` vs `topk10`），不会冲突。

### Q2: 修改召回方法后是否需要删除旧 cache？
**A**: 不需要。新的召回方法会使用新的 cache 目录（如 `_bge` vs `_random`），不会冲突。

### Q3: 如何确认是否使用了正确的 cache？
**A**: 查看启动日志中的 "Preprocess cache" 部分，确认 Path 是否符合预期。

### Q4: 不同实验的结果文件会冲突吗？
**A**: 不会。结果文件名包含 `topk` 参数（如 `FusionRAG_global_topk_10_rate_0.1_revert_rope.csv`），不同 topk 的结果会保存在不同文件中。

### Q5: Cache 占用空间过大怎么办？
**A**: 可以定期清理不需要的配置的 cache：
```bash
# 只保留 topk=10 的 cache
rm -rf /mnt/data3/tmp/fusionrag/Qwen2.5-7B-Instruct/musique/preprocess_kv_cache_*topk5*/
rm -rf /mnt/data3/tmp/fusionrag/Qwen2.5-7B-Instruct/musique/preprocess_kv_cache_*topk20*/
```

---

## 设计优势

1. **防止 Cache 混用**: 不同配置自动使用独立 cache
2. **可扩展性强**: 轻松添加新的召回方法，只需修改字符串生成逻辑
3. **清晰可读**: 目录名直接说明配置（scope, topk, recall_method）
4. **向后兼容**: 可以保留旧 cache，不影响新实验
5. **实验可复现**: Cache 名称包含完整配置信息，便于追溯

---

## 未来可能的扩展方向

### 1. 添加 `revert_rope` 参数到 Cache 名称
如果 `revert_rope` 也影响 preprocess cache 内容：
```python
rope_str = "rope" if revert_rope else "norope"
cache_dir_name = f"preprocess_kv_cache_{scope_str}_topk{topk}_{recall_method}_{rope_str}"
```

### 2. 支持混合召回方法
```python
# 示例：70% BM25 + 30% BGE
recall_method = "hybrid_bm25_0.7_bge_0.3"
```

### 3. 添加 Random Seed 到 Cache 名称
如果需要测试不同随机种子：
```python
if recall_method == "random":
    recall_method = f"random_seed{random_seed}"
```

### 4. 使用 Hash 生成 Cache 名称（适合参数特别多的情况）
```python
import hashlib

config_str = f"{scope_str}_{topk}_{recall_method}_{revert_rope}_{random_seed}"
config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
cache_dir_name = f"preprocess_kv_cache_{config_hash}"

# 同时保存配置文件方便查询
with open(f"{preprocess_save_path}/config.json", 'w') as f:
    json.dump({
        'scope': scope_str,
        'topk': topk,
        'recall_method': recall_method,
        'revert_rope': revert_rope,
        'random_seed': random_seed
    }, f, indent=2)
```

---

**文档创建时间**: 2026-01-14
**最后更新**: 2026-01-14
