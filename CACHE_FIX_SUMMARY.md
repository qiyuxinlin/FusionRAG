# FusionRAG Cache Reuse 问题修复总结

## 问题诊断

### 原始问题
两个实验（random recall vs BGE similarity）产生了完全相同的结果：
- `preprocess_ramdom` 实验: 79/95 correct (83.16%)
- 正常路径实验: 前95行也是 79/95 correct (83.16%)
- 预测结果完全相同！

### 根本原因
**Cache 复用导致不同实验使用了相同的 preprocess 数据：**

1. 两个实验都使用了相同的 cache 目录：`/mnt/data3/tmp/fusionrag`
2. Preprocess cache 路径相同：`/mnt/data3/tmp/fusionrag/Qwen2.5-7B-Instruct/musique/preprocess_kv_cache_global/`
3. 代码逻辑：`if os.path.exists(preprocess_key_path): continue`
4. 第二个实验直接复用了第一个实验的 cache，无论 `use_random_recall` 如何设置

### Random 代码验证
✅ **Random 逻辑本身是正确的**
- `random.sample()` 和 `random.choices()` 确实产生随机选择
- 问题不在 random 实现，而在 cache 复用机制

---

## 解决方案

### 代码修改

**文件**: `/home/shm/document/exp/FusionRAG/test_fusionrag_reflect.py`

**修改位置**: Lines 1054-1072

**修改内容**:
```python
# 修改前（简单方案）：
if preprocess_scope == PreprocessScope.GLOBAL:
    preprocess_save_path = os.path.join(model_cache_root, 'preprocess_kv_cache_global')

# 修改后（智能命名方案）：
# Cache naming includes all parameters that affect KV cache content:
# - scope: global/per_example/skip_untested
# - topk: number of documents to fuse
# - recall_method: random/bge (extensible for future methods)
recall_method = "random" if use_random_recall else "bge"
scope_str = "global"  # or per_example, skip_untested

cache_dir_name = f"preprocess_kv_cache_{scope_str}_topk{topk}_{recall_method}"
preprocess_save_path = os.path.join(model_cache_root, cache_dir_name)
```

**效果**:
- Random recall, topk=10: `preprocess_kv_cache_global_topk10_random/`
- BGE similarity, topk=10: `preprocess_kv_cache_global_topk10_bge/`
- BGE similarity, topk=5: `preprocess_kv_cache_global_topk5_bge/`
- **完全独立的 cache**，确保真正独立的实验
- **支持未来扩展**：可轻松添加新的召回方法

---

## 后续实验步骤

### 选项 1: 重新运行 Random 实验（推荐）

由于现有的实验结果实际上都是使用 BGE similarity 的结果，建议：

1. **确认现有 cache**:
```bash
ls -lh /mnt/data3/tmp/fusionrag/Qwen2.5-7B-Instruct/musique/
```

2. **运行新的 Random 实验**:
```bash
cd /home/shm/document/exp/FusionRAG
./run_fusionrag_sweep.sh
```

现在代码会自动创建新的 `preprocess_kv_cache_global_random/` 目录，不会复用旧 cache。

3. **监控输出**:
查看控制台输出，确认使用了正确的 cache 路径：
```
Preprocess cache (global, Random Recall): .../preprocess_kv_cache_global_random
```

### 选项 2: 同时测试两种方法

如果想对比 Random vs BGE：

**测试 BGE (已修改的代码)**:
```bash
# 修改 run_fusionrag_sweep.sh:
USE_RANDOM_RECALL="false"
RATE_LIST=(0.1 0.15 0.3)

# 运行
./run_fusionrag_sweep.sh
```

**测试 Random**:
```bash
# 修改 run_fusionrag_sweep.sh:
USE_RANDOM_RECALL="true"
RATE_LIST=(0.1 0.15 0.3)

# 运行
./run_fusionrag_sweep.sh
```

### 选项 3: 清理旧 Cache（如果需要）

如果想完全重新开始：
```bash
# ⚠️ 注意：这会删除所有已生成的 cache，需要重新生成
rm -rf /mnt/data3/tmp/fusionrag/Qwen2.5-7B-Instruct/musique/preprocess_kv_cache_global/

# 然后重新运行实验
```

---

## 验证方法

### 1. 检查 Cache 目录
```bash
# 运行实验后，应该看到不同配置的独立目录：
ls /mnt/data3/tmp/fusionrag/Qwen2.5-7B-Instruct/musique/

# 输出示例：
# preprocess_kv_cache_global_topk10_bge/      （BGE 相似度召回，topk=10）
# preprocess_kv_cache_global_topk10_random/   （随机召回，topk=10）
# preprocess_kv_cache_global_topk5_bge/       （BGE 相似度召回，topk=5）
```

### 2. 比较实验结果
运行完两组实验后，对比结果文件：
```bash
# BGE 结果
result/Qwen2.5-7B-Instruct/musique/results/FusionRAG_global_topk_10_rate_0.15_revert_rope.csv

# Random 结果
result/preprocess_ramdom2/Qwen2.5-7B-Instruct/musique/results/FusionRAG_global_topk_10_rate_0.15_revert_rope.csv
```

现在这两个文件应该有显著差异（不再是相同的准确率）。

### 3. 监控日志输出
启动时应该看到详细的 cache 配置信息：
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
和召回方法信息：
```
========================================
Using RANDOM recall (seed=42, scope: global)...
========================================
```
或
```
========================================
Using BGE model to compute document similarities...
========================================
```

---

## 预期结果

修复后，Random 和 BGE 两种方法应该产生**不同**的结果：

- **BGE Similarity**: 基于语义相似度选择文档，理论上应该更准确
- **Random Recall**: 随机选择文档，作为消融实验的 baseline

如果结果仍然相同，可能说明文档选择对最终准确率影响不大（这本身也是有价值的发现）。

---

## 技术细节

### Cache 路径命名规则
```python
# Format: {scope}_topk{topk}_{recall_method}
preprocess_kv_cache_{scope}_topk{topk}_{recall_method}

# 示例：
preprocess_kv_cache_global_topk10_random      # Global scope + TopK=10 + Random
preprocess_kv_cache_global_topk10_bge         # Global scope + TopK=10 + BGE
preprocess_kv_cache_global_topk5_bge          # Global scope + TopK=5 + BGE
preprocess_kv_cache_per_example_topk10_random # Per-example + TopK=10 + Random
```

**命名包含的参数**：
1. **scope**: 文档检索范围 (global/per_example/skip_untested)
2. **topk**: 融合的文档数量（直接影响 KV cache 内容）
3. **recall_method**: 召回方法 (bge/random，未来可扩展)

这确保了所有影响 preprocess cache 内容的参数都体现在路径名中。

### 为什么需要独立 Cache
1. **Preprocess 阶段**会将 system prompt + 多个相似文档融合生成 KV Cache
2. **相似文档的选择**（random vs BGE）会影响融合后的 KV Cache 内容
3. **TopK 参数**决定融合多少个文档，直接影响 KV Cache 的大小和内容
4. **不同的配置**必须使用不同的 preprocess cache，否则会导致错误的实验结果

### Cache 检查逻辑
```python
# 代码会检查 cache 是否已存在
if os.path.exists(preprocess_key_path):
    continue  # 跳过已有 cache，直接加载

# 修复后，不同方法的 cache 路径不同，不会相互干扰
```

---

## 问题回顾

**用户原问题**: "你检查一下现在代码里面random是真的有random吗"

**答案**:
✅ Random 代码本身正确
❌ 但由于 cache 复用，实际实验中未真正使用 random
✅ 现已修复，确保 random 和 BGE 使用独立 cache

---

## 下一步建议

1. **立即执行**: 重新运行 random 实验，使用修复后的代码
2. **对比分析**: 比较 Random vs BGE 的准确率差异
3. **消融实验**: 分析文档选择策略对 FusionRAG 效果的影响
4. **论文结果**: 可以用这个对比作为消融实验的重要数据

---

---

## 修改总结

### 改进的优势
1. ✅ **完全避免 cache 混用**：不同配置自动使用独立 cache
2. ✅ **支持 TopK 实验**：不同 topk 值产生独立 cache，无需手动清理
3. ✅ **易于扩展**：未来添加新召回方法（BM25、TF-IDF 等）只需修改少量代码
4. ✅ **实验可复现**：cache 目录名完整描述配置，便于追溯和对比
5. ✅ **清晰可读**：目录名直接说明 scope、topk、recall_method

### 相关文档
- **详细设计文档**: `CACHE_NAMING_DESIGN.md` - 包含命名规范、扩展指南、示例代码
- **修复总结**: 本文档 `CACHE_FIX_SUMMARY.md`

---

修复完成时间: 2026-01-14
