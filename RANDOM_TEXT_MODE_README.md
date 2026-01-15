# RANDOM_TEXT召回模式使用说明

## 功能概述

**RANDOM_TEXT模式**是一个新的召回方法，用于验证"位置适应假设"（Position Adaptation Hypothesis）。

### 实验设计思路

- **召回阶段**：使用BGE相似度召回topk个文档，获取这些文档的token长度
- **融合阶段**：不使用被召回文档的KV cache，而是使用**等长度的无关随机文本**的KV cache进行融合
- **目的**：如果性能提升主要来自位置适应而非语义相关性，那么使用无关文本的KV cache也应该能获得类似的提升

### 与其他召回方法的对比

| 召回方法 | 召回逻辑 | 融合内容 | 验证目的 |
|---------|---------|---------|---------|
| **BGE** | BGE相似度 | 召回文档KV | 语义相似度假设 |
| **Random** | 随机召回 | 召回文档KV | 排除语义相关性 |
| **Repeat_Self** | 重复自己 | 自己的KV×K次 | 纯位置扩展效应 |
| **RANDOM_TEXT** | BGE相似度（仅获取长度） | **无关文本KV** | 位置适应假设 |

## 文件说明

### 1. 无关文本库
- **路径**: `./data/random_text_library.json`
- **内容**: 20个不同主题的文本（科学、历史、文学、技术、自然等）
- **长度范围**: 199-893 characters (约50-250 tokens)
- **生成脚本**: `generate_random_text_corpus.py`

### 2. KV Cache存储
- **原始文档KV**: `/mnt/data3/tmp/fusionrag/{model}/{dataset}/kv_cache/`
- **随机文本KV**: `/mnt/data3/tmp/fusionrag/{model}/{dataset}/kv_cache/random_texts/`
  - 格式: `text_text_{id}_key.pt` / `text_text_{id}_value.pt`
- **预处理KV**: `/mnt/data3/tmp/fusionrag/{model}/{dataset}/preprocess_kv_cache_global_topk{K}_random_text/`

### 3. 结果目录
- **路径**: `result/{model}/{dataset}/FusionRAG_global_topk{K}_random_text/`

## 使用方法

### 方法1: 修改启动脚本运行

编辑 `run_fusionrag.sh`:
```bash
RECALL_METHOD="random_text"  # 修改这一行
```

运行:
```bash
bash run_fusionrag.sh
```

### 方法2: 直接运行Python脚本

```bash
/home/shm/anaconda3/envs/fusionrag/bin/python test_fusionrag_reflect.py \
    --model_path /mnt/data/models/Qwen2.5-7B-Instruct \
    --data_path ./data/result_reflect.json \
    --cache_path /mnt/data3/tmp/fusionrag \
    --result_path ./result \
    --rate 0.15 \
    --topk 10 \
    --preprocess \
    --recall_method random_text \
    --preprocess_scope global \
    --revert_rope \
    --device cuda:0
```

### 方法3: TopK扫描实验

编辑 `run_fusionrag_topk_sweep.sh`:
```bash
RECALL_METHOD="random_text"  # 修改这一行
```

运行:
```bash
bash run_fusionrag_topk_sweep.sh
```

## 实现细节

### 1. 长度匹配机制
```python
# 对于每个被BGE召回的文档:
recalled_doc_len = questions_data[corpus_i]['doc_tensors'][c_id].shape[0]

# 从random text库中选择长度最接近的文本:
best_random_idx = min(range(len(random_text_lengths)),
                     key=lambda i: abs(random_text_lengths[i] - recalled_doc_len))
```

### 2. 索引编码
- **正常文档**: 正数索引 (0, 1, 2, ...)
- **Random Text**: 负数索引 `-(text_id + 2)`
  - 例如: text_0 → -2, text_1 → -3, text_2 → -4
  - -1保留用于padding

### 3. KV Cache加载流程
1. 加载system KV
2. **遍历topk召回**:
   - 如果是负数索引 < -1: 加载random text KV
   - 如果是正数索引: 加载文档KV (正常逻辑)
3. 加载当前文档KV
4. 进行FusionRAG预处理

## 预期实验结果

### 假设1: 位置适应占主导
如果**RANDOM_TEXT的性能 ≈ BGE的性能**，说明:
- 性能提升主要来自位置适应，而非语义相关性
- 只要长度匹配，内容无关的KV也能提供类似的收益

### 假设2: 语义相关性重要
如果**RANDOM_TEXT的性能 < BGE的性能**，说明:
- 语义相关性确实重要
- BGE召回的相关文档提供了额外的语义信息

### 假设3: 性能比较
预期性能排序（从高到低）:
1. **BGE** (语义+位置)
2. **RANDOM_TEXT** (仅位置)
3. **Random** (随机文档的语义+位置)
4. **No_Preprocess** (无融合)

如果 RANDOM_TEXT ≈ BGE > Random，则强烈支持位置适应假设。

## 注意事项

1. **首次运行耗时较长**
   - 需要生成20个random text的KV cache
   - 每个文本约50-250 tokens
   - 之后会使用缓存，速度加快

2. **仅支持GLOBAL scope**
   - 当前只实现了 `preprocess_scope=global` 模式
   - PER_EXAMPLE模式会抛出NotImplementedError

3. **文本库可扩展**
   - 如需更多/更长的随机文本，修改 `generate_random_text_corpus.py`
   - 重新运行生成脚本
   - 删除旧的random_texts KV cache目录

4. **结果对比**
   ```bash
   # 对比不同召回方法的结果
   cat result/Qwen2.5-7B-Instruct/musique/FusionRAG_global_topk10_bge/rate_0.15_revert_rope.txt
   cat result/Qwen2.5-7B-Instruct/musique/FusionRAG_global_topk10_random_text/rate_0.15_revert_rope.txt
   cat result/Qwen2.5-7B-Instruct/musique/FusionRAG_global_topk10_random/rate_0.15_revert_rope.txt
   cat result/no_preprocess/Qwen2.5-7B-Instruct/musique/nopreprocess/rate_0.15_revert_rope.txt
   ```

## 论文实验建议

### 实验设置
1. **对比组**:
   - BGE (baseline)
   - RANDOM_TEXT (本实验)
   - Random
   - No_Preprocess

2. **变量控制**:
   - 固定: rate=0.15, topk=10, scope=global, revert_rope=true
   - 对比不同召回方法的性能

3. **评估指标**:
   - Main Question Accuracy
   - Sub Question Accuracy
   - F1 Score
   - EM Score

### 预期贡献
- 提供更强的证据支持位置适应假设
- 区分"语义融合"和"位置扩展"的贡献
- 解释为什么Random召回也有效果

## 代码修改记录

1. `test_fusionrag_reflect.py`:
   - 添加 `RecallMethod.RANDOM_TEXT` 枚举
   - 在 `prepare_reflect_data()` 中添加RANDOM_TEXT处理逻辑
   - 修改KV cache生成和加载逻辑，支持负数索引

2. `generate_random_text_corpus.py`:
   - 新建：生成无关文本库

3. 启动脚本:
   - `run_fusionrag.sh`: 添加random_text选项说明
   - `run_fusionrag_topk_sweep.sh`: 添加random_text选项说明

## 故障排除

### 问题1: FileNotFoundError: Random text library not found
**解决**: 确保生成了文本库
```bash
cd /home/shm/document/exp/FusionRAG
/home/shm/anaconda3/envs/fusionrag/bin/python generate_random_text_corpus.py
```

### 问题2: NotImplementedError for PER_EXAMPLE mode
**解决**: 使用GLOBAL scope
```bash
PREPROCESS_SCOPE="global"
```

### 问题3: Random text KV cache重新生成
**解决**: 删除缓存目录
```bash
rm -rf /mnt/data3/tmp/fusionrag/Qwen2.5-7B-Instruct/musique/kv_cache/random_texts/
```
