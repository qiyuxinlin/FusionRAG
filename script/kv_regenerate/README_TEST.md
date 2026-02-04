# KV Cache 单元测试说明

## 概述

本测试脚本用于验证 KV cache 加载的正确性。测试方法是对每个文档加载其 KV cache，然后让 LLM 重复文档内容，最后对比生成结果与原始文档。

## 文件说明

- `test_kv_cache_unit.py` - 主测试脚本
- `demo.py` - KV cache 加载和生成函数（已修复）
- `test_results.json` - 测试结果输出

## 使用方法

### 1. 配置参数

编辑 `test_kv_cache_unit.py` 中的配置（`main()` 函数开头）：

```python
KV_CACHE_DIR = "/mnt/data3/tmp/fusionrag_online_lazy/Qwen2.5-7B-Instruct/2wikimqa/kv_cache"
DOC_POOL_PATH = "/home/shm/document/exp/FusionRAG/data/2wiki_input_rebuilt.json"
MODEL_PATH = "/mnt/data/models/Qwen2.5-7B-Instruct"
NUM_TEST_DOCS = 10  # 测试文档数量
```

### 2. 运行测试

```bash
cd /home/shm/document/exp/FusionRAG/script/kv_regenerate
python test_kv_cache_unit.py
```

### 3. 查看结果

测试过程中会实时输出每个文档的测试结果：
- 原始文本长度
- 生成的 token 数
- 字符准确率
- 词准确率
- 完全匹配状态

最终会生成 `test_results.json`，包含所有测试的详细数据。

## 测试原理

### 1. KV Cache 加载

对于每个文档：
1. 从文件名提取 doc_id（例如 `doc_1000_key.pt` → `doc_id=1000`）
2. 加载 `doc_{doc_id}_key.pt` 和 `doc_{doc_id}_value.pt`
3. 使用 `load_kv_and_generate` 函数加载 KV cache 并生成

### 2. Prompt 设计

```
Instruction: Please repeat the text from the previous conversation exactly as it is,
word for word, maintaining the original order and format. Output the result in JSON format,
with a key-value pair where the key is 'repeated_text' and the value is the repeated text
from the prior conversation:

{original_document_text}
```

### 3. 评估指标

- **Exact Match**: 原始文本 == 生成文本
- **Char Accuracy**: 匹配字符数 / 总字符数
- **Word Accuracy**: 匹配词数 / 总词数

## 预期结果

如果 KV cache 加载正确：
- 字符准确率应 > 95%
- 词准确率应 > 90%
- 大部分文档应该完全匹配

## 常见问题

### Q: 提示 "KV cache 目录不存在"
A: 检查 `KV_CACHE_DIR` 路径是否正确

### Q: 提示 "没有可测试的文档"
A: 确认 KV cache 目录中的 doc_id 与文档池中的 id 有交集

### Q: 准确率很低（< 50%）
A: 可能的原因：
1. KV cache 文件损坏
2. doc_id 映射错误
3. 模型没有正确加载 KV cache

### Q: 内存不足
A: 减少 `NUM_TEST_DOCS` 或使用更小的批次

## 故障排查

1. **检查 KV cache 文件是否存在**
   ```bash
   ls /mnt/data3/tmp/fusionrag_online_lazy/Qwen2.5-7B-Instruct/2wikimqa/kv_cache/doc_*_key.pt | head -5
   ```

2. **检查文档池格式**
   ```python
   import json
   with open('/home/shm/document/exp/FusionRAG/data/2wiki_input_rebuilt.json') as f:
       data = json.load(f)
   print(data[0])  # 应包含 'id' 和 'text' 字段
   ```

3. **验证 doc_id 映射**
   ```python
   # 检查 KV cache 的 doc_id 是否在文档池中
   doc_ids_in_cache = [1000, 1001, ...]
   doc_map = {doc['id']: doc['text'] for doc in doc_pool}
   common_ids = set(doc_ids_in_cache) & set(doc_map.keys())
   print(f"可测试的文档数: {len(common_ids)}")
   ```

## 输出示例

```
================================================================================
KV Cache 单元测试
================================================================================
KV Cache 目录: /mnt/data3/tmp/.../kv_cache
文档池路径: /home/shm/.../2wiki_input_rebuilt.json
模型路径: /mnt/data/models/Qwen2.5-7B-Instruct
测试文档数: 10

[1/5] 加载文档池...
  加载了 3659 个文档

[2/5] 扫描 KV cache 文件...
  找到 1435 个文档的 KV cache

[3/5] 匹配 KV cache 和文档池...
  可测试的文档数: 1200

[4/5] 加载模型...
  模型加载完成

[5/5] 开始测试前 10 个文档...
================================================================================

[Doc 1000] Testing...
  Original text length: 456 chars
  Prompt tokens: 234
  Generated tokens: 245
  Generated text length: 512 chars
  Exact Match: False
  Char Accuracy: 98.24%
  Word Accuracy: 97.56%

...

================================================================================
测试报告
================================================================================

总体统计:
  测试总数: 10
  成功: 10
  失败: 0

准确性统计:
  完全匹配: 8/10 (80.0%)
  平均字符准确率: 97.85%
  平均词准确率: 96.32%
  高准确率 (≥95%): 9/10 (90.0%)

详细结果已保存至: test_results.json
```

## 扩展测试

要测试所有文档：

```python
NUM_TEST_DOCS = len(test_doc_ids)  # 测试所有可用文档
```

或在运行时指定：

```bash
python test_kv_cache_unit.py --num_docs 100
```
