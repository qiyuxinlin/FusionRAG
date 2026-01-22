# 数据集优化说明

## 优化概述

将 `result_reflect.json` 中存储的完整文档文本替换为文档池索引引用，大幅减少存储空间并提高数据访问效率。

## 优化效果

- **原始文件大小**: 4749.82 KB
- **优化后文件大小**: 1304.93 KB
- **空间节省**: 72.53%
- **文档匹配成功率**: 100% (0个文档未找到)

## 文件说明

### 输入文件

1. **musique_input.json**: 文档池，包含6060个文档
   - 每个文档有唯一的 `id` (索引)
   - 包含文档的 `text` 和 `metadata`

2. **result_reflect.json**: 原始结果数据
   - 包含200个样本
   - 每个样本的文档以完整文本形式存储

### 输出文件

**result_reflect_optimized.json**: 优化后的结果数据
- 所有文档文本已替换为索引引用
- 数据结构完全保持一致

## 优化的字段

优化脚本处理了以下字段中的文档文本：

1. `gold_docs` - 金标准文档
2. `retrieved_results` - 最终召回的文档
3. `intermediate_context[].retrieve docs` - 中间步骤召回的文档
4. `intermediate_context[].retrieve docs supported` - 中间步骤支持的文档

### 优化前后对比

**优化前 (原始格式):**
```json
{
    "gold_docs": [
        "National Cycle Route 57 is part of the United Kingdom's National Cycle Network...",
        "The National Cycle Network (NCN) is the national cycling route network..."
    ]
}
```

**优化后 (索引格式):**
```json
{
    "gold_docs": [174, 797]
}
```

## 使用方法

### 方法1: 使用提供的 DocumentPool 类

```python
from usage_example import DocumentPool, load_json

# 初始化文档池
doc_pool = DocumentPool('/path/to/musique_input.json')

# 加载优化后的数据
samples = load_json('/path/to/result_reflect_optimized.json')

# 通过索引获取文档文本
sample = samples[0]
gold_doc_texts = doc_pool.get_docs_texts(sample['gold_docs'])

for idx, text in zip(sample['gold_docs'], gold_doc_texts):
    print(f"Doc {idx}: {text}")
```

### 方法2: 直接访问

```python
import json

# 加载文档池
with open('musique_input.json', 'r') as f:
    doc_pool = json.load(f)
    idx_to_doc = {doc['id']: doc for doc in doc_pool}

# 加载优化数据
with open('result_reflect_optimized.json', 'r') as f:
    samples = json.load(f)

# 访问文档
sample = samples[0]
for idx in sample['gold_docs']:
    doc_text = idx_to_doc[idx]['text']
    print(f"Doc {idx}: {doc_text}")
```

## 脚本说明

### optimize_with_fuzzy_match.py

优化脚本，使用多种匹配策略：

1. **精确匹配**: 基于去除空白后的文本精确匹配
2. **标准化匹配**: 标准化空白字符后匹配
3. **前缀匹配**: 处理可能被截断的文档

**运行方式:**
```bash
python optimize_with_fuzzy_match.py
```

### usage_example.py

使用示例脚本，展示如何访问优化后的数据。

**运行方式:**
```bash
python usage_example.py
```

## 数据统计

- **总样本数**: 200
- **金标准文档引用**: 400个
- **召回文档引用**: 423个
- **中间步骤文档引用**: 5330个
- **文档池总文档数**: 6060个

## 优势

1. **存储效率**: 减少72.53%的存储空间
2. **加载速度**: 更小的文件，更快的加载速度
3. **内存占用**: 大幅减少内存使用
4. **数据一致性**: 文档内容统一管理，避免重复
5. **易于维护**: 文档内容更新只需修改文档池

## 注意事项

1. 使用优化后的数据时，必须配合文档池 (`musique_input.json`) 使用
2. 文档索引从0开始
3. 如果某个索引在文档池中不存在，访问时会返回错误信息
4. 原始文件 `result_reflect.json` 已被保留，可以随时回退

## 恢复原始格式

如果需要恢复原始格式（索引转换回完整文本），可以使用以下脚本：

```python
import json

def restore_original_format(optimized_path, doc_pool_path, output_path):
    # 加载文档池
    with open(doc_pool_path, 'r') as f:
        doc_pool = json.load(f)
        idx_to_doc = {doc['id']: doc['text'] for doc in doc_pool}

    # 加载优化数据
    with open(optimized_path, 'r') as f:
        samples = json.load(f)

    # 恢复文本
    for sample in samples:
        if 'gold_docs' in sample:
            sample['gold_docs'] = [idx_to_doc[idx] for idx in sample['gold_docs']]
        # ... 其他字段类似处理

    # 保存
    with open(output_path, 'w') as f:
        json.dump(samples, f, ensure_ascii=False, indent=4)
```

## 联系方式

如有问题或建议，请联系开发团队。
