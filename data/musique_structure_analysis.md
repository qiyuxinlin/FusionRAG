# Musique-200 数据集结构分析

## 📋 数据集概述

- **文件格式**: JSONL (每行一个JSON对象)
- **总问题数**: 200
- **数据集类型**: Multi-hop QA (多跳问答)
- **语言**: 英语

## 🔍 数据结构

### 顶层字段

```json
{
  "input": "问题文本",
  "context": "包含所有相关文档的长文本",
  "answers": ["答案1", "答案2"],
  "length": 11054,
  "dataset": "musique",
  "language": "en",
  "all_classes": null,
  "_id": "唯一标识符"
}
```

### 字段详解

| 字段 | 类型 | 说明 | 示例值 |
|------|------|------|--------|
| `input` | string | 多跳问题文本 | "Who plays the wife of the producer of Here Comes the Boom in Grown Ups?" |
| `context` | string | 包含多个Passage的长文本，每个Passage以"Passage N"标记 | "Passage 1\nGrown Ups (film)...\nPassage 2\n..." |
| `answers` | list[string] | 问题的正确答案列表（可能有多个可接受的答案） | ["Maria Bello"] |
| `length` | int | Context的token/字符长度 | 11054 |
| `dataset` | string | 数据集名称 | "musique" |
| `language` | string | 语言代码 | "en" |
| `all_classes` | null | 保留字段 | null |
| `_id` | string | 唯一标识符（48字符） | "4c114f84ca41284c0845705943adc10b70a4fd3e8d0ca295" |

## 📊 统计信息

### 问题统计
- **最短问题**: 35 字符
- **最长问题**: 187 字符
- **平均长度**: 85.5 字符

### Context 统计
- **Passage 数量**: 19-31 个/问题
- **平均 Passage 数**: 23.7 个/问题
- **Context 长度**: 22,978 - 82,409 字符
- **平均长度**: 69,932.5 字符

### 答案统计
- **答案数量**: 1-5 个/问题
- **平均数量**: 1.4 个/问题

## 📝 Context 结构

Context 字段包含多个文档段落（Passages），格式如下：

```
Passage 1
文档标题 Part 1
文档内容...

Passage 2
文档标题 Part 2
文档内容...

...

Passage N
文档标题
文档内容...
```

### Passage 特点
- 每个 Passage 由标记行 "Passage N" 分隔
- Passage 通常包含标题行（如 "Grown Ups (film) Part 1"）
- 同一主题的文档可能被分成多个 Part
- 长文档会被拆分成多个 Passage（Part 1, Part 2, Part 3...）

## 💡 使用场景

### 1. **多跳问答 (Multi-hop QA)**
问题需要跨多个文档片段推理才能回答。

**示例**:
```
问题: "Who plays the wife of the producer of Here Comes the Boom in Grown Ups?"

推理步骤:
1. 从 Passage 7-8 找到: "Here Comes the Boom" 的制片人是 Kevin James
2. 从 Passage 1-3 找到: Kevin James 在 "Grown Ups" 中的妻子由 Maria Bello 扮演
3. 答案: Maria Bello
```

### 2. **长上下文理解**
- 每个问题包含平均 23.7 个 Passages
- 总长度约 70,000 字符
- 测试模型的长文本处理能力

### 3. **检索增强生成 (RAG)**
Context 提供了预先收集的相关文档，可用于:
- 测试检索系统的准确性
- 评估文档排序能力
- 验证答案生成质量

## 🔄 与 result_reflect.json 的关系

### 重叠情况
- **30/200** (15%) 的问题在两个数据集中都存在
- result_reflect.json 中的其余问题来自其他数据集

### 主要区别

| 特性 | musique-200.jsonl | result_reflect.json |
|------|-------------------|---------------------|
| 格式 | JSONL (每行一个问题) | JSON (单个数组) |
| 文档存储 | 长文本字符串，Passage分隔 | 数组，每个文档是独立字符串 |
| 子问题 | 无 | 有 intermediate_context |
| 问题分解 | 无 | 有详细的推理步骤 |
| 用途 | 源数据集 | 预处理后的评估数据集 |

## 🎯 使用示例

### 解析单个问题

```python
import json

with open('musique-200.jsonl', 'r') as f:
    for line in f:
        item = json.loads(line)

        # 提取问题和答案
        question = item['input']
        answers = item['answers']

        # 解析 Passages
        import re
        passages = re.split(r'Passage \d+\n', item['context'])
        passages = [p.strip() for p in passages if p.strip()]

        print(f"问题: {question}")
        print(f"答案: {answers}")
        print(f"文档数: {len(passages)}")
        break
```

### 提取所有文档

```python
import json
import re

all_docs = []
with open('musique-200.jsonl', 'r') as f:
    for line in f:
        item = json.loads(line)
        passages = re.split(r'Passage \d+\n', item['context'])
        passages = [p.strip() for p in passages if p.strip()]
        all_docs.extend(passages)

print(f"总文档数: {len(all_docs)}")
```

## 📚 数据集来源

Musique (Multi-hop QUestions In Streaming Environments) 是一个专门设计用于多跳推理的问答数据集，特点包括:

1. **多跳推理**: 需要跨多个文档进行推理
2. **可回答性**: 所有问题都基于提供的 Context 可回答
3. **答案多样性**: 部分问题有多个可接受的答案形式
4. **文档相关性**: Context 包含相关和干扰文档

## 🔗 相关文件

- `/home/shm/document/exp/FusionRAG/data/musique-200.jsonl` - 原始数据集
- `/home/shm/document/exp/FusionRAG/data/result_reflect.json` - 处理后的评估数据集
- `/home/shm/document/exp/FusionRAG/test_fusionrag_reflect.py` - 测试脚本
