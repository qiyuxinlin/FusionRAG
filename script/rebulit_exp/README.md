# KV Cache 重建实验脚本

## 概述

此脚本用于测试 KV cache 重建能力，通过以下两个阶段：

1. **阶段1**：构建上下文 `[doc1, ..., docK, doc_o]` 并只保存 `doc_o` 的 KV cache
2. **阶段2**：加载 `doc_o` 的 KV cache，使用 repeat_prompt 尝试重建文档

## 文件说明

- `kv_cache_rebuilt_exp.py` - 主实验脚本
- `run_kv_cache_rebuilt_exp.sh` - 完整启动脚本（支持各种参数）
- `quick_test.sh` - 快速测试脚本（2次实验，K=2）
- `README.md` - 本说明文档

## 快速开始

### 1. 快速测试（2次实验）

```bash
cd /home/shm/document/exp/FusionRAG
./script/rebulit_exp/quick_test.sh
```

### 2. 单次实验（K=5, 10次实验）

```bash
./script/rebulit_exp/run_kv_cache_rebuilt_exp.sh -K 5 -n 10
```

### 3. 批量测试（多个K值）

```bash
./script/rebulit_exp/run_kv_cache_rebuilt_exp.sh --batch 0,1,2,5,10 -n 20
```

### 4. 使用多GPU

```bash
./script/rebulit_exp/run_kv_cache_rebuilt_exp.sh -K 5 -n 20 --multi_gpu
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-K, --K` | 5 | 目标文档前的随机文档数量 |
| `-n, --num_experiments` | 10 | 实验次数 |
| `--model_path` | /mnt/data/models/Qwen2.5-7B-Instruct | 模型路径 |
| `--device` | cuda:0 | 设备 |
| `--output_dir` | ./results/ | 输出目录 |
| `--multi_gpu` | false | 使用多GPU |

## 评估指标

脚本会计算以下指标：

1. **F1 分数**：字面一致性（token级别）
2. **语义相似度**：使用 BGE-M3 模型计算余弦相似度
3. **匹配分类**：
   - `literal_match`: F1 >= 0.9（字面一致）
   - `semantic_match`: 语义相似度 >= 0.85（语义一致）
   - `partial_match`: 语义相似度 0.6-0.85（部分一致）
   - `no_match`: 语义相似度 < 0.6（不一致）

## 输出结果

结果保存在 JSON 文件中，包含：

```json
{
  "config": { ... },
  "summary": {
    "total_experiments": 10,
    "success_count": 8,
    "success_rate": 0.8,
    "avg_f1": 0.9234,
    "avg_semantic_similarity": 0.9156,
    "match_counts": {
      "literal_match": 6,
      "semantic_match": 1,
      "partial_match": 1,
      "no_match": 2
    }
  },
  "results": [ ... ]
}
```

## 直接使用 Python 脚本

```bash
cd /home/shm/document/exp/FusionRAG

# 使用 fusionrag 环境
/home/shm/anaconda3/envs/fusionrag/bin/python script/rebulit_exp/kv_cache_rebuilt_exp.py \
    --K 5 \
    --num_experiments 10 \
    --model_path /mnt/data/models/Qwen2.5-7B-Instruct \
    --output_file ./results/rebuilt_k5.json
```

## RoPE 编码处理

脚本会自动处理 RoPE (Rotary Position Embedding) 编码：
- **保存时**：将 KV cache 从绝对位置转换到相对位置 0
- **加载时**：直接使用相对位置 0（无需再次转换）

## 环境要求

- Python 3.10
- PyTorch
- Transformers
- FlagEmbedding (BGE-M3)
- CUDA (推荐)

## 常见问题

### Q: 如何更改 prompt？

A: 修改 `run_kv_cache_rebuilt_exp.sh` 中的 `REPEAT_PROMPT` 变量，或直接使用 Python 脚本时添加 `--repeat_prompt` 参数。

### Q: 实验失败怎么办？

A: 检查：
1. 模型路径是否正确
2. 数据文件是否存在
3. GPU 内存是否足够
4. 使用 `quick_test.sh` 先进行小规模测试

### Q: 如何查看详细日志？

A: 脚本会输出详细的运行日志，包括每个实验的进度和结果。

## 目录结构

```
/home/shm/document/exp/FusionRAG/
├── script/rebulit_exp/
│   ├── kv_cache_rebuilt_exp.py   # 主脚本
│   ├── run_kv_cache_rebuilt_exp.sh  # 启动脚本
│   ├── quick_test.sh              # 快速测试
│   └── README.md                  # 本文档
├── cache/rebuilt_exp/             # KV cache 存储目录
└── results/                       # 结果输出目录
```
