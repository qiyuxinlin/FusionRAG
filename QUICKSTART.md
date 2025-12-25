# FusionRAG 快速开始指南

本指南帮助你快速了解并运行FusionRAG的完整pipeline，获得性能和质量指标。

## 📋 目录
1. [5分钟快速开始](#5分钟快速开始)
2. [详细步骤](#详细步骤)
3. [查看和分析结果](#查看和分析结果)
4. [常见问题](#常见问题)

---

## 5分钟快速开始

### 前提条件
- Python 3.10+
- 至少 16GB 显存 (推荐 24GB+)
- CUDA 11.8+ (可选，使用CPU会慢很多)

### 安装

```bash
# 1. 进入项目目录
cd /home/shm/document/FusionRAG

# 2. 创建虚拟环境
conda create -n ktransformers python=3.11 -y
conda activate ktransformers

# 3. 安装项目
pip install -e .
```

### 运行

```bash
# 使用默认配置运行(Mistral + ProcessCache)
python run_complete_pipeline.py

# 或指定模型
python run_complete_pipeline.py \
  --model_type mistral \
  --model_path ./models/Mistral-7B-Instruct-v0.3 \
  --model_name Mistral-7B-Instruct-v0.3
```

### 查看结果

```bash
# 分析结果(交互式)
python analyze_results.py --interactive

# 或直接分析特定文件
python analyze_results.py --csv_file output/cache/musique-200.jsonl/Mistral-7B/reprocess_method_processCache_rate_0.15.csv
```

---

## 详细步骤

### 步骤 1: 环境准备

#### 1.1 克隆/进入项目

```bash
cd /home/shm/document/FusionRAG
```

#### 1.2 创建虚拟环境

```bash
# 使用 conda (推荐)
conda create -n ktransformers python=3.11
conda activate ktransformers

# 或使用 venv
python3.11 -m venv venv
source venv/bin/activate
```

#### 1.3 验证GPU

```bash
python << 'EOF'
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Devices: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Device 0: {torch.cuda.get_device_name(0)}")
    print(f"Device 0 Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
EOF
```

### 步骤 2: 安装项目

#### 2.1 安装依赖

```bash
# 安装PyTorch (如果还未安装)
pip install torch>=2.3.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装项目
pip install -e .
```

#### 2.2 验证安装

```bash
python -c "import ktransformers; print('✓ KTransformers installed successfully')"
```

### 步骤 3: 准备数据和模型

#### 3.1 数据已包含

项目已包含4个示例数据集在 `./data/` 目录:
- `musique-200.jsonl` (推荐，200个样本)
- `2wikimqa-200.jsonl`
- `hotpotqa-260-100-10-doc.jsonl`
- `triviaqa-270-100-10-doc.jsonl`

#### 3.2 准备模型

**方式A: 使用现有模型路径**

如果你已经有模型权重，直接指定路径:

```bash
python run_complete_pipeline.py \
  --model_path /your/model/path \
  --model_name YourModelName
```

**方式B: 从HuggingFace下载**

```bash
# 使用 huggingface-cli
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.3 \
  --local-dir ./models/Mistral-7B-Instruct-v0.3

# 或使用Python
python << 'EOF'
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "mistralai/Mistral-7B-Instruct-v0.3"
save_dir = "./models/Mistral-7B-Instruct-v0.3"

# 只下载tokenizer和config(不下载权重，节省空间)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(save_dir)

print(f"✓ Model preparation completed: {save_dir}")
EOF
```

### 步骤 4: 运行Pipeline

#### 最简单的方式

```bash
python run_complete_pipeline.py
```

这将使用默认配置:
- 模型: Mistral-7B
- 数据: musique-200.jsonl
- 方法: ProcessCache
- 重计算率: 15%

#### 使用不同配置

```bash
# 使用Qwen模型 + CacheBlend方法
python run_complete_pipeline.py \
  --model_type qwen \
  --model_path ./models/Qwen2.5-7B \
  --model_name Qwen2.5-7B \
  --reprocess_method cacheBlend \
  --no-preprocess

# 使用Llama + 完全重计算(基线)
python run_complete_pipeline.py \
  --model_type llama \
  --model_path ./models/Llama-3.1-8B \
  --model_name Llama-3.1-8B \
  --rate 1.0

# 启用对比模式(与完全重计算对比)
python run_complete_pipeline.py \
  --compare-with-full-recompute
```

#### 参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| `--model_type` | 模型架构 | `mistral`, `qwen`, `llama`, `pangu` |
| `--model_path` | 模型权重路径 | `./models/Mistral-7B` |
| `--model_name` | 显示名称 | `Mistral-7B` |
| `--data_name` | 数据集文件 | `musique-200.jsonl` |
| `--rate` | 重计算比例(0-1) | `0.15`, `1.0` |
| `--reprocess_method` | 重处理方法 | `processCache`, `cacheBlend` |
| `--preprocess` / `--no-preprocess` | 启用预处理 | 默认启用 |
| `--compare-with-full-recompute` | 对比模式 | 启用时对比完全重计算 |

### 步骤 5: 查看结果

#### 5.1 实时输出

运行时，终端会显示:

```
======================================================================
Processing: Question 1/200
======================================================================
Question: Who was the first president?
Real Answer: George Washington
Pred Answer: George Washington
EM Score: 1.0000
ROUGE Score: 1.0000
...

======================================================================
FINAL RESULTS SUMMARY
======================================================================
Total Prefill Time: 123.45s
Average Prefill Time: 0.62s per question

Final EM Score: 0.8500
Final ROUGE Score: 0.9200
======================================================================
```

#### 5.2 保存的文件

结果保存在 `output/cache/` 目录下:

```
output/cache/
├── musique-200.jsonl/
│   └── Mistral-7B/
│       ├── reprocess_method_processCache_rate_0.15_revert_rope_False_topk_10.csv
│       ├── reprocess_method_processCache_rate_0.15_revert_rope_False_topk_10.txt
│       └── ...
└── data/
    └── musique-200.jsonl/
        └── Mistral-7B/
            ├── 0_0_key.pt
            ├── 0_0_value.pt
            └── ...
```

#### 5.3 分析结果

使用提供的分析脚本:

```bash
# 交互式分析
python analyze_results.py --interactive

# 分析特定文件
python analyze_results.py --csv_file output/cache/musique-200.jsonl/Mistral-7B/reprocess_method_processCache_rate_0.15_revert_rope_False_topk_10.csv

# 对比多个方法
python analyze_results.py --compare \
  output/cache/musique-200.jsonl/Mistral-7B/reprocess_method_processCache_rate_0.15.csv \
  output/cache/musique-200.jsonl/Mistral-7B/reprocess_method_cacheBlend_rate_0.15.csv

# 导出汇总报告
python analyze_results.py --export-summary
```

---

## 查看和分析结果

### 指标说明

#### 1. **EM (Exact Match) 分数**
- 精确匹配分数，范围 0-1
- 越高越好
- 计算: (完全匹配答案数) / (总样本数)

```
EM Score: 0.8500  # 表示85%的答案完全匹配
```

#### 2. **ROUGE 分数**
- 衡量生成文本与参考文本的相似度
- 范围 0-1
- 越高越好

```
ROUGE Score: 0.9200  # 表示92%的相似度
```

#### 3. **缓存命中率**
- 缓存中使用的token比例
- 越高表示缓存效率越好

```
Cache Hit Rate: 0.85  # 表示85%的token来自缓存
```

### 查看详细结果

#### 方式 1: 查看CSV文件

```bash
# 前20行
head -20 output/cache/musique-200.jsonl/Mistral-7B/reprocess_method_processCache_rate_0.15.csv

# 输出:
# Question,Real Answer,Pred Answer
# Who was the first president?,George Washington,George Washington
# ...
```

#### 方式 2: 使用Python分析

```python
import csv
import json

# 读取和分析结果
csv_file = "output/cache/musique-200.jsonl/Mistral-7B/reprocess_method_processCache_rate_0.15.csv"

with open(csv_file, 'r') as f:
    reader = csv.DictReader(f)
    results = list(reader)

# 计算指标
total = len(results)
matches = sum(1 for r in results if r['Pred Answer'].lower() == r['Real Answer'].lower())
em_score = matches / total

print(f"Total: {total}")
print(f"Matches: {matches}")
print(f"EM Score: {em_score:.4f}")

# 显示几个例子
for i, r in enumerate(results[:5]):
    print(f"\nExample {i+1}:")
    print(f"  Q: {r['Question']}")
    print(f"  Expected: {r['Real Answer']}")
    print(f"  Got: {r['Pred Answer']}")
    print(f"  Match: {'✓' if r['Pred Answer'].lower() == r['Real Answer'].lower() else '✗'}")
```

#### 方式 3: 查看对比结果

运行对比模式时:

```bash
python run_complete_pipeline.py --compare-with-full-recompute
```

输出:
```
======================================================================
QUALITY COMPARISON RESULTS
======================================================================

Full Recompute (Baseline):
  - EM Score: 0.8700
  - ROUGE Score: 0.9300

Cache Reuse (rate=0.15):
  - EM Score: 0.8650
  - ROUGE Score: 0.9250

Quality Difference:
  EM Score Difference: -0.0050 (-0.57%)
  ROUGE Score Difference: -0.0050 (-0.54%)

✓ Quality preserved: cache reuse maintains generation quality
======================================================================
```

---

## 常见问题

### Q1: 运行时出现显存不足错误

**A:** 显存不足有几个解决方案:

```bash
# 1. 减少缓存长度
python run_complete_pipeline.py --max_cache_len 16384

# 2. 使用更小的模型
python run_complete_pipeline.py --model_path ./models/Qwen2.5-1.5B

# 3. 减少重计算比例
python run_complete_pipeline.py --rate 0.05
```

### Q2: 为什么模型加载很慢?

**A:** 这是正常的。首次加载会:
- 初始化模型权重 (可能需要几分钟)
- 加载到GPU显存 (取决于模型大小和网速)

可以通过以下方式加速:
- 确保模型文件在本地(不需要下载)
- 使用更快的存储设备(SSD)
- 使用更小的模型

### Q3: 预处理需要的BGE模型在哪里?

**A:** 有两个选择:

```bash
# 方式1: 指定现有模型路径
python run_complete_pipeline.py \
  --bge_model_path /your/bge/model/path

# 方式2: 禁用预处理
python run_complete_pipeline.py --no-preprocess
```

### Q4: 如何比较不同方法的性能?

**A:** 运行多个配置并对比:

```bash
# 方法1: ProcessCache
python run_complete_pipeline.py --reprocess_method processCache

# 方法2: CacheBlend
python run_complete_pipeline.py --reprocess_method cacheBlend

# 方法3: 完全重计算(基线)
python run_complete_pipeline.py --rate 1.0

# 然后对比结果
python analyze_results.py --export-summary
```

### Q5: 结果文件中包含什么?

**A:** 结果包括:

```
CSV文件: question, real_answer, predicted_answer
- 用于计算EM和ROUGE分数

TXT文件: 详细统计信息
- 总指标汇总
- 性能指标
- 时间统计

PT文件: KV缓存
- 预计算的key和value张量
- 用于加速后续推理
```

### Q6: 如何使用自定义数据集?

**A:** 按照格式创建JSONL文件:

```jsonl
{"question": "Who was the first president?", "passages": [{"passage_id": 0, "text": "George Washington was..."}, ...], "answers": ["George Washington"], "supporting_facts": [[0, 1]]}
```

然后:

```bash
python run_complete_pipeline.py \
  --data_name custom_data.jsonl \
  --data_path ./data/
```

### Q7: 推理速度如何测试?

**A:** 脚本会自动测量:

```
Total Prefill Time: 123.45s      # 总耗时
Average Prefill Time: 0.62s      # 平均每个问题的耗时
```

### Q8: 如何部署为服务?

**A:** 使用FastAPI服务器:

```bash
python -m ktransformers.server.main \
  --host 0.0.0.0 \
  --port 8000 \
  --model_name Mistral-7B \
  --model_path ./models/Mistral-7B

# 然后调用
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Mistral-7B", "messages": [{"role": "user", "content": "Hello"}]}'
```

---

## 下一步

现在你已经了解如何:
1. ✓ 安装FusionRAG
2. ✓ 运行完整pipeline
3. ✓ 获得性能和质量指标
4. ✓ 分析和对比结果

### 推荐的后续操作

1. **尝试不同配置**
   - 不同的模型 (Qwen, Llama, PanGu)
   - 不同的方法 (ProcessCache, CacheBlend, Cache-Craft)
   - 不同的重计算比例

2. **性能优化**
   - 对比不同方法的速度和准确性
   - 找到最适合你的平衡点

3. **生产部署**
   - 使用FastAPI服务器部署
   - 集成到你的应用中

4. **深度学习**
   - 阅读论文了解原理
   - 自定义重处理方法

---

## 获取帮助

- **查看README**: `README_COMPLETE.md`
- **查看源代码**: `ktransformers/unified_process_cache.py`
- **提交Issue**: https://github.com/kvcache-ai/ktransformers/issues
- **阅读论文**: https://kvcache.ai/

---

**祝你使用愉快!** 🚀
