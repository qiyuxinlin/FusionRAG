# FusionRAG - KV缓存优化和检索增强生成框架

**FusionRAG** 是一个基于 PyTorch 和 Transformers 的高性能 LLM 优化框架，专注于 **KV缓存优化** 和 **检索增强生成(RAG)** 功能。通过查询引导的缓存重处理技术，显著降低显存占用并加速推理。

## 项目概览

### 核心特性

- **统一接口**: 单一脚本支持 Mistral、Qwen、PanGu、Llama 等多种模型架构
- **KV缓存优化**: 多种缓存重处理方法(processCache、cacheBlend、Cache-Craft等)
- **RAG支持**: 内置KV缓存管理和检索能力
- **灵活部署**: FastAPI服务器，支持OpenAI兼容接口
- **多种优化方法**:
  - **ProcessCache**: 查询引导的KV缓存重处理(论文方法)
  - **CacheBlend**: SOTA基线，混合注意力缓存
  - **Cache-Craft**: SOTA基线，精心设计的缓存选择
  - **Speculative Prefill**: 推测预填充加速
  - **FrontRow**: 前排缓存基线

### 支持的模型

| 模型 | 类别 | 说明 |
|------|------|------|
| Mistral-7B | Encoder-Decoder | 开源LLM |
| Qwen2.5-7B | Encoder-Decoder | 阿里开源模型 |
| Llama-3.1-8B | Decoder-Only | Meta开源模型 |
| OpenPanGu-1B | Decoder-Only | 内向开源模型 |
| Deepseek | Mixture-of-Experts | MoE模型 |
| Mixtral | Mixture-of-Experts | MoE模型 |

---

## 安装指南

### 环境要求

- **Python**: >= 3.10
- **CUDA**: 11.8+ (用于GPU推理)
- **PyTorch**: >= 2.3.0
- **显存**: 至少 16GB (推荐 24GB+)

### 步骤 1: 克隆仓库

```bash
git clone https://github.com/kvcache-ai/ktransformers.git
cd FusionRAG
```

### 步骤 2: 创建虚拟环境

```bash
# 使用 conda
conda create -n ktransformers python=3.11
conda activate ktransformers

# 或使用 venv
python3.11 -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows
```

### 步骤 3: 安装依赖

#### 方式A: 使用安装脚本 (推荐)

```bash
# Linux/Mac
bash install.sh

# Windows
install.bat
```

#### 方式B: 手动安装

```bash
# 安装必要的依赖
pip install torch>=2.3.0 transformers==4.43.2 fastapi>=0.111.0

# 安装项目
pip install -e .

# 或仅安装（不进行源码编译）
pip install .
```

#### 方式C: 从源码编译 (如需CUDA优化)

```bash
# 设置编译标志
export KTRANSFORMERS_FORCE_BUILD=TRUE
pip install -e .
```

### 步骤 4: 验证安装

```bash
python -c "import ktransformers; print(ktransformers.__version__)"
```

---

## 数据准备

### 数据集说明

项目提供了4个基准数据集，位于 `/data/` 目录：

| 数据集 | 文件 | 样本数 | 用途 |
|--------|------|--------|------|
| MuSiQue | `musique-200.jsonl` | 200 | 多跳问答 |
| 2WikiMQA | `2wikimqa-200.jsonl` | 200 | 多跳问答 |
| HotpotQA | `hotpotqa-260-100-10-doc.jsonl` | 260 | 多跳问答 |
| TriviaQA | `triviaqa-270-100-10-doc.jsonl` | 270 | 开放域问答 |

### 数据格式

每个JSONL文件的格式为：

```json
{
  "question": "What is the capital of France?",
  "passages": [
    {"passage_id": 0, "text": "Paris is..."},
    {"passage_id": 1, "text": "France is..."},
    ...
  ],
  "answers": ["Paris"],
  "supporting_facts": [[0, 1]]
}
```

### 使用自定义数据集

如需使用自定义数据，请按照上述格式创建JSONL文件，放入 `./data/` 目录。

---

## 快速开始

### 最简单的方式: 运行预设配置

```bash
cd /home/shm/document/FusionRAG

# 方式1: 直接Python脚本运行
python -m ktransformers.unified_process_cache

# 方式2: 作为模块导入运行
python << 'EOF'
from ktransformers.unified_process_cache import main

# Mistral + ProcessCache 默认配置
main()
EOF
```

---

## 完整使用指南

### 方法1: Python脚本调用 (推荐用于实验)

#### 示例 1: Mistral + ProcessCache + 预处理

```python
from ktransformers.unified_process_cache import main

main(
    # 模型配置
    model_type='mistral',
    model_path='/path/to/Mistral-7B-Instruct-v0.3',
    model_name='Mistral-7B-Instruct-v0.3',

    # 数据配置
    data_name='musique-200.jsonl',
    data_path='./data/',

    # 缓存配置
    cache_path='./cache/',
    max_cache_len=32768,

    # 重处理方法
    rate=0.15,  # 15%重计算比例
    reprocess_method='processCache',

    # 预处理
    preprocess=True,
    topk=10,
    bge_model_path='/path/to/bge-m3-FP16',

    # 其他配置
    revert_rope=True,
    device='cuda:0'
)
```

#### 示例 2: Qwen + CacheBlend

```python
from ktransformers.unified_process_cache import main

main(
    model_type='qwen',
    model_path='/path/to/Qwen2.5-7B-Instruct',
    model_name='Qwen2.5-7B-Instruct',

    data_name='hotpotqa-260-100-10-doc.jsonl',
    cache_path='./cache/',

    rate=0.2,
    reprocess_method='cacheBlend',
    preprocess=False,  # CacheBlend不需要预处理

    device='cuda:0'
)
```

#### 示例 3: Llama + 完全重计算

```python
from ktransformers.unified_process_cache import main

main(
    model_type='llama',
    model_path='/path/to/Llama-3.1-8B-Instruct',
    model_name='Llama-3.1-8B-Instruct',

    data_name='2wikimqa-200.jsonl',
    rate=1.0,  # 完全重计算(性能基线)

    preprocess=False,
    reprocess_method='processCache',

    device='cuda:0'
)
```

#### 示例 4: PanGu + 推测预填充

```python
from ktransformers.unified_process_cache import main

main(
    model_type='pangu',
    model_path='/path/to/openPangu-Embedded-1B-V1.1',
    model_name='openPangu-Embedded-1B-V1.1',

    # Draft模型(可以是任意模型)
    draft_model_path='/path/to/Qwen2.5-1.5B-Instruct',

    data_name='triviaqa-270-100-10-doc.jsonl',
    rate=0.1,
    reprocess_method='speculative_prefill',

    device='cuda:0'
)
```

#### 示例 5: 对比模式 (性能评估)

```python
from ktransformers.unified_process_cache import main

# 自动与完全重计算进行对比
main(
    model_type='mistral',
    model_path='/path/to/Mistral-7B-Instruct-v0.3',
    data_name='musique-200.jsonl',

    rate=0.15,
    reprocess_method='processCache',

    compare_with_full_recompute=True,  # 启用对比模式

    device='cuda:0'
)
```

### 方法2: FastAPI服务器部署

#### 启动服务器

```bash
# 默认配置启动
ktransformers \
  --host 0.0.0.0 \
  --port 8000 \
  --model_name Mistral-7B-Instruct-v0.3 \
  --model_path /path/to/Mistral-7B-Instruct-v0.3

# 或使用Python直接启动
python -m ktransformers.server.main \
  --host 0.0.0.0 \
  --port 8000 \
  --model_name Mistral-7B-Instruct-v0.3 \
  --model_path /path/to/model
```

#### 调用API

```bash
# OpenAI兼容接口
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Mistral-7B",
    "messages": [{"role": "user", "content": "What is AI?"}],
    "temperature": 0.7
  }'

# 或使用Python客户端
from openai import OpenAI

client = OpenAI(api_key="fake", base_url="http://localhost:8000/v1")

response = client.chat.completions.create(
    model="Mistral-7B",
    messages=[{"role": "user", "content": "What is AI?"}]
)

print(response.choices[0].message.content)
```

---

## 参数详解

### main() 函数参数

#### 模型配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_type` | str | `'mistral'` | 模型类型: `'mistral'`、`'qwen'`、`'pangu'`、`'llama'` |
| `model_path` | str | 必需 | 模型权重路径 |
| `model_name` | str | 必需 | 模型名称(用于日志输出) |
| `draft_model_path` | str | None | Draft模型路径(仅`speculative_prefill`需要) |

#### 数据配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `data_name` | str | `'musique-200.jsonl'` | 数据集文件名 |
| `data_path` | str | `'./data/'` | 数据集目录 |

#### 缓存配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `cache_path` | str | `'./cache/'` | 缓存存储目录 |
| `max_cache_len` | int | 32768 | 最大缓存长度 |

#### 重处理配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `rate` | float | 0.2 | 重计算比例(0.0-1.0) |
| `reprocess_method` | str | `'cacheBlend'` | 重处理方法 |
| `revert_rope` | bool | False | 是否还原RoPE位置编码 |
| `use_sparse_attention` | bool | True | 是否使用稀疏注意力 |

#### 预处理配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `preprocess` | bool | True | 是否启用KV缓存预处理 |
| `topk` | int | 10 | 预处理选择的相似段落数 |
| `bge_model_path` | str | 必需(if preprocess=True) | BGE嵌入模型路径 |

#### 其他配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `device` | str | `'cuda:0'` | 计算设备 |
| `compare_with_full_recompute` | bool | False | 是否与完全重计算进行对比 |

---

## 输出和指标

### 输出文件结构

```
cache_path/
├── {data_name}/
│   └── {model_name}/
│       ├── reprocess_method_{method}_rate_{rate}_revert_rope_{rope}_topk_{topk}.csv
│       ├── reprocess_method_{method}_rate_{rate}_revert_rope_{rope}_topk_{topk}.txt
│       └── ...
└── data/
    └── {data_name}/
        └── {model_name}/
            ├── {example_id}_{chunk_id}_key.pt      # KV缓存key
            ├── {example_id}_{chunk_id}_value.pt    # KV缓存value
            └── ...
```

### 评估指标

#### 1. **精确匹配 (EM, Exact Match)**
- 衡量模型答案与标准答案完全匹配的比例
- 范围: 0-1 (越高越好)
- 计算方式: `(完全匹配答案数) / (总样本数)`

#### 2. **ROUGE分数 (ROUGE-L)**
- 衡量模型答案与标准答案的词序相似度
- 范围: 0-1 (越高越好)
- 计算方式: 最长公共子序列

#### 3. **F1分数** (用于某些模型)
- 精确率和召回率的调和平均数
- 范围: 0-1 (越高越好)

#### 4. **缓存效率指标**
- **缓存命中率**: 缓存中命中的token比例
- **显存节省**: 使用缓存相比完全计算的显存节省比例
- **推理加速比**: 使用缓存相比完全计算的推理加速倍数

### 查看指标结果

#### 实时终端输出

运行脚本时，终端会实时输出：

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
Final F1 Score: 0.8800 (if applicable)

======================================================================
```

#### CSV文件分析

查看 `cache_path/{data_name}/{model_name}/` 目录下的CSV文件：

```bash
# 查看结果CSV
head -20 cache_path/musique-200.jsonl/Mistral-7B/reprocess_method_processCache_rate_0.15_revert_rope_True_topk_10.csv

# 输出格式:
# Question,Real Answer,Pred Answer
# Who was the first president?,George Washington,George Washington
# ...
```

#### 读取评估指标

```python
import csv
import json

# 读取CSV结果
csv_file = "cache_path/musique-200.jsonl/Mistral-7B/reprocess_method_processCache_rate_0.15_revert_rope_True_topk_10.csv"

with open(csv_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(f"Q: {row['Question']}")
        print(f"Expected: {row['Real Answer']}")
        print(f"Got: {row['Pred Answer']}")
        print()
```

---

## 完整运行流程示例

### Step 1: 准备环境

```bash
# 进入项目目录
cd /home/shm/document/FusionRAG

# 激活虚拟环境
conda activate ktransformers

# 验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

### Step 2: 下载模型 (可选)

```bash
# 使用huggingface-cli下载
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.3 \
  --local-dir ./models/Mistral-7B-Instruct-v0.3

# 或使用Python下载
python << 'EOF'
from transformers import AutoModel
model = AutoModel.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.3",
    cache_dir="./models/"
)
EOF
```

### Step 3: 运行完整Pipeline

```python
# save as run_example.py
from ktransformers.unified_process_cache import main
import torch

print("="*80)
print("FusionRAG Complete Pipeline Example")
print("="*80)

# 检查CUDA
print(f"\nCUDA Available: {torch.cuda.is_available()}")
print(f"Device Count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current Device: {torch.cuda.get_device_name(0)}")

# 运行Pipeline
print("\n" + "="*80)
print("Starting Process Cache Experiment")
print("="*80 + "\n")

main(
    # 模型配置
    model_type='mistral',
    model_path='./models/Mistral-7B-Instruct-v0.3',
    model_name='Mistral-7B-Instruct-v0.3',

    # 数据配置
    data_name='musique-200.jsonl',
    data_path='./data/',

    # 缓存配置
    cache_path='./output/cache/',

    # 实验配置
    rate=0.15,
    reprocess_method='processCache',
    preprocess=True,
    topk=10,
    bge_model_path='./models/bge-m3-FP16',

    device='cuda:0'
)

print("\n" + "="*80)
print("Pipeline Completed!")
print("="*80)
```

### Step 4: 运行脚本

```bash
python run_example.py
```

### Step 5: 查看结果

```bash
# 查看输出目录
ls -la output/cache/musique-200.jsonl/Mistral-7B-Instruct-v0.3/

# 查看CSV结果
head -20 output/cache/musique-200.jsonl/Mistral-7B-Instruct-v0.3/reprocess_method_processCache_rate_0.15_revert_rope_True_topk_10.csv

# 查看详细结果
cat output/cache/musique-200.jsonl/Mistral-7B-Instruct-v0.3/reprocess_method_processCache_rate_0.15_revert_rope_True_topk_10.txt
```

---

## 性能对比和基准测试

### 运行对比实验

```python
from ktransformers.unified_process_cache import main

# 自动与完全重计算对比
main(
    model_type='mistral',
    model_path='./models/Mistral-7B-Instruct-v0.3',
    model_name='Mistral-7B-Instruct-v0.3',
    data_name='musique-200.jsonl',

    rate=0.15,
    reprocess_method='processCache',
    compare_with_full_recompute=True,  # 启用对比

    device='cuda:0'
)
```

### 基准结果示例

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

## 故障排除

### 常见问题

#### 1. CUDA显存不足

```
RuntimeError: CUDA out of memory
```

**解决方案:**
- 减少 `max_cache_len`
- 减少批处理大小
- 使用更小的模型
- 启用量化(如果支持)

```python
main(
    max_cache_len=16384,  # 从32768降低
    ...
)
```

#### 2. 模型加载失败

```
FileNotFoundError: model not found
```

**解决方案:**
- 确保模型路径正确
- 使用绝对路径而非相对路径
- 检查文件权限

#### 3. 数据集未找到

```
FileNotFoundError: data file not found
```

**解决方案:**
- 确保数据文件在 `data_path` 目录
- 检查文件名拼写
- 验证JSONL格式

#### 4. 内存错误

```
MemoryError
```

**解决方案:**
- 减少数据集大小
- 增加系统虚拟内存
- 使用更小的模型

### 调试技巧

```python
import logging

# 启用详细日志
logging.basicConfig(level=logging.DEBUG)

# 运行时打印调试信息
main(
    model_type='mistral',
    ...
    device='cuda:0'  # 使用CUDA调试
)
```

---

## 高级用法

### 自定义重处理方法

```python
# 在 unified_process_cache.py 中实现自定义方法
from ktransformers.unified_process_cache import main

# 支持的方法:
# - 'processCache': 查询引导重处理
# - 'cacheBlend': 混合注意力
# - 'Cache-Craft': 精心设计选择
# - 'speculative_prefill': 推测预填充
# - 'frontRow': 前排基线
```

### 使用量化模型

```python
main(
    model_path='./models/mistral-7b-gguf',  # GGUF格式
    model_type='mistral',
    ...
)
```

### 多卡推理

```python
# 支持多个GPU设备
for device_id in range(torch.cuda.device_count()):
    main(
        model_type='mistral',
        device=f'cuda:{device_id}',
        ...
    )
```

---

## API文档

### 核心模块

#### 1. `unified_process_cache.py`

**主函数:** `main(**kwargs)`

用于运行完整的Process Cache实验。参见上面的参数详解。

#### 2. `util/utils.py`

**关键函数:**
- `prefill_and_generate()`: 执行预填充和生成
- `load_kv_and_generate()`: 加载KV缓存并生成
- `compute_f1()`: 计算F1分数
- `_exact_match_score()`: 计算精确匹配分数

#### 3. `models/custom_cache.py`

**类:** `StaticCache`

管理KV缓存的生命周期。

#### 4. `server/main.py`

**应用:** FastAPI应用实例

提供REST API接口。

---

## 贡献指南

### 报告问题

如发现问题，请通过以下方式报告：
1. GitHub Issues: https://github.com/kvcache-ai/ktransformers/issues
2. 邮件: support@kvcache.ai

### 提交代码

1. Fork 仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 打开Pull Request

---

## 许可证

MIT License - 详见 LICENSE 文件

---

## 引用

如使用此项目，请引用以下论文：

```bibtex
@article{fusionrag2024,
  title={FusionRAG: Fusing Retrieval-Augmented Generation with Process Cache},
  author={...},
  journal={...},
  year={2024}
}
```

---

## 更新日志

### v0.1.4 (最新)
- 新增统一脚本支持所有模型
- 优化内存管理
- 修复CUDA显存泄漏

### v0.1.3
- 支持PanGu和Qwen模型
- 新增CacheBlend方法

### v0.1.2
- 基础Mistral支持
- Process Cache实现

---

## 联系方式

- **官网**: https://kvcache.ai
- **GitHub**: https://github.com/kvcache-ai/ktransformers
- **文档**: https://docs.kvcache.ai
- **邮件**: support@kvcache.ai

---

**祝你使用愉快!** 🚀
