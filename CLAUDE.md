# FusionRAG 项目文档

## 项目背景

### RAG 场景问题
在 RAG (Retrieval-Augmented Generation) 场景下，单次文本信息通常由以下部分组成：
- **System Prompt**: 系统提示词
- **多个召回文本块**: 从知识库检索到的相关文档
- **用户提问**: 用户的实际问题

### 优化目标
将 `system prompt + 文本块` 提前处理成 KV Cache，在 prefill 阶段直接召回 KV Cache 拼接然后输出，以加速推理。

### 核心挑战
这种方法破坏了 prefix cache 特性：
- **优势**: 可以显著加速 prefill 阶段
- **劣势**: 生成质量会下降（因为直接使用预计算的 KV Cache，丢失了完整的 attention 信息）

### FusionRAG 基础方案
原始 FusionRAG 方法：**所有层所有头固定挑选一部分 token 重算**，以此一定程度上挽回生成质量。

---

## 已完成工作

### 1. 核心实现文件

#### 1.1 `/mnt/data/wjh/FusionRAG/per_head_generation.py`
实现了逐层、逐头的 token 选择和重计算功能：

**主要功能模块**:
- `analyze_attention_distribution()`: 分析 attention 分布特征
- `compute_query_attention_scores()`: 计算查询相关的 attention 分数
- `find_connected_components()`: 查找高 attention 位置的连通分量
- `smart_query_selection()`: 智能查询选择（Oracle 方法核心算法）
  - 基于连通分量的 token 选择
  - 边界扩展策略
  - 动态比例计算支持
- `compute_layerwise_token_selection()`: 计算逐层 token 选择
- `sparse_prefill_per_head()`: 稀疏 prefill（支持分层分头选择）
- `generate_with_sparse_prefill()`: 使用稀疏 prefill 生成答案

**DraftModel 相关功能**:
- `compute_draft_model_attention()`: 使用小模型计算 attention
- `select_tokens_from_draft_attention()`: 从 draft model attention 中选择 token
- `analyze_draft_model_selection()`: 分析 draft model 的选择效果
- `main_with_draft_model()`: DraftModel 方法的完整测试流程

#### 1.2 `/mnt/data/wjh/FusionRAG/test_fusionrag_reflect.py`
主测试文件，实现了完整的 FusionRAG 测试流程：

**数据处理**:
- `prepare_reflect_data()`: 准备 result_reflect.json 数据集
- `load_system_prompt()`: 加载系统提示词
- 支持多种文档检索范围（GLOBAL, PER_EXAMPLE, SKIP_UNTESTED）

**评估框架**:
- `judge_answer_with_openai()`: 使用 OpenAI API 判断答案正确性
- 支持多线程异步评估
- 计算 F1 Score 和 Exact Match (EM)
- 主问题 + 子问题的层次化评估

### 2. 已实现的方法

#### 2.1 **DraftModel 方法**
- **核心思想**: 用小模型（如 Qwen2.5-3B）指导大模型（如 Qwen2.5-7B）的 token 选择
- **实现位置**:
  - `test_fusionrag_reflect.py`: lines 579-587, 920, 1224-1247
  - `per_head_generation.py`: `compute_draft_model_attention()`, `select_tokens_from_draft_attention()`
- **特点**:
  - 使用小模型完整 prefill 获取 attention 分布
  - 基于 attention 分数选择重要 token
  - 支持熵选层（entropy selection）或最后一层选择
  - 固定重算比例（如 rate=0.3）

#### 2.2 **Oracle 方法**
- **核心思想**: 用主模型自身做完整 prefill 获取 attention，指导 token 选择
- **实现位置**:
  - `test_fusionrag_reflect.py`: lines 589-590, 1198-1222
  - `per_head_generation.py`: `smart_query_selection()`
- **特点**:
  - 无需额外的 draft model
  - 使用主模型的真实 attention 分布
  - 基于连通分量和边界扩展的智能选择
  - 固定重算比例
- **优势**: 选择质量最高（因为使用真实 attention）

#### 2.3 **vAttention 方法**
- **论文参考**: "vAttention: Verified Sparse Attention" (arXiv:2510.05688)
- **核心思想**: 结合 top-k 选择和随机采样
- **实现位置**: `test_fusionrag_reflect.py`: lines 1169-1196
- **特点**:
  - `vattention_topk_ratio` 控制 top-k vs 随机采样的比例（默认 0.5 = 各占 50%）
  - 固定总重算比例
  - 平衡确定性和多样性

#### 2.4 **OracleDynamic 方法**
- **核心思想**: Oracle 选择 + 动态计算重算比例
- **实现位置**: `test_fusionrag_reflect.py`: lines 503-508
- **特点**:
  - 基于 attention 分布动态计算每个问题的最优 rate
  - 参数化控制：
    - `epsilon`: 误差容忍度（如 0.1 = 10% 相对误差）
    - `delta`: 置信度（如 0.05 = 95% 置信度）
    - `min_rate`: 最小重算比例
    - `max_rate`: 最大重算比例

#### 2.5 **OracleAdaptive 方法**
- **核心思想**: Oracle 选择 + 综合多特征动态比例计算
- **实现位置**: `test_fusionrag_reflect.py`: lines 1132-1167
- **特点**:
  - 使用和 Oracle 完全相同的选择策略（smart_query_selection）
  - 动态比例综合以下特征：
    1. **Coverage-based ratio**: 达到 85% attention 覆盖所需比例
    2. **Connected components**: 高 attention 位置的分散程度
    3. **Spread factor**: 高 attention 位置在文档中的跨度
    4. **Gini coefficient**: Attention 集中度
  - 保证 safety buffer（min_rate）确保不遗漏 long-tail 重要信息

#### 2.6 **QueryAttention 方法**
- **核心思想**: 基于查询相关性的 attention 选择
- **实现位置**: `test_fusionrag_reflect.py`: lines 1249-1269
- **特点**:
  - 使用 query attention 指导选择
  - 支持熵选层（entropy selection）消融实验
  - 保留原始 FusionRAG 的文档融合机制

#### 2.7 **FusionRAG 基础方法**
- **核心思想**: 所有层所有头固定比例重算 token
- **实现位置**: `test_fusionrag_reflect.py`: main()
- **特点**:
  - 固定 rate（如 0.2, 0.3）
  - 所有层使用相同的选择策略
  - 作为 baseline 对比方法

---

## 方法对比总结

| 方法 | 选择策略 | 重算比例 | 额外开销 | 优势 | 劣势 |
|------|---------|---------|---------|------|------|
| **FusionRAG** | 固定随机/top-k | 固定 | 无 | 简单、快速 | 选择不够智能 |
| **DraftModel** | Draft model attention | 固定 | 小模型 prefill | 成本低于 Oracle | 需要额外模型 |
| **Oracle** | 主模型 attention + 连通分量 | 固定 | 主模型 prefill | 选择质量最高 | 额外 prefill 开销 |
| **vAttention** | Top-k + 随机采样 | 固定 | 主模型 prefill | 平衡确定性与多样性 | 随机性可能不稳定 |
| **OracleDynamic** | Oracle 选择 | 动态（基于统计） | 主模型 prefill | 自适应不同问题 | 动态算法复杂度 |
| **OracleAdaptive** | Oracle 选择 | 动态（综合特征） | 主模型 prefill | 最智能、多维度考虑 | 计算最复杂 |

---

## 测试流程

### 数据集
- **来源**: `result_reflect.json`
- **结构**: 主问题 → 多个子问题 → 每个子问题关联多个检索文档
- **评估方式**:
  - 主问题正确 = 所有子问题都正确
  - 使用 OpenAI API (DeepSeek) 判断答案正确性
  - 计算 F1 Score 和 Exact Match

### 预处理模式
- **GLOBAL**: 从所有问题的文档中检索相似文档（原始行为）
- **PER_EXAMPLE**: 仅从当前问题的文档中检索
- **SKIP_UNTESTED**: 跳过不测试的问题的文档

### 实验参数
- `rate`: 重算比例（0.1 ~ 1.0）
- `topk`: FusionRAG 预处理时融合的相似文档数量
- `revert_rope`: 是否在预处理时还原 RoPE
- `use_entropy_selection`: 是否使用熵选层
- `entropy_top_k`: 熵选层选择的层数
- `draft_layer_selection`: DraftModel/Oracle 选层方式（'entropy' 或 'last'）

---

## 效果总结
- **DraftModel 方法**: 效果很不错，在较低的额外开销下获得了接近 Oracle 的选择质量
- **Oracle 方法**: 选择质量最高，但需要额外的完整 prefill 开销
- **OracleAdaptive 方法**: 综合多维度特征，自适应不同问题的特点，是最智能的方法

---

## 下一步计划
等待用户指示进一步改进方向...
