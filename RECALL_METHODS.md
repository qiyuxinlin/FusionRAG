# FusionRAG 文档召回方法说明

## 概述

FusionRAG 在 preprocess 阶段需要为每个文档召回 topK 个相似文档进行融合。我们实现了 **4 种不同的召回方法**，用于不同的实验场景。

---

## 召回方法列表

### 1. BGE 相似度召回 (`bge`)

**描述**: 使用 BGE embedding 模型计算文档语义相似度，召回最相似的 topK 个文档。

**适用场景**:
- 正常实验（默认方法）
- 追求最佳召回质量
- 基于语义相似度的文档融合

**实现原理**:
1. 使用 BGE-M3 模型将所有文档编码为向量
2. 使用 FAISS 构建相似度索引
3. 对每个文档，查找语义最相似的 topK 个文档

**优点**:
- ✅ 召回质量高，基于语义相似度
- ✅ 符合实际应用场景

**缺点**:
- ❌ 需要额外的 BGE 模型
- ❌ 计算开销较大

**配置示例**:
```bash
RECALL_METHOD="bge"
```

---

### 2. 随机召回 (`random`)

**描述**: 从候选文档池中随机抽取 topK 个文档。

**适用场景**:
- 消融实验：测试召回方法的重要性
- 对比实验：与 BGE 方法对比
- Baseline 实验

**实现原理**:
1. 确定候选池（根据 `preprocess_scope` 设置）
2. 从候选池中随机抽取 topK 个文档（排除自身）
3. 使用固定随机种子保证可复现

**优点**:
- ✅ 实现简单，无需额外模型
- ✅ 计算快速
- ✅ 可复现（固定随机种子）

**缺点**:
- ❌ 召回质量差，无法保证相关性

**配置示例**:
```bash
RECALL_METHOD="random"
RANDOM_SEED="42"  # 固定种子保证可复现
```

---

### 3. 重复自身 (`repeat_self`)

**描述**: 每个文档召回自身 topK 次。

**适用场景**:
- **消融实验**: 测试文档融合的必要性
- **上界实验**: 理论上最相似的文档就是自己
- **Baseline**: 不进行跨文档融合的情况

**实现原理**:
1. 对每个文档，将其自身的索引重复 topK 次
2. 相当于在 preprocess 阶段，每个文档只与自己融合

**优点**:
- ✅ 实现最简单
- ✅ 计算最快
- ✅ 理论上保证相关性（自己最相似）

**缺点**:
- ❌ 无法利用其他文档的信息
- ❌ 失去了文档融合的意义

**使用场景说明**:

这个方法主要用于回答以下实验问题：
- **问题**: 文档融合真的有用吗？还是单个文档就够了？
- **方法**: 对比 `repeat_self` vs `bge`，如果两者效果接近，说明融合其他文档的收益有限

**配置示例**:
```bash
RECALL_METHOD="repeat_self"
```

---

### 4. 固定文档召回 (`fixed_doc`)

**描述**: 所有文档都召回同一个固定的文档 topK 次。

**适用场景**:
- **消融实验**: 测试召回多样性的重要性
- **极端 Baseline**: 最差的召回策略
- **噪声实验**: 测试无关文档的影响

**实现原理**:
1. 指定一个固定文档索引 `fixed_doc_idx`
2. 对所有文档，都召回这个固定文档 topK 次
3. 相当于用一个固定文档的信息"污染"所有文档

**优点**:
- ✅ 可以测试召回多样性的重要性
- ✅ 提供最差情况的 Baseline

**缺点**:
- ❌ 召回质量极差
- ❌ 无实际应用价值（仅用于消融）

**使用场景说明**:

这个方法主要用于回答以下实验问题：
- **问题**: 召回的文档多样性重要吗？
- **方法**: 对比 `fixed_doc` vs `random` vs `bge`，如果 `fixed_doc` 显著更差，说明多样性很重要

**配置示例**:
```bash
RECALL_METHOD="fixed_doc"
FIXED_DOC_IDX="0"  # 使用第 0 个文档
```

---

## 对比总结

| 召回方法 | 召回质量 | 计算开销 | 多样性 | 适用场景 |
|---------|---------|---------|--------|---------|
| **BGE** | ⭐⭐⭐⭐⭐ 最高 | 🔥🔥🔥 高 | ⭐⭐⭐⭐ 高 | 正常实验、最佳性能 |
| **Random** | ⭐⭐ 随机 | 🔥 低 | ⭐⭐⭐⭐⭐ 最高 | 消融实验、Baseline |
| **Repeat Self** | ⭐⭐⭐⭐ 理论最高 | 🔥 最低 | ⭐ 最低 | 测试融合必要性 |
| **Fixed Doc** | ⭐ 最差 | 🔥 最低 | ⭐ 最低 | 极端 Baseline |

---

## 使用方法

### 方法 1: 修改 Shell 脚本

编辑 `run_fusionrag.sh` 或 `run_fusionrag_sweep.sh`:

```bash
# 选择召回方法
RECALL_METHOD="bge"          # 或 random, repeat_self, fixed_doc
RANDOM_SEED="42"             # random 模式的种子
FIXED_DOC_IDX="0"            # fixed_doc 模式的文档索引
```

### 方法 2: 命令行参数

```bash
python test_fusionrag_reflect.py \
    --recall_method bge \
    --topk 10 \
    ...
```

或

```bash
python test_fusionrag_reflect.py \
    --recall_method random \
    --random_seed 42 \
    --topk 10 \
    ...
```

或

```bash
python test_fusionrag_reflect.py \
    --recall_method repeat_self \
    --topk 10 \
    ...
```

或

```bash
python test_fusionrag_reflect.py \
    --recall_method fixed_doc \
    --fixed_doc_idx 0 \
    --topk 10 \
    ...
```

---

## 实验设计建议

### 消融实验 1: 召回方法的重要性

**目标**: 测试不同召回方法对 FusionRAG 性能的影响

**实验组**:
1. BGE 相似度召回（最佳）
2. 随机召回（无语义）
3. 重复自身（无融合）
4. 固定文档（最差）

**预期结果**:
- BGE > Random > Repeat Self ≈ Fixed Doc
- 如果 BGE 显著优于 Random，说明语义相似度很重要
- 如果 Repeat Self 接近 BGE，说明融合收益有限

**配置**:
```bash
# 保持其他参数相同，只改变 RECALL_METHOD
RATE_LIST=(0.15 0.3 0.5)
TOPK="10"

# 运行 4 次，分别使用 4 种召回方法
for method in "bge" "random" "repeat_self" "fixed_doc"; do
    RECALL_METHOD="$method"
    ./run_fusionrag_sweep.sh
done
```

### 消融实验 2: TopK 的影响

**目标**: 测试融合文档数量的影响

**实验组**:
- TopK=5, 10, 20（使用 BGE 和 Random 对比）

**配置**:
```bash
for topk in 5 10 20; do
    for method in "bge" "random"; do
        TOPK="$topk"
        RECALL_METHOD="$method"
        ./run_fusionrag_sweep.sh
    done
done
```

### 消融实验 3: 文档融合的必要性

**目标**: 测试是否真的需要融合其他文档

**对比**:
- Repeat Self（只用自己）vs BGE（融合相似文档）

**预期结果**:
- 如果 BGE 显著优于 Repeat Self：融合有用
- 如果两者接近：单文档信息已足够，融合收益有限

---

## Cache 和结果目录命名

不同的召回方法会使用不同的 cache 和结果目录：

### Cache 目录

```
/mnt/data3/tmp/fusionrag/Qwen2.5-7B-Instruct/musique/
├── preprocess_kv_cache_global_topk10_bge/         # BGE 召回
├── preprocess_kv_cache_global_topk10_random/      # 随机召回
├── preprocess_kv_cache_global_topk10_repeat_self/ # 重复自身
└── preprocess_kv_cache_global_topk10_fixed_doc/   # 固定文档
```

### 结果目录

```
result/Qwen2.5-7B-Instruct/musique/
├── FusionRAG_global_topk10_bge/
│   ├── rate_0.15_revert_rope.csv
│   └── ...
├── FusionRAG_global_topk10_random/
│   ├── rate_0.15_revert_rope.csv
│   └── ...
├── FusionRAG_global_topk10_repeat_self/
│   ├── rate_0.15_revert_rope.csv
│   └── ...
└── FusionRAG_global_topk10_fixed_doc/
    ├── rate_0.15_revert_rope.csv
    └── ...
```

---

## 技术细节

### RecallMethod 枚举

```python
class RecallMethod(Enum):
    BGE = "bge"
    RANDOM = "random"
    REPEAT_SELF = "repeat_self"
    FIXED_DOC = "fixed_doc"
```

### 函数签名

```python
def prepare_reflect_data(
    ...,
    recall_method: RecallMethod = RecallMethod.BGE,
    random_seed: int = 42,
    fixed_doc_idx: int = 0,
    ...
)
```

### 向后兼容

保留了 `use_random_recall` 参数用于向后兼容：
```python
# 旧代码仍然可以工作
python test_fusionrag_reflect.py --use_random_recall true

# 新代码推荐使用
python test_fusionrag_reflect.py --recall_method random
```

---

## 常见问题

### Q1: 哪种召回方法最好？

**A**: 对于实际应用，**BGE 相似度召回**最好。其他方法主要用于消融实验。

### Q2: Repeat Self 和 Fixed Doc 有什么区别？

**A**:
- **Repeat Self**: 每个文档召回自己（相关性高，但无多样性）
- **Fixed Doc**: 所有文档召回同一个文档（相关性低，且无多样性）

### Q3: 为什么需要 Fixed Doc 这种极端方法？

**A**: 用于建立最差情况的 Baseline，帮助理解：
- 召回质量的下界
- 噪声文档的影响
- 多样性的重要性

### Q4: 不同召回方法的结果可以直接对比吗？

**A**: 可以，但要注意：
- 保持其他参数相同（rate, topk, reprocess_method 等）
- 使用相同的数据集和测试样本
- 最好在同一次实验中运行所有方法

### Q5: 如何选择 Fixed Doc 的文档索引？

**A**: 建议：
- 使用 `fixed_doc_idx=0`（第一个文档）
- 或者使用数据集中间的文档（如 `total_docs//2`）
- 实际上选哪个文档影响不大，重点是测试"固定召回"的影响

---

**创建时间**: 2026-01-14
**文档版本**: 1.0
