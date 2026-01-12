# 为什么无法通过 Draft Attention 分布动态确定重算比例

## 问题背景

在 FusionRAG 的 token 重算策略中，我们尝试使用 Draft Model 的 attention 分布来动态决定每个问题需要重算多少比例的 token。直觉上，这个想法很有吸引力：

- 如果 attention 集中在少数 token 上 → 只需重算少量 token
- 如果 attention 分散在很多 token 上 → 需要重算更多 token

然而，实验表明这个思路存在根本性问题。

---

## 核心问题：Attention 分布的长尾特性导致 Coverage 需求不可控

### 1. 数据验证

我们分析了 44 个样本的 attention 分布特征：

| Coverage 目标 | 最小需求 | 平均需求 | 最大需求 | 超过 30% 预算的样本 |
|--------------|---------|---------|---------|-------------------|
| 覆盖 90% attention | 15.8% | 28.9% | 41.9% | **45% (20/44)** |
| 覆盖 95% attention | 27.0% | 43.4% | 59.3% | **98% (43/44)** |

### 2. 问题本质

假设我们采用"覆盖 attention 前 90%"作为动态 rate 的依据：

```
样本 A: 达到 90% coverage 需要 16% token → rate = 0.16
样本 B: 达到 90% coverage 需要 42% token → rate = 0.42
```

这意味着：
- **波动范围过大**：同样的 coverage 目标，不同样本的 rate 可以相差 2.6 倍
- **经常超出预算**：如果设定 max_rate = 30%，则 45% 的样本会超标

### 3. OracleDynamic 方法的实际表现

OracleDynamic 方法正是基于这个思路实现的（参考 vAttention 论文）：

```python
# 理论：根据 coverage 动态计算 rate
coverage_threshold = 1.0 - epsilon  # 如 ε=0.1 → 覆盖 90%
dynamic_rate = (topk_budget + random_budget) / doc_len

# 现实：必须强制裁剪，否则会超出预算
dynamic_rate = max(min_rate, min(max_rate, dynamic_rate))
```

裁剪后的分布：
- 被裁剪到 max_rate=30%：**20 个样本 (45%)**
- 在 [5%, 30%] 范围内：24 个样本 (55%)
- 被裁剪到 min_rate=5%：0 个样本

**结论**：近一半样本被强制 clip 到上限，所谓"动态"实际上退化为固定 rate。

---

## 根本原因分析

### 1. Attention 分布的长尾特性

Attention 分布通常呈现幂律分布（power-law）：
- 少数 token 占据大部分 attention 权重
- 但剩余的"长尾"token 虽然单个权重小，累计起来却很可观

这导致：
- 覆盖前 50% attention 只需约 4% 的 token
- 覆盖前 90% attention 需要约 29% 的 token
- 覆盖前 95% attention 需要约 43% 的 token

从 90% 到 95%，额外 5% 的 coverage 却需要额外 14% 的 token！

### 2. Coverage ≠ 信息覆盖

即使我们能精确控制 coverage，也存在更深层的问题：

**Attention 覆盖 ≠ 关键信息覆盖**

实验发现：
- cov_85（达到 85% coverage 需要的 token 比例）在"答对组"和"答错组"之间几乎没有差异
- Cohen's d = 0.16，属于"几乎无区分能力"
- 两组的 cov_85 均值差异仅 0.007（0.206 vs 0.213）

这说明：attention 集中程度与是否能答对问题没有直接关系。

### 3. 信息论的根本限制

要判断是否需要更高的 rate，我们需要知道"关键 token 是否被选中"。

但要知道哪些是关键 token，我们需要知道正确答案。

而知道正确答案后，就不需要预测 rate 了！

```
预测 rate → 需要知道关键 token → 需要知道正确答案 → 不需要预测 rate
         ↑___________________________|
              信息论死结
```

---

## 结论

1. **Coverage-based 动态 rate 不可行**：attention 长尾特性导致 coverage 需求波动过大（16% ~ 42%），无法作为可控的 rate 依据

2. **Attention 分布无法预测答案质量**：cov_85 等指标在"答对/答错"两组之间几乎无差异，说明 attention 集中程度与答案正确性无关

3. **存在信息论限制**：要预测最优 rate，需要知道哪些 token 包含正确答案，但这正是我们试图通过重算来获取的信息

4. **实用建议**：
   - 放弃基于 attention 分布的动态 rate 策略
   - 采用固定 rate 或逐层递减（layerwise decay）等更可控的方案
   - 如果需要动态调整，考虑基于问题类型、文档长度等外部特征，而非 attention 内部分布
