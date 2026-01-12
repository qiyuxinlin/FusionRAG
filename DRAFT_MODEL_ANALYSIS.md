# 3B vs 7B DraftModel Token 选择分析报告

## 1. 背景

在 FusionRAG 的 DraftModel 方法中，我们使用小模型（3B）的 attention 来指导大模型（7B）的 token 重算选择。实验发现，在 rate=0.2（仅重算 20% 的 tokens）时，3B 和 7B 作为 draft model 的效果差距明显：

| Draft Model | Rate=0.2 准确率 | Rate=0.3 准确率 |
|-------------|-----------------|-----------------|
| 7B          | 86.40%          | 88.40%          |
| 3B          | 83.60%          | 86.80%          |
| 1.5B        | -               | 86.00%          |
| Rate=1 基线 | 87.90%          | 87.90%          |

本文档记录了对 3B 和 7B draft model token 选择差异的深入分析。

---

## 2. 分析方法

### 2.1 差异案例筛选
从 200 个测试样本中，筛选出 7B draft model 答对但 3B draft model 答错的案例，共 9 个典型案例。

### 2.2 分析维度
对每个案例，从以下维度对比 3B 和 7B 的选择：
1. **层选择差异**：熵选层选择了哪些层
2. **Token 选择重叠率**：两个模型选择的 tokens 有多少重合
3. **Attention 分布特征**：Gini 系数（集中度）、最大值、均值
4. **答案覆盖率**：选中的 tokens 中有多少与标准答案相关

---

## 3. 关键发现

### 3.1 Token 选择位置偏好差异

**核心发现：3B 模型倾向于选择输入开头的格式化 tokens，而 7B 模型选择的 tokens 更加语义相关。**

具体表现：
- 3B 选择的 tokens 位置分布偏向文档开头（system prompt 后的格式化标记）
- 7B 选择的 tokens 更均匀地分布在文档的语义关键位置
- 3B 经常选中 `Document`、`\n\n`、标点符号等格式化 tokens
- 7B 更多选中实体名称、关键动词、关系描述等语义 tokens

**示例（案例 1）**：
```
问题: Who were the siblings of Alice de Lusignan, Countess of Surrey?
标准答案: Alice de Lusignan had a uterine half-brother, King Henry III of England.

7B 预测 (正确): King Henry III of England
3B 预测 (错误): John of England and Joan de Geneville, 2nd Baroness of March

仅 7B 选择的 tokens: 'Henry' 'III' 'England' 'half' 'brother'
仅 3B 选择的 tokens: 'Document' '\n' 'title' ':' 'The'
```

### 3.2 Attention 分布集中度差异

**核心发现：7B 的 attention 分布更加集中，能够更准确地识别关键信息。**

| 指标 | 7B 平均值 | 3B 平均值 | 结论 |
|------|-----------|-----------|------|
| Gini 系数 | 0.85+ | 0.75-0.80 | 7B 更集中 |
| Attention 最大值 | 更高 | 较低 | 7B 峰值更明显 |
| Token 重叠率 | - | ~60-70% | 约 30-40% 选择不同 |

### 3.3 答案相关 Token 覆盖率差异

**核心发现：7B 对答案相关 tokens 的覆盖率显著高于 3B。**

典型案例统计：
```
案例 1 (Alice de Lusignan):
  答案相关 tokens: 15 个
  7B 覆盖: 12/15 (80.0%)
  3B 覆盖: 5/15 (33.3%)

案例 3 (Natalie Wood):
  答案相关 tokens: 8 个
  7B 覆盖: 6/8 (75.0%)
  3B 覆盖: 2/8 (25.0%)
```

### 3.4 层选择情况

两个模型的层数不同，但熵选层策略都倾向于选择中间层：
- **7B 模型**: 共 27 层，熵选层通常选择中间层（如 layer 12-16 附近）
- **3B 模型**: 共 36 层，熵选层同样选择中间层（如 layer 16-20 附近）

熵选层策略在两个模型上都正常工作，选择的是 attention 分布最集中（熵最低）的层。层选择本身不是导致性能差异的主要原因。

---

## 4. 问题根因分析

### 4.1 3B 模型的 Attention 偏差

3B 模型由于参数量较小，其 attention 机制存在以下问题：

1. **格式敏感性过高**：对文档格式标记（如 `Document`、换行符）分配了过高的 attention 权重
2. **语义理解不足**：无法准确识别与查询语义相关的关键 tokens
3. **长距离依赖弱**：在长文档中，3B 更容易"遗忘"前面的关键信息

### 4.2 根本原因

**模型能力差距**：3B 模型的表示能力有限，其 attention 分布无法准确反映 token 的语义重要性。7B 模型由于参数量更大，能够更好地理解查询意图并定位相关信息。

---

## 5. 改进尝试与结果

### 5.1 尝试方案：相似度重排序

**思路**：用 Query-Token 语义相似度来修正 3B 的 attention 偏差

**实现方法**：
1. 使用 3B 模型中间层的 hidden states 计算 query 和 doc tokens 的语义相似度
2. 先用 `smart_query_selection` 选择 2x 候选（保留连通分量分析）
3. 在候选中按相似度重排序，选择最相关的 tokens

**结果**：
| 方法 | Rate=0.2 准确率 |
|------|-----------------|
| 原始 3B | 83.60% |
| 3B + 相似度重排序 | 74.40% |

**失败原因分析**：
1. 3B 模型的 hidden states 质量不足以准确计算语义相似度
2. 相似度重排序丢弃了 `smart_query_selection` 选中的边界 tokens
3. 离线分析的答案覆盖率提升没有转化为实际生成质量提升

### 5.2 其他可能方向

以下方向尚未尝试，可能有效：

1. **层融合策略改进**：
   - 不仅使用熵选层，还考虑层间一致性
   - 给不同层分配不同权重

2. **多模型集成**：
   - 同时使用 1.5B 和 3B 的 attention 投票
   - 只选择两个模型都认为重要的 tokens

3. **位置惩罚**：
   - 对文档开头的 tokens 施加惩罚
   - 避免选择格式化标记

4. **Query-aware 选择**：
   - 先用 query embedding 过滤明显无关的 tokens
   - 然后再用 attention 排序

---

## 6. 结论

### 6.1 主要结论

1. **3B 和 7B 的 attention 质量存在本质差距**：3B 模型的 attention 无法准确识别语义关键 tokens
2. **3B 倾向选择格式化 tokens**：而 7B 选择的是语义相关的内容 tokens
3. **答案覆盖率差距显著**：7B 覆盖 75-80% 的答案相关 tokens，3B 仅覆盖 25-35%
4. **简单的后处理难以弥补这一差距**：相似度重排序等方法无法有效改进

### 6.2 实用建议

在资源允许的情况下：
- **推荐使用 7B 作为 draft model**：在 rate=0.3 时可达到 88.40% 准确率，超过 rate=1 基线
- **如果必须使用 3B**：建议 rate 不低于 0.3，以确保足够的 token 覆盖

---

## 7. 附录：分析脚本

相关分析脚本：
- `analyze_token_selection_diff.py`：3B vs 7B token 选择对比分析
- `improved_draft_v2.py`：相似度重排序改进方案的离线测试
- `diff_cases_7b_vs_3b.json`：7B 答对但 3B 答错的差异案例

---

*文档生成日期：2026-01-05*
