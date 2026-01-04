# 动态重算比例功能说明

## 功能概述

基于 draft model 的 attention 分布，自动计算每个测试样例的最优重算比例，替代固定的 30% 比例。

## 核心思想

### 为什么需要动态比例？

从实际测试发现，固定 30% 的重算比例刚好能让某个示例答对（正确答案是 "1216 and 1220"），但：

1. **不同样例的 attention 分布差异很大**
   - 有些样例 attention 非常集中（少数 tokens 很重要）
   - 有些样例 attention 很分散（需要更多 tokens）

2. **固定比例的问题**
   - 对于集中的 attention：30% 可能浪费计算
   - 对于分散的 attention：30% 可能不够

3. **Long-tail 问题**
   - 答案 tokens 不一定有最高 attention
   - 单纯用累积覆盖率（如 90%）可能遗漏关键信息
   - 需要 safety buffer 确保捕获 long-tail tokens

### 解决方案

我们的动态比例方法综合考虑：

1. **Coverage-based ratio**: 达到 85% attention 覆盖率所需的比例
2. **Connected components**: 高 attention 位置的分散程度（连通分量数量）
3. **Spread factor**: 高 attention 位置在文档中的跨度
4. **Gini coefficient**: Attention 的集中度（0=均匀，1=集中）
5. **Safety buffer**: 最小比例 (默认 20%) 确保不遗漏 long-tail

## 使用方法

### 方法 1: 使用固定比例（原方法）

```python
from per_head_generation import main_with_draft_model

results = main_with_draft_model(
    target_model_path='/mnt/data/models/Qwen2.5-7B-Instruct',
    draft_model_path='/mnt/data/models/Qwen2.5-3B-Instruct',
    data_path='./result_reflect.json',
    example_idx=4,
    sub_question_idx=1,
    total_ratio=0.3,           # 固定 30%
    use_dynamic_ratio=False,   # 不使用动态比例
    output_path='./results/fixed_30.json'
)
```

### 方法 2: 使用动态比例（新方法）

```python
from per_head_generation import main_with_draft_model

results = main_with_draft_model(
    target_model_path='/mnt/data/models/Qwen2.5-7B-Instruct',
    draft_model_path='/mnt/data/models/Qwen2.5-3B-Instruct',
    data_path='./result_reflect.json',
    example_idx=4,
    sub_question_idx=1,
    total_ratio=0.3,           # 作为 base_ratio，用于参考
    use_dynamic_ratio=True,    # 启用动态比例
    min_ratio=0.20,            # 最小比例 20%
    max_ratio=0.50,            # 最大比例 50%
    output_path='./results/dynamic.json'
)
```

### 运行测试脚本

```bash
# 对比固定比例和动态比例的效果
python3 test_dynamic_ratio.py
```

## 参数说明

### `use_dynamic_ratio`
- `False` (默认): 使用固定的 `total_ratio`
- `True`: 基于 attention 分析动态计算比例

### `total_ratio`
- 在固定模式下：直接使用此比例
- 在动态模式下：作为 `base_ratio`，用于对比和参考

### `min_ratio`
- 动态比例的最小值（默认 0.20）
- 确保即使 attention 非常集中，也有足够的 safety buffer
- **关键**：太低会遗漏 long-tail 重要信息

### `max_ratio`
- 动态比例的最大值（默认 0.50）
- 防止动态比例过高，浪费计算

## 算法细节

### 动态比例计算公式

```
1. 计算 coverage_based_ratio (达到 85% attention 覆盖)
2. 分析连通分量：
   component_adjustment = normalized_components * 0.10
3. 分析分散度：
   spread_adjustment = spread_ratio * 0.08
4. 分析集中度 (Gini):
   gini_adjustment = -0.03 (if gini > 0.7, 高度集中)
                   = +0.03 (if gini < 0.5, 分散)
                   = 0     (otherwise)
5. 综合计算：
   ratio = coverage_based_ratio + component_adjustment + spread_adjustment + gini_adjustment
6. 应用约束：
   ratio = max(ratio, min_ratio)
   ratio = min(ratio, max_ratio)
7. 四舍五入到 0.05 倍数
```

### 为什么这样设计？

#### 1. 使用 85% coverage 而不是 90% 或 95%
- **原因**：更高的覆盖率（如 95%）在 attention 集中时可能只需 5-10% tokens
- **问题**：会遗漏 long-tail 但重要的 tokens（如答案本身）
- **解决**：85% coverage + adjustments + min_ratio (20%) 提供更好的平衡

#### 2. 连通分量调整
- **原理**：如果重要信息分散在多个位置（多个连通分量），需要更高比例
- **示例**：答案在文档中有 4 处出现，每处是一个连通分量
- **调整**：最多增加 10%

#### 3. Spread 调整
- **原理**：高 attention 位置跨度大 → 信息分散 → 需要更多 tokens
- **调整**：最多增加 8%

#### 4. Gini 调整
- **保守设计**：即使 Gini 很高（很集中），也只减少 3%
- **原因**：集中不代表可以忽略其他信息，需要谨慎

#### 5. Safety Buffer (min_ratio = 20%)
- **关键设计**：确保始终有足够的覆盖
- **对比原来的 5%**：提高到 20% 是基于分析发现，答案 tokens 往往在 top 20-30% 范围内

## 实验结果说明

运行 `test_dynamic_ratio.py` 后，你会看到：

```
==============================================
COMPARISON: Fixed vs Dynamic Ratio
==============================================

Configuration:
  Fixed ratio:    30%
  Dynamic ratio:  25%  (示例，实际会根据 attention 分布计算)

Token Selection:
  Fixed:   390 tokens (30.0%)
  Dynamic: 325 tokens (25.0%)

Dynamic Ratio Analysis Details:
  Coverage Analysis:
    80% coverage: 18.5%
    85% coverage: 22.3%
    90% coverage: 28.1%

  Distribution Features:
    Connected components: 5
    Spread ratio: 0.7234
    Gini coefficient: 0.6234

  Adjustments Applied:
    component: +0.0211
    spread: +0.0579
    gini: +0.0000

  Final Ratio: 25%

SUMMARY
✓ Both methods produced CORRECT answers
  Dynamic ratio saved 16.7% tokens while maintaining correctness!
```

## 注意事项

1. **最小比例设置**
   - 建议不低于 15-20%
   - 太低可能遗漏 long-tail 重要信息

2. **不同数据集可能需要调整**
   - 当前参数基于 2WikiMQA 数据集调优
   - 其他数据集可能需要调整权重和阈值

3. **性能开销**
   - 动态比例计算本身很快（< 1秒）
   - 主要开销仍然是 draft model prefill

## 文件说明

- `compute_dynamic_ratio.py`: 动态比例计算模块
- `per_head_generation.py`: 主要生成代码（已集成动态比例）
- `test_dynamic_ratio.py`: 测试脚本
- `analyze_oracle_selection.py`: 理论分析（为什么动态比例有效）

## 未来改进方向

1. **自适应权重**：根据数据集特征自动调整 component/spread/gini 的权重
2. **多样例学习**：从多个样例学习最优参数
3. **分层动态比例**：不同层使用不同的动态比例
4. **Position-aware selection**：考虑答案位置的上下文扩展

## 参考

基于以下分析设计：
- `analyze_oracle_selection.py`: 为什么固定 30% 有效的理论分析
- `attention_heatmap.png`: 实际 attention 分布可视化
- `oracle_analysis/dynamic_selection_problem.png`: 问题示意图
