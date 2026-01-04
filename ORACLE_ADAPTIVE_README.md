# OracleAdaptive 综合多特征动态比例方法

## 更新说明

已将综合多特征动态比例算法集成到 `test_fusionrag_reflect.py` 的 `OracleAdaptive` 方法中。

## 核心改进

### 旧版本 (基于 vAttention 覆盖率方法)
- 使用 epsilon 控制累积覆盖率阈值（如 epsilon=0.1 → 90% 覆盖）
- Top-k budget + Random sampling budget
- 问题：可能遗漏 long-tail 重要信息

### 新版本 (综合多特征方法)
- **特征 1**: Coverage-based ratio (达到 85% attention 覆盖)
- **特征 2**: Connected components (高 attention 位置的连通分量数量)
- **特征 3**: Spread factor (高 attention 位置在文档中的跨度)
- **特征 4**: Gini coefficient (Attention 集中度)
- **Safety buffer**: 最小比例 20% (确保 long-tail)

### 设计原理

基于实际分析发现：
- 答案 "1216 and 1220" 刚好需要 30% 才能答对
- 答案 tokens 不一定有最高 attention
- 需要综合多个特征，不能只看单一指标（如覆盖率或熵）

## 使用方法

### 方法 1: 直接运行测试脚本

```bash
# 给脚本添加执行权限
chmod +x run_oracle_adaptive_test.sh

# 运行测试
./run_oracle_adaptive_test.sh
```

### 方法 2: 直接运行 Python

```bash
# 设置 GPU
export CUDA_VISIBLE_DEVICES=0

# 运行测试
python3 test_fusionrag_reflect.py
```

### 方法 3: 修改参数后运行

编辑 `test_fusionrag_reflect.py`，修改 main() 函数的参数：

```python
main(
    model_type='qwen',
    model_path='/mnt/data/models/Qwen2.5-7B-Instruct',
    data_path='./result_reflect.json',
    cache_path='/mnt/data/reflect/',
    model_name='Qwen2.5-7B-Instruct',
    rate=0.3,              # 作为 base_ratio 参考
    reprocess_method='OracleAdaptive',

    # 动态比例参数
    min_rate=0.20,         # 最小 20%（关键！确保 long-tail）
    max_rate=0.50,         # 最大 50%

    max_samples=200,       # 测试样本数（可改为小值快速测试，如 10）
    # ...其他参数
)
```

## 输出说明

### 运行时输出

每个样例会输出详细的分析信息：

```
============================================================
OracleAdaptive: Oracle selection + Dynamic rate
============================================================
  Parameters: base_ratio=30%
  Budget range: [20.0%, 50.0%]

  Attention 分布特征分析:
    Coverage Analysis:
      80% coverage: 18.50%
      85% coverage: 22.30%
      90% coverage: 28.10%
    Connected Components: 5
    Position Span: 867 tokens
    Spread Ratio: 0.7234
    Gini Coefficient: 0.6234

  动态比例计算:
    Coverage-based ratio: 22.30%
    Adjustments:
      component: +0.0211
      spread: +0.0579
      gini: +0.0000
    Raw computed ratio: 30.10%
    FINAL DYNAMIC RATIO: 30%

  OracleAdaptive 选择结果:
    选中 360 tokens (30.0%)
    (smart_query_selection: 连通分量 + 边界扩展)
  select_time: 1.234s
```

### 结果文件

测试完成后会生成：
1. **CSV 文件**: `./result/reprocess_method_OracleAdaptive_*.csv`
   - 每行记录一个样例的详细结果

2. **统计文件**: `./result/reprocess_method_OracleAdaptive_*.txt`
   - 汇总统计：准确率、F1 分数等
   - 动态比例统计：每个样例的实际比例

3. **动态比例分布** (如果有统计):
   ```
   ============================================================
   OracleAdaptive 动态比例统计
   ============================================================
   样例数量: 200

   动态比例分布:
     Min: 20.0%
     25%: 25.0%
     50%: 30.0%
     75%: 35.0%
     Max: 45.0%
     Mean: 30.5%

   平均选择 tokens: 365.2
   平均 doc_len: 1200.5
   ```

## 参数调优建议

### min_rate（最小比例）
- **默认**: 0.20 (20%)
- **建议范围**: 0.15 - 0.25
- **过低风险**: < 0.15 可能遗漏 long-tail 答案
- **过高影响**: > 0.25 减少动态调整空间

### max_rate（最大比例）
- **默认**: 0.50 (50%)
- **建议范围**: 0.40 - 0.60
- **说明**: 很少触及上限，主要防止异常情况

### base_ratio（参考比例）
- **默认**: 0.30 (30%)
- **作用**: 用于对比，不直接影响计算
- **说明**: 实际比例由特征动态决定

## 对比其他方法

### vs Oracle (固定 30%)
```bash
# 修改 test_fusionrag_reflect.py，取消注释 Oracle 部分
# 对比：固定 30% vs 动态 20-50%
```

### vs OracleDynamic (vAttention 覆盖率方法)
- OracleDynamic: 基于 epsilon 覆盖率 + CLT 采样
- OracleAdaptive: 综合多特征（更全面）

## 预期效果

基于设计原理，OracleAdaptive 应该：
1. **集中分布样例**: 动态降低到 20-25%，节省计算
2. **分散分布样例**: 动态提高到 35-40%，确保准确性
3. **中等分布样例**: 保持 25-35%，接近固定 30%

与固定 30% 相比：
- 平均准确率：**相当或略高**（因为更好适应分布）
- 平均 token 选择：**略低**（集中样例节省了计算）
- 鲁棒性：**更好**（适应不同分布特征）

## 故障排查

### 1. 导入错误
```
ModuleNotFoundError: No module named 'torch'
```
**解决**: 确保激活了正确的虚拟环境

### 2. GPU 内存不足
```
RuntimeError: CUDA out of memory
```
**解决**:
- 减少 `max_samples`
- 使用更少的 GPU: `export CUDA_VISIBLE_DEVICES=0`

### 3. 未找到 KV cache
```
FileNotFoundError: /mnt/data/reflect/...
```
**解决**: 确保已经预先生成了 KV cache

## 修改记录

### 文件修改
1. **`ktransformers/util/utils.py`**:
   - 新增函数: `compute_dynamic_ratio_comprehensive()`
   - 修改: OracleAdaptive 部分，使用新算法

2. **`test_fusionrag_reflect.py`**:
   - 更新: OracleAdaptive 配置参数
   - 更新: 注释说明

### 关键变化
- `min_rate`: 0.05 → 0.20 (提高 safety buffer)
- 动态计算: vAttention 覆盖率方法 → 综合多特征方法
- 新增统计: num_components, gini_coefficient, spread_ratio

## 参考

- 设计文档: `DYNAMIC_RATIO_README.md`
- 算法实现: `compute_dynamic_ratio.py`
- 理论分析: `analyze_oracle_selection.py`
- 可视化: `attention_heatmap.png`, `oracle_analysis/dynamic_selection_problem.png`
