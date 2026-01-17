# Steering Alpha 参数实现总结

## 概述

成功添加了 `--steering_alpha` 命令行参数，用于控制 steering vector 的强度系数（α）。

## 实现的公式

**Steering Vector 干预公式**：
```
h' = h + α * r_M
```

其中：
- `h`: 原始 KV cache
- `r_M`: Manifold 投影后的 steering vector
- `α`: 可调节的强度系数（新增参数）
- `h'`: 修正后的 KV cache

## 代码修改

### 1. test_fusionrag_reflect.py

#### 1.1 添加命令行参数 (第 3064-3065 行)
```python
parser.add_argument('--steering_alpha', type=float, default=1.0,
                    help='Steering vector strength (alpha) for no_preprocess_with_bias method (default: 1.0)')
```

#### 1.2 更新 main() 函数签名 (第 1324 行)
```python
def main(
    ...
    kv_stats_path=None,
    steering_alpha=1.0,  # 新增参数
    ...
):
```

#### 1.3 参数传递 (第 3170 行)
```python
main(
    ...
    kv_stats_path=args.kv_stats_path,
    steering_alpha=args.steering_alpha,  # 从命令行传入
    ...
)
```

#### 1.4 使用参数 (第 2057 行)
```python
# 旧代码：
# alpha = 1.0  # Steering strength (can be tuned)

# 新代码：
alpha = steering_alpha  # Use parameter from command line
```

### 2. run_cross_dataset_experiment.sh

#### 2.1 添加配置变量 (第 46 行)
```bash
STEERING_ALPHA="1.0"  # Steering vector strength (α): 1.0 = full steering, <1.0 = weaker, >1.0 = stronger
```

#### 2.2 在实验 2 中使用 (第 191 行)
```bash
${PYTHON_PATH} test_fusionrag_reflect.py \
    ...
    --kv_stats_path "${STEERING_NO_PROJ}" \
    --steering_alpha "${STEERING_ALPHA}" \
    ...
```

#### 2.3 在实验 3 中使用 (第 226 行)
```bash
${PYTHON_PATH} test_fusionrag_reflect.py \
    ...
    --kv_stats_path "${STEERING_MANIFOLD}" \
    --steering_alpha "${STEERING_ALPHA}" \
    ...
```

### 3. CROSS_DATASET_EXPERIMENT_README.md

#### 3.1 更新参数表格 (第 83 行)
```markdown
| Steering strength (α) | 1.0 (可调) |
```

#### 3.2 添加参数说明章节 (第 86-119 行)
详细解释了：
- Alpha 参数的作用
- 不同取值范围的效果
- 调参建议
- 超参数搜索示例

#### 3.3 更新使用示例 (第 261-282 行)
添加了：
- 基础使用示例（带 --steering_alpha 1.0）
- Alpha 参数调优的完整示例

## 参数说明

### Alpha 取值指南

| Alpha 值 | 效果 | 适用场景 |
|---------|------|---------|
| α < 1.0<br>(如 0.5, 0.8) | 更弱的干预<br>更保守的修正 | - 担心 steering vector 过度修改原始表示<br>- 需要保留更多原始信息 |
| α = 1.0<br>(默认) | 完全应用 steering vector | - 论文标准方法<br>- 首选配置 |
| α > 1.0<br>(如 1.2, 1.5) | 更强的干预<br>更激进的修正 | - 原始分布偏离理想状态较远<br>- 需要更大幅度修正 |

## 使用示例

### 基础使用
```bash
python test_fusionrag_reflect.py \
    --recall_method no_preprocess_with_bias \
    --kv_stats_path ./kv_stats/steering.pt \
    --steering_alpha 1.0 \
    ...
```

### Alpha 超参数搜索
```bash
# 方法 1: 在脚本中修改
STEERING_ALPHA="0.8"  # 修改配置变量
bash run_cross_dataset_experiment.sh

# 方法 2: 命令行循环测试
for ALPHA in 0.5 0.8 1.0 1.2 1.5; do
    python test_fusionrag_reflect.py \
        --steering_alpha ${ALPHA} \
        --result_path ./result/alpha_${ALPHA}/ \
        --kv_stats_path ./kv_stats/steering.pt \
        --recall_method no_preprocess_with_bias \
        ...
done
```

## 验证

### 命令行帮助信息
```bash
python test_fusionrag_reflect.py --help | grep -A 2 "steering_alpha"
```

输出：
```
  --steering_alpha STEERING_ALPHA
                        Steering vector strength (alpha) for
                        no_preprocess_with_bias method (default: 1.0)
```

### 运行时输出
程序会在 Step 2.5 显示使用的 alpha 值：
```
Step 2.5: Apply steering/bias to KV cache
  Using manifold steering vectors (α=1.0)
```

## 与论文的对应关系

**论文中的公式（Equation 1）**：
```
h' = h + α · r
```

**我们的实现（带 Manifold 投影）**：
```
h' = h + α · r_M
```

其中：
- `r = mean(h_bge) - mean(h_no_preprocess)` (原始 steering vector)
- `r_M = P_M @ r` (Manifold 投影后的 steering vector)
- `P_M = U_eff @ U_eff^T` (PCA 投影矩阵)

**关键区别**：
- 论文：在隐藏层表示上操作
- 我们：在 KV cache 上操作（迁移相同思想）

## 理论依据

根据论文 Theorem 4.1：
- **流形上的信号**：主要的、有用的信息（任务通用）
- **正交补空间的噪声**：数据集特定的干扰

通过调整 α：
- 控制从"过度思考"分布向"简洁"分布的修正程度
- 在保留原始信息和应用修正之间取得平衡

## 下一步实验建议

1. **基线对比**：α = 1.0（论文标准）
2. **弱干预**：α = 0.5, 0.8（保守修正）
3. **强干预**：α = 1.2, 1.5（激进修正）
4. **零干预**：α = 0.0（等同于 no_preprocess，验证作用）

观察不同 alpha 值对以下指标的影响：
- 准确率（Accuracy）
- 跨数据集泛化能力
- 与 BGE 方法的性能差距

## 完成状态

✅ **全部完成**：
- [x] 添加命令行参数
- [x] 更新函数签名
- [x] 参数传递链路
- [x] 实际应用逻辑
- [x] 实验脚本集成
- [x] 文档更新
- [x] 使用示例
- [x] 验证测试
