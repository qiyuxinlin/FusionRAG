# Learn Rate Sweep 实验指南

## 概述

`run_fusionrag_learn_rate_sweep.sh` 是专门设计用于研究**用少量样本学习的steering vectors在完整数据集上的泛化能力**的实验脚本。

## 实验设计

### 核心问题

**能否用少量样本计算的steering vectors，在完整数据集上取得好的效果？**

### 实验方法

1. **统计阶段（Learning）**: 使用 `LEARN_RATE%` 的样本计算steering vectors
2. **测试阶段（Testing）**: 在**完整数据集**上评估性能

### 与其他脚本的区别

| 脚本 | 统计样本 | 测试样本 | 适用场景 |
|------|---------|---------|---------|
| `run_fusionrag_steering_muliti.sh` | 全部或指定 | 全部 | 跨数据集实验 |
| `run_fusionrag_learn_rate_sweep.sh` | Learn_rate% | 100% | 同数据集泛化实验 |

## 配置参数

### 核心参数

```bash
# 数据集配置（同数据集实验）
DATASET="musique"                        # 数据集名称
TEST_DATA_PATH="./data/result_reflect.json"  # 测试数据路径

# Learn Rate 扫描配置（最重要！）
LEARN_RATE_LIST=(10 20 50 100)  # 学习样本比例列表（百分比）
# 10  = 用10%样本学习，在100%样本上测试
# 20  = 用20%样本学习，在100%样本上测试
# 50  = 用50%样本学习，在100%样本上测试
# 100 = 用100%样本学习，在100%样本上测试（baseline）

# Steering 强度扫描
STEERING_ALPHA_LIST=(0.3 0.5 0.7 1.0)

# PCA 配置
PCA_VARIANCE_THRESHOLD="0.7"
PER_HEAD_STATS="false"  # 建议先用false快速测试
```

### 结果目录配置

```bash
BASE_RESULT_DIR="/home/shm/document/exp/FusionRAG/result/learn_rate_sweep"
```

结果会按以下结构组织：
```
result/learn_rate_sweep/
├── learn_rate_10pct/      # 用10%样本学习
│   ├── alpha_0.3/
│   ├── alpha_0.5/
│   ├── alpha_0.7/
│   └── alpha_1.0/
├── learn_rate_20pct/      # 用20%样本学习
│   └── ...
├── learn_rate_50pct/      # 用50%样本学习
│   └── ...
└── learn_rate_100pct/     # 用100%样本学习（baseline）
    └── ...
```

## 使用方法

### 快速开始

```bash
# 1. 确保数据集KV cache已生成
ls /mnt/data3/tmp/fusionrag/Qwen2.5-7B-Instruct/musique/kv_cache/

# 2. 编辑配置（可选）
vim run_fusionrag_learn_rate_sweep.sh

# 3. 运行实验
bash run_fusionrag_learn_rate_sweep.sh
```

### 自定义实验

#### 实验1: 快速验证（最小配置）

```bash
# 只测试2个learn_rate，2个alpha
LEARN_RATE_LIST=(20 100)  # 20%样本 vs 100%样本
STEERING_ALPHA_LIST=(0.5 1.0)  # 只测试2个强度
RATE_LIST=(0.0)

# 总测试数: 2 × 2 × 1 = 4次
```

#### 实验2: 详细泛化研究

```bash
# 测试多个learn_rate点
LEARN_RATE_LIST=(5 10 15 20 30 50 75 100)
STEERING_ALPHA_LIST=(0.3 0.5 0.7 1.0)
RATE_LIST=(0.0)

# 总测试数: 8 × 4 × 1 = 32次
```

#### 实验3: 与PCA阈值联合实验

```bash
# 测试不同PCA阈值对泛化的影响
# 第一轮: PCA=0.5
PCA_VARIANCE_THRESHOLD="0.5"
LEARN_RATE_LIST=(10 20 50 100)
bash run_fusionrag_learn_rate_sweep.sh

# 第二轮: PCA=0.7
PCA_VARIANCE_THRESHOLD="0.7"
LEARN_RATE_LIST=(10 20 50 100)
bash run_fusionrag_learn_rate_sweep.sh

# 第三轮: PCA=0.9
PCA_VARIANCE_THRESHOLD="0.9"
LEARN_RATE_LIST=(10 20 50 100)
bash run_fusionrag_learn_rate_sweep.sh
```

## 工作流程

### 阶段1: 初始化

1. 检查数据集KV cache是否存在
2. 自动统计可用样本总数（如musique有100个样本）
3. 计算每个learn_rate需要的样本数
   - `LEARN_RATE=10` → 需要10个样本
   - `LEARN_RATE=20` → 需要20个样本
   - `LEARN_RATE=100` → 需要100个样本

### 阶段2: 对每个Learn Rate

#### Step 0: 计算Steering Vectors（学习阶段）

```bash
# 假设 LEARN_RATE=20, 总样本数=100
# → 使用前20个样本（ID: 0-19）

python compute_kv_distribution_stats.py \
    --cache_dir /mnt/data3/tmp/fusionrag \
    --dataset musique \
    --sample_ids 0 1 2 3 ... 19 \  # 前20个
    --output_path musique_learn20of100_20pct_manifold_pca7.pt
```

**文件命名格式**:
```
{dataset}_learn{used_samples}of{total_samples}_{learn_rate}pct_{suffix}.pt

示例:
musique_learn10of100_10pct_manifold_pca7.pt  # 用10/100样本
musique_learn20of100_20pct_manifold_pca7.pt  # 用20/100样本
musique_learn100of100_100pct_manifold_pca7.pt  # 用100/100样本
```

#### Step 1-N: 测试（在完整数据集上）

```bash
# 使用learn_rate=20计算的steering vectors
# 但在全部100个样本上测试

for alpha in 0.3 0.5 0.7 1.0; do
    python test_fusionrag_reflect.py \
        --data_path ./data/result_reflect.json \  # 完整测试集
        --kv_stats_path musique_learn20of100_20pct_manifold_pca7.pt \
        --steering_alpha ${alpha}
        # 会在所有100个样本上测试！
done
```

## 输出示例

### 运行日志

```bash
######################################################################
#                 FusionRAG Learn Rate Sweep 实验
######################################################################

实验目标: 研究用少量样本学习的steering vectors的泛化能力

时间: 2026-01-17 18:00:00
GPU: 7
模型: Qwen2.5-7B-Instruct
数据集: musique
方法: FusionRAG

实验配置:
  Learn Rate 列表: 10 20 50 100% (用于计算steering vectors的样本比例)
  Alpha 列表: 0.3 0.5 0.7 1.0 (steering强度)
  Rate 列表: 0.0 (KV cache压缩率)
  PCA 阈值: 0.7
  Per-head 统计: false

==========================================
检查数据集可用样本...
==========================================
✓ 数据集 musique 可用样本数: 100

将进行 16 次测试:
  4 个 Learn Rates × 4 个 Alphas × 1 个 Rates

######################################################################
# Learn Rate = 10% (用10%样本学习，在100%样本上测试)
######################################################################
时间: 2026-01-17 18:01:00

样本配置:
  总样本数: 100
  Learn Rate: 10%
  学习样本数: 10 (前10个样本用于计算steering vectors)
  测试样本数: 100 (在完整数据集上测试)

==========================================
Step 0: 计算 Steering Vectors
==========================================
数据集: musique
模型: Qwen2.5-7B-Instruct
学习样本数: 10 / 100 (10%)

输出路径: /mnt/data3/tmp/fusionrag/kv_stats/musique_learn10of100_10pct_manifold_pca7.pt

开始计算 steering vectors...

使用样本ID: 0 1 2 3 4 5 6 7 8 9

  Manifold projection: 启用 (PCA 阈值: 0.7)
  Per-head statistics: 禁用

✓ Steering vectors 计算成功！
  保存位置: /mnt/data3/tmp/fusionrag/kv_stats/musique_learn10of100_10pct_manifold_pca7.pt

------------------------------------------------------------
测试: Learn Rate=10%, Alpha=0.3
时间: 2026-01-17 18:05:00
------------------------------------------------------------

✓ Learn Rate 10%, Alpha 0.3, Rate 0.0 测试完成

[... 继续测试其他alpha值 ...]

######################################################################
✓ Learn Rate 10% 的所有测试完成！
  结果保存在: result/learn_rate_sweep/learn_rate_10pct/
######################################################################

[... 继续测试其他learn_rate值 ...]
```

### 最终总结

```bash
========================================================================
                       所有测试完成！
========================================================================
时间: 2026-01-17 20:00:00

实验总结:
  数据集: musique
  总样本数: 100
  Learn Rate 列表: 10 20 50 100%
  Alpha 列表: 0.3 0.5 0.7 1.0
  Rate 列表: 0.0
  总测试数: 16

存储位置:
  KV Cache: /mnt/data3/tmp/fusionrag/Qwen2.5-7B-Instruct/musique/
  Steering Vectors: /mnt/data3/tmp/fusionrag/kv_stats/
  Results: result/learn_rate_sweep/

结果目录结构:
  Learn Rate 10%: result/learn_rate_sweep/learn_rate_10pct/
    ├─ Alpha 0.3: learn_rate_10pct/alpha_0.3/
    ├─ Alpha 0.5: learn_rate_10pct/alpha_0.5/
    ├─ Alpha 0.7: learn_rate_10pct/alpha_0.7/
    └─ Alpha 1.0: learn_rate_10pct/alpha_1.0/
  Learn Rate 20%: result/learn_rate_sweep/learn_rate_20pct/
    └─ ...
  Learn Rate 50%: result/learn_rate_sweep/learn_rate_50pct/
    └─ ...
  Learn Rate 100%: result/learn_rate_sweep/learn_rate_100pct/
    └─ ...

Steering Vectors 文件:
  Learn Rate 10% (10 samples):
    /mnt/data3/tmp/fusionrag/kv_stats/musique_learn10of100_10pct_manifold_pca7.pt
  Learn Rate 20% (20 samples):
    /mnt/data3/tmp/fusionrag/kv_stats/musique_learn20of100_20pct_manifold_pca7.pt
  Learn Rate 50% (50 samples):
    /mnt/data3/tmp/fusionrag/kv_stats/musique_learn50of100_50pct_manifold_pca7.pt
  Learn Rate 100% (100 samples):
    /mnt/data3/tmp/fusionrag/kv_stats/musique_learn100of100_100pct_manifold_pca7.pt
========================================================================
```

## 结果分析

### 查看结果

```bash
# 查看所有结果文件
ls -lh result/learn_rate_sweep/learn_rate_*/alpha_*/Qwen2.5-7B-Instruct/musique/FusionRAG_*/rate_0.0_revert_rope.txt

# 对比不同learn_rate的性能
for lr in 10 20 50 100; do
    echo "Learn Rate ${lr}%:"
    tail -n 5 result/learn_rate_sweep/learn_rate_${lr}pct/alpha_1.0/Qwen2.5-7B-Instruct/musique/FusionRAG_*/rate_0.0_revert_rope.txt
done
```

### 预期发现

#### 1. 泛化能力曲线

绘制 learn_rate vs 性能 曲线：

```
性能
 ^
 |     ________________  ← 100% baseline
 |    /
 |   /
 |  /
 | /
 |/_____|_____|_____|_____|→ Learn Rate
      10%   20%   50%  100%
```

**关键问题**：
- 性能何时饱和？（如20%就接近100%的性能）
- 最小有效样本数是多少？（如至少需要10%）

#### 2. 样本效率

计算 **每个样本的边际收益**：

```python
# 如果：
# 10样本 → 60分
# 20样本 → 75分
# 50样本 → 85分
# 100样本 → 88分

# 边际收益：
# 10→20样本: +15分 / 10样本 = 1.5分/样本
# 20→50样本: +10分 / 30样本 = 0.33分/样本
# 50→100样本: +3分 / 50样本 = 0.06分/样本

# 结论：前20个样本最有价值！
```

#### 3. 不同alpha的泛化

对比不同alpha在不同learn_rate下的表现：

```bash
# alpha=0.3: 可能对样本数不敏感（泛化好）
# alpha=1.0: 可能需要更多样本（容易过拟合）
```

## 高级用法

### 与Per-Head联合实验

```bash
# 测试per-head模式的泛化能力
PER_HEAD_STATS="true"
LEARN_RATE_LIST=(10 20 50 100)

# 预期：per-head可能需要更多样本才能泛化
```

### 多数据集对比

```bash
# 第一轮：musique
DATASET="musique"
bash run_fusionrag_learn_rate_sweep.sh

# 第二轮：2wikimqa
DATASET="2wikimqa"
TEST_DATA_PATH="./data/2wikimqa_reflect.json"
bash run_fusionrag_learn_rate_sweep.sh

# 对比两个数据集的泛化曲线
```

### 与跨数据集泛化对比

```bash
# 实验A: 同数据集泛化（本脚本）
# musique 10样本 → musique 100样本测试

# 实验B: 跨数据集泛化（用run_fusionrag_steering_muliti.sh）
# 2wikimqa 全部样本 → musique 100样本测试

# 对比：哪个泛化更难？
```

## 故障排查

### 问题1: 找不到KV cache

```
✗ 错误: 未找到 musique 的KV cache文件
```

**解决**: 先运行数据预处理生成KV cache

### 问题2: 样本数计算错误

```bash
# 检查实际可用样本数
ls /mnt/data3/tmp/fusionrag/Qwen2.5-7B-Instruct/musique/kv_cache/*_1_key.pt | wc -l
```

### 问题3: Steering vectors文件冲突

不同learn_rate的文件名不同，不会冲突：
```
musique_learn10of100_10pct_manifold_pca7.pt  # 10%
musique_learn20of100_20pct_manifold_pca7.pt  # 20%
```

## 总结

### 这个脚本做什么？

研究**用少量样本学习的steering vectors能否在完整数据集上取得好效果**。

### 为什么重要？

1. **降低计算成本**: 如果10%样本就够用，可以节省90%的统计时间
2. **理解泛化能力**: 验证steering vectors学到了什么通用特征
3. **指导实践**: 确定最小有效样本数

### 核心优势

✅ **自动化**: 自动计算样本数，生成文件名
✅ **可重复**: 文件命名包含完整配置信息
✅ **易分析**: 结果按learn_rate组织，便于对比
✅ **灵活**: 支持与其他参数（alpha, PCA, per-head）联合实验

### 下一步

运行实验后，绘制泛化曲线，找到最优的learn_rate！
