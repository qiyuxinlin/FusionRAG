# FusionRAG 实验脚本对比

## 三个主要脚本

### 1. `run_fusionrag_steering_muliti.sh` - 跨数据集实验

**用途**: 研究steering vectors在不同数据集间的泛化能力

**典型配置**:
```bash
TRAIN_DATASET="2wikimqa"     # 在2wikimqa上学习
TEST_DATASET="musique"       # 在musique上测试
USE_ALL_SAMPLES="true"       # 使用所有可用样本
STEERING_ALPHA_LIST=(0.1 0.2 ... 1.0)  # 遍历多个alpha
```

**实验问题**: 在数据集A上学习的steering vectors能否应用到数据集B？

**文件命名**:
```
2wikimqa_to_musique_steering_manifold_pca7.pt
```

**结果目录**:
```
result/steering/
├── alpha_0.1/
├── alpha_0.2/
└── ...
```

---

### 2. `run_fusionrag_steering_same_dataset.sh` - 同数据集实验

**用途**: 在同一数据集上学习和测试

**典型配置**:
```bash
TRAIN_DATASET="musique"      # 在musique上学习
TEST_DATASET="musique"       # 在musique上测试
USE_ALL_SAMPLES="true"       # 使用所有样本
STEERING_ALPHA_LIST=(0.1 0.2 ... 1.0)
```

**实验问题**: Steering vectors在同数据集上的最优效果是什么？

**文件命名**:
```
musique_steering_manifold_pca7.pt
```

**结果目录**:
```
result/steering_same_dataset/
├── alpha_0.1/
├── alpha_0.2/
└── ...
```

---

### 3. `run_fusionrag_learn_rate_sweep.sh` - 样本泛化实验（新）

**用途**: 研究用少量样本学习的steering vectors在完整数据集上的泛化能力

**典型配置**:
```bash
DATASET="musique"                      # 同一数据集
LEARN_RATE_LIST=(10 20 50 100)        # 用不同比例的样本学习
STEERING_ALPHA_LIST=(0.3 0.5 0.7 1.0)
```

**实验问题**: 需要多少样本才能学到有效的steering vectors？

**文件命名**:
```
musique_learn10of100_10pct_manifold_pca7.pt   # 用10/100样本
musique_learn20of100_20pct_manifold_pca7.pt   # 用20/100样本
musique_learn50of100_50pct_manifold_pca7.pt   # 用50/100样本
musique_learn100of100_100pct_manifold_pca7.pt # 用100/100样本
```

**结果目录**:
```
result/learn_rate_sweep/
├── learn_rate_10pct/
│   ├── alpha_0.3/
│   ├── alpha_0.5/
│   └── ...
├── learn_rate_20pct/
│   └── ...
├── learn_rate_50pct/
│   └── ...
└── learn_rate_100pct/
    └── ...
```

---

## 详细对比表

| 特性 | Multi (跨数据集) | Same Dataset (同数据集) | Learn Rate Sweep (样本泛化) |
|------|----------------|---------------------|------------------------|
| **统计数据集** | TRAIN_DATASET | TRAIN_DATASET | DATASET |
| **测试数据集** | TEST_DATASET | TEST_DATASET | DATASET (完整) |
| **典型配置** | 2wikimqa→musique | musique→musique | musique (10%→100%) |
| **样本控制** | 全部或指定 | 全部或指定 | 按比例自动计算 |
| **研究问题** | 跨数据集泛化 | 同数据集最优性能 | 样本效率 |
| **文件命名** | {train}_to_{test}_ | {dataset}_ | {dataset}_learn{n}of{m}_{pct}% |
| **结果组织** | 按alpha | 按alpha | 按learn_rate→alpha |
| **计算成本** | 高 (跨数据集) | 中 (全样本) | 可变 (10%-100%) |
| **适用场景** | 研究泛化能力 | 建立性能基线 | 优化计算成本 |

## 实验组合建议

### 完整研究流程

```bash
# Step 1: 同数据集基线（了解上限）
bash run_fusionrag_steering_same_dataset.sh
# 配置: musique全样本 → musique测试
# 得到: 最优性能基线

# Step 2: 样本效率研究（优化成本）
bash run_fusionrag_learn_rate_sweep.sh
# 配置: musique (10%, 20%, 50%, 100%)
# 得到: 最小有效样本数

# Step 3: 跨数据集泛化（验证通用性）
bash run_fusionrag_steering_muliti.sh
# 配置: 2wikimqa → musique
# 得到: 跨数据集性能

# 对比三个实验的结果
```

### 快速验证流程

```bash
# 只想快速验证steering vectors是否有效

# 使用 Same Dataset 脚本
DATASET="musique"
STEERING_ALPHA_LIST=(0.5 1.0)  # 只测试2个alpha
RATE_LIST=(0.0)
bash run_fusionrag_steering_same_dataset.sh
```

### 深入研究流程

```bash
# 研究steering vectors的各种特性

# 1. 样本效率
bash run_fusionrag_learn_rate_sweep.sh
# 问题：需要多少样本？
# 发现：如10%样本就达到80%性能

# 2. PCA阈值影响
for pca in 0.5 0.7 0.9; do
    PCA_VARIANCE_THRESHOLD=$pca
    bash run_fusionrag_learn_rate_sweep.sh
done
# 问题：PCA阈值如何影响泛化？

# 3. Per-head vs Layer-level
PER_HEAD_STATS="true"
bash run_fusionrag_learn_rate_sweep.sh
# 问题：per-head需要更多样本吗？

# 4. 跨数据集验证
bash run_fusionrag_steering_muliti.sh
# 问题：能否跨数据集使用？
```

## 选择指南

### 什么时候用哪个脚本？

#### 用 `run_fusionrag_steering_muliti.sh` 当你想:
- ✅ 测试跨数据集泛化（如 2wikimqa → musique）
- ✅ 研究不同数据集的相似性
- ✅ 验证steering vectors学到的是通用特征还是数据集特定特征

#### 用 `run_fusionrag_steering_same_dataset.sh` 当你想:
- ✅ 建立性能基线（同数据集最优性能）
- ✅ 快速验证方法是否有效
- ✅ 调优参数（alpha, PCA阈值等）

#### 用 `run_fusionrag_learn_rate_sweep.sh` 当你想:
- ✅ 研究样本效率（需要多少样本？）
- ✅ 优化计算成本（能否用少量样本？）
- ✅ 理解泛化曲线（性能如何随样本数变化）
- ✅ 发现最小有效样本数

## 参数共性

### 所有脚本都支持的参数

```bash
# Steering配置
USE_MANIFOLD_PROJECTION="true"
PCA_VARIANCE_THRESHOLD="0.7"
PER_HEAD_STATS="false"

# Alpha扫描
STEERING_ALPHA_LIST=(...)

# 层选择
STEERING_KEY_LAYERS="all"
STEERING_VALUE_LAYERS="all"

# 压缩率
RATE_LIST=(0.0)
```

### 脚本特有参数

```bash
# Multi脚本特有
TRAIN_DATASET="2wikimqa"  # 训练数据集
TEST_DATASET="musique"    # 测试数据集

# Same Dataset脚本特有
TRAIN_DATASET="musique"   # 与TEST_DATASET相同
TEST_DATASET="musique"

# Learn Rate Sweep脚本特有
DATASET="musique"         # 单一数据集
LEARN_RATE_LIST=(10 20 50 100)  # 学习样本比例
```

## 输出对比

### Steering Vectors文件

#### Multi脚本:
```
/mnt/data3/tmp/fusionrag/kv_stats/
└── 2wikimqa_to_musique_steering_manifold_pca7.pt
```

#### Same Dataset脚本:
```
/mnt/data3/tmp/fusionrag/kv_stats/
└── musique_steering_manifold_pca7.pt
```

#### Learn Rate Sweep脚本:
```
/mnt/data3/tmp/fusionrag/kv_stats/
├── musique_learn10of100_10pct_manifold_pca7.pt
├── musique_learn20of100_20pct_manifold_pca7.pt
├── musique_learn50of100_50pct_manifold_pca7.pt
└── musique_learn100of100_100pct_manifold_pca7.pt
```

### 结果目录

#### Multi脚本:
```
result/steering/
└── alpha_{value}/
    └── Qwen2.5-7B-Instruct/musique/...
```

#### Same Dataset脚本:
```
result/steering_same_dataset/
└── alpha_{value}/
    └── Qwen2.5-7B-Instruct/musique/...
```

#### Learn Rate Sweep脚本:
```
result/learn_rate_sweep/
└── learn_rate_{pct}pct/
    └── alpha_{value}/
        └── Qwen2.5-7B-Instruct/musique/...
```

## 实验示例

### 示例1: 完整性能评估

```bash
# 目标：了解steering vectors的全部能力

# 1. 同数据集最优性能
bash run_fusionrag_steering_same_dataset.sh

# 2. 不同样本数的性能
bash run_fusionrag_learn_rate_sweep.sh

# 3. 跨数据集性能
bash run_fusionrag_steering_muliti.sh

# 分析：
# - 同数据集100%: 90分（上限）
# - 同数据集20%: 85分（样本效率高！）
# - 跨数据集: 75分（泛化能力中等）
```

### 示例2: 成本优化

```bash
# 目标：找到最小有效样本数

# 运行learn_rate sweep
LEARN_RATE_LIST=(5 10 15 20 30 50 100)
bash run_fusionrag_learn_rate_sweep.sh

# 分析结果，发现：
# - 5%: 60分
# - 10%: 75分
# - 20%: 85分
# - 100%: 88分

# 结论：使用20%样本即可，节省80%计算！
```

### 示例3: 泛化能力研究

```bash
# 目标：理解什么影响泛化

# 实验A: 同数据集，不同样本数
bash run_fusionrag_learn_rate_sweep.sh

# 实验B: 跨数据集，全样本
bash run_fusionrag_steering_muliti.sh

# 发现：
# - 同数据集泛化（10%→100%）：性能下降5分
# - 跨数据集泛化（2wikimqa→musique）：性能下降15分
# 结论：跨数据集泛化更难！
```

## 总结

### 三个脚本的定位

| 脚本 | 核心价值 | 主要用途 |
|------|---------|---------|
| Multi | 研究通用性 | 验证跨数据集泛化 |
| Same Dataset | 建立基线 | 获得最优性能 |
| Learn Rate Sweep | 优化效率 | 找到最小样本数 |

### 推荐使用顺序

1. **First**: `run_fusionrag_steering_same_dataset.sh` - 快速验证方法
2. **Second**: `run_fusionrag_learn_rate_sweep.sh` - 优化样本数
3. **Third**: `run_fusionrag_steering_muliti.sh` - 验证通用性

### 关键区别

- **Multi**: 不同数据集间的泛化
- **Same**: 同数据集的最优性能
- **Learn Rate**: 同数据集内的样本效率

选择合适的脚本，进行合适的实验！
