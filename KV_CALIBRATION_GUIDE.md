# KV Cache Calibration 使用指南

## 📋 概述

KV Calibration是一种类似BatchNorm的优化方案，旨在**避免preprocess的计算开销，同时获得接近preprocess的性能提升**。

### 核心思路

```
类比BatchNorm：
  offline: 统计 E[x], Var[x]
  online:  y = (x - E[x]) / sqrt(Var[x] + eps)

KV Calibration：
  offline: 统计 Δ = E[KV_preprocess - KV_no_preprocess]
  online:  KV_calibrated = KV_no_preprocess + Δ
```

### 方案优势

- ✅ **降低计算开销**: online阶段无需重新计算preprocess，直接加偏移即可
- ✅ **保留性能收益**: 通过统计偏移分布，接近preprocess的效果
- ✅ **灵活可配置**: 支持多种粒度(per-layer/per-head/per-position)和层选择策略
- ✅ **一次统计，多次使用**: offline统计一次，online可重复使用

---

## 🚀 快速开始

### Step 1: Offline阶段 - 统计偏移分布

首先，你需要有两组KV cache：
- `no_preprocess`: 未经preprocess的KV cache
- `bge` (或其他方法): 经过preprocess的KV cache

运行offline统计脚本：

```bash
cd /home/shm/document/exp/FusionRAG
bash run_kv_calibration_offline.sh
```

**关键参数配置** (在 `run_kv_calibration_offline.sh` 中修改):

```bash
SAMPLE_RATIO="0.1"           # 使用10%样本统计（可改为0.2, 0.5等）
REFERENCE_METHOD="bge"        # 参考方法（bge, oracle, repeat_self等）
GRANULARITY="per_layer"       # 统计粒度（per_layer, per_head, per_position）
AUTO_SELECT_LAYERS="false"    # 是否自动选择显著层
```

**输出**:
- `calibration_stats_bge_per_layer.pt` - 统计量文件（用于online加载）
- `calibration_stats_bge_per_layer.json` - 配置和层范数（便于查看）

### Step 2: 查看统计摘要

```bash
python kv_calibration.py \
    --mode summary \
    --stats_path /mnt/data3/tmp/fusionrag/musique/Qwen2.5-7B-Instruct/calibration_stats_bge_per_layer.pt
```

输出示例：
```
============================================================
Calibration Statistics Summary
============================================================
Granularity: per_layer
Reference method: bge

Key Layer Norms (L2):
  Layer  0: 0.123456
  Layer  1: 0.234567
  ...
  Layer 27: 0.087654

Value Layer Norms (L2):
  Layer  0: 0.098765
  ...
============================================================
```

### Step 3: Online阶段 - 应用校准

**注意**: Online模式的完整集成需要修改 `test_fusionrag_reflect.py`，目前提供了核心API `apply_calibration_online()`。

修改 `run_fusionrag_sweep.sh` 配置：

```bash
# 启用KV校准
ENABLE_KV_CALIBRATION="true"

# 设置为online模式
KV_CALIBRATION_MODE="online"

# 指定统计量文件路径
CALIBRATION_STATS_PATH="/mnt/data3/tmp/fusionrag/musique/Qwen2.5-7B-Instruct/calibration_stats_bge_per_layer.pt"

# （可选）指定要校准的层
CALIBRATION_KEY_LAYERS="0,1,2,3,4,5,6,7,8,9"    # 只校准前10层的Key
CALIBRATION_VALUE_LAYERS="0,1,2,3,4,5,6,7,8,9"  # 只校准前10层的Value
```

然后正常运行sweep脚本：

```bash
bash run_fusionrag_sweep.sh
```

---

## 📊 参数详解

### Offline阶段参数

| 参数 | 说明 | 可选值 | 推荐 |
|------|------|--------|------|
| `SAMPLE_RATIO` | 用于统计的样本比例 | 0.0-1.0 | 0.1 (10%) |
| `REFERENCE_METHOD` | 参考preprocess方法 | bge, oracle, repeat_self等 | bge |
| `GRANULARITY` | 统计粒度 | per_layer, per_head, per_position | per_layer |
| `AGGREGATION` | 聚合方式 | mean, mean_std, weighted | mean |
| `AUTO_SELECT_LAYERS` | 自动选择显著层 | true, false | false |
| `THRESHOLD` | 自动选择的L2范数阈值 | 浮点数 | 0.1 |

### Online阶段参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `CALIBRATION_STATS_PATH` | 统计量文件路径 | `/path/to/calibration_stats_bge_per_layer.pt` |
| `CALIBRATION_KEY_LAYERS` | 要校准的Key层 | `"0,1,2,3,4,5"` (留空=所有层) |
| `CALIBRATION_VALUE_LAYERS` | 要校准的Value层 | `"0,1,2,3,4,5"` (留空=所有层) |

---

## 🔬 实验建议

### 1. 探索不同统计粒度

**per_layer** (推荐开始):
```bash
GRANULARITY="per_layer"
```
- ✅ 最简单，计算开销最小
- ✅ 统计稳定，需要样本量少
- ❌ 粗粒度，可能损失一些精度

**per_head**:
```bash
GRANULARITY="per_head"
```
- ✅ 更精细，每个attention head单独校准
- ❌ 统计量更大，需要更多样本
- 适合：模型较大，有足够样本时

**per_position** (暂未实现):
- 最精细，但需要对齐不同文档的token位置
- 计算和存储开销大

### 2. 自适应层选择

如果不确定哪些层的偏移最显著：

```bash
AUTO_SELECT_LAYERS="true"
THRESHOLD="0.1"  # 只校准L2范数 > 0.1的层
```

Offline统计后会自动输出：
```
Selected 8 key layers: [0, 1, 2, 3, 4, 5, 6, 7]
Selected 6 value layers: [0, 1, 2, 5, 6, 7]
```

### 3. 对比实验

建议进行以下对比：

```bash
# Baseline 1: 纯no_preprocess
ENABLE_KV_CALIBRATION="false"
PREPROCESS="false"

# Baseline 2: 完整preprocess (BGE)
ENABLE_KV_CALIBRATION="false"
PREPROCESS="true"
RECALL_METHOD="bge"

# 实验组: Calibrated no_preprocess
ENABLE_KV_CALIBRATION="true"
KV_CALIBRATION_MODE="online"
PREPROCESS="false"  # 关键：不使用preprocess
```

**期望结果**:
- 校准后性能接近Baseline 2
- 计算开销接近Baseline 1

---

## 🛠️ 核心API使用

如果你想在自己的代码中使用KV Calibration：

```python
from kv_calibration import KVCalibrator, CalibrationConfig, KVCalibrationStats

# ========== Offline: 统计偏移 ==========
config = CalibrationConfig(
    granularity="per_layer",
    reference_method="bge",
    sample_ratio=0.1,
    auto_select_layers=True,
    threshold=0.1,
)

calibrator = KVCalibrator(config)
calibrator.compute_offset_offline(
    cache_dir="/mnt/data3/tmp/fusionrag",
    dataset_name="musique",
    model_name="Qwen2.5-7B-Instruct",
    example_ids=[0, 1, 2, ...],  # 要统计的样本ID
    num_layers=28,
)

# 保存统计量
calibrator.stats.save("calibration_stats.pt")

# ========== Online: 应用校准 ==========
# 加载统计量
stats = KVCalibrationStats.load("calibration_stats.pt")
calibrator_online = KVCalibrator(stats.config)
calibrator_online.stats = stats

# 加载原始no_preprocess的KV cache
kv_key = torch.load("example_0_0_key.pt")    # List[Tensor]
kv_val = torch.load("example_0_0_value.pt")

# 应用校准
calibrated_key, calibrated_val = calibrator_online.apply_calibration_online(
    kv_cache_key=kv_key,
    kv_cache_value=kv_val,
    device='cuda:0'
)

# 使用校准后的KV cache进行推理
# ... (后续推理代码)
```

---

## 📁 文件说明

| 文件 | 用途 |
|------|------|
| `kv_calibration.py` | 核心实现：Offline统计 + Online校准API |
| `run_kv_calibration_offline.sh` | Offline阶段运行脚本 |
| `run_fusionrag_sweep.sh` | 主实验脚本（已添加KV calibration参数） |
| `KV_CALIBRATION_GUIDE.md` | 本文档 |

生成的文件：
| 文件 | 内容 |
|------|------|
| `calibration_stats_<method>_<granularity>.pt` | 统计量（Tensor格式） |
| `calibration_stats_<method>_<granularity>.json` | 配置和层范数（可读格式） |

---

## ❓ 常见问题

### Q1: Offline统计需要多少样本？

**建议**:
- 快速验证: 10% (SAMPLE_RATIO=0.1)
- 稳定统计: 20-30%
- 最佳效果: 50%

### Q2: 如何选择参考方法？

通常选择性能最好的preprocess方法：
- `bge`: BGE相似度召回（推荐）
- `oracle`: Oracle方法（如果可用）
- `repeat_self`: 自重复（用于消融实验）

### Q3: per_layer vs per_head，如何选择？

| 场景 | 推荐 |
|------|------|
| 初次尝试 | per_layer |
| 样本数 < 100 | per_layer |
| 样本数 > 200 | 可尝试 per_head |
| 追求极致性能 | per_head |

### Q4: 为什么只处理chunk_id=0？

`chunk_id=0` 对应 system prompt，所有样本都有。
`chunk_id > 0` 对应文档，不同样本的文档数量不同，不适合全局统计。

未来可以扩展到对每个样本的所有chunk单独统计。

### Q5: Online模式需要修改哪些代码？

**已完成**:
- ✅ `kv_calibration.py`: 核心API
- ✅ `run_fusionrag_sweep.sh`: 参数传递

**待完成** (需要你手动集成):
- ⬜ `test_fusionrag_reflect.py`: 在生成KV cache后调用 `apply_calibration_online()`

集成示例见下一节。

---

## 🔌 集成到test_fusionrag_reflect.py

在KV cache生成后、使用前应用校准：

```python
# 1. 在文件开头导入
from kv_calibration import KVCalibrator, KVCalibrationStats

# 2. 在main函数开始时加载统计量（如果启用）
calibrator = None
if args.enable_kv_calibration and args.kv_calibration_mode == 'online':
    stats = KVCalibrationStats.load(args.calibration_stats_path)
    calibrator = KVCalibrator(stats.config)
    calibrator.stats = stats
    print(f"✓ Loaded KV calibration stats from {args.calibration_stats_path}")

# 3. 在加载KV cache后应用校准
# 找到类似这样的代码段:
chunk_key_cache = torch.load(f"{save_path}/{example_id}_{chunk_id}_key.pt", weights_only=True)
chunk_value_cache = torch.load(f"{save_path}/{example_id}_{chunk_id}_value.pt", weights_only=True)

# 添加校准逻辑:
if calibrator is not None:
    chunk_key_cache, chunk_value_cache = calibrator.apply_calibration_online(
        kv_cache_key=chunk_key_cache,
        kv_cache_value=chunk_value_cache,
        device=str(input_device)
    )

# 然后继续使用calibrated的KV cache
```

---

## 📈 性能预期

理想情况下：

| 方法 | 准确率 | 计算开销 | 备注 |
|------|--------|----------|------|
| no_preprocess | Baseline | 1.0× | 基线 |
| bge preprocess | Baseline + δ | 1.5× - 2.0× | 重新计算KV |
| **KV Calibration** | **Baseline + 0.7δ ~ 0.9δ** | **~1.05×** | **目标** |

实际效果取决于：
- 统计样本数量和质量
- 选择的粒度和层
- 具体任务和数据集

---

## 📝 TODO

- [ ] 实现 `per_position` 粒度支持
- [ ] 支持对所有chunk（不仅是chunk_id=0）统计
- [ ] 集成到 `test_fusionrag_reflect.py` (自动化)
- [ ] 添加 `mean_std` 聚合方式支持
- [ ] 可视化不同层的偏移分布

---

## 📞 反馈

如有问题或建议，请联系FusionRAG团队。

**创建日期**: 2026-01-16
**版本**: v1.0
**作者**: FusionRAG Team
