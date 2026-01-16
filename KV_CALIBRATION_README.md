# KV Cache Calibration - 实现总结

## 🎯 方案概述

KV Calibration是一种**类似BatchNorm的KV Cache校准方案**，旨在**避免preprocess的重计算开销，同时保留性能提升**。

### 核心思路

```python
# BatchNorm做法
offline: 统计 E[x], Var[x]
online:  y = (x - E[x]) / sqrt(Var[x] + eps)

# KV Calibration做法
offline: 统计 Δ = E[KV_preprocess - KV_no_preprocess]  # 偏移分布
online:  KV_calibrated = KV_no_preprocess + Δ           # 应用偏移
```

### 预期效果

| 方法 | 准确率 | 计算开销 |
|------|--------|----------|
| no_preprocess | Baseline | 1.0× |
| bge preprocess | Baseline + δ | 1.5-2.0× |
| **KV Calibration** | **Baseline + 0.7δ~0.9δ** | **~1.05×** |

---

## 📦 实现文件清单

| 文件 | 行数 | 说明 |
|------|------|------|
| `kv_calibration.py` | 700+ | 核心实现：Offline统计 + Online校准API |
| `run_kv_calibration_offline.sh` | 65 | Offline阶段运行脚本 |
| `demo_kv_calibration_workflow.sh` | 180 | 完整演示流程（offline→summary→online配置） |
| `KV_CALIBRATION_GUIDE.md` | 400+ | 详细使用指南 |
| `KV_CALIBRATION_README.md` | 本文件 | 实现总结 |

修改的文件：
| 文件 | 修改内容 |
|------|----------|
| `run_fusionrag_sweep.sh` | 添加11个KV Calibration参数 |

---

## 🚀 快速开始

### 3步使用流程

```bash
# Step 1: Offline统计（一次性）
bash run_kv_calibration_offline.sh

# Step 2: 查看统计摘要
python kv_calibration.py --mode summary \
    --stats_path /mnt/data3/tmp/fusionrag/musique/Qwen2.5-7B-Instruct/calibration_stats_bge_per_layer.pt

# Step 3: Online应用（修改run_fusionrag_sweep.sh）
ENABLE_KV_CALIBRATION="true"
KV_CALIBRATION_MODE="online"
CALIBRATION_STATS_PATH="<path_to_stats.pt>"
# 然后运行
bash run_fusionrag_sweep.sh
```

### 完整演示

```bash
bash demo_kv_calibration_workflow.sh
```

会自动执行：
1. 三种粒度的offline统计 (per_layer, per_head, 自适应)
2. 查看统计摘要
3. 显示online配置说明

---

## 🔧 核心功能

### 1. 支持三种统计粒度

| 粒度 | 统计量大小 | 精度 | 推荐场景 |
|------|-----------|------|----------|
| **per_layer** | 小 | 中 | 初次尝试、样本少 |
| **per_head** | 中 | 高 | 样本充足、追求性能 |
| **per_position** | 大 | 极高 | 暂未实现 |

### 2. 自适应层选择

根据偏移L2范数自动选择显著的层：

```bash
AUTO_SELECT_LAYERS="true"
THRESHOLD="0.1"  # 只校准||Δ|| > 0.1的层
```

输出示例：
```
Selected 8 key layers: [0, 1, 2, 3, 4, 5, 6, 7]
Selected 6 value layers: [0, 1, 2, 5, 6, 7]
```

### 3. 灵活的层选择

可以对Key和Value选择不同的层：

```bash
# 只校准前10层的Key
CALIBRATION_KEY_LAYERS="0,1,2,3,4,5,6,7,8,9"

# 只校准后10层的Value
CALIBRATION_VALUE_LAYERS="18,19,20,21,22,23,24,25,26,27"
```

### 4. 多种聚合方式

```python
aggregation = "mean"       # 简单均值（推荐）
aggregation = "mean_std"   # 均值+标准差归一化（待实现）
aggregation = "weighted"   # 加权均值（待实现）
```

---

## 📊 参数配置

### Offline阶段参数

在 `run_kv_calibration_offline.sh` 中配置：

```bash
SAMPLE_RATIO="0.1"              # 统计样本比例（0.1 = 10%）
REFERENCE_METHOD="bge"          # 参考preprocess方法
GRANULARITY="per_layer"         # 统计粒度
AUTO_SELECT_LAYERS="false"      # 是否自动选择层
THRESHOLD="0.1"                 # 自动选择阈值
```

### Online阶段参数

在 `run_fusionrag_sweep.sh` 中配置：

```bash
# 启用校准
ENABLE_KV_CALIBRATION="true"
KV_CALIBRATION_MODE="online"

# 统计量路径
CALIBRATION_STATS_PATH="/path/to/calibration_stats_bge_per_layer.pt"

# 参考方法（用于offline，online时自动从stats读取）
CALIBRATION_REFERENCE_METHOD="bge"

# 粒度（offline统计时的粒度，online时自动从stats读取）
CALIBRATION_GRANULARITY="per_layer"

# 校准层选择（可选，留空=使用stats中的所有层）
CALIBRATION_KEY_LAYERS=""       # 如 "0,1,2,3,4,5"
CALIBRATION_VALUE_LAYERS=""

# 自适应层选择（offline时使用）
CALIBRATION_AUTO_SELECT_LAYERS="false"
CALIBRATION_THRESHOLD="0.1"
```

---

## 🔬 核心API

### Python API

```python
from kv_calibration import KVCalibrator, CalibrationConfig, KVCalibrationStats

# ===== Offline: 统计偏移 =====
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
    example_ids=list(range(50)),  # 前50个样本
    num_layers=28,
)

calibrator.stats.save("calibration_stats.pt")

# ===== Online: 应用校准 =====
stats = KVCalibrationStats.load("calibration_stats.pt")
calibrator_online = KVCalibrator(stats.config)
calibrator_online.stats = stats

# 加载KV cache
kv_key = torch.load("example_100_0_key.pt")
kv_val = torch.load("example_100_0_value.pt")

# 应用校准
calibrated_key, calibrated_val = calibrator_online.apply_calibration_online(
    kv_cache_key=kv_key,
    kv_cache_value=kv_val,
    device='cuda:0'
)

# 使用校准后的KV进行推理...
```

---

## 📈 实验建议

### 1. Baseline对比实验

```bash
# Baseline 1: 纯no_preprocess（最快，性能较差）
ENABLE_KV_CALIBRATION="false"
PREPROCESS="false"

# Baseline 2: 完整BGE preprocess（最慢，性能最好）
ENABLE_KV_CALIBRATION="false"
PREPROCESS="true"
RECALL_METHOD="bge"

# 实验组: KV Calibration（速度快，性能接近Baseline 2）
ENABLE_KV_CALIBRATION="true"
KV_CALIBRATION_MODE="online"
PREPROCESS="false"  # 关键：不使用preprocess
```

### 2. 消融实验

**粒度消融**:
```bash
# 对比 per_layer vs per_head
GRANULARITY="per_layer"   # 快速、粗粒度
GRANULARITY="per_head"    # 精细、更准确
```

**层选择消融**:
```bash
# 全层校准
CALIBRATION_KEY_LAYERS=""
CALIBRATION_VALUE_LAYERS=""

# 仅前10层
CALIBRATION_KEY_LAYERS="0,1,2,3,4,5,6,7,8,9"
CALIBRATION_VALUE_LAYERS="0,1,2,3,4,5,6,7,8,9"

# 仅后10层
CALIBRATION_KEY_LAYERS="18,19,20,21,22,23,24,25,26,27"
CALIBRATION_VALUE_LAYERS="18,19,20,21,22,23,24,25,26,27"

# 自适应选择
AUTO_SELECT_LAYERS="true"
THRESHOLD="0.1"
```

**样本数消融**:
```bash
SAMPLE_RATIO="0.05"  # 5%
SAMPLE_RATIO="0.1"   # 10%
SAMPLE_RATIO="0.2"   # 20%
SAMPLE_RATIO="0.5"   # 50%
```

---

## 🔌 集成到现有代码

### test_fusionrag_reflect.py集成点

在KV cache加载后、使用前应用校准：

```python
# 文件开头导入
from kv_calibration import KVCalibrator, KVCalibrationStats

# main函数中加载统计量
if args.enable_kv_calibration and args.kv_calibration_mode == 'online':
    stats = KVCalibrationStats.load(args.calibration_stats_path)
    calibrator = KVCalibrator(stats.config)
    calibrator.stats = stats

# 在加载KV cache后应用（约在1925行附近）
chunk_key_cache = torch.load(f"{save_path}/{corpus_i}_{similar_chunk_id}_key.pt", weights_only=True)
chunk_value_cache = torch.load(f"{save_path}/{corpus_i}_{similar_chunk_id}_value.pt", weights_only=True)

# 添加校准
if calibrator is not None:
    chunk_key_cache, chunk_value_cache = calibrator.apply_calibration_online(
        kv_cache_key=chunk_key_cache,
        kv_cache_value=chunk_value_cache,
        device=str(input_device)
    )
```

---

## 📁 生成文件

运行offline统计后会生成：

```bash
/mnt/data3/tmp/fusionrag/musique/Qwen2.5-7B-Instruct/
├── calibration_stats_bge_per_layer.pt       # Tensor格式（用于online加载）
├── calibration_stats_bge_per_layer.json     # JSON格式（便于查看配置）
├── calibration_stats_bge_per_head.pt
├── calibration_stats_bge_per_head.json
└── ...
```

**文件内容**:
- `.pt`: 包含完整统计量（均值、标准差、层范数等）
- `.json`: 只包含配置和层范数（人类可读）

---

## ❓ 常见问题

### Q: 为什么只统计chunk_id=0？

A: `chunk_id=0`对应system prompt，所有样本都有。`chunk_id > 0`对应文档，不同样本的文档数量不同，不适合全局统计。未来可扩展到per-example的chunk统计。

### Q: 需要多少样本才够？

A:
- 快速验证: 10% (50个样本左右)
- 稳定统计: 20-30%
- 最佳效果: 50%

### Q: 参考方法如何选择？

A: 选择性能最好的preprocess方法，通常是`bge`。也可以是`oracle`等。

### Q: per_layer vs per_head如何选？

A:
- 样本 < 100: 用 per_layer
- 样本 > 200: 可尝试 per_head
- 初次尝试: per_layer

### Q: Online模式需要修改代码吗？

A: 是的，需要在`test_fusionrag_reflect.py`中集成`apply_calibration_online()`调用。参见"集成到现有代码"章节。

---

## 📝 实现细节

### 类设计

```python
@dataclass
class CalibrationConfig:
    """配置类"""
    granularity: str
    aggregation: str
    sample_ratio: float
    reference_method: str
    key_layers: List[int]
    value_layers: List[int]
    auto_select_layers: bool
    threshold: float

class KVCalibrationStats:
    """统计量存储类"""
    key_stats: Dict[int, Dict[str, Tensor]]    # {layer_idx: {mean, std, count}}
    value_stats: Dict[int, Dict[str, Tensor]]
    key_layer_norms: Dict[int, float]          # {layer_idx: L2_norm}
    value_layer_norms: Dict[int, float]

    def save(path)
    def load(path)

class KVCalibrator:
    """校准器主类"""
    def compute_offset_offline(...)   # Offline统计
    def apply_calibration_online(...) # Online校准
```

### 统计量计算

**per_layer**:
```python
# 对每一层
key_offset = KV_bge[layer] - KV_no_preprocess[layer]
# Shape: [batch, num_heads, seq_len, head_dim]

# 对所有维度求平均，只保留head_dim
offset_mean = key_offset.mean(dim=(0, 1, 2))  # [head_dim]

# 跨样本累加
all_offsets.append(offset_mean)

# 最终统计
final_mean = mean(all_offsets)  # [head_dim]
final_std = std(all_offsets)
```

**per_head**:
```python
# 对每一层每个head
key_offset = KV_bge[layer] - KV_no_preprocess[layer]
# [batch, num_heads, seq_len, head_dim]

# 对每个head分别求平均
offset_per_head = key_offset.mean(dim=(0, 2))  # [num_heads, head_dim]

# 分别统计每个head
for head in range(num_heads):
    all_offsets[head].append(offset_per_head[head])

# 最终统计
for head in range(num_heads):
    stats[head]['mean'] = mean(all_offsets[head])
    stats[head]['std'] = std(all_offsets[head])
```

### 校准应用

```python
# per_layer校准
offset = stats[layer]['mean']  # [head_dim]
offset = offset.view(1, 1, 1, -1)  # Broadcast to [batch, heads, seq, head_dim]
kv_calibrated = kv_original + offset

# per_head校准
for head_idx in range(num_heads):
    offset = stats[layer][head_idx]['mean']  # [head_dim]
    offset = offset.view(1, 1, 1, -1)
    kv_calibrated[:, head_idx, :, :] = kv_original[:, head_idx, :, :] + offset
```

---

## 🎯 TODO / 未来改进

- [ ] 实现 `per_position` 粒度
- [ ] 实现 `mean_std` 聚合（类似完整的BatchNorm）
- [ ] 实现 `weighted` 聚合（基于相似度加权）
- [ ] 支持对所有chunk统计（不仅chunk_id=0）
- [ ] 自动集成到 `test_fusionrag_reflect.py`
- [ ] 可视化不同层的偏移分布（类似PCA工具）
- [ ] 支持动态调整校准强度（如 `alpha * offset`）

---

## 📞 联系方式

如有问题或建议，请联系FusionRAG团队。

**创建日期**: 2026-01-16
**版本**: v1.0
**作者**: FusionRAG Team
**实现时间**: ~2小时
