# KV Calibration 集成完成总结

## ✅ 完成的工作

### 1. 核心实现完成
- ✅ `kv_calibration.py` (700+ 行) - 完整的offline/online实现
- ✅ 支持3种粒度：per_layer, per_head, per_position (架构)
- ✅ 自适应层选择功能
- ✅ K/V分离校准配置

### 2. test_fusionrag_reflect.py 集成完成

#### 修改点 1: 导入模块 (Line 29-35)
```python
# KV Calibration support
try:
    from kv_calibration import KVCalibrator, KVCalibrationStats, CalibrationConfig
    KV_CALIBRATION_AVAILABLE = True
except ImportError:
    KV_CALIBRATION_AVAILABLE = False
```

#### 修改点 2: main函数参数 (Line 1284-1295)
添加11个KV Calibration参数：
```python
enable_kv_calibration=False,
kv_calibration_mode='online',
calibration_reference_method='bge',
calibration_sample_ratio=0.1,
calibration_granularity='per_layer',
calibration_aggregation='mean',
calibration_key_layers='',
calibration_value_layers='',
calibration_auto_select_layers=False,
calibration_threshold=0.1,
calibration_stats_path='',
```

#### 修改点 3: argparse参数 (Line 2919-2944)
添加完整的命令行参数解析。

#### 修改点 4: main函数调用 (Line 3018-3029)
传递所有KV Calibration参数。

#### 修改点 5: Calibrator初始化 (Line 1403-1486)
- Online模式：加载统计量
- Offline模式：创建配置（用于后续统计）
- 错误处理和友好提示

#### 修改点 6: KV Cache加载点1 - 普通文档 (Line 2033-2039)
```python
# Apply KV Calibration if enabled (NEW - 2026-01-16)
if calibrator is not None and kv_calibration_mode == 'online':
    chunk_key_cache, chunk_value_cache = calibrator.apply_calibration_online(
        kv_cache_key=chunk_key_cache,
        kv_cache_value=chunk_value_cache,
        device=str(input_device)
    )
```

#### 修改点 7: KV Cache加载点2 - Random Text (Line 1978-1984)
在random_text的KV cache加载后也应用校准。

### 3. Shell脚本和文档

| 文件 | 说明 |
|------|------|
| `run_kv_calibration_offline.sh` | Offline统计脚本 |
| `test_kv_calibration_e2e.sh` | **完整端到端测试** (Phase 0→1→2→3) |
| `test_kv_calibration_quick.sh` | **快速测试** (仅Online) |
| `demo_kv_calibration_workflow.sh` | 演示流程 |
| `run_fusionrag_sweep.sh` | 已添加KV Calibration参数 |
| `KV_CALIBRATION_GUIDE.md` | 完整使用指南 |
| `KV_CALIBRATION_README.md` | 实现总结 |

---

## 🚀 端到端测试

### 方案 1: 完整测试 (推荐首次使用)

```bash
cd /home/shm/document/exp/FusionRAG
bash test_kv_calibration_e2e.sh
```

**这个脚本会自动执行**:
1. **Phase 0**: 生成基础KV cache (no_preprocess 和 bge)
2. **Phase 1**: Offline统计KV偏移
3. **Phase 2**: 查看统计摘要
4. **Phase 3**: Online应用KV校准进行推理

**测试范围**:
- 只测试5个样本（快速验证）
- 对比3种方法：no_preprocess, bge, kv_calibrated

**预期时间**: 约10-20分钟（取决于GPU速度）

### 方案 2: 快速测试 (假设offline已完成)

```bash
cd /home/shm/document/exp/FusionRAG
bash test_kv_calibration_quick.sh
```

**前提**: 已经运行过offline统计，存在calibration stats文件。

**测试范围**: 只测试3个样本的online推理

---

## 📋 使用流程

### Step 1: Offline统计（一次性）

**使用独立脚本**:
```bash
bash run_kv_calibration_offline.sh
```

**或使用test_fusionrag_reflect.py** (高级用法):
```bash
python test_fusionrag_reflect.py \
    --enable_kv_calibration true \
    --kv_calibration_mode offline \
    --calibration_reference_method bge \
    --calibration_granularity per_layer \
    --calibration_sample_ratio 0.1 \
    --calibration_auto_select_layers true \
    --max_samples 50
```

### Step 2: 查看统计摘要

```bash
python kv_calibration.py --mode summary \
    --stats_path /mnt/data3/tmp/fusionrag/musique/Qwen2.5-7B-Instruct/calibration_stats_bge_per_layer.pt
```

### Step 3: Online推理

**修改run_fusionrag_sweep.sh**:
```bash
ENABLE_KV_CALIBRATION="true"
KV_CALIBRATION_MODE="online"
CALIBRATION_STATS_PATH="/mnt/data3/tmp/fusionrag/musique/Qwen2.5-7B-Instruct/calibration_stats_bge_per_layer.pt"

# (可选) 指定要校准的层
CALIBRATION_KEY_LAYERS="0,1,2,3,4,5,6,7,8,9"
CALIBRATION_VALUE_LAYERS="0,1,2,3,4,5,6,7,8,9"
```

**然后运行**:
```bash
bash run_fusionrag_sweep.sh
```

---

## 🔍 集成测试检查清单

在运行测试前，请确认以下几点：

### ✅ 环境检查
- [ ] Python环境: `/home/shm/anaconda3/envs/fusionrag/bin/python`
- [ ] GPU可用: `CUDA_VISIBLE_DEVICES=5,6`
- [ ] 数据集: `./data/result_reflect.json` 存在
- [ ] 模型: `/mnt/data/models/Qwen2.5-7B-Instruct` 存在
- [ ] BGE模型: `/mnt/data/models/bge-m3-FP16` 存在

### ✅ 文件检查
- [ ] `kv_calibration.py` 存在并可导入
- [ ] `test_fusionrag_reflect.py` 已更新
- [ ] 测试脚本有执行权限

### ✅ 功能检查
运行以下命令验证导入正常：
```bash
cd /home/shm/document/exp/FusionRAG
python -c "from kv_calibration import KVCalibrator; print('✓ Import OK')"
```

---

## 📊 预期结果

### 准确率对比

| 方法 | 准确率 | 计算开销 | 备注 |
|------|--------|----------|------|
| no_preprocess | Baseline | 1.0× | 基线 |
| bge preprocess | Baseline + δ | 1.5-2.0× | 重计算KV |
| **kv_calibrated** | **Baseline + 0.7δ~0.9δ** | **~1.05×** | **目标** |

### 输出文件位置

**test_kv_calibration_e2e.sh 生成**:
```
/mnt/data3/tmp/fusionrag/musique/Qwen2.5-7B-Instruct/
├── calibration_stats_bge_per_layer.pt
├── calibration_stats_bge_per_layer.json
└── kv_cache/
    ├── 0_0_key.pt
    ├── 0_0_value.pt
    └── ...

/home/shm/document/exp/FusionRAG/result/kv_calibration_test/Qwen2.5-7B-Instruct/musique/
├── nopreprocess/                    # no_preprocess结果
├── FusionRAG_global_topk10_bge/     # bge preprocess结果
└── (在kv_calibrated目录下)          # KV calibrated结果
```

---

## 🐛 故障排除

### 问题 1: 找不到kv_calibration模块

**错误信息**:
```
Warning: kv_calibration module not found. KV Calibration features will be disabled.
```

**解决方案**:
```bash
cd /home/shm/document/exp/FusionRAG
python -c "import sys; print(sys.path)"
# 确保当前目录在sys.path中
```

### 问题 2: Calibration stats不存在

**错误信息**:
```
⚠️  Warning: Calibration stats not found: /path/to/calibration_stats_bge_per_layer.pt
```

**解决方案**:
1. 先运行offline统计:
   ```bash
   bash run_kv_calibration_offline.sh
   ```
2. 或指定正确的stats路径

### 问题 3: KV cache shape不匹配

**可能原因**: calibrator返回的KV cache shape与expected不一致

**解决方案**: 检查calibration_granularity是否与offline统计时一致

### 问题 4: CUDA OOM

**解决方案**:
1. 减少MAX_SAMPLES
2. 使用单GPU: `GPUS="5"`
3. 降低batch size（如果有的话）

---

## 🎯 下一步

### 1. 运行完整测试
```bash
bash test_kv_calibration_e2e.sh
```

### 2. 查看结果对比
```bash
# 查看准确率
cat /home/shm/document/exp/FusionRAG/result/kv_calibration_test/*/musique/*/results_*.csv
```

### 3. 调整参数实验

**尝试不同粒度**:
```bash
# per_head (更精细)
CALIBRATION_GRANULARITY="per_head"

# 重新运行offline
bash run_kv_calibration_offline.sh
```

**尝试不同层选择**:
```bash
# 只校准前10层
CALIBRATION_KEY_LAYERS="0,1,2,3,4,5,6,7,8,9"
CALIBRATION_VALUE_LAYERS="0,1,2,3,4,5,6,7,8,9"
```

**尝试自适应层选择**:
```bash
CALIBRATION_AUTO_SELECT_LAYERS="true"
CALIBRATION_THRESHOLD="0.05"  # 更严格的阈值
```

### 4. 扩展到完整数据集

如果5样本测试效果好，修改测试脚本：
```bash
MAX_SAMPLES=500  # 或留空使用全部
```

---

## 📝 修改记录

| 日期 | 修改内容 | 行数 |
|------|----------|------|
| 2026-01-16 | 添加kv_calibration导入 | test_fusionrag_reflect.py:29-35 |
| 2026-01-16 | 添加main函数参数 | test_fusionrag_reflect.py:1284-1295 |
| 2026-01-16 | 添加argparse参数 | test_fusionrag_reflect.py:2919-2944 |
| 2026-01-16 | 添加main调用参数 | test_fusionrag_reflect.py:3018-3029 |
| 2026-01-16 | 添加calibrator初始化 | test_fusionrag_reflect.py:1403-1486 |
| 2026-01-16 | 添加KV校准应用点1 | test_fusionrag_reflect.py:2033-2039 |
| 2026-01-16 | 添加KV校准应用点2 | test_fusionrag_reflect.py:1978-1984 |

---

## ✨ 特性总结

- ✅ **完全集成**: 无需修改核心推理代码，通过参数开关
- ✅ **向后兼容**: 不启用calibration时，行为完全不变
- ✅ **灵活配置**: 支持所有kv_calibration.py的功能
- ✅ **友好提示**: 完善的错误处理和用户提示
- ✅ **端到端测试**: 提供完整的测试脚本
- ✅ **文档齐全**: 使用指南、API文档、集成总结

---

## 🎉 准备就绪！

你现在可以开始端到端测试了：

```bash
cd /home/shm/document/exp/FusionRAG

# 完整测试（推荐）
bash test_kv_calibration_e2e.sh

# 或快速测试（如果offline已完成）
bash test_kv_calibration_quick.sh
```

**预计测试时间**: 10-20分钟

**预期输出**: 3种方法的准确率对比，验证KV Calibration效果

祝测试顺利！ 🚀
