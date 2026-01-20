# Online Lazy Loading - Rate Sweep 使用指南

## 快速开始

### 1. 基本使用

```bash
cd /home/shm/document/exp/FusionRAG

# 使用默认配置运行（测试所有样本）
bash run_online_lazy_sweep.sh

# 快速测试（只测试5个样本）
# 修改脚本中的 MAX_SAMPLES="" 为 MAX_SAMPLES="5"
```

### 2. 配置参数

打开 `run_online_lazy_sweep.sh` 编辑以下关键参数：

```bash
# GPU设置
GPUS="4"                    # 使用的GPU编号

# Rate列表 - 要测试的重算比例
RATE_LIST=(0.1 0.15 0.3 0.5 0.7 1.0)

# 测试样本数
MAX_SAMPLES=""              # 留空=全部，或设置为数字如"5"

# 是否清除旧缓存
CLEAR_CACHE_BEFORE_START="false"  # true=清除旧缓存重新开始
```

### 3. 执行流程

脚本会自动：

1. **遍历每个 rate 值**
   - 0.1 → 0.15 → 0.3 → 0.5 → 0.7 → 1.0

2. **对每个 rate**：
   - 跳过offline预处理（零预处理时间）
   - 运行测试并按需生成KV
   - 新文档自动强制重算（rate=1.0）
   - 已缓存文档按设定rate重算
   - 保存结果到单独的CSV文件

3. **完成后显示**：
   - 总耗时
   - 缓存统计（文档数、大小）
   - 结果保存位置

## 输出位置

### KV Cache
```
/mnt/data3/tmp/fusionrag_online_lazy/
└── Qwen2.5-7B-Instruct/
    └── musique/
        └── kv_cache/
            ├── 0_0_key.pt     # System KV
            ├── 0_1_key.pt     # 文档1 KV
            ├── 0_2_key.pt     # 文档2 KV
            └── ...
```

### 结果文件
```
/home/shm/document/exp/FusionRAG/result/online_lazy_sweep/
└── Qwen2.5-7B-Instruct/
    └── musique/
        ├── rate_0.1_results.csv
        ├── rate_0.15_results.csv
        ├── rate_0.3_results.csv
        ├── rate_0.5_results.csv
        ├── rate_0.7_results.csv
        └── rate_1.0_results.csv
```

## 预期行为

### 第一次运行（冷启动）
```
Rate 0.1:
  Sub-question 1/2
    ⚠ Chunk 1: KV cache not found, will generate
    ⚠ Chunk 2: KV cache not found, will generate
    ⚡ Forward pass with 2 missing doc(s) + question
      ✓ Chunk 1 KV generated and saved (1024 tokens)
      ✓ Chunk 2 KV generated and saved (856 tokens)
    → Forcing recompute for 2 new chunks
    → Adjusting budget from 53 to 1880 to cover all new documents

  平均查询时间: ~1.5s
  ✓ Rate 0.1 测试完成
  当前已缓存文档数: 15

Rate 0.15:
  Sub-question 1/2
    (已有缓存，直接加载)

  平均查询时间: ~0.35s
  ✓ Rate 0.15 测试完成
  当前已缓存文档数: 15  (无新增)
```

### 第二次运行（热启动）
```
当前已缓存文档数: 15
当前缓存大小: 120M

Rate 0.1:
  (全部从缓存加载，无需生成)
  平均查询时间: ~0.35s  ← 快5倍！

Rate 0.15:
  平均查询时间: ~0.35s
  ...
```

## 性能特点

| 指标 | 第一次运行（冷启动） | 第二次运行（热启动） |
|------|-------------------|-------------------|
| 预处理时间 | 0分钟 ✅ | 0分钟 ✅ |
| 首次查询 | ~1.5s（需生成KV） | ~0.35s（从缓存加载） |
| 后续查询 | ~0.35s | ~0.35s |
| KV生成 | 按需生成 | 无需生成 |

## 常见场景

### 1. 完整测试（第一次运行）
```bash
# 清除旧缓存，从头开始
# 编辑脚本：CLEAR_CACHE_BEFORE_START="true"
bash run_online_lazy_sweep.sh
```

### 2. 快速验证（只测试几个样本）
```bash
# 编辑脚本：MAX_SAMPLES="5"
bash run_online_lazy_sweep.sh
```

### 3. 基于现有缓存继续测试
```bash
# 保持默认配置（CLEAR_CACHE_BEFORE_START="false"）
# 会复用之前生成的KV cache
bash run_online_lazy_sweep.sh
```

### 4. 测试特定rate范围
```bash
# 编辑脚本中的 RATE_LIST
RATE_LIST=(0.3 0.5 0.7)  # 只测试这三个值
bash run_online_lazy_sweep.sh
```

## 与 Offline 模式对比

| 特性 | Offline预处理 | Online Lazy Loading |
|------|--------------|-------------------|
| **启动时间** | 需要10-30分钟预处理 | 立即开始测试 ✅ |
| **首次查询** | 0.3s | 1.5s |
| **后续查询** | 0.3s | 0.35s ✅ |
| **KV复用** | 仅当次测试 | 跨测试复用 ✅ |
| **空间占用** | 全量预生成 | 按需生成 ✅ |
| **灵活性** | 需预知测试集 | 无需预知 ✅ |

## 缓存管理

### 查看缓存状态
```bash
# 查看缓存文档数
find /mnt/data3/tmp/fusionrag_online_lazy -name "*_key.pt" | wc -l

# 查看缓存大小
du -sh /mnt/data3/tmp/fusionrag_online_lazy
```

### 清除缓存
```bash
# 完全清除
rm -rf /mnt/data3/tmp/fusionrag_online_lazy/*

# 只清除特定数据集
rm -rf /mnt/data3/tmp/fusionrag_online_lazy/Qwen2.5-7B-Instruct/musique/kv_cache/*
```

## 故障排查

### 问题1：缓存文件找不到
```
⚠ Chunk 1: KV cache not found, will generate
```
**正常行为** - 第一次使用该文档时会自动生成

### 问题2：Budget调整消息
```
→ Adjusting budget from 53 to 330 to cover all new documents
```
**正常行为** - 确保新文档完全重算（rate=1.0）

### 问题3：GPU内存不足
```
CUDA out of memory
```
**解决方案**：
- 减少 MAX_SAMPLES（如设为5）
- 使用更大GPU或减小batch size

### 问题4：结果CSV文件被覆盖
**说明**：每次运行相同rate会覆盖对应的CSV文件
**解决方案**：测试前备份 `result/online_lazy_sweep/` 目录

## 高级用法

### 并行测试多个rate（不推荐）
由于KV会被复用，顺序执行更高效。第二个rate可以利用第一个rate生成的缓存。

### 自定义结果目录
```bash
# 编辑脚本
RESULT_DIR="/home/shm/document/exp/FusionRAG/result/my_experiment"
```

### 使用不同数据集
```bash
# 编辑脚本
DATA_PATH="./data/2wikimqa_reflect.json"
DATASET_NAME="2wikimqa"
```

## 注意事项

1. **首次运行较慢**：需要即时生成KV，建议先用小样本（MAX_SAMPLES="5"）测试
2. **缓存持久化**：KV保存在 `/mnt/data3/tmp/`，定期清理以释放空间
3. **跨测试复用**：不同rate的测试会共享相同的KV cache，第二个rate会更快
4. **RoPE调整**：已自动处理，保存时revert，加载时apply
5. **强制重算**：新文档自动设为rate=1.0，无需手动配置

## 总结

Online Lazy Loading 模式最适合：
- ✅ 不想等待长时间预处理
- ✅ 需要多次测试不同rate
- ✅ 测试集未完全确定
- ✅ 存储空间有限
- ✅ 真实应用场景模拟

一次性测试所有rate只需：
```bash
bash run_online_lazy_sweep.sh
```
