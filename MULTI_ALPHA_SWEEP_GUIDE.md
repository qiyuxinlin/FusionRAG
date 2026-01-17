# Multi-Alpha Sweep Guide

## Overview

`run_fusionrag_steering_muliti.sh` supports iterating through multiple steering alpha values in a single run, with automatic organization of results into separate directories.

## Configuration

### Alpha List Setup

```bash
# Define multiple alpha values to test
STEERING_ALPHA_LIST=(0.1 0.2 0.3 0.4 0.5 0.6 0.8 0.9 1.0)

# Define rate values to test
RATE_LIST=(0.0)
```

### Result Directory Structure

Results are automatically organized by alpha value:

```
${RESULT_DIR}/
├── alpha_0.1/
│   ├── FusionRAG_Qwen2.5-7B-Instruct_musique_rate0.0.csv
│   └── ...
├── alpha_0.2/
│   ├── FusionRAG_Qwen2.5-7B-Instruct_musique_rate0.0.csv
│   └── ...
├── alpha_0.3/
│   └── ...
...
```

## Usage

### Example 1: Standard Multi-Alpha Sweep

```bash
# Configure in script
STEERING_ALPHA_LIST=(0.1 0.2 0.3 0.4 0.5)
RATE_LIST=(0.0)

# Run
bash run_fusionrag_steering_muliti.sh
```

**Total tests**: 5 alphas × 1 rate = 5 tests

### Example 2: Multi-Alpha + Multi-Rate

```bash
STEERING_ALPHA_LIST=(0.1 0.3 0.5 0.7 1.0)
RATE_LIST=(0.0 0.3 0.5 1.0)

bash run_fusionrag_steering_muliti.sh
```

**Total tests**: 5 alphas × 4 rates = 20 tests

### Example 3: Fine-Grained Alpha Search

```bash
# Test alpha values around 0.3 with fine granularity
STEERING_ALPHA_LIST=(0.1 0.2 0.25 0.3 0.35 0.4 0.5)
RATE_LIST=(0.0)

bash run_fusionrag_steering_muliti.sh
```

**Total tests**: 7 alphas × 1 rate = 7 tests

## Execution Flow

The script executes in this order:

```
1. Display configuration
   - Show all alpha values
   - Show all rate values
   - Calculate total test count

2. Step 0: Compute Steering Vectors (once)
   - Uses TRAIN_DATASET
   - Saves to: {TRAIN_DATASET}_to_{TEST_DATASET}_steering_manifold.pt
   - Reused for all alpha/rate combinations

3. For each alpha in STEERING_ALPHA_LIST:
   3.1. Create alpha-specific result directory
   3.2. For each rate in RATE_LIST:
        3.2.1. Run test with current alpha and rate
        3.2.2. Save results to alpha-specific directory

4. Summary
   - List all result directories
   - Show completion status
```

## Output Example

```
==========================================
FusionRAG Multi-Parameter Sweep
==========================================
时间: 2026-01-17 10:30:00
GPU: 6
模型: Qwen2.5-7B-Instruct
方法: FusionRAG
召回方法: no_preprocess_with_bias
Rate 列表: 0.0

数据集配置: 跨数据集 (2wikimqa → musique)
  训练数据集: 2wikimqa (用于计算 steering vectors)
  测试数据集: musique (用于实际测试)

Steering Vector 配置:
  计算 Steering: true
  训练数据集: 2wikimqa
  使用所有样本: true
  Manifold 投影: true
  PCA 阈值: 0.7
  Alpha 列表: 0.1 0.2 0.3 0.4 0.5 0.6 0.8 0.9 1.0
  Key 层: all
  Value 层: all
==========================================

将遍历 9 个 Alpha 值 × 1 个 Rate 值
总计: 9 次测试

======================================================================
开始测试 Steering Alpha = 0.1
时间: 2026-01-17 10:35:00
======================================================================

当前 Alpha 结果目录: ./result/steering/alpha_0.1

==========================================
测试 Alpha = 0.1, Rate = 0.0
时间: 2026-01-17 10:35:05
==========================================

✓ Alpha 0.1, Rate 0.0 测试完成
==========================================

======================================================================
✓ Alpha 0.1 的所有测试完成！
  结果保存在: ./result/steering/alpha_0.1
======================================================================

[... 继续其他 alpha 值 ...]

==========================================
所有测试完成！
时间: 2026-01-17 12:30:00
==========================================

存储位置：
  KV Cache: /mnt/data3/tmp/fusionrag
  Results Base: ./result/steering
  Steering Vectors: /mnt/data3/tmp/fusionrag/kv_stats/2wikimqa_to_musique_steering_manifold.pt

测试的参数：
  Alpha 列表: 0.1 0.2 0.3 0.4 0.5 0.6 0.8 0.9 1.0
  Rate 列表: 0.0
  总测试数: 9

结果目录：
  Alpha 0.1: ./result/steering/alpha_0.1/
  Alpha 0.2: ./result/steering/alpha_0.2/
  Alpha 0.3: ./result/steering/alpha_0.3/
  Alpha 0.4: ./result/steering/alpha_0.4/
  Alpha 0.5: ./result/steering/alpha_0.5/
  Alpha 0.6: ./result/steering/alpha_0.6/
  Alpha 0.8: ./result/steering/alpha_0.8/
  Alpha 0.9: ./result/steering/alpha_0.9/
  Alpha 1.0: ./result/steering/alpha_1.0/
```

## Cross-Dataset Support

The multi-alpha sweep fully supports cross-dataset experiments:

```bash
# Train on 2wikimqa, test on musique, sweep alpha values
TRAIN_DATASET="2wikimqa"
TEST_DATASET="musique"
TEST_DATA_PATH="./data/result_reflect.json"

STEERING_ALPHA_LIST=(0.1 0.3 0.5 0.7 1.0)
RATE_LIST=(0.0)

bash run_fusionrag_steering_muliti.sh
```

Each alpha value will use the same steering vectors (computed once on TRAIN_DATASET) but with different intervention strengths.

## Comparing Results Across Alpha Values

After running the sweep, you can compare results:

```bash
# View all result files
ls -lh ./result/steering/alpha_*/FusionRAG_*.csv

# Compare metrics across alpha values
for alpha_dir in ./result/steering/alpha_*; do
    echo "=== $alpha_dir ==="
    tail -n 5 "$alpha_dir"/FusionRAG_*.csv
done
```

## Best Practices

### 1. Start with Wide Range

```bash
# First, test a wide range to identify promising regions
STEERING_ALPHA_LIST=(0.0 0.25 0.5 0.75 1.0)
```

### 2. Refine Around Best Value

```bash
# If 0.5 works best, refine around it
STEERING_ALPHA_LIST=(0.3 0.4 0.45 0.5 0.55 0.6 0.7)
```

### 3. Consider Computational Cost

Each alpha value requires running the full test pipeline:
- For quick validation: Use 3-5 alpha values
- For thorough search: Use 7-10 alpha values
- For publication: Use fine-grained sweep (10-15 values)

### 4. Use Meaningful Ranges

- **Low intervention** (0.1-0.3): Subtle steering, minimal distribution shift
- **Medium intervention** (0.4-0.6): Balanced steering
- **High intervention** (0.7-1.0): Strong steering, may cause instability
- **Very high** (>1.0): Risk of degrading model performance

## Troubleshooting

### Problem: Results are identical across alpha values

**Possible causes**:
1. Steering vectors not being loaded properly
2. `--kv_stats_path` not set correctly
3. `RECALL_METHOD` not set to `no_preprocess_with_bias`

**Solution**: Check logs for "Loaded steering vectors" message

### Problem: Out of memory errors

**Solution**: Reduce the number of concurrent tests or use smaller alpha lists

### Problem: Path conflicts or overwrites

**Solution**: The script automatically creates separate directories - check that `RESULT_DIR` is set correctly

## Comparison with Single-Alpha Script

| Feature | run_fusionrag_steering.sh | run_fusionrag_steering_muliti.sh |
|---------|---------------------------|----------------------------------|
| Alpha values | Single value | Multiple values (list) |
| Result organization | Flat directory | Nested by alpha |
| Total tests | N rates | N alphas × M rates |
| Best for | Single experiment | Parameter search |

## Related Documentation

- [RUN_FUSIONRAG_STEERING_README.md](./RUN_FUSIONRAG_STEERING_README.md) - Single-alpha script guide
- [CROSS_DATASET_STEERING_GUIDE.md](./CROSS_DATASET_STEERING_GUIDE.md) - Cross-dataset experiments
- [NEW_FEATURES.md](./NEW_FEATURES.md) - Layer selection and auto-scan features
- [STEERING_ALPHA_IMPLEMENTATION.md](./STEERING_ALPHA_IMPLEMENTATION.md) - Alpha parameter details

## Summary

The multi-alpha sweep feature enables:
- Efficient parameter search across multiple alpha values
- Automatic result organization by alpha
- Cross-dataset compatibility
- Clear progress tracking and logging

This makes it easy to find the optimal steering strength for your specific use case.
