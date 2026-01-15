# KV Cache PCA 分析 - 使用指南

## 📝 三种运行方式

### 方式 1: 快速分析（指定样本数量）

**脚本**: `run_kv_pca_quick.sh`

**用法**:
```bash
bash run_kv_pca_quick.sh <样本数量> [输出目录]
```

**示例**:
```bash
# 分析前5个样本（默认）
bash run_kv_pca_quick.sh 5

# 分析前10个样本
bash run_kv_pca_quick.sh 10

# 分析前20个样本，保存到指定目录
bash run_kv_pca_quick.sh 20 ./my_analysis

# 分析前50个样本
bash run_kv_pca_quick.sh 50
```

**说明**: 自动分析样本 0, 1, 2, ..., N-1

---

### 方式 2: 范围分析（指定样本范围）

**脚本**: `run_kv_pca_range.sh`

**用法**:
```bash
bash run_kv_pca_range.sh <起始ID> <结束ID> [输出目录]
```

**示例**:
```bash
# 分析样本 0-5
bash run_kv_pca_range.sh 0 5

# 分析样本 10-20
bash run_kv_pca_range.sh 10 20

# 分析样本 50-100，保存到指定目录
bash run_kv_pca_range.sh 50 100 ./range_50_100

# 分析单个样本
bash run_kv_pca_range.sh 5 5
```

**说明**: 分析从 start_id 到 end_id（包含两端）的所有样本

---

### 方式 3: 自定义分析（完全控制）

**脚本**: `run_kv_pca_analysis.sh` 或直接调用 Python

#### 3a. 修改配置脚本

编辑 `run_kv_pca_analysis.sh`:
```bash
# 指定任意样本 ID（可以不连续）
SAMPLE_IDS="0 2 5 10 15 20"

# 指定分析的层
LAYERS="0 7 14 21 27"

# 调整采样token数
MAX_TOKENS="1000"
```

然后运行:
```bash
bash run_kv_pca_analysis.sh
```

#### 3b. 直接调用 Python

```bash
/home/shm/anaconda3/envs/fusionrag/bin/python visualize_kv_pca.py \
    --sample_ids 0 2 5 10 15 \
    --chunk_id 1 \
    --layers 0 7 14 21 27 \
    --max_tokens 500 \
    --output_dir ./my_custom_analysis
```

---

## 🎯 常见使用场景

### 场景 1: 快速验证（少量样本）

**目标**: 快速看看趋势，不需要精确统计

**方法**:
```bash
# 分析前3个样本
bash run_kv_pca_quick.sh 3
```

**预期时间**: 1-2 分钟

---

### 场景 2: 标准分析（中等样本）

**目标**: 获得可信的统计结果

**方法**:
```bash
# 分析前10-20个样本
bash run_kv_pca_quick.sh 15
```

**预期时间**: 5-10 分钟

**建议**: 论文使用推荐这个规模

---

### 场景 3: 全面分析（大量样本）

**目标**: 获得最可靠的统计，发现边缘情况

**方法**:
```bash
# 分析前50-100个样本
bash run_kv_pca_quick.sh 50
```

**预期时间**: 20-40 分钟

**建议**:
- 运行在后台: `bash run_kv_pca_quick.sh 50 > analysis.log 2>&1 &`
- 或使用 screen/tmux

---

### 场景 4: 特定样本分析

**目标**: 分析特定感兴趣的样本

**方法**:
```bash
# 方法1: 使用范围脚本
bash run_kv_pca_range.sh 10 20

# 方法2: 修改 run_kv_pca_analysis.sh
# 编辑 SAMPLE_IDS="10 11 12 13 14 15 16 17 18 19 20"
bash run_kv_pca_analysis.sh

# 方法3: 直接指定（不连续）
python visualize_kv_pca.py --sample_ids 5 10 15 20 25
```

---

### 场景 5: 层级详细分析

**目标**: 详细分析每一层的变化

**方法**:
```bash
# 分析所有28层（耗时较长）
python visualize_kv_pca.py \
    --sample_ids 0 1 2 \
    --layers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 \
    --max_tokens 200 \
    --output_dir ./all_layers
```

**预期时间**: 15-30 分钟

---

## 📊 输出文件说明

运行后会在输出目录生成以下文件：

### 1. PCA 散点图 (每个样本一组)

**文件名**:
- `pca_key_example{N}_chunk1.png`
- `pca_value_example{N}_chunk1.png`

**内容**: 多个子图，每个子图显示一层的 2D PCA 投影
- 🔵 蓝色点 = No Preprocess
- 🔴 红色点 = BGE

**如何解读**:
- 重叠多 → 差异小
- 分离明显 → 差异大
- 看 L2 距离数值

### 2. L2 距离趋势图

**文件名**: `l2_distance_trends.png`

**内容**:
- 左图: Key 的 L2 距离随层数变化
- 右图: Value 的 L2 距离随层数变化

**如何解读**:
- 递增趋势 → 深层影响更大
- 平坦 → 各层影响相似
- 多条线重合 → 样本间一致

### 3. 方差解释比例

**文件名**: `pca_variance_explained.png`

**内容**: 前2个主成分解释的方差比例

**如何解读**:
- > 0.6: 2D 投影很好
- 0.4-0.6: 可接受
- < 0.4: 2D 投影损失较多信息

### 4. 统计摘要

**文件名**: `summary_statistics.json`

**内容**: 每层的 L2 距离和方差数值

**用途**:
- 提取数值用于表格
- 进一步统计分析

---

## ⚡ 性能优化建议

### 如果内存不足

```bash
# 减少采样 token 数
MAX_TOKENS="200"  # 默认是 500

# 或减少同时分析的样本数
bash run_kv_pca_quick.sh 5  # 而不是 50
```

### 如果速度太慢

```bash
# 减少分析的层数
LAYERS="0 14 27"  # 只分析首中尾3层

# 减少采样
MAX_TOKENS="200"

# 减少样本
bash run_kv_pca_quick.sh 5
```

### 如果需要更高精度

```bash
# 增加采样 token 数
MAX_TOKENS="1000"  # 或更多

# 分析所有层
LAYERS="0 1 2 3 ... 27"  # 28层全部

# 更多样本
bash run_kv_pca_quick.sh 50
```

---

## 🔍 检查可用样本

如果不确定有多少样本可用：

```bash
# 查看 No Preprocess 的样本
ls /mnt/data3/tmp/fusionrag/Qwen2.5-7B-Instruct/musique/kv_cache/ | grep "_1_key.pt" | wc -l

# 查看 BGE 的样本
ls /mnt/data3/tmp/fusionrag/Qwen2.5-7B-Instruct/musique/preprocess_kv_cache_global_topk10_bge/ | grep "_1_key.pt" | wc -l

# 查看具体的样本 ID
ls /mnt/data3/tmp/fusionrag/Qwen2.5-7B-Instruct/musique/kv_cache/ | grep "_1_key.pt" | sed 's/_1_key.pt//' | head -20
```

---

## 🎓 推荐工作流

### 第一步: 快速探索（5分钟）
```bash
# 分析3个样本，看看大致趋势
bash run_kv_pca_quick.sh 3
# 查看 l2_distance_trends.png
```

### 第二步: 标准分析（10分钟）
```bash
# 分析15个样本，获得可信结果
bash run_kv_pca_quick.sh 15 ./standard_analysis
# 查看所有生成的图表
```

### 第三步: 深入分析（按需）

根据第二步的发现，选择性地：

**如果某层特别有趣**:
```bash
# 详细分析该层附近的层
python visualize_kv_pca.py --sample_ids 0 1 2 3 4 --layers 10 11 12 13 14 15
```

**如果某个样本异常**:
```bash
# 单独分析该样本的所有层
python visualize_kv_pca.py --sample_ids 5 --layers 0 1 2 3 ... 27 --max_tokens 1000
```

---

## 📚 参数速查表

| 参数 | 说明 | 默认值 | 推荐值 |
|------|------|--------|--------|
| `样本数量` | 分析多少个样本 | 5 | 10-20 (论文) |
| `--layers` | 分析哪些层 | 自动6层 | 0 5 11 16 22 27 |
| `--max_tokens` | 每层采样token数 | 500 | 200-1000 |
| `--chunk_id` | 分析哪个chunk | 1 | 1 (文档), 0 (system) |

---

## 🐛 常见问题

### Q1: "FileNotFoundError: KV cache not found"

**原因**: 指定的样本没有 KV cache

**解决**:
```bash
# 检查可用样本
ls /mnt/data3/tmp/fusionrag/Qwen2.5-7B-Instruct/musique/kv_cache/ | grep "_1_key.pt" | head -10

# 使用存在的样本 ID
```

### Q2: "MemoryError" 或 "CUDA out of memory"

**原因**: 内存不足

**解决**:
```bash
# 减少采样
MAX_TOKENS="200"

# 一次分析更少样本
bash run_kv_pca_quick.sh 3
```

### Q3: PCA 方差解释率很低 (< 30%)

**原因**: KV 特征高度复杂，2D 投影损失信息多

**解决**:
- 仍可参考 L2 距离数值
- 或尝试 t-SNE (需修改代码)
- 或增加到 3D PCA (需修改代码)

---

## ✅ 快速参考

```bash
# 最简单：分析5个样本
bash run_kv_pca_quick.sh 5

# 论文推荐：分析15个样本
bash run_kv_pca_quick.sh 15

# 范围分析：样本 10-20
bash run_kv_pca_range.sh 10 20

# 自定义：指定所有参数
python visualize_kv_pca.py \
    --sample_ids 0 1 2 \
    --layers 0 14 27 \
    --max_tokens 500 \
    --output_dir ./output
```

---

**祝分析顺利！🎉**
