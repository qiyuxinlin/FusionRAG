# PCA 可视化颜色方案更新说明

## 🎨 更新内容

为了让多方法PCA对比图更加清晰易读，我们更新了颜色方案，特别是**解决了红色和橙色容易混淆**的问题。

## 新旧颜色对比

### ❌ 旧颜色方案的问题

| 方法 | 旧颜色 | 颜色代码 | 问题 |
|------|--------|----------|------|
| no_preprocess | 🔵 蓝色 | `#1f77b4` | - |
| bge | 🟠 **橙色** | `#ff7f0e` | ⚠️ 与红色接近 |
| random | 🟢 绿色 | `#2ca02c` | - |
| repeat_self | 🔴 **红色** | `#d62728` | ⚠️ 与橙色接近 |
| fixed_doc | 🟣 紫色 | `#9467bd` | 对比度不够 |
| random_docs | 🟤 棕色 | `#8c564b` | 不够鲜明 |
| random_text | 🩷 粉色 | `#e377c2` | 不够鲜明 |
| bge_shuffled | ⚫ 灰色 | `#7f7f7f` | - |

**主要问题**：
- 🟠 橙色 (bge) 和 🔴 红色 (repeat_self) 在图上容易混淆
- 某些颜色对比度不够，不够鲜明

---

### ✅ 新颜色方案（高对比度）

| 方法 | 新颜色 | 颜色代码 | 改进 |
|------|--------|----------|------|
| no_preprocess | 🔵 **深蓝色** | `#0066CC` | 更深更清晰 |
| bge | 🟣 **深紫色** | `#9933FF` | ✅ 取代橙色，与红色明显区分 |
| random | 🟢 **亮绿色** | `#00CC44` | 更鲜艳，容易识别 |
| repeat_self | 🔴 **鲜红色** | `#FF0033` | ✅ 更亮，与紫色明显区分 |
| fixed_doc | 🩵 **深青色** | `#00CCCC` | 独特的青色，高辨识度 |
| random_docs | 🟠 **深橙色** | `#FF6600` | 鲜明的橙色 |
| random_text | 💗 **品红色** | `#FF0099` | 更鲜艳的粉色 |
| bge_shuffled | ⚫ **深灰色** | `#666666` | 中性色 |

**核心改进**：
- ✅ **bge 从橙色改为紫色**：与红色有明显的色相差异
- ✅ **所有颜色饱和度增加**：更鲜艳，更容易区分
- ✅ **高对比度组合**：任意两种颜色都容易区分

---

## 📊 可视化对比

查看生成的对比图：
```bash
ls -lh /home/shm/document/exp/FusionRAG/color_scheme_comparison.png
```

或使用图片查看器打开：
```bash
eog /home/shm/document/exp/FusionRAG/color_scheme_comparison.png
```

---

## 🎯 实际效果示例

### 2种方法对比

**旧方案**：
```
Layer 18 - Value
● ● ●  (橙色 bge)
▲ ▲ ▲  (红色 repeat_self) ⚠️ 容易混淆
```

**新方案**：
```
Layer 18 - Value
● ● ●  (紫色 bge) ✅ 清晰
■ ■ ■  (红色 repeat_self) ✅ 清晰
```

### 4种方法对比

**新方案效果**：
```
Layer 27 - Value
🔵 深蓝色 (no_preprocess)
🟣 深紫色 (bge)          ✅ 与红色区分明显
🟢 亮绿色 (random)
🔴 鲜红色 (repeat_self)  ✅ 与紫色区分明显
```

**色彩对比度提升**：
- 蓝色 ↔ 紫色：色相差异大
- 紫色 ↔ 红色：色相差异大（取代了橙色↔红色的小差异）
- 绿色 ↔ 红色：互补色，最大对比
- 青色：独特色相，易识别

---

## 🚀 如何使用新颜色方案

### 自动使用

所有使用 `visualize_kv_pca_multi.py` 的脚本都会自动使用新颜色方案：

```bash
# 预设对比
bash run_kv_pca_compare_preset.sh baseline 5

# 自定义方法
bash run_kv_pca_multi.sh bge repeat_self 5

# Python直接调用
python visualize_kv_pca_multi.py --methods bge repeat_self --sample_ids 0 1 2
```

### 验证新颜色

运行任意多方法对比，查看生成的图：

```bash
# 对比bge和repeat_self（最容易混淆的两个）
bash run_kv_pca_multi.sh bge repeat_self 3

# 查看生成的图
ls ./kv_pca_multi_bge_repeat_self/pca_*.png
```

现在应该能清楚地看到**紫色(bge)**和**红色(repeat_self)**的明显区别！

---

## 🎨 颜色选择原理

### 色相分布

我们选择了在色轮上均匀分布的颜色：

```
        绿色 (120°)
         🟢
         |
蓝色 ----+---- 橙色
🔵      |      🟠
        |
       红色/紫色
      🔴  🟣
```

**关键颜色对**：
- 蓝色(240°) vs 橙色(30°)：互补色，最大对比
- 绿色(120°) vs 红色(0°)：互补色，最大对比
- 紫色(270°) vs 黄绿色：高对比

### 饱和度和亮度

- **高饱和度**：所有颜色使用鲜艳的版本（00、33、CC、FF）
- **适中亮度**：避免过亮（难看清）或过暗（不突出）
- **平衡设计**：既鲜艳又不刺眼

---

## 📚 相关文档更新

以下文档已同步更新颜色信息：

- ✅ `visualize_kv_pca_multi.py` - 核心代码已更新
- ✅ `MULTI_METHOD_PCA_GUIDE.md` - 使用指南已更新
- ✅ `PCA_TOOLS_README.md` - 工具总览已更新

---

## 🐛 如果还是觉得颜色不够分明

### 方案 1: 调整点的大小和透明度

编辑 `visualize_kv_pca_multi.py` 第236和254行：

```python
# 当前设置
ax_key.scatter(..., alpha=0.5, s=20, ...)

# 改为更大的点，更低的透明度
ax_key.scatter(..., alpha=0.7, s=30, ...)
```

### 方案 2: 使用不同的标记

编辑 `visualize_kv_pca_multi.py`，为不同方法使用不同形状：

```python
METHOD_MARKERS = {
    'no_preprocess': 'o',  # 圆形
    'bge': 's',             # 方形
    'random': '^',          # 三角形
    'repeat_self': 'v',     # 倒三角
    'fixed_doc': 'D',       # 菱形
    'random_docs': 'p',     # 五边形
    'random_text': '*',     # 星形
    'bge_shuffled': 'X',    # X形
}
```

### 方案 3: 减少对比方法数量

```bash
# 不要一次对比太多方法（≤4种最佳）
bash run_kv_pca_multi.sh bge repeat_self random
```

### 方案 4: 自定义颜色

直接修改 `visualize_kv_pca_multi.py` 中的 `METHOD_COLORS` 字典，选择你喜欢的颜色组合。

---

## ✅ 总结

| 改进项 | 旧方案 | 新方案 |
|--------|--------|--------|
| bge颜色 | 🟠 橙色 | 🟣 **紫色** ✅ |
| 与红色区分 | ⚠️ 容易混淆 | ✅ **明显区分** |
| 整体对比度 | 中等 | ✅ **高对比度** |
| 颜色饱和度 | 中等 | ✅ **高饱和度** |
| 视觉清晰度 | 一般 | ✅ **非常清晰** |

**核心改进**：bge从橙色改为紫色，解决了与红色混淆的问题！

---

现在就试试新颜色方案吧！🎉

```bash
bash run_kv_pca_multi.sh bge repeat_self 5
```
