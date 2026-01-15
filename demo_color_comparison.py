#!/usr/bin/env python3
"""演示新旧颜色方案的对比"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 旧颜色方案
OLD_COLORS = {
    'no_preprocess': '#1f77b4',
    'bge': '#ff7f0e',
    'random': '#2ca02c',
    'repeat_self': '#d62728',
    'fixed_doc': '#9467bd',
    'random_docs': '#8c564b',
    'random_text': '#e377c2',
    'bge_shuffled': '#7f7f7f',
}

# 新颜色方案
NEW_COLORS = {
    'no_preprocess': '#0066CC',
    'bge': '#9933FF',
    'random': '#00CC44',
    'repeat_self': '#FF0033',
    'fixed_doc': '#00CCCC',
    'random_docs': '#FF6600',
    'random_text': '#FF0099',
    'bge_shuffled': '#666666',
}

# 方法名称
methods = ['no_preprocess', 'bge', 'random', 'repeat_self',
           'fixed_doc', 'random_docs', 'random_text', 'bge_shuffled']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 绘制旧颜色方案
y_pos = list(range(len(methods)))
for i, method in enumerate(methods):
    ax1.barh(i, 1, color=OLD_COLORS[method], height=0.8)
    ax1.text(0.5, i, method, ha='center', va='center', fontsize=11,
             color='white', fontweight='bold')

ax1.set_xlim(0, 1)
ax1.set_ylim(-0.5, len(methods) - 0.5)
ax1.set_yticks([])
ax1.set_xticks([])
ax1.set_title('旧颜色方案\n(橙色和红色较接近)', fontsize=14, fontweight='bold')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)

# 绘制新颜色方案
for i, method in enumerate(methods):
    ax2.barh(i, 1, color=NEW_COLORS[method], height=0.8)
    ax2.text(0.5, i, method, ha='center', va='center', fontsize=11,
             color='white', fontweight='bold')

ax2.set_xlim(0, 1)
ax2.set_ylim(-0.5, len(methods) - 0.5)
ax2.set_yticks([])
ax2.set_xticks([])
ax2.set_title('新颜色方案 ✅\n(高对比度，更易区分)', fontsize=14, fontweight='bold', color='green')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['left'].set_visible(False)

plt.suptitle('PCA 可视化颜色方案对比', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('/home/shm/document/exp/FusionRAG/color_scheme_comparison.png', dpi=120, bbox_inches='tight')
print("✓ 颜色方案对比图已生成: color_scheme_comparison.png")
print("\n新颜色方案特点：")
print("  • 深蓝色 (no_preprocess) - 清晰的baseline颜色")
print("  • 深紫色 (bge) - 取代橙色，与红色区分明显")
print("  • 亮绿色 (random) - 更鲜艳，容易识别")
print("  • 鲜红色 (repeat_self) - 更亮，与紫色区分明显")
print("  • 深青色 (fixed_doc) - 独特的青色")
print("  • 深橙色 (random_docs) - 鲜明的橙色")
print("  • 品红色 (random_text) - 更鲜艳的粉色")
print("  • 深灰色 (bge_shuffled) - 中性色")
