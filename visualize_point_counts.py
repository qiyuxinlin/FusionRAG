#!/usr/bin/env python3
"""
对比不同点数的视觉效果
"""
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 左图：90个点
np.random.seed(42)
points_90 = np.random.randn(90, 2)
axes[0].scatter(points_90[:, 0], points_90[:, 1], alpha=0.5, s=20, c='blue')
axes[0].set_title('90 points (AI_Tech)', fontsize=14)
axes[0].grid(True, alpha=0.3)
axes[0].text(0.5, -0.1, f'Actual count: {len(points_90)}',
            ha='center', transform=axes[0].transAxes, fontsize=12)

# 右图：183个点（90个重叠 + 93个新点）
np.random.seed(42)
points_183_overlap = np.random.randn(90, 2)  # 前90个与左图位置相同
np.random.seed(43)
points_183_new = np.random.randn(93, 2) * 0.5 + 0.5  # 后93个新点
points_183 = np.vstack([points_183_overlap, points_183_new])

# 先画重叠的90个点（红色）
axes[1].scatter(points_183_overlap[:, 0], points_183_overlap[:, 1],
               alpha=0.5, s=20, c='red', label='Overlap with AI_Tech (90)')
# 再画新增的93个点（红色）
axes[1].scatter(points_183_new[:, 0], points_183_new[:, 1],
               alpha=0.5, s=20, c='darkred', label='Unique to AI_Tech2 (93)')
axes[1].set_title('183 points (AI_Tech2)', fontsize=14)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)
axes[1].text(0.5, -0.1, f'Actual count: {len(points_183)}',
            ha='center', transform=axes[1].transAxes, fontsize=12)

plt.tight_layout()
plt.savefig('/home/shm/document/exp/FusionRAG/point_count_comparison.png', dpi=150, bbox_inches='tight')
print("✓ 保存到 point_count_comparison.png")
print("\n观察要点：")
print("  - 左图：90个蓝色点")
print("  - 右图：183个红色点，其中90个与左图重叠（浅红）+ 93个新点（深红）")
print("  - 视觉上：两边的密度看起来差异不大")
print("  - 这就是为什么你看到的可视化效果看起来点数差不多")
