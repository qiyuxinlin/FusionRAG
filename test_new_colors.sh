#!/bin/bash

#####################################################################
# 快速测试新颜色方案
#
# 这个脚本会生成一个小规模的对比图，让你快速验证新颜色是否清晰
#####################################################################

echo "==========================================="
echo "测试新颜色方案"
echo "==========================================="
echo ""
echo "生成测试图片，对比最容易混淆的两个方法："
echo "  • bge (紫色) - 原来是橙色"
echo "  • repeat_self (红色)"
echo ""
echo "如果新颜色方案有效，你应该能清楚地看到紫色和红色的区别！"
echo ""
echo "==========================================="
echo ""

cd /home/shm/document/exp/FusionRAG

# 对比bge和repeat_self（原来最容易混淆）
echo "运行分析（分析3个样本，只需1-2分钟）..."
bash run_kv_pca_multi.sh bge repeat_self 3

EXIT_CODE=$?

if [ ${EXIT_CODE} -eq 0 ]; then
    echo ""
    echo "==========================================="
    echo "✓ 测试完成！"
    echo "==========================================="
    echo ""
    echo "查看生成的图片："
    OUTPUT_DIR="/home/shm/document/exp/FusionRAG/kv_pca_analysis6"
    echo "  ls ${OUTPUT_DIR}/*.png"
    echo ""
    echo "使用图片查看器打开："
    echo "  eog ${OUTPUT_DIR}/pca_*.png"
    echo ""
    echo "颜色对比："
    echo "  🟣 紫色 = bge（新颜色，原来是橙色）"
    echo "  🔴 红色 = repeat_self"
    echo ""
    echo "如果两种颜色清晰可辨，说明新方案成功！✨"
    echo "==========================================="
else
    echo "✗ 测试失败"
fi
