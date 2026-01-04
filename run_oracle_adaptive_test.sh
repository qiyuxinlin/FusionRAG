#!/bin/bash
# 快速测试 OracleAdaptive (综合多特征动态比例)

# 设置 GPU
export CUDA_VISIBLE_DEVICES=0

echo "========================================="
echo "Testing OracleAdaptive (Comprehensive Multi-Feature Dynamic Ratio)"
echo "========================================="
echo ""
echo "Configuration:"
echo "  - Method: OracleAdaptive"
echo "  - Base ratio: 30% (reference for dynamic calculation)"
echo "  - Min ratio: 20% (safety buffer for long-tail)"
echo "  - Max ratio: 50%"
echo "  - Features: Coverage + Components + Spread + Gini"
echo ""
echo "Starting test..."
echo ""

python3 test_fusionrag_reflect.py

echo ""
echo "Test completed!"
echo ""
echo "查看结果文件："
echo "  - CSV: ./result/reprocess_method_OracleAdaptive_*.csv"
echo "  - 统计: ./result/reprocess_method_OracleAdaptive_*.txt"
