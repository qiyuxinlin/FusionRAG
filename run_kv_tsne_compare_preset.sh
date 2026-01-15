#!/bin/bash

#####################################################################
# KV Cache t-SNE 预设对比分析脚本
#
# 提供常用的对比预设，方便快速分析
#
# 使用方法：
#   bash run_kv_tsne_compare_preset.sh <preset> [样本数] [输出目录]
#
# 可用预设：
#   baseline    - no_preprocess vs bge (默认)
#   ablation2   - bge vs random
#   ablation3   - bge vs random vs repeat_self
#   ablation4   - bge vs random vs repeat_self vs fixed_doc
#   ablation_all - 对比所有ablation方法（6个方法）
#   random_methods - 对比所有随机相关方法
#
# 示例：
#   bash run_kv_tsne_compare_preset.sh baseline 3
#   bash run_kv_tsne_compare_preset.sh ablation3 5
#####################################################################

PRESET=${1:-baseline}
NUM_SAMPLES=${2:-5}
OUTPUT_DIR=${3:-""}

echo "==========================================="
echo "KV Cache t-SNE 预设对比分析"
echo "==========================================="
echo "预设: ${PRESET}"
echo "样本数: ${NUM_SAMPLES}"
echo "==========================================="
echo ""

case $PRESET in
    baseline)
        echo "📊 对比: no_preprocess (baseline) vs bge (主方法)"
        METHODS="no_preprocess bge"
        [ -z "$OUTPUT_DIR" ] && OUTPUT_DIR="./tsne_baseline_vs_bge"
        ;;

    ablation2)
        echo "📊 对比: bge vs random (2种方法)"
        METHODS="bge random"
        [ -z "$OUTPUT_DIR" ] && OUTPUT_DIR="./tsne_bge_vs_random"
        ;;

    ablation3)
        echo "📊 对比: bge vs random vs repeat_self (3种方法)"
        METHODS="bge random repeat_self"
        [ -z "$OUTPUT_DIR" ] && OUTPUT_DIR="./tsne_ablation3"
        ;;

    ablation4)
        echo "📊 对比: bge vs random vs repeat_self vs fixed_doc (4种方法)"
        METHODS="bge random repeat_self fixed_doc"
        [ -z "$OUTPUT_DIR" ] && OUTPUT_DIR="./tsne_ablation4"
        ;;

    ablation_all)
        echo "📊 对比: 所有ablation方法 (6种)"
        echo "   no_preprocess, bge, random, repeat_self, fixed_doc, random_docs"
        METHODS="no_preprocess bge random repeat_self fixed_doc random_docs"
        [ -z "$OUTPUT_DIR" ] && OUTPUT_DIR="./tsne_ablation_all"
        ;;

    random_methods)
        echo "📊 对比: 所有随机相关方法 (4种)"
        echo "   no_preprocess, random, random_docs, random_text"
        METHODS="no_preprocess random random_docs random_text"
        [ -z "$OUTPUT_DIR" ] && OUTPUT_DIR="./tsne_random_methods"
        ;;

    bge_variants)
        echo "📊 对比: BGE及其变体 (2种)"
        echo "   bge, bge_shuffled"
        METHODS="bge bge_shuffled"
        [ -z "$OUTPUT_DIR" ] && OUTPUT_DIR="./tsne_bge_variants"
        ;;

    *)
        echo "❌ 错误: 未知预设 '${PRESET}'"
        echo ""
        echo "可用预设："
        echo "  baseline      - no_preprocess vs bge"
        echo "  ablation2     - bge vs random"
        echo "  ablation3     - bge vs random vs repeat_self"
        echo "  ablation4     - bge vs random vs repeat_self vs fixed_doc"
        echo "  ablation_all  - 所有ablation方法（6个）"
        echo "  random_methods - 所有随机相关方法"
        echo "  bge_variants  - BGE及其变体"
        exit 1
        ;;
esac

echo ""
echo "⚠️  注意: t-SNE比PCA慢，建议使用较少的样本（3-5个）"
echo ""
echo "调用: bash run_kv_tsne_multi.sh ${METHODS} ${NUM_SAMPLES} ${OUTPUT_DIR}"
echo ""

# 调用主脚本
bash /home/shm/document/exp/FusionRAG/run_kv_tsne_multi.sh ${METHODS} ${NUM_SAMPLES} ${OUTPUT_DIR}
