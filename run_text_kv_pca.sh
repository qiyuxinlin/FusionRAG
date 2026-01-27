#!/bin/bash

#####################################################################
# 文本段落 KV Cache PCA 可视化脚本
#
# 使用方法：
#   bash run_text_kv_pca.sh <选项>
#
# 示例：
#   # 直接在命令行指定文本
#   bash run_text_kv_pca.sh --texts "这是第一个段落。" "这是第二个段落。"
#
#   # 从文件读取文本
#   bash run_text_kv_pca.sh --text_file /home/shm/document/exp/FusionRAG/example_texts.txt --start_token "0,90" --end_token "-1,130" --all_layers 
#   bash run_text_kv_pca.sh --text_file example_texts.txt --all_layers  --start_token "0,0" --end_token "-1,90" 
#
#   # 指定GPU和输出目录
#   bash run_text_kv_pca.sh --texts "Text 1" "Text 2" --device cuda:0 --output_dir ./results
#
#   # 指定要分析的层
#   bash run_text_kv_pca.sh --texts "Text 1" "Text 2" --layers 0 5 10 15 20 27
#####################################################################

cd /home/shm/document/exp/FusionRAG

# 环境配置
PYTHON_PATH="/home/shm/anaconda3/envs/fusionrag/bin/python"
SCRIPT_PATH="./visualize_text_kv_pca.py"

# 模型配置 (根据需要修改)
MODEL_NAME="Qwen2.5-7B-Instruct"
MODEL_PATH="/mnt/data/models/Qwen2.5-7B-Instruct"
MODEL_TYPE="qwen2"

# 默认参数
DEVICE="cuda"
TORCH_DTYPE="float16"
MAX_TOKENS=50000
OUTPUT_DIR="./text_kv_pca_analysis"

# 解析命令行参数
PASSED_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --texts)
            TEXTS_SPECIFIED=true
            PASSED_ARGS+=("$1" "$2")
            shift 2
            ;;
        --text_file)
            TEXT_FILE_SPECIFIED=true
            PASSED_ARGS+=("$1" "$2")
            shift 2
            ;;
        --labels)
            PASSED_ARGS+=("$1" "$2")
            shift 2
            ;;
        --layers)
            # Convert comma-separated to space-separated
            if [[ $2 == *","* ]]; then
                LAYERS_ARG=$(echo $2 | tr ',' ' ')
                PASSED_ARGS+=("--layers" $LAYERS_ARG)
            else
                PASSED_ARGS+=("$1" "$2")
            fi
            shift 2
            ;;
        --all_layers)
            PASSED_ARGS+=("--all_layers")
            shift
            ;;
        --max_tokens)
            MAX_TOKENS=$2
            PASSED_ARGS+=("$1" "$2")
            shift 2
            ;;
        --start_token)
            # Use = syntax to avoid argparse treating negative values as options
            PASSED_ARGS+=("${1}=${2}")
            shift 2
            ;;
        --end_token)
            # Use = syntax to avoid argparse treating negative values as options
            if [[ -z "$2" ]]; then
                echo "❌ 错误: --end_token 需要一个参数值"
                echo "   正确用法: --end_token \"-1,130\""
                exit 1
            fi
            PASSED_ARGS+=("${1}=${2}")
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR=$2
            PASSED_ARGS+=("$1" "$2")
            shift 2
            ;;
        --device)
            DEVICE=$2
            PASSED_ARGS+=("$1" "$2")
            shift 2
            ;;
        --torch_dtype)
            TORCH_DTYPE=$2
            PASSED_ARGS+=("$1" "$2")
            shift 2
            ;;
        --model_path)
            MODEL_PATH=$2
            shift 2
            ;;
        --model_type)
            MODEL_TYPE=$2
            shift 2
            ;;
        -h|--help)
            echo "用法: bash run_text_kv_pca.sh [选项]"
            echo ""
            echo "选项:"
            echo "  --texts \"文本1\" \"文本2\" ...     直接指定文本段落"
            echo "  --text_file FILE                从文件读取文本（每行一个，或 \"标签: 文本\" 格式）"
            echo "  --labels \"标签1\" \"标签2\" ...     为每个文本指定标签"
            echo "  --layers L1 L2 L3 ...           指定要分析的层（默认：均匀分布的6层）"
            echo "  --all_layers                    分析所有层（覆盖 --layers）"
            echo "  --max_tokens NUM                每层最大token数（默认：500）"
            echo "  --start_token \"VALS\"            从第N个token开始（逗号分隔需引号，如 \"0,90\"，默认：0）"
            echo "  --end_token \"VALS\"              到第N个token结束（逗号分隔需引号，如 \"-1,130\"，默认：-1）"
            echo "  --output_dir DIR                输出目录（默认：./text_kv_pca_analysis）"
            echo "  --device DEVICE                 设备（默认：cuda）"
            echo "  --torch_dtype TYPE              数据类型（默认：float16）"
            echo "  --model_path PATH               模型路径"
            echo "  --model_type TYPE               模型类型（qwen2/llama/mistral）"
            echo ""
            echo "示例:"
            echo "  # 对比两个段落"
            echo '  bash run_text_kv_pca.sh --texts "段落一内容" "段落二内容"'
            echo ""
            echo "  # 从第90个token开始（所有文本）"
            echo '  bash run_text_kv_pca.sh --text_file texts.txt --start_token 90'
            echo ""
            echo "  # 为每个文本单独设置范围（逗号分隔，需引号）"
            echo '  bash run_text_kv_pca.sh --text_file texts.txt --start_token "0,90" --end_token "-1,130"'
            echo "    # 文本1: [0, end) 全部"
            echo "    # 文本2: [90, 130) 只看40个token"
            echo ""
            echo "  # 只看特定段落（所有文本）"
            echo '  bash run_text_kv_pca.sh --text_file texts.txt --start_token 90 --end_token 130'
            echo ""
            echo "  # 分析所有层"
            echo '  bash run_text_kv_pca.sh --text_file texts.txt --all_layers'
            echo ""
            echo "  # 指定层"
            echo '  bash run_text_kv_pca.sh --text_file texts.txt --layers 0 5 10 15 20 27'
            echo ""
            echo "  # 指定标签"
            echo '  bash run_text_kv_pca.sh --texts "Text 1" "Text 2" --labels "Baseline" "Method A"'
            exit 0
            ;;
        *)
            echo "⚠️  警告: 未知参数 '$1' (将被忽略)"
            shift
            ;;
    esac
done

# 检查是否提供了文本或文本文件
if [[ -z "$TEXTS_SPECIFIED" ]] && [[ -z "$TEXT_FILE_SPECIFIED" ]]; then
    echo "❌ 错误: 必须指定 --texts 或 --text_file"
    echo "使用 --help 查看使用说明"
    exit 1
fi

# 构建命令（使用数组避免引号问题）
CMD_ARRAY=("${PYTHON_PATH}" "${SCRIPT_PATH}")
CMD_ARRAY+=("--model_path" "${MODEL_PATH}")
CMD_ARRAY+=("--model_name" "${MODEL_NAME}")
CMD_ARRAY+=("--model_type" "${MODEL_TYPE}")
CMD_ARRAY+=("--device" "${DEVICE}")
CMD_ARRAY+=("--torch_dtype" "${TORCH_DTYPE}")
CMD_ARRAY+=("--max_tokens" "${MAX_TOKENS}")
CMD_ARRAY+=("--output_dir" "${OUTPUT_DIR}")

# 添加PASSED_ARGS中的所有参数
for arg in "${PASSED_ARGS[@]}"; do
    CMD_ARRAY+=("${arg}")
done

# 显示配置
echo "=========================================="
echo "文本段落 KV Cache PCA 可视化"
echo "=========================================="
echo "模型: ${MODEL_NAME}"
echo "模型路径: ${MODEL_PATH}"
echo "模型类型: ${MODEL_TYPE}"
echo "设备: ${DEVICE}"
echo "数据类型: ${TORCH_DTYPE}"
echo "输出目录: ${OUTPUT_DIR}"
echo "=========================================="
echo ""

# 运行分析（使用数组，避免引号问题）
"${CMD_ARRAY[@]}"

EXIT_CODE=$?

echo ""
echo "==========================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✓ 分析完成！"
    echo ""
    echo "生成的可视化文件："
    echo "  1. pca_key_cache.png   - Key cache PCA 可视化图"
    echo "  2. pca_value_cache.png - Value cache PCA 可视化图"
    echo "  3. summary_statistics.json - 统计摘要"
    echo ""
    echo "查看结果："
    echo "  ls -lh ${OUTPUT_DIR}/"
else
    echo "✗ 分析失败！退出码: ${EXIT_CODE}"
fi
echo "==========================================="

exit ${EXIT_CODE}
