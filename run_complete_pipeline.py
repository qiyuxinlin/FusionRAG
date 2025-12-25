#!/usr/bin/env python
"""
FusionRAG Complete Pipeline Example
运行完整的process cache实验pipeline，获得性能和质量指标

使用方式:
    python run_complete_pipeline.py

或指定参数:
    python run_complete_pipeline.py --model_type mistral --rate 0.15 --data_name musique-200.jsonl
"""

import os
import sys
import torch
import argparse
import json
from pathlib import Path
from datetime import datetime

# 确保能导入ktransformers
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)


def print_header(text, char="="):
    """打印格式化的标题"""
    width = 80
    print("\n" + char * width)
    print(text.center(width))
    print(char * width + "\n")


def print_section(text, char="-"):
    """打印格式化的小标题"""
    width = 80
    print("\n" + char * width)
    print(text)
    print(char * width + "\n")


def check_system():
    """检查系统配置"""
    print_header("系统配置检查")

    print(f"Python 版本: {sys.version}")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA 设备数: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  Device {i}: {props.name}")
            print(f"    显存: {props.total_memory / 1024**3:.2f} GB")
    else:
        print("⚠️  警告: 未检测到CUDA，将使用CPU推理(速度较慢)")

    print(f"当前工作目录: {os.getcwd()}")


def check_environment(args):
    """检查依赖和文件"""
    print_section("环境依赖检查")

    # 检查模型路径
    if not os.path.exists(args.model_path):
        print(f"❌ 错误: 模型路径不存在: {args.model_path}")
        print(f"   请下载模型到该目录，或通过 --model_path 指定正确路径")
        return False
    else:
        print(f"✓ 模型路径正确: {args.model_path}")

    # 检查数据路径
    data_file = os.path.join(args.data_path, args.data_name)
    if not os.path.exists(data_file):
        print(f"❌ 错误: 数据文件不存在: {data_file}")
        print(f"   可用数据集:")
        data_files = list(Path(args.data_path).glob("*.jsonl"))
        for f in data_files:
            print(f"   - {f.name}")
        return False
    else:
        print(f"✓ 数据文件正确: {data_file}")

    # 检查缓存目录
    os.makedirs(args.cache_path, exist_ok=True)
    print(f"✓ 缓存目录: {args.cache_path}")

    # 检查BGE模型(如果需要预处理)
    if args.preprocess:
        if not os.path.exists(args.bge_model_path):
            print(f"❌ 错误: BGE模型路径不存在: {args.bge_model_path}")
            print(f"   预处理需要BGE模型，请指定正确路径或禁用预处理 (--no-preprocess)")
            return False
        else:
            print(f"✓ BGE模型路径正确: {args.bge_model_path}")

    # 检查Draft模型(如果使用speculative_prefill)
    if args.reprocess_method == "speculative_prefill" and args.draft_model_path:
        if not os.path.exists(args.draft_model_path):
            print(f"❌ 错误: Draft模型路径不存在: {args.draft_model_path}")
            return False
        else:
            print(f"✓ Draft模型路径正确: {args.draft_model_path}")

    return True


def display_config(args):
    """显示运行配置"""
    print_header("运行配置")

    config_dict = {
        "模型配置": {
            "模型类型": args.model_type,
            "模型路径": args.model_path,
            "模型名称": args.model_name,
        },
        "数据配置": {
            "数据集": args.data_name,
            "数据路径": args.data_path,
        },
        "缓存配置": {
            "缓存目录": args.cache_path,
            "最大缓存长度": args.max_cache_len,
        },
        "重处理配置": {
            "重处理方法": args.reprocess_method,
            "重计算比例": f"{args.rate:.2f}",
            "还原RoPE": args.revert_rope,
            "稀疏注意力": args.use_sparse_attention,
        },
        "预处理配置": {
            "启用预处理": args.preprocess,
            "Top-K": args.topk if args.preprocess else "N/A",
            "BGE模型": args.bge_model_path if args.preprocess else "N/A",
        },
        "其他配置": {
            "计算设备": args.device,
            "对比模式": args.compare_with_full_recompute,
        }
    }

    for section, configs in config_dict.items():
        print(f"\n{section}:")
        for key, value in configs.items():
            print(f"  {key:.<30} {value}")


def run_pipeline(args):
    """运行完整pipeline"""
    from ktransformers.unified_process_cache import main

    print_header("开始运行Pipeline", "=")

    start_time = datetime.now()

    try:
        results = main(
            model_type=args.model_type,
            model_path=args.model_path,
            model_name=args.model_name,
            draft_model_path=args.draft_model_path,

            data_name=args.data_name,
            data_path=args.data_path,

            cache_path=args.cache_path,
            max_cache_len=args.max_cache_len,

            rate=args.rate,
            revert_rope=args.revert_rope,
            reprocess_method=args.reprocess_method,
            use_sparse_attention=args.use_sparse_attention,

            preprocess=args.preprocess,
            topk=args.topk,
            bge_model_path=args.bge_model_path,

            device=args.device,
            compare_with_full_recompute=args.compare_with_full_recompute
        )

        elapsed_time = datetime.now() - start_time

        print_header("Pipeline执行完成", "=")

        print(f"总耗时: {elapsed_time}")
        print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        return True

    except Exception as e:
        print(f"\n❌ 错误: Pipeline执行失败")
        print(f"错误信息: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def display_results(args):
    """显示结果文件信息"""
    print_section("结果文件位置")

    # 构建结果文件路径
    result_dir = os.path.join(
        args.cache_path,
        args.data_name.replace('.jsonl', ''),
        args.model_name
    )

    if os.path.exists(result_dir):
        print(f"结果保存在: {result_dir}\n")

        # 列出所有CSV文件
        csv_files = list(Path(result_dir).glob("*.csv"))
        if csv_files:
            print("📊 结果文件:")
            for f in csv_files:
                file_size = f.stat().st_size
                print(f"  - {f.name} ({file_size} bytes)")

        # 列出所有TXT文件
        txt_files = list(Path(result_dir).glob("*.txt"))
        if txt_files:
            print("\n📄 详细结果:")
            for f in txt_files:
                print(f"  - {f.name}")

        print("\n" + "="*80)
        print("查看结果:")
        print("="*80)
        print(f"\n# 查看CSV (完整预测和答案)")
        if csv_files:
            print(f"head -20 {csv_files[0]}")

        print(f"\n# 查看详细统计")
        if txt_files:
            print(f"cat {txt_files[0]}")

    else:
        print(f"⚠️  结果目录不存在: {result_dir}")


def generate_report(args, success):
    """生成运行报告"""
    print_section("生成运行报告")

    report_file = os.path.join(
        args.cache_path,
        f"pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    report = {
        "timestamp": datetime.now().isoformat(),
        "status": "success" if success else "failed",
        "config": {
            "model_type": args.model_type,
            "model_name": args.model_name,
            "data_name": args.data_name,
            "reprocess_method": args.reprocess_method,
            "rate": args.rate,
            "preprocess": args.preprocess,
            "topk": args.topk if args.preprocess else None,
        },
        "environment": {
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "pytorch_version": torch.__version__,
            "python_version": sys.version,
        }
    }

    os.makedirs(args.cache_path, exist_ok=True)
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"✓ 报告已保存: {report_file}")

    return report_file


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="FusionRAG Complete Pipeline - 完整的Process Cache实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:

  # 默认配置(Mistral + ProcessCache)
  python run_complete_pipeline.py

  # Qwen + CacheBlend
  python run_complete_pipeline.py \\
    --model_type qwen \\
    --model_path /path/to/Qwen2.5-7B \\
    --model_name Qwen2.5-7B \\
    --reprocess_method cacheBlend \\
    --no-preprocess

  # Llama + 完全重计算(性能基线)
  python run_complete_pipeline.py \\
    --model_type llama \\
    --model_path /path/to/Llama-3.1-8B \\
    --model_name Llama-3.1-8B \\
    --rate 1.0

  # 启用对比模式
  python run_complete_pipeline.py \\
    --compare-with-full-recompute
        """
    )

    # 模型配置
    parser.add_argument('--model_type', type=str, default='mistral',
                        choices=['mistral', 'qwen', 'pangu', 'llama'],
                        help='模型架构类型 (默认: mistral)')
    parser.add_argument('--model_path', type=str,
                        default='./models/Mistral-7B-Instruct-v0.3',
                        help='模型权重路径')
    parser.add_argument('--model_name', type=str,
                        default='Mistral-7B-Instruct-v0.3',
                        help='模型名称(用于日志和输出)')
    parser.add_argument('--draft_model_path', type=str, default=None,
                        help='Draft模型路径(仅用于speculative_prefill)')

    # 数据配置
    parser.add_argument('--data_name', type=str, default='musique-200.jsonl',
                        choices=['musique-200.jsonl', '2wikimqa-200.jsonl',
                                 'hotpotqa-260-100-10-doc.jsonl',
                                 'triviaqa-270-100-10-doc.jsonl'],
                        help='数据集文件名')
    parser.add_argument('--data_path', type=str, default='./data/',
                        help='数据集目录')

    # 缓存配置
    parser.add_argument('--cache_path', type=str, default='./output/cache/',
                        help='缓存存储目录')
    parser.add_argument('--max_cache_len', type=int, default=32768,
                        help='最大缓存长度')

    # 重处理配置
    parser.add_argument('--rate', type=float, default=0.15,
                        help='重计算比例 (0.0-1.0, 默认: 0.15)')
    parser.add_argument('--reprocess_method', type=str, default='cacheBlend',
                        choices=['processCache', 'cacheBlend', 'Cache-Craft',
                                 'speculative_prefill', 'frontRow'],
                        help='重处理方法')
    parser.add_argument('--revert_rope', action='store_true',
                        help='是否还原RoPE位置编码')
    parser.add_argument('--no-sparse-attention', dest='use_sparse_attention',
                        action='store_false', default=True,
                        help='禁用稀疏注意力')

    # 预处理配置
    parser.add_argument('--no-preprocess', dest='preprocess',
                        action='store_false', default=True,
                        help='禁用KV缓存预处理')
    parser.add_argument('--topk', type=int, default=10,
                        help='预处理的Top-K相似段落数')
    parser.add_argument('--bge_model_path', type=str,
                        default='./models/bge-m3-FP16',
                        help='BGE嵌入模型路径')

    # 其他配置
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='计算设备 (默认: cuda:0)')
    parser.add_argument('--compare-with-full-recompute', action='store_true',
                        help='与完全重计算进行对比')

    args = parser.parse_args()

    print_header("FusionRAG 完整Pipeline运行脚本", "=")

    # 第一步: 检查系统
    check_system()

    # 第二步: 检查环境
    if not check_environment(args):
        print_header("❌ 环境检查失败，终止运行", "=")
        return 1

    # 第三步: 显示配置
    display_config(args)

    # 确认继续
    print("\n")
    input("按 Enter 键继续运行Pipeline... (Ctrl+C 取消)")

    # 第四步: 运行Pipeline
    success = run_pipeline(args)

    # 第五步: 显示结果
    if success:
        display_results(args)

    # 第六步: 生成报告
    report_file = generate_report(args, success)

    # 最后: 总结
    print_header("运行总结", "=")
    if success:
        print("✓ Pipeline执行成功!\n")
        print("后续步骤:")
        print("1. 查看结果文件:")
        result_dir = os.path.join(
            args.cache_path,
            args.data_name.replace('.jsonl', ''),
            args.model_name
        )
        print(f"   ls -la {result_dir}/\n")

        print("2. 分析结果:")
        csv_file = os.path.join(
            result_dir,
            f"reprocess_method_{args.reprocess_method}_rate_{args.rate}_revert_rope_{args.revert_rope}.csv"
        )
        if args.preprocess:
            csv_file = csv_file.replace(".csv", f"_topk_{args.topk}.csv")
        print(f"   cat {csv_file}\n")

        print("3. 查看报告:")
        print(f"   cat {report_file}\n")

        return 0
    else:
        print("❌ Pipeline执行失败!\n")
        print("检查要点:")
        print("1. 模型路径是否正确")
        print("2. 数据文件是否存在")
        print("3. 是否有足够的显存")
        print("4. 检查上面的错误信息\n")
        print(f"详细报告: {report_file}\n")

        return 1


if __name__ == "__main__":
    sys.exit(main())
