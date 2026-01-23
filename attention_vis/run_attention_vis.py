#!/usr/bin/env python3
"""
命令行接口：注意力可视化工具
支持通过命令行参数快速可视化语言模型的注意力分布
"""

import argparse
import os
from visualize_attention import AttentionVisualizer


def main():
    parser = argparse.ArgumentParser(
        description="可视化语言模型的注意力分布",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 可视化平均注意力
  python run_attention_vis.py --model gpt2 --text "Hello world" --mode avg

  # 可视化特定层
  python run_attention_vis.py --model gpt2 --text "Hello world" --mode layer --layer 5

  # 可视化特定层的特定头
  python run_attention_vis.py --model gpt2 --text "Hello world" --mode head --layer 5 --head 3

  # 可视化所有层
  python run_attention_vis.py --model gpt2 --text "Hello world" --mode all-layers

  # 可视化特定层的所有头
  python run_attention_vis.py --model gpt2 --text "Hello world" --mode all-heads --layer 5

  # 使用自定义模型路径
  python run_attention_vis.py --model /path/to/model --text "Your text here" --mode avg

  # 不遮蔽未来tokens（用于编码器模型）
  python run_attention_vis.py --model bert-base-uncased --text "Hello world" --mode avg --no-mask
        """
    )

    # 必需参数
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='模型名称或路径（Hugging Face模型ID或本地路径）'
    )

    parser.add_argument(
        '--text', '-t',
        type=str,
        required=True,
        help='要分析的文本'
    )

    # 可视化模式
    parser.add_argument(
        '--mode',
        type=str,
        choices=['avg', 'layer', 'head', 'all-layers', 'all-heads', 'info'],
        default='avg',
        help="""可视化模式:
  avg: 平均注意力（所有层和头）
  layer: 特定层的平均注意力
  head: 特定层特定头的注意力
  all-layers: 所有层的注意力
  all-heads: 特定层所有头的注意力
  info: 仅显示模型信息"""
    )

    # 可选参数
    parser.add_argument(
        '--layer', '-l',
        type=int,
        default=None,
        help='层索引（用于layer、head、all-heads模式）'
    )

    parser.add_argument(
        '--head',
        type=int,
        default=None,
        help='头索引（用于head模式）'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='输出图片路径（默认根据模式自动生成）'
    )

    parser.add_argument(
        '--no-mask',
        action='store_true',
        help='不遮蔽未来tokens（用于编码器模型如BERT）'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='运行设备'
    )

    parser.add_argument(
        '--dtype',
        type=str,
        default='float16',
        choices=['float16', 'float32', 'bfloat16'],
        help='模型数据类型'
    )

    parser.add_argument(
        '--figsize',
        type=str,
        default='12,10',
        help='图片大小（格式: width,height）'
    )

    args = parser.parse_args()

    # 解析图片大小
    try:
        figsize = tuple(map(int, args.figsize.split(',')))
        if len(figsize) != 2:
            raise ValueError
    except:
        print("错误: figsize格式应为 'width,height'")
        return

    # 解析数据类型
    import torch
    dtype_map = {
        'float16': torch.float16,
        'float32': torch.float32,
        'bfloat16': torch.bfloat16
    }
    torch_dtype = dtype_map[args.dtype]

    # 创建可视化器
    print(f"初始化注意力可视化器...")
    visualizer = AttentionVisualizer(
        args.model,
        device=args.device,
        torch_dtype=torch_dtype
    )

    # 获取模型信息
    info = visualizer.get_model_info()
    print("\n" + "="*60)
    print("模型信息:")
    print(f"  模型名称: {info['model_name']}")
    print(f"  层数: {info['num_layers']}")
    print(f"  每层注意力头数: {info['num_heads']}")
    print(f"  设备: {info['device']}")
    print("="*60 + "\n")

    # 如果只是查询信息，直接返回
    if args.mode == 'info':
        return

    # 检查参数
    mask_future = not args.no_mask

    if args.mode == 'layer' and args.layer is None:
        print("错误: layer模式需要指定 --layer 参数")
        return

    if args.mode == 'head' and (args.layer is None or args.head is None):
        print("错误: head模式需要指定 --layer 和 --head 参数")
        return

    if args.mode == 'all-heads' and args.layer is None:
        print("错误: all-heads模式需要指定 --layer 参数")
        return

    # 生成默认输出路径
    if args.output is None:
        if args.mode == 'avg':
            output_path = "attention_avg.png"
        elif args.mode == 'layer':
            output_path = f"attention_layer_{args.layer}.png"
        elif args.mode == 'head':
            output_path = f"attention_layer_{args.layer}_head_{args.head}.png"
        elif args.mode == 'all-layers':
            output_path = "attention_all_layers.png"
        elif args.mode == 'all-heads':
            output_path = f"attention_all_heads_layer_{args.layer}.png"
    else:
        output_path = args.output

    # 执行可视化
    print(f"文本: {args.text}")
    print(f"可视化模式: {args.mode}")
    print(f"输出路径: {output_path}\n")

    try:
        if args.mode == 'avg':
            visualizer.plot_average_attention(
                args.text,
                output_path=output_path,
                mask_future=mask_future,
                figsize=figsize
            )

        elif args.mode == 'layer':
            visualizer.plot_layer_attention(
                args.text,
                layer_idx=args.layer,
                output_path=output_path,
                mask_future=mask_future,
                figsize=figsize
            )

        elif args.mode == 'head':
            visualizer.plot_head_attention(
                args.text,
                layer_idx=args.layer,
                head_idx=args.head,
                output_path=output_path,
                mask_future=mask_future,
                figsize=figsize
            )

        elif args.mode == 'all-layers':
            visualizer.plot_all_layers(
                args.text,
                output_path=output_path,
                mask_future=mask_future
            )

        elif args.mode == 'all-heads':
            visualizer.plot_all_heads(
                args.text,
                layer_idx=args.layer,
                output_path=output_path,
                mask_future=mask_future
            )

        print(f"\n✓ 可视化完成！图片已保存到: {output_path}")

    except Exception as e:
        print(f"\n✗ 错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
