"""
使用示例：展示如何使用AttentionVisualizer进行各种注意力可视化
"""

from visualize_attention import AttentionVisualizer


def example_1_basic_usage():
    """示例1: 基本使用 - 可视化GPT-2的平均注意力"""
    print("=" * 60)
    print("示例1: 基本使用 - 可视化GPT-2的平均注意力")
    print("=" * 60)

    # 创建可视化器
    visualizer = AttentionVisualizer("gpt2")

    # 输入文本
    text = "The quick brown fox jumps over the lazy dog."

    # 可视化平均注意力
    visualizer.plot_average_attention(
        text,
        output_path="example1_avg_attention.png"
    )

    print("✓ 完成! 查看 example1_avg_attention.png\n")


def example_2_specific_layer():
    """示例2: 可视化特定层的注意力"""
    print("=" * 60)
    print("示例2: 可视化特定层的注意力")
    print("=" * 60)

    visualizer = AttentionVisualizer("gpt2")
    text = "Attention is all you need."

    # 获取模型信息
    info = visualizer.get_model_info()
    print(f"模型有 {info['num_layers']} 层")

    # 可视化第5层
    layer_idx = 5
    visualizer.plot_layer_attention(
        text,
        layer_idx=layer_idx,
        output_path=f"example2_layer_{layer_idx}.png"
    )

    print(f"✓ 完成! 查看 example2_layer_{layer_idx}.png\n")


def example_3_specific_head():
    """示例3: 可视化特定层特定头的注意力"""
    print("=" * 60)
    print("示例3: 可视化特定层特定头的注意力")
    print("=" * 60)

    visualizer = AttentionVisualizer("gpt2")
    text = "Machine learning is fascinating."

    # 可视化第5层第3个头
    layer_idx = 5
    head_idx = 3
    visualizer.plot_head_attention(
        text,
        layer_idx=layer_idx,
        head_idx=head_idx,
        output_path=f"example3_layer_{layer_idx}_head_{head_idx}.png"
    )

    print(f"✓ 完成! 查看 example3_layer_{layer_idx}_head_{head_idx}.png\n")


def example_4_all_layers():
    """示例4: 可视化所有层的注意力"""
    print("=" * 60)
    print("示例4: 可视化所有层的注意力")
    print("=" * 60)

    visualizer = AttentionVisualizer("gpt2")
    text = "Deep learning rocks!"

    # 可视化所有层
    visualizer.plot_all_layers(
        text,
        output_path="example4_all_layers.png"
    )

    print("✓ 完成! 查看 example4_all_layers.png\n")


def example_5_all_heads():
    """示例5: 可视化特定层的所有头"""
    print("=" * 60)
    print("示例5: 可视化特定层的所有头")
    print("=" * 60)

    visualizer = AttentionVisualizer("gpt2")
    text = "Transformers are powerful models."

    # 可视化第5层的所有头
    layer_idx = 5
    visualizer.plot_all_heads(
        text,
        layer_idx=layer_idx,
        output_path=f"example5_all_heads_layer_{layer_idx}.png"
    )

    print(f"✓ 完成! 查看 example5_all_heads_layer_{layer_idx}.png\n")


def example_6_custom_model():
    """示例6: 使用自定义模型路径"""
    print("=" * 60)
    print("示例6: 使用自定义模型路径")
    print("=" * 60)

    # 可以使用本地模型路径或其他Hugging Face模型
    # 例如: "/path/to/your/local/model"
    # 或者: "meta-llama/Llama-2-7b-hf"

    model_name = "gpt2"  # 这里用gpt2示例，可以替换为你的模型
    visualizer = AttentionVisualizer(model_name)

    text = "Custom models work great!"

    visualizer.plot_average_attention(
        text,
        output_path="example6_custom_model.png"
    )

    print("✓ 完成! 查看 example6_custom_model.png\n")


def example_7_no_mask():
    """示例7: 不遮蔽未来tokens（用于编码器模型）"""
    print("=" * 60)
    print("示例7: 不遮蔽未来tokens")
    print("=" * 60)

    visualizer = AttentionVisualizer("gpt2")
    text = "See the full attention matrix."

    # 设置mask_future=False可以看到完整的注意力矩阵
    visualizer.plot_average_attention(
        text,
        output_path="example7_no_mask.png",
        mask_future=False
    )

    print("✓ 完成! 查看 example7_no_mask.png\n")


def example_8_long_text():
    """示例8: 处理较长的文本"""
    print("=" * 60)
    print("示例8: 处理较长的文本")
    print("=" * 60)

    visualizer = AttentionVisualizer("gpt2")

    # 较长的文本
    text = """
    The transformer architecture has revolutionized natural language processing.
    It relies on self-attention mechanisms to process sequential data.
    """

    visualizer.plot_average_attention(
        text,
        output_path="example8_long_text.png",
        figsize=(16, 14)  # 使用更大的图片尺寸
    )

    print("✓ 完成! 查看 example8_long_text.png\n")


def run_all_examples():
    """运行所有示例"""
    examples = [
        example_1_basic_usage,
        example_2_specific_layer,
        example_3_specific_head,
        example_4_all_layers,
        example_5_all_heads,
        example_6_custom_model,
        example_7_no_mask,
        example_8_long_text,
    ]

    print("\n" + "🚀 开始运行所有示例...\n")

    for i, example in enumerate(examples, 1):
        try:
            example()
        except Exception as e:
            print(f"✗ 示例{i}运行失败: {str(e)}\n")
            continue

    print("=" * 60)
    print("🎉 所有示例运行完成!")
    print("=" * 60)


if __name__ == "__main__":
    # 运行所有示例
    run_all_examples()

    # 或者只运行特定示例
    # example_1_basic_usage()
