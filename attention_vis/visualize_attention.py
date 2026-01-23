"""
通用的注意力可视化工具
支持给定任意句子和语言模型，可视化其注意力分布
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from typing import Optional, Union, List
import warnings

warnings.filterwarnings('ignore')


class AttentionVisualizer:
    """注意力可视化器"""

    def __init__(self, model_name: str, device: str = "auto", torch_dtype=torch.float16):
        """
        初始化注意力可视化器

        Args:
            model_name: 模型名称或路径
            device: 设备 ("auto", "cuda", "cpu")
            torch_dtype: 模型数据类型
        """
        print(f"Loading model: {model_name}")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # 尝试加载因果语言模型或普通模型
        # 注意: 必须使用 attn_implementation="eager" 才能获取注意力权重
        # SDPA (Scaled Dot Product Attention) 不支持 output_attentions
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device,
                torch_dtype=torch_dtype,
                attn_implementation="eager"  # 使用 eager 模式以支持注意力输出
            )
        except:
            try:
                self.model = AutoModel.from_pretrained(
                    model_name,
                    device_map=device,
                    torch_dtype=torch_dtype,
                    attn_implementation="eager"  # 使用 eager 模式以支持注意力输出
                )
            except Exception as e:
                print(f"Warning: Failed to load model with eager attention. Trying without it...")
                # 最后尝试不指定 attn_implementation
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        device_map=device,
                        torch_dtype=torch_dtype
                    )
                except:
                    self.model = AutoModel.from_pretrained(
                        model_name,
                        device_map=device,
                        torch_dtype=torch_dtype
                    )

        self.model.eval()
        print(f"Model loaded successfully on {self.model.device}")

    def get_attention_weights(self, text: str) -> tuple:
        """
        获取文本的注意力权重

        Args:
            text: 输入文本

        Returns:
            (tokens, attentions) - tokens列表和注意力张量
        """
        # 编码输入
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

        # 获取模型输出 - 显式请求注意力权重
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)

        # 提取注意力权重
        # attentions: tuple of (batch_size, num_heads, seq_len, seq_len)
        attentions = outputs.attentions

        # 检查是否成功获取注意力权重
        if attentions is None:
            raise ValueError(
                f"Model {self.model_name} did not return attention weights. "
                "This might be because the model doesn't support attention output."
            )

        return tokens, attentions

    def plot_average_attention(
        self,
        text: str,
        output_path: str = "attention_avg.png",
        mask_future: bool = True,
        figsize: tuple = (12, 10)
    ):
        """
        绘制平均注意力分布（对所有层和所有头平均）

        Args:
            text: 输入文本
            output_path: 输出图片路径
            mask_future: 是否遮蔽未来tokens（用于自回归模型）
            figsize: 图片大小
        """
        tokens, attentions = self.get_attention_weights(text)

        # 将所有层的注意力堆叠并平均
        # attentions: tuple of tensors (batch, heads, seq, seq)
        attentions_stack = torch.stack(attentions)  # (layers, batch, heads, seq, seq)

        # 对层和头维度平均
        avg_attention = attentions_stack.mean(dim=0).mean(dim=1)[0].cpu().numpy()  # (seq, seq)

        # 绘图
        plt.figure(figsize=figsize)

        # 如果是自回归模型，遮蔽未来tokens
        mask = None
        if mask_future:
            mask = np.triu(np.ones_like(avg_attention, dtype=bool), k=1)

        sns.heatmap(
            avg_attention,
            xticklabels=tokens,
            yticklabels=tokens,
            mask=mask,
            cmap="YlOrRd",
            cbar_kws={'label': 'Attention Weight'},
            square=True
        )

        plt.title(f"Average Attention Distribution\n{self.model_name}")
        plt.xlabel("Key Tokens")
        plt.ylabel("Query Tokens")
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Average attention plot saved to: {output_path}")
        plt.close()

    def plot_layer_attention(
        self,
        text: str,
        layer_idx: int,
        output_path: str = "attention_layer.png",
        mask_future: bool = True,
        figsize: tuple = (12, 10)
    ):
        """
        绘制特定层的平均注意力分布

        Args:
            text: 输入文本
            layer_idx: 层索引（从0开始）
            output_path: 输出图片路径
            mask_future: 是否遮蔽未来tokens
            figsize: 图片大小
        """
        tokens, attentions = self.get_attention_weights(text)

        if layer_idx >= len(attentions):
            raise ValueError(f"Layer index {layer_idx} out of range. Model has {len(attentions)} layers.")

        # 获取特定层并对头维度平均
        layer_attention = attentions[layer_idx][0].mean(dim=0).cpu().numpy()  # (seq, seq)

        # 绘图
        plt.figure(figsize=figsize)

        mask = None
        if mask_future:
            mask = np.triu(np.ones_like(layer_attention, dtype=bool), k=1)

        sns.heatmap(
            layer_attention,
            xticklabels=tokens,
            yticklabels=tokens,
            mask=mask,
            cmap="YlOrRd",
            cbar_kws={'label': 'Attention Weight'},
            square=True
        )

        plt.title(f"Layer {layer_idx} Attention Distribution\n{self.model_name}")
        plt.xlabel("Key Tokens")
        plt.ylabel("Query Tokens")
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Layer {layer_idx} attention plot saved to: {output_path}")
        plt.close()

    def plot_head_attention(
        self,
        text: str,
        layer_idx: int,
        head_idx: int,
        output_path: str = "attention_head.png",
        mask_future: bool = True,
        figsize: tuple = (12, 10)
    ):
        """
        绘制特定层特定头的注意力分布

        Args:
            text: 输入文本
            layer_idx: 层索引
            head_idx: 头索引
            output_path: 输出图片路径
            mask_future: 是否遮蔽未来tokens
            figsize: 图片大小
        """
        tokens, attentions = self.get_attention_weights(text)

        if layer_idx >= len(attentions):
            raise ValueError(f"Layer index {layer_idx} out of range. Model has {len(attentions)} layers.")

        num_heads = attentions[layer_idx].shape[1]
        if head_idx >= num_heads:
            raise ValueError(f"Head index {head_idx} out of range. Layer {layer_idx} has {num_heads} heads.")

        # 获取特定层特定头的注意力
        head_attention = attentions[layer_idx][0, head_idx].cpu().numpy()  # (seq, seq)

        # 绘图
        plt.figure(figsize=figsize)

        mask = None
        if mask_future:
            mask = np.triu(np.ones_like(head_attention, dtype=bool), k=1)

        sns.heatmap(
            head_attention,
            xticklabels=tokens,
            yticklabels=tokens,
            mask=mask,
            cmap="YlOrRd",
            cbar_kws={'label': 'Attention Weight'},
            square=True
        )

        plt.title(f"Layer {layer_idx}, Head {head_idx} Attention\n{self.model_name}")
        plt.xlabel("Key Tokens")
        plt.ylabel("Query Tokens")
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Layer {layer_idx}, Head {head_idx} attention plot saved to: {output_path}")
        plt.close()

    def plot_all_layers(
        self,
        text: str,
        output_path: str = "attention_all_layers.png",
        mask_future: bool = True,
        max_cols: int = 4
    ):
        """
        绘制所有层的注意力分布（每层对头维度平均）

        Args:
            text: 输入文本
            output_path: 输出图片路径
            mask_future: 是否遮蔽未来tokens
            max_cols: 最大列数
        """
        tokens, attentions = self.get_attention_weights(text)

        num_layers = len(attentions)
        num_cols = min(max_cols, num_layers)
        num_rows = (num_layers + num_cols - 1) // num_cols

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 4*num_rows))
        if num_layers == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for layer_idx in range(num_layers):
            layer_attention = attentions[layer_idx][0].mean(dim=0).cpu().numpy()

            mask = None
            if mask_future:
                mask = np.triu(np.ones_like(layer_attention, dtype=bool), k=1)

            sns.heatmap(
                layer_attention,
                xticklabels=tokens if layer_idx >= num_layers - num_cols else [],
                yticklabels=tokens if layer_idx % num_cols == 0 else [],
                mask=mask,
                cmap="YlOrRd",
                ax=axes[layer_idx],
                cbar=True,
                square=True
            )

            axes[layer_idx].set_title(f"Layer {layer_idx}")
            if layer_idx >= num_layers - num_cols:
                axes[layer_idx].set_xlabel("Keys")
                plt.setp(axes[layer_idx].get_xticklabels(), rotation=90, fontsize=6)
            if layer_idx % num_cols == 0:
                axes[layer_idx].set_ylabel("Queries")
                plt.setp(axes[layer_idx].get_yticklabels(), rotation=0, fontsize=6)

        # 隐藏多余的子图
        for idx in range(num_layers, len(axes)):
            axes[idx].axis('off')

        plt.suptitle(f"Attention Distribution Across All Layers\n{self.model_name}", fontsize=14)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"All layers attention plot saved to: {output_path}")
        plt.close()

    def plot_all_heads(
        self,
        text: str,
        layer_idx: int,
        output_path: str = "attention_all_heads.png",
        mask_future: bool = True,
        max_cols: int = 4
    ):
        """
        绘制特定层所有头的注意力分布

        Args:
            text: 输入文本
            layer_idx: 层索引
            output_path: 输出图片路径
            mask_future: 是否遮蔽未来tokens
            max_cols: 最大列数
        """
        tokens, attentions = self.get_attention_weights(text)

        if layer_idx >= len(attentions):
            raise ValueError(f"Layer index {layer_idx} out of range. Model has {len(attentions)} layers.")

        num_heads = attentions[layer_idx].shape[1]
        num_cols = min(max_cols, num_heads)
        num_rows = (num_heads + num_cols - 1) // num_cols

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 4*num_rows))
        if num_heads == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for head_idx in range(num_heads):
            head_attention = attentions[layer_idx][0, head_idx].cpu().numpy()

            mask = None
            if mask_future:
                mask = np.triu(np.ones_like(head_attention, dtype=bool), k=1)

            sns.heatmap(
                head_attention,
                xticklabels=tokens if head_idx >= num_heads - num_cols else [],
                yticklabels=tokens if head_idx % num_cols == 0 else [],
                mask=mask,
                cmap="YlOrRd",
                ax=axes[head_idx],
                cbar=True,
                square=True
            )

            axes[head_idx].set_title(f"Head {head_idx}")
            if head_idx >= num_heads - num_cols:
                axes[head_idx].set_xlabel("Keys")
                plt.setp(axes[head_idx].get_xticklabels(), rotation=90, fontsize=6)
            if head_idx % num_cols == 0:
                axes[head_idx].set_ylabel("Queries")
                plt.setp(axes[head_idx].get_yticklabels(), rotation=0, fontsize=6)

        # 隐藏多余的子图
        for idx in range(num_heads, len(axes)):
            axes[idx].axis('off')

        plt.suptitle(f"Attention Distribution Across All Heads (Layer {layer_idx})\n{self.model_name}", fontsize=14)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"All heads (Layer {layer_idx}) attention plot saved to: {output_path}")
        plt.close()

    def get_model_info(self):
        """获取模型信息"""
        tokens, attentions = self.get_attention_weights("Hello world")
        num_layers = len(attentions)
        num_heads = attentions[0].shape[1]

        return {
            "model_name": self.model_name,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "device": str(self.model.device)
        }


def main():
    """示例用法"""
    # 示例：使用GPT-2模型
    model_name = "gpt2"  # 可以替换为任何Hugging Face模型
    text = "The quick brown fox jumps over the lazy dog."

    # 创建可视化器
    visualizer = AttentionVisualizer(model_name)

    # 打印模型信息
    info = visualizer.get_model_info()
    print("\nModel Information:")
    print(f"  Model: {info['model_name']}")
    print(f"  Layers: {info['num_layers']}")
    print(f"  Heads per layer: {info['num_heads']}")
    print(f"  Device: {info['device']}")

    # 可视化平均注意力
    print("\n生成平均注意力可视化...")
    visualizer.plot_average_attention(text, "attention_avg.png")

    # 可视化特定层
    print("\n生成层级注意力可视化...")
    visualizer.plot_layer_attention(text, layer_idx=5, output_path="attention_layer_5.png")

    # 可视化特定头
    print("\n生成注意力头可视化...")
    visualizer.plot_head_attention(text, layer_idx=5, head_idx=3, output_path="attention_layer_5_head_3.png")

    # 可视化所有层
    print("\n生成所有层注意力可视化...")
    visualizer.plot_all_layers(text, "attention_all_layers.png")

    # 可视化特定层的所有头
    print("\n生成所有头注意力可视化...")
    visualizer.plot_all_heads(text, layer_idx=5, output_path="attention_all_heads_layer_5.png")

    print("\n所有可视化完成！")


if __name__ == "__main__":
    main()
