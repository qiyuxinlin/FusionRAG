import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM

def plot_re2_attention(model_name, question):
    # 1. 加载分词器和模型
    # 注意：需要输出注意力权重 (output_attentions=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto", 
        torch_dtype=torch.float16, 
        output_attentions=True
    )

    # 2. 构建 RE2 格式的输入
    # 格式参考论文图1和公式：Q: {q} \n Read the question again: {q} \n A: Let's think step by step.
    re2_input = f"Q: {question}\nRead the question again: {question}\nA: Let's think step by step."
    inputs = tokenizer(re2_input, return_tensors="pt").to(model.device)
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    # 3. 获取模型输出
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 4. 提取并处理注意力权重 
    # outputs.attentions 是一个元组，包含每一层 (L, B, H, S, S)
    # 我们将其堆叠并对“层”和“头”维度取平均值
    attentions = torch.stack(outputs.attentions)  # (layers, batch, heads, seq, seq)
    avg_attention = attentions.mean(dim=0).mean(dim=1)[0].cpu().numpy()  # (seq, seq)

    # 5. 绘图
    plt.figure(figsize=(12, 10))
    
    # 使用掩码只显示下三角（自回归模型的特性，对应论文中的三角形展示）
    mask = np.triu(np.ones_like(avg_attention, dtype=bool), k=1)
    
    sns.heatmap(
        avg_attention, 
        xticklabels=tokens, 
        yticklabels=tokens, 
        mask=mask,
        cmap="YlGnBu",  # 类似论文的“越深代表注意力越高” [cite: 56]
        cbar_kws={'label': 'Average Attention Weight'}
    )

    # 标记第一遍和第二遍阅读的范围（根据实际生成的 token 长度手动或自动调整）
    # 这里只是演示，实际需要计算 question token 的索引
    plt.title(f"RE2 Attention Distribution ({model_name})")
    plt.xlabel("Key Tokens")
    plt.ylabel("Query Tokens")
    
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("re2_attention_reproduction.png")
    print("图像已保存为 re2_attention_reproduction.png")

# 示例调用
example_q = "Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?"
# 建议在有显存的环境下运行，替换为你的本地模型路径或 Llama 系列模型
plot_re2_attention("meta-llama/Llama-2-7b-hf", example_q)