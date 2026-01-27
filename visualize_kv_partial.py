#!/usr/bin/env python3
"""
分段可视化KV cache - 验证后半部分是否有差异
"""
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

model_path = "/mnt/data/models/Qwen2.5-7B-Instruct"
device = "cuda"

print("加载模型...")
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    device_map=device,
    trust_remote_code=True,
    local_files_only=True
)
model.eval()

# 读取文本
with open("/home/shm/document/exp/FusionRAG/example_texts.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

texts = {}
for line in lines:
    line = line.strip()
    if not line or line.startswith('#'):
        continue
    if ':' in line and len(line.split(':', 1)) == 2:
        label, text = line.split(':', 1)
        texts[label.strip()] = text.strip()

# 只取AI_Tech和AI_Tech2
texts = {k: v for k, v in texts.items() if k in ['AI_Tech', 'AI_Tech2']}

print(f"\n文本数量: {len(texts)}")

# 生成KV cache
all_key_caches = []
for label, text in texts.items():
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
    key_cache = outputs.past_key_values
    all_key_caches.append(key_cache)
    print(f"{label}: {len(inputs['input_ids'][0])} tokens")

# 分析第0层的前半部分和后半部分
layer_idx = 0
print(f"\n分析Layer {layer_idx}")

# key_cache的结构: past_key_values[layer_idx][0] where 0=key, 1=value
# Shape: [batch_size, num_kv_heads, seq_len, head_dim]
key_cache_1 = all_key_caches[0][layer_idx][0].squeeze(0).cpu()  # [num_kv_heads, seq_len, head_dim]
key_cache_2 = all_key_caches[1][layer_idx][0].squeeze(0).cpu()

seq_len_1 = key_cache_1.shape[1]
seq_len_2 = key_cache_2.shape[1]

print(f"AI_Tech序列长度: {seq_len_1}")
print(f"AI_Tech2序列长度: {seq_len_2}")

# 提取特征
def extract_features(kv_cache, start_idx=0, end_idx=None):
    # kv_cache shape: [num_heads, seq_len, head_dim]
    if end_idx is None:
        end_idx = kv_cache.shape[1]
    subset = kv_cache[:, start_idx:end_idx, :]  # [num_heads, subset_len, head_dim]
    # Transpose to [subset_len, num_heads, head_dim], then flatten
    subset = subset.permute(1, 0, 2)  # [subset_len, num_heads, head_dim]
    subset = subset.reshape(subset.shape[0], -1)  # [subset_len, num_heads * head_dim]
    return subset.numpy()

# 1. 全部token（原始行为）
feat1_all = extract_features(key_cache_1)
feat2_all = extract_features(key_cache_2)
all_feat = np.vstack([feat1_all, feat2_all])
pca = PCA(n_components=2)
pca_result = pca.fit_transform(all_feat)
pca1 = pca_result[:seq_len_1]
pca2 = pca_result[seq_len_1:]

# 2. 只看后半部分（从token 90开始）
feat2_second_half = extract_features(key_cache_2, start_idx=90, end_idx=seq_len_2)
print(f"\nAI_Tech2后半部分: {feat2_second_half.shape[0]} tokens (从90到{seq_len_2})")

# 对后半部分做PCA（单独）
pca2_half = PCA(n_components=2).fit_transform(feat2_second_half)

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 图1: 全部token（两个文本）
ax = axes[0]
ax.scatter(pca1[:, 0], pca1[:, 1], alpha=0.6, s=20, c='blue', label='AI_Tech (90 tokens)')
ax.scatter(pca2[:, 0], pca2[:, 1], alpha=0.6, s=20, c='red', label='AI_Tech2 (164 tokens)')
ax.set_title('全部token - 两个文本')
ax.legend()
ax.grid(True, alpha=0.3)

# 图2: AI_Tech2的前90个token vs 后74个token
feat2_first_90 = extract_features(key_cache_2, start_idx=0, end_idx=90)
feat2_last_74 = extract_features(key_cache_2, start_idx=90, end_idx=seq_len_2)

combined_half = np.vstack([feat2_first_90, feat2_last_74])
pca_half = PCA(n_components=2).fit_transform(combined_half)

ax = axes[1]
ax.scatter(pca_half[:90, 0], pca_half[:90, 1], alpha=0.6, s=20, c='blue', label='AI_Tech2前90个token')
ax.scatter(pca_half[90:, 0], pca_half[90:, 1], alpha=0.6, s=20, c='green', label='AI_Tech2后74个token')
ax.set_title('AI_Tech2内部: 前90 vs 后74')
ax.legend()
ax.grid(True, alpha=0.3)

# 图3: 只看AI_Tech2的后74个token
ax = axes[2]
ax.scatter(pca2_half[:, 0], pca2_half[:, 1], alpha=0.6, s=30, c='green', label='AI_Tech2后74个token（医学内容）')
ax.set_title('AI_Tech2后半部分单独可视化')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/shm/document/exp/FusionRAG/debug_partial_tokens.png', dpi=150, bbox_inches='tight')
print("\n✓ 保存到 debug_partial_tokens.png")
print("\n结论:")
print("  - 图1: 如果红蓝点重合，说明整体分布被前90个相同token主导")
print("  - 图2: 绿色点（医学内容）应该与蓝色点（AI内容）有差异")
print("  - 图3: 单独看后半部分的分布")
