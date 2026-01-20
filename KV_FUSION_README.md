# KV Fusion预处理模块使用指南

本文档说明如何使用独立的KV融合预处理模块来研究不同预处理方式对KV cache的影响。

## 模块概述

### 1. `kv_fusion_preprocess.py` - KV融合预处理核心模块

从 `analyse_data.py` 抽离的核心KV融合逻辑,包含:

**主要类:**
- `KVCacheFusion`: KV cache融合处理器
- `DocumentRetriever`: 文档召回器
- `RecallMethod`: 召回方法枚举(BGE/Random/RepeatSelf等)
- `PreprocessScope`: 检索范围枚举(Global/PerExample/SkipUntested)

**主要功能:**
1. 生成文档的独立KV cache
2. 基于不同策略召回相关文档
3. 融合多个文档的KV cache
4. 应用分布对齐steering vector

### 2. `analyse_kv_preprocessing.py` - KV分析工具

分析和可视化不同预处理方法的效果,包含:

**主要类:**
- `KVCacheAnalyzer`: 单个方法的KV分析器
- `KVComparisonAnalyzer`: 多方法对比分析器

**主要功能:**
1. 统计KV cache的分布特征(均值、方差、范数等)
2. 计算不同方法间的KV差异(L2距离、余弦相似度等)
3. 分析KV在不同层的变化模式
4. 可视化对比结果(热图、曲线图等)

## 支持的召回方法

| 方法 | 说明 | 研究目的 |
|------|------|----------|
| `BGE` | 基于BGE模型的语义相似度召回 | 原始FusionRAG方法,作为baseline |
| `RANDOM` | 随机采样文档 | 研究语义相关性的重要性 |
| `REPEAT_SELF` | 重复当前文档K次 | 研究多样性vs重复的影响 |
| `FIXED_DOC` | 使用固定文档召回 | 控制变量,研究特定文档的影响 |
| `RANDOM_TEXT` | BGE召回长度,但使用随机文本KV | 研究语义内容vs长度的影响 |
| `BGE_SHUFFLED` | BGE召回后打乱KV位置 | 研究位置信息的重要性 |
| `RANDOM_DOCS` | 随机文档,保持原始KV长度 | 研究长度匹配的重要性 |
| `NO_PREPROCESS_WITH_BIAS` | 不融合,仅应用分布对齐 | 研究分布对齐的独立效果 |

## 使用方法

### 方法1: 使用完整的pipeline函数

这是最简单的方式,适合快速实验:

```python
from kv_fusion_preprocess import preprocess_documents_pipeline, RecallMethod
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch

# 加载模型
model_path = "/path/to/your/model"
config = AutoConfig.from_pretrained(model_path)
config.torch_dtype = torch.float16

model = AutoModelForCausalLM.from_pretrained(
    model_path, config=config, torch_dtype=torch.float16, device_map='auto'
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 准备文档
documents = [
    {'text': 'Document 1 content...', 'id': 'doc_0'},
    {'text': 'Document 2 content...', 'id': 'doc_1'},
    {'text': 'Document 3 content...', 'id': 'doc_2'},
]
system_prompt = "You are a helpful assistant."

# 运行预处理
result = preprocess_documents_pipeline(
    model=model,
    tokenizer=tokenizer,
    model_type='qwen2',  # 或 'mistral', 'llama'等
    documents=documents,
    system_prompt=system_prompt,
    output_dir='./kv_cache_output',
    bge_model_path='/path/to/bge-m3',  # BGE方法需要
    topk=3,  # 融合top-3文档
    recall_method=RecallMethod.BGE,
    device='cuda:0'
)

print(f"处理完成! KV cache保存在: {result['preprocess_dir']}")
```

### 方法2: 使用单独的类进行细粒度控制

适合需要更灵活控制的研究:

```python
from kv_fusion_preprocess import KVCacheFusion, DocumentRetriever, RecallMethod

# 1. 初始化KV融合器
kv_fusion = KVCacheFusion(
    model=model,
    tokenizer=tokenizer,
    model_type='qwen2',
    device='cuda:0'
)

# 2. 生成独立KV cache
for doc_idx, doc_text in enumerate(doc_texts):
    doc_tensor = tokenizer.encode(doc_text, return_tensors='pt')[0]
    input_tensor = torch.cat([system_tensor, doc_tensor])

    kv_fusion.generate_independent_kv_cache(
        input_tensor=input_tensor,
        save_path='./kv_cache/no_preprocess',
        cache_id=f'doc_{doc_idx}',
        system_len=system_tensor.shape[0]
    )

# 3. 初始化文档召回器
retriever = DocumentRetriever(
    bge_model_path='/path/to/bge-m3',
    recall_method=RecallMethod.BGE
)

# 4. 计算相似度并召回
similarity_matrix = retriever.compute_similarity_matrix(doc_texts)
similar_indices = retriever.retrieve_similar_documents(
    query_idx=0, similarity_matrix=similarity_matrix, topk=3
)

# 5. 融合KV cache
similar_doc_info = [
    {
        'cache_path_prefix': f'./kv_cache/no_preprocess/doc_{idx}',
        'doc_tensor': doc_tensors[idx],
        'is_random_text': False
    }
    for idx in similar_indices
]

kv_fusion.fuse_kv_caches(
    current_doc_tensor=doc_tensors[0],
    similar_doc_info=similar_doc_info,
    system_tensor=system_tensor,
    kv_cache_dir='./kv_cache/no_preprocess',
    output_path='./kv_cache/preprocess_topk3_bge',
    output_id='doc_0',
    topk=3,
    recall_method=RecallMethod.BGE
)
```

### 方法3: 批量对比多种召回方法

快速对比不同方法的效果:

```python
from kv_fusion_preprocess import preprocess_documents_pipeline, RecallMethod

methods_to_compare = [
    RecallMethod.BGE,
    RecallMethod.RANDOM,
    RecallMethod.REPEAT_SELF,
    RecallMethod.BGE_SHUFFLED
]

for method in methods_to_compare:
    print(f"\n处理方法: {method.value}")

    result = preprocess_documents_pipeline(
        model=model,
        tokenizer=tokenizer,
        model_type='qwen2',
        documents=documents,
        system_prompt=system_prompt,
        output_dir=f'./kv_cache_output/{method.value}',
        bge_model_path='/path/to/bge-m3',
        topk=3,
        recall_method=method,
        device='cuda:0',
        random_seed=42
    )

    print(f"完成! 输出目录: {result['preprocess_dir']}")
```

## 分析不同方法的效果

使用 `analyse_kv_preprocessing.py` 来分析和可视化结果:

### 方法1: 命令行使用

```bash
python analyse_kv_preprocessing.py \
    --base_dir ./kv_cache_output \
    --cache_id doc_0 \
    --methods no_preprocess preprocess_topk3_bge preprocess_topk3_random preprocess_topk3_repeat_self \
    --output_dir ./analysis_results
```

### 方法2: Python脚本

```python
from analyse_kv_preprocessing import KVComparisonAnalyzer

# 创建对比分析器
analyzer = KVComparisonAnalyzer(base_dir='./kv_cache_output')

# 添加要对比的方法
analyzer.add_method('no_preprocess', './kv_cache_output/no_preprocess')
analyzer.add_method('bge', './kv_cache_output/preprocess_topk3_bge')
analyzer.add_method('random', './kv_cache_output/preprocess_topk3_random')
analyzer.add_method('repeat_self', './kv_cache_output/preprocess_topk3_repeat_self')
analyzer.add_method('bge_shuffled', './kv_cache_output/preprocess_topk3_bge_shuffled')

# 对比分析
comparison = analyzer.compare_all_methods(
    cache_id='doc_0',
    reference_method='no_preprocess'
)

# 生成可视化结果
analyzer.visualize_comparison(comparison, output_dir='./analysis_results')

# 查看具体数值
import json
print(json.dumps(comparison, indent=2))
```

### 输出结果

分析工具会生成以下文件:

1. **layer_difference_heatmap.png** - 每层的KV差异热图
2. **statistics_comparison.png** - 统计量对比曲线图
3. **layer_changes.png** - 层级变化模式图
4. **layer_differences.csv** - 层级差异数值表
5. **statistics.csv** - 统计量数值表
6. **comparison_results.json** - 完整的对比结果

## 研究示例

### 示例1: 研究语义相关性的重要性

对比 `BGE` (语义召回) vs `RANDOM` (随机召回):

```python
# 生成两种方法的KV cache
for method in [RecallMethod.BGE, RecallMethod.RANDOM]:
    preprocess_documents_pipeline(
        model=model, tokenizer=tokenizer, model_type='qwen2',
        documents=documents, system_prompt=system_prompt,
        output_dir=f'./experiment1/{method.value}',
        bge_model_path='/path/to/bge-m3',
        topk=3, recall_method=method, device='cuda:0'
    )

# 分析差异
analyzer = KVComparisonAnalyzer('./experiment1')
analyzer.add_method('no_preprocess', './experiment1/no_preprocess')
analyzer.add_method('bge', './experiment1/preprocess_topk3_bge')
analyzer.add_method('random', './experiment1/preprocess_topk3_random')

comparison = analyzer.compare_all_methods('doc_0', 'no_preprocess')
analyzer.visualize_comparison(comparison, './experiment1/analysis')
```

**研究问题:**
- BGE和RANDOM相比no_preprocess的差异有多大?
- 哪些层受影响最大?
- 语义相关性是否带来更一致的KV变化?

### 示例2: 研究位置信息的重要性

对比 `BGE` vs `BGE_SHUFFLED` (打乱KV位置):

```python
for method in [RecallMethod.BGE, RecallMethod.BGE_SHUFFLED]:
    preprocess_documents_pipeline(
        model=model, tokenizer=tokenizer, model_type='qwen2',
        documents=documents, system_prompt=system_prompt,
        output_dir=f'./experiment2/{method.value}',
        bge_model_path='/path/to/bge-m3',
        topk=3, recall_method=method, device='cuda:0', random_seed=42
    )
```

**研究问题:**
- 打乱KV位置是否显著改变最终的KV cache?
- 哪些层对位置信息更敏感?
- 位置信息对模型生成的影响有多大?

### 示例3: 研究topk数量的影响

对比不同的topk值:

```python
for topk in [1, 3, 5, 10]:
    preprocess_documents_pipeline(
        model=model, tokenizer=tokenizer, model_type='qwen2',
        documents=documents, system_prompt=system_prompt,
        output_dir=f'./experiment3/topk{topk}',
        bge_model_path='/path/to/bge-m3',
        topk=topk, recall_method=RecallMethod.BGE, device='cuda:0'
    )

# 批量分析
analyzer = KVComparisonAnalyzer('./experiment3')
analyzer.add_method('no_preprocess', './experiment3/no_preprocess')
for topk in [1, 3, 5, 10]:
    analyzer.add_method(f'topk{topk}', f'./experiment3/preprocess_topk{topk}_bge')

comparison = analyzer.compare_all_methods('doc_0', 'no_preprocess')
```

**研究问题:**
- topk增加时,KV的变化是否线性增长?
- 是否存在饱和点?
- 不同层对topk的敏感度如何?

## 高级用法

### 1. 使用NO_PREPROCESS_WITH_BIAS方法

这个方法不进行KV融合,而是应用分布对齐steering vector:

```python
# 首先需要准备KV分布统计文件
kv_stats = torch.load('/path/to/kv_distribution_stats.pt')

# 运行预处理
from kv_fusion_preprocess import KVCacheFusion

kv_fusion = KVCacheFusion(model, tokenizer, 'qwen2', device='cuda:0')

# 对每个文档应用steering
for doc_idx in range(len(documents)):
    kv_fusion.apply_distribution_steering(
        no_preprocess_kv_path_prefix=f'./kv_cache/no_preprocess/doc_{doc_idx}',
        output_path_prefix=f'./kv_cache/with_steering/doc_{doc_idx}',
        kv_distribution_stats=kv_stats,
        steering_alpha=1.0,  # 调整强度
        key_layers='all',  # 或指定层 '0-10', '0,5,10'
        value_layers='all',
        use_per_head=False  # 是否使用per-head steering
    )
```

### 2. 自定义召回策略

你可以继承 `DocumentRetriever` 类来实现自定义召回:

```python
from kv_fusion_preprocess import DocumentRetriever, RecallMethod

class CustomRetriever(DocumentRetriever):
    def retrieve_similar_documents(self, query_idx, similarity_matrix, topk, exclude_self):
        # 实现你的自定义召回逻辑
        # 例如: 基于主题模型、基于图结构、混合策略等
        custom_indices = your_custom_logic(query_idx, topk)
        return custom_indices
```

### 3. 分析特定层的变化

```python
from analyse_kv_preprocessing import KVCacheAnalyzer

analyzer = KVCacheAnalyzer('./kv_cache/preprocess_topk3_bge')

# 加载KV
key, value = analyzer.load_kv_cache('doc_0')

# 只分析特定层 (如第10-15层)
for layer_idx in range(10, 16):
    layer_key = key[layer_idx]
    layer_value = value[layer_idx]

    print(f"Layer {layer_idx}:")
    print(f"  Key shape: {layer_key.shape}")
    print(f"  Key mean: {layer_key.mean().item():.4f}")
    print(f"  Key std: {layer_key.std().item():.4f}")
    print(f"  Value mean: {layer_value.mean().item():.4f}")
    print(f"  Value std: {layer_value.std().item():.4f}")
```

## 关键API参考

### KVCacheFusion类

```python
class KVCacheFusion:
    def __init__(self, model, tokenizer, model_type, device, device_map=None)

    def generate_independent_kv_cache(
        self, input_tensor, save_path, cache_id,
        system_len=0, passage_len=None, reprocess_method=None
    )

    def fuse_kv_caches(
        self, current_doc_tensor, similar_doc_info, system_tensor,
        kv_cache_dir, output_path, output_id, topk=3,
        recall_method=RecallMethod.BGE, revert_rope=False,
        reprocess_method=None, shuffle_seed=None
    )

    def apply_distribution_steering(
        self, no_preprocess_kv_path_prefix, output_path_prefix,
        kv_distribution_stats, steering_alpha=1.0,
        key_layers='all', value_layers='all', use_per_head=False
    )
```

### DocumentRetriever类

```python
class DocumentRetriever:
    def __init__(self, bge_model_path=None, recall_method=RecallMethod.BGE, random_seed=42)

    def compute_similarity_matrix(
        self, documents, preprocess_scope=PreprocessScope.GLOBAL, corpus_lens=None
    ) -> np.ndarray

    def retrieve_similar_documents(
        self, query_idx, similarity_matrix, topk=3, exclude_self=True
    ) -> List[int]
```

### KVComparisonAnalyzer类

```python
class KVComparisonAnalyzer:
    def __init__(self, base_dir)

    def add_method(self, method_name, kv_cache_dir)

    def compare_all_methods(self, cache_id, reference_method='no_preprocess') -> Dict

    def visualize_comparison(self, comparison, output_dir)
```

## 完整的实验流程示例

```python
#!/usr/bin/env python3
"""
完整的KV预处理对比实验
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from kv_fusion_preprocess import preprocess_documents_pipeline, RecallMethod
from analyse_kv_preprocessing import KVComparisonAnalyzer

# 1. 加载模型
model_path = "/path/to/qwen-7b"
config = AutoConfig.from_pretrained(model_path)
config.torch_dtype = torch.float16

model = AutoModelForCausalLM.from_pretrained(
    model_path, config=config, torch_dtype=torch.float16, device_map='auto'
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 2. 准备数据
documents = [
    {'text': 'Document 1: AI safety is important...'},
    {'text': 'Document 2: Machine learning algorithms...'},
    {'text': 'Document 3: Natural language processing...'},
    {'text': 'Document 4: Computer vision techniques...'},
    {'text': 'Document 5: Reinforcement learning methods...'},
]
system_prompt = "You are a helpful AI assistant."

# 3. 对比多种方法
methods = {
    'bge': RecallMethod.BGE,
    'random': RecallMethod.RANDOM,
    'repeat_self': RecallMethod.REPEAT_SELF,
    'bge_shuffled': RecallMethod.BGE_SHUFFLED
}

for method_name, method_enum in methods.items():
    print(f"\n{'='*60}")
    print(f"Processing method: {method_name}")
    print('='*60)

    result = preprocess_documents_pipeline(
        model=model,
        tokenizer=tokenizer,
        model_type='qwen2',
        documents=documents,
        system_prompt=system_prompt,
        output_dir=f'./final_experiment/{method_name}',
        bge_model_path='/path/to/bge-m3',
        topk=3,
        recall_method=method_enum,
        device='cuda:0',
        random_seed=42
    )

    print(f"✓ Completed: {result['preprocess_dir']}")

# 4. 分析结果
print(f"\n{'='*60}")
print("Analyzing results...")
print('='*60)

analyzer = KVComparisonAnalyzer('./final_experiment')

# 添加所有方法
analyzer.add_method('no_preprocess', './final_experiment/bge/no_preprocess')
for method_name in methods.keys():
    analyzer.add_method(
        method_name,
        f'./final_experiment/{method_name}/preprocess_topk3_{method_name}'
    )

# 对每个文档进行分析
for doc_idx in range(len(documents)):
    print(f"\nAnalyzing document {doc_idx}...")

    comparison = analyzer.compare_all_methods(
        cache_id=f'doc_{doc_idx}',
        reference_method='no_preprocess'
    )

    analyzer.visualize_comparison(
        comparison,
        output_dir=f'./final_experiment/analysis/doc_{doc_idx}'
    )

print("\n✓ All analysis completed!")
print("Check results in: ./final_experiment/analysis/")
```

## 注意事项

1. **内存管理**: 生成KV cache需要较大内存,建议:
   - 使用 `device_map='auto'` 进行多GPU分布
   - 处理大量文档时分批进行
   - 及时释放不用的KV cache

2. **文件路径**: 所有路径都使用绝对路径或相对于工作目录的相对路径

3. **模型兼容性**: 目前支持 Qwen, Mistral, Llama 等主流模型,其他模型可能需要调整

4. **BGE模型**: BGE相关方法需要下载BGE-M3模型,约2GB

5. **随机种子**: 对于随机方法,确保设置相同的随机种子以保证可复现性

## 故障排查

### 问题1: CUDA out of memory

解决方案:
```python
# 使用多GPU
model = AutoModelForCausalLM.from_pretrained(
    model_path, device_map='auto', max_memory={0: "20GB", 1: "20GB"}
)

# 或减少batch size / 文档数量
```

### 问题2: BGE模型加载失败

解决方案:
```python
# 确保安装了FlagEmbedding
# pip install -U FlagEmbedding

# 指定正确的模型路径
bge_model_path = '/path/to/BAAI/bge-m3'
```

### 问题3: KV cache shape不匹配

解决方案:
```python
# 确保所有文档使用相同的system prompt
# 检查tokenizer的padding设置
tokenizer.padding_side = 'left'
```

## 扩展研究方向

1. **层级选择性**: 只在特定层进行KV融合
2. **动态topk**: 根据相似度动态调整topk
3. **混合策略**: 组合多种召回方法
4. **压缩比例**: 研究KV压缩对融合的影响
5. **跨模型对比**: 不同规模模型的KV融合效果

## 相关论文和资源

- FusionRAG: 原始论文和方法
- BGE-M3: 多语言embedding模型
- Overthinking: Steering vector方法
- KV Cache优化相关工作

---

如有问题,请参考 `analyse_data.py` 中的完整实现。
