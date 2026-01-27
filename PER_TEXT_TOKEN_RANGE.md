# 每个文本单独设置Token范围功能

## 新功能

现在可以为**每个文本单独指定**`--start_token`和`--end_token`，而不是一刀切！

## 使用方法

### 格式

使用**逗号分隔**的值，每个值对应一个文本：

```bash
--start_token 值1,值2,值3,...
--end_token 值1,值2,值3,...
```

值的数量必须与文本数量相同！

### 示例

#### 示例1：对比不同文本的不同部分
```bash
# 文本1：全部显示（0到结尾）
# 文本2：只看90-130token
bash run_text_kv_pca.sh --text_file example_texts.txt \
    --start_token 0,90 \
    --end_token -1,130 \
    --layers 0

# 输出legend显示：
# - AI_Tech (90t)        # 全部90个token
# - AI_Tech2 (40t)       # 只显示90-130，共40个token
```

#### 示例2：都从0开始，但长度不同
```bash
# 文本1：显示0-90
# 文本2：显示0-130
bash run_text_kv_pca.sh --text_file example_texts.txt \
    --start_token 0,0 \
    --end_token 90,130 \
    --layers 0

# 输出legend显示：
# - AI_Tech (90t)        # 0-90，共90个
# - AI_Tech2 (130t)      # 0-130，共130个
```

#### 示例3：向后兼容（所有文本使用相同值）
```bash
# 所有文本都从第90个token开始
bash run_text_kv_pca.sh --text_file example_texts.txt \
    --start_token 90 \
    --end_token -1 \
    --layers 0

# 等价于：
# --start_token 90,90
# --end_token -1,-1
```

## 参数说明

| 参数 | 单个值 | 多个值 | 说明 |
|------|--------|--------|------|
| `--start_token` | `90` | `0,90` | 单个值应用到所有文本；多个值用逗号分隔，每个对应一个文本 |
| `--end_token` | `130` | `-1,130` | `-1`表示到结尾 |

## 规则

1. **值的数量必须等于文本数量**
   ```bash
   # 2个文本
   --start_token 0,90   # ✓ 正确
   --start_token 0,90,50 # ✗ 错误（3个值但只有2个文本）
   ```

2. **特殊值：-1**
   ```bash
   -1 表示到文本结尾
   --end_token -1,130  # 第1个到结尾，第2个到130
   ```

3. **索引从0开始**
   ```bash
   --start_token 0     # 从第1个token开始
   --start_token 90    # 从第91个token开始
   ```

## 实用场景

### 场景1：忽略共同前缀，只看差异部分
```bash
# AI_Tech: 全部显示
# AI_Tech2: 只看不同部分（90-183）
bash run_text_kv_pca.sh --text_file example_texts.txt \
    --start_token 0,90 \
    --end_token -1,-1 \
    --layers 0
```

### 场景2：对比特定段落
```bash
# 文本1：显示0-100
# 文本2：显示200-300
# 文本3：全部显示
bash run_text_kv_pca.sh --text_file example_texts.txt \
    --start_token 0,200,0 \
    --end_token 100,300,-1 \
    --layers 0
```

### 场景3：只分析中间部分
```bash
# 所有文本都只看50-150token的部分
bash run_text_kv_pca.sh --text_file example_texts.txt \
    --start_token 50 \
    --end_token 150 \
    --layers 0
```

## 输出说明

运行时会在控制台打印每个文本的token范围：
```
Token ranges:
  AI_Tech: [0, end)
  AI_Tech2: [90, 130)
```

legend中会显示实际使用的token数：
```
- AI_Tech (90t)
- AI_Tech2 (40t)
```

## 错误处理

如果值数量不匹配：
```bash
$ bash run_text_kv_pca.sh --text_file example_texts.txt \
    --start_token 0,90,50 \
    --layers 0

ValueError: Token range list has 3 values but there are 2 texts
```

## 同时适用于

- `visualize_text_kv_pca.py` / `run_text_kv_pca.sh` ✅
- `visualize_text_kv_tsne.py` / `run_text_kv_tsne.sh` ✅
