# Introduction

## Dataset

The data is organized in files such as `hotpotqa-260-100-10-random.jsonl` and `triviaqa-270-100-10-random.jsonl`. Each example contains 10 text passages selected by KILT that are most relevant to the given question. These passages are shuffled during data loading.

## Code

Under the `ktransformers` directory, two scripts are provided for testing different model series:
- `process_cache.py` supports Mistral-based models.
- `qwen_process_cache.py` supports the Qwen model series.

## Launch Command Explanation

- `rate`: Specifies the recomputation ratio.
- `preprocess`: Indicates whether to apply KVCache preprocessing during the preprocessing stage. This corresponds to the *similarity-guided* approach described in the paper.
- `topk`: Specifies the number of similar passages selected during the preprocessing stage. This is only effective when `preprocess` is set to `True`.
- `reprocess_method`:  
  - `processCache` corresponds to the *Query-guided Reprocessing* method in the paper.  
  - `cacheBlend` refers to the SOTA *CacheBlend* method used for comparison.
  - `Cache-Craft` refers to the SOTA *CacheBlend* method used for comparison.
- `data_name`: Specifies which dataset to use.
