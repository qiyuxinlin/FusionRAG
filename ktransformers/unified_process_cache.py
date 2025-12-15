"""
Unified process cache script supporting multiple models (Mistral, PanGu, Qwen, Llama)
"""
import shutil
import torch
import os
import csv
import sys
from transformers import (
    AutoTokenizer,
    AutoConfig,
)

project_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_dir)

from ktransformers.util.utils import (
    prefill_and_generate, prefill_and_save_kv_cache, load_kv_and_generate,
    rotate_half, prefill_with_cache_and_save_preprocess,
    prepare_data, _exact_match_score, _metric_max_over_ground_truths,
    find_group_and_index, remove_unused_tokens, compute_f1
)
from ktransformers.models.custom_cache import StaticCache


def load_model(model_type, model_path, config):
    """
    Load model based on model type

    Args:
        model_type: one of ['mistral', 'pangu', 'qwen', 'llama']
        model_path: path to model
        config: model config

    Returns:
        model instance
    """
    if model_type == 'mistral':
        from models.modeling_mistral import MistralForCausalLM
        with torch.no_grad():
            model = MistralForCausalLM.from_pretrained(model_path, config=config, torch_dtype=config.torch_dtype)
    elif model_type == 'pangu':
        from models.modeling_openpangu_dense import PanguEmbeddedForCausalLM
        torch.set_default_dtype(config.torch_dtype)
        with torch.no_grad():
            model = PanguEmbeddedForCausalLM.from_pretrained(model_path, config=config, torch_dtype=config.torch_dtype)
    elif model_type == 'qwen':
        from models.modeling_qwen2 import Qwen2ForCausalLM
        torch.set_default_dtype(config.torch_dtype)
        with torch.no_grad():
            model = Qwen2ForCausalLM.from_pretrained(model_path, config=config, torch_dtype=config.torch_dtype)
    elif model_type == 'llama':
        from models.modeling_llama import LlamaForCausalLM
        torch.set_default_dtype(config.torch_dtype)
        with torch.no_grad():
            model = LlamaForCausalLM.from_pretrained(model_path, config=config, torch_dtype=config.torch_dtype)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return model


def post_process_answer(answer, model_type):
    """
    Post-process answer based on model type

    Args:
        answer: raw answer string
        model_type: one of ['mistral', 'pangu', 'qwen', 'llama']

    Returns:
        processed answer string
    """
    if model_type == 'pangu':
        answer = remove_unused_tokens(answer)
        if "Answer:" in answer:
            answer = answer.split('Answer:')[1]
    elif model_type == 'llama':
        answer = answer.split('[/INST]')[0]
    # mistral and qwen don't need special processing
    return answer


def compute_score(answer, real_answer_list, tokenizer, model_type):
    """
    Compute exact match score based on model type

    Args:
        answer: predicted answer
        real_answer_list: list of ground truth answers
        tokenizer: tokenizer instance
        model_type: one of ['mistral', 'pangu', 'qwen', 'llama']

    Returns:
        exact match score
    """
    if model_type == 'llama':
        return max([compute_f1(answer, real_answer, tokenizer) for real_answer in real_answer_list])
    else:
        return max([_exact_match_score(answer, real_answer) for real_answer in real_answer_list])


def main(model_type='mistral',
         model_path='/mnt/data/models/Mistral-7B-Instruct-v0.3',
         data_name='musique-200.jsonl',
         data_path='/mnt/data/benchmark/data/',
         cache_path='/mnt/data/processCache/',
         model_name='Mistral-7B-Instruct-v0.3',
         max_cache_len=32768,
         rate=0.2,
         dense=2,
         revert_rope=False,
         topk=10,
         preprocess=True,
         reprocess_method='cacheBlend',
         bge_model_path='/mnt/data/models/bge-m3-FP16',
         draft_model_path=None):
    """
    Main function for process cache experiments

    Args:
        model_type: one of ['mistral', 'pangu', 'qwen', 'llama']
        model_path: path to the model
        data_name: name of the dataset file
        data_path: path to the data directory
        cache_path: path to cache directory
        model_name: name of the model (for logging)
        max_cache_len: maximum cache length
        rate: compression rate
        dense: dense parameter
        revert_rope: whether to revert rope
        topk: top-k for preprocessing
        preprocess: whether to use preprocessing
        reprocess_method: reprocessing method ('cacheBlend', 'processCache', 'Cache-Craft', 'speculative_prefill')
        bge_model_path: path to BGE model for embedding
        draft_model_path: path to draft model for speculative_prefill (optional, any model can use any draft model)
    """

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # Prepare data
    prompt_data, tokens_data, question_list, real_answer_list, stop_token_id, \
    save_path, preporcess_save_path, csv_path, data_name_prefix, rouge_metrics, \
    context_rank, corpus_lens = prepare_data(
        model_name, data_path, data_name, cache_path, tokenizer,
        topk, revert_rope, preprocess, bge_model_path
    )

    # Model-specific configuration
    config._attn_implementation = "sdpa"

    # Load model
    model = load_model(model_type, model_path, config)
    model = model.to('cuda')

    # Load draft model for speculative_prefill (all models can use any draft model)
    draft_model = None
    if reprocess_method == "speculative_prefill" and draft_model_path is not None:
        # Automatically detect draft model type from path or use AutoModelForCausalLM
        from transformers import AutoModelForCausalLM
        with torch.no_grad():
            draft_config = AutoConfig.from_pretrained(draft_model_path, trust_remote_code=True)
            draft_model = AutoModelForCausalLM.from_pretrained(
                draft_model_path, config=draft_config, torch_dtype=config.torch_dtype, trust_remote_code=True
            )
        draft_model = draft_model.to('cuda')

    answer_list = []
    rouge_score = 0
    normalized_em = 0

    # Generate preprocess kv cache
    if preprocess:
        csv_file = f"{csv_path}/reprocess_method_{reprocess_method}_rate_{rate}_revert_rope_{revert_rope}_topk_{topk}.csv"
    else:
        csv_file = f"{csv_path}/reprocess_method_{reprocess_method}_rate_{rate}_revert_rope_{revert_rope}.csv"

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Question', 'Real Answer', 'Pred Answer'])

    # Adjust passage_len for different models
    passage_len_config = {
        'mistral': 32768,
        'pangu': 32768,
        'qwen': 32768,
        'llama': 32768
    }

    past_key_values = StaticCache(
        config=model.config, max_batch_size=1, max_cache_len=max_cache_len,
        device='cuda', dtype=model.dtype,
        passage_len=passage_len_config.get(model_type, 32768),
    )

    for i, iter in enumerate(tokens_data):
        system_len = iter[0].shape[0]

        if rate == 1:
            # Full Cache Recompute
            inputs = torch.cat(iter).to('cuda').unsqueeze(0)
            generated_tokens, _ = prefill_and_generate(model, tokenizer, inputs, max_new_tokens=50)
        else:
            # Cache Reuse
            # Generate KV Cache and importance
            for chunk_id, chunk in enumerate(iter[:-1]):
                if not os.path.exists(f'{save_path}/{i+1}_{chunk_id}_key.pt') or \
                   (reprocess_method == "Cache-Craft" and not os.path.exists(f'{save_path}/cachecraftattn_{i+1}_{chunk_id}.pt')):
                    passage_len = chunk.shape[0]
                    if chunk_id == 0:
                        input_tensor = chunk.unsqueeze(0)
                    else:
                        input_tensor = torch.cat((iter[0], chunk)).unsqueeze(0)
                    prefill_and_save_kv_cache(
                        model, tokenizer, past_key_values, input_tensor.cuda(), save_path=save_path,
                        example_id=i+1, chunk_id=chunk_id, system_len=iter[0].shape[0],
                        passage_len=passage_len, reprocess_method=reprocess_method
                    )

                if preprocess == True and chunk_id == 0:
                    if not os.path.exists(f"{preporcess_save_path}/{i+1}_{chunk_id}_key.pt"):
                        shutil.copy(f"{save_path}/{i+1}_{chunk_id}_key.pt", f"{preporcess_save_path}/{i+1}_{chunk_id}_key.pt")
                        shutil.copy(f"{save_path}/{i+1}_{chunk_id}_value.pt", f"{preporcess_save_path}/{i+1}_{chunk_id}_value.pt")

                elif preprocess == True and chunk_id > 0:
                    past_len = 0
                    for layer_idx in range(len(past_key_values.key_cache)):
                        past_key_values.past_tokens[layer_idx] = 0

                    # 不需要生成 preprocess
                    if os.path.exists(f"{preporcess_save_path}/{i+1}_{chunk_id}_key.pt"):
                        continue

                    corpus_passages = [iter[0]]
                    system_key_cache = torch.load(f"{save_path}/{i+1}_{0}_key.pt", weights_only=True)
                    system_value_cache = torch.load(f"{save_path}/{i+1}_{0}_value.pt", weights_only=True)

                    for layer_idx in range(len(past_key_values.key_cache)):
                        past_key_values.key_cache[layer_idx].narrow(2, 0, system_len).copy_(system_key_cache[layer_idx])
                        past_key_values.value_cache[layer_idx].narrow(2, 0, system_len).copy_(system_value_cache[layer_idx])
                        past_key_values.past_tokens[layer_idx] += system_len
                    past_len += system_len
                    id = 1

                    # 检查下 context 的 topk 有没有准备好，没有的现场生成
                    for corpus_id in context_rank[sum(corpus_lens[:i])+chunk_id-1]:
                        corpus_i, c_id = find_group_and_index(corpus_lens, corpus_id)
                        corpus_i += 1
                        c_id += 1
                        corpus_len = tokens_data[corpus_i-1][c_id].shape[0]

                        # 存在，更新到 past_key_value 中
                        if corpus_i - 1 == i and c_id == chunk_id:
                            continue

                        corpus_passages.append(tokens_data[corpus_i-1][c_id])

                        if os.path.exists(f"{save_path}/{corpus_i}_{c_id}_key.pt") and \
                           ((reprocess_method == "Cache-Craft" and os.path.exists(f'{save_path}/cachecraftattn_{corpus_i}_{c_id}.pt')) or \
                            reprocess_method != "Cache-Craft"):
                            chunk_key_cache = torch.load(f"{save_path}/{corpus_i}_{c_id}_key.pt", weights_only=True)
                            chunk_value_cache = torch.load(f"{save_path}/{corpus_i}_{c_id}_value.pt", weights_only=True)
                        else:
                            tmp_past_key_values = StaticCache(
                                config=model.config, max_batch_size=1,
                                max_cache_len=corpus_len+iter[0].shape[0]+5, device='cuda', dtype=model.dtype
                            )
                            input_tensor = torch.cat((iter[0], tokens_data[corpus_i-1][c_id])).unsqueeze(0)
                            chunk_key_cache, chunk_value_cache = prefill_and_save_kv_cache(
                                model, tokenizer, tmp_past_key_values, input_tensor.cuda(), save_path=save_path,
                                example_id=corpus_i, chunk_id=c_id, system_len=iter[0].shape[0],
                                passage_len=corpus_len, reprocess_method=reprocess_method,
                            )

                        # rope 修正
                        if revert_rope and id > 1:
                            position_ids = torch.full((1, chunk_key_cache[0].shape[2]), past_len - system_len, device='cuda')
                            # Different models have different rotary_emb access patterns
                            if model_type in ['mistral', 'qwen']:
                                cos, sin = model.model.layers[0].self_attn.rotary_emb(chunk_key_cache[0], position_ids)
                            else:  # pangu, llama
                                cos, sin = model.model.rotary_emb(chunk_key_cache[0], position_ids)
                            cos = cos.unsqueeze(1)
                            sin = sin.unsqueeze(1)
                            chunk_key_cache = (chunk_key_cache * cos) + (rotate_half(chunk_key_cache) * sin)

                        for layer_idx in range(len(past_key_values.key_cache)):
                            past_key_values.key_cache[layer_idx].narrow(2, past_len, corpus_len).copy_(chunk_key_cache[layer_idx])
                            past_key_values.value_cache[layer_idx].narrow(2, past_len, corpus_len).copy_(chunk_value_cache[layer_idx])
                            past_key_values.past_tokens[layer_idx] += corpus_len
                        past_len += corpus_len
                        id += 1

                    corpus_passages.append(chunk)
                    prefill_with_cache_and_save_preprocess(
                        model, tokenizer, past_key_values,
                        corpus_passages, preporcess_save_path,
                        i+1, chunk_id, system_len=system_len, revert_rope=revert_rope,
                        reprocess_method=reprocess_method
                    )
                    print(f'preprocess batch: {i+1}, context_id: {chunk_id}')

            if preprocess:
                load_path = preporcess_save_path
            else:
                load_path = save_path

            generated_tokens = load_kv_and_generate(
                model, tokenizer, past_key_values, iter, load_path, i+1,
                max_new_tokens=50, revert_rope=revert_rope, reprocess_method=reprocess_method,
                rate=rate, dense=dense, draft_model=draft_model, preprocess=preprocess
            )

        answer = tokenizer.decode(torch.tensor(generated_tokens[:-1]))

        # Post-process answer based on model type
        answer = post_process_answer(answer, model_type)

        print(model_name, data_name.split('.')[0], rate, topk)
        if data_name_prefix != 'samsum':
            print("question: " + question_list[i])
        print(f'batch: {i+1} preprocess: {preprocess} reprocess_method: {reprocess_method}')
        print("real answer: " + real_answer_list[i][0])
        print("answer: " + answer)

        if answer == '':
            answer_list.append(' ')
        else:
            answer_list.append(answer)

        # Compute score based on model type
        local_em = compute_score(answer, real_answer_list[i], tokenizer, model_type)
        normalized_em += local_em

        local_rouge = _metric_max_over_ground_truths(
            rouge_metrics, answer, real_answer_list[i]
        )
        rouge_score += local_rouge

        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([question_list[i], real_answer_list[i][0], answer])

        torch.cuda.empty_cache()

    # Print results
    print(rouge_score/len(tokens_data))
    print(f'em: {normalized_em/len(tokens_data)}')

    if preprocess:
        file_path = f"{csv_path}/reprocess_method_{reprocess_method}_rate_{rate}_revert_rope_{revert_rope}_topk_{topk}.txt"
    else:
        file_path = f"{csv_path}/reprocess_method_{reprocess_method}_rate_{rate}_revert_rope_{revert_rope}.txt"

    with open(file_path, 'w') as f:
        print(f'num_in_batch: {10}', file=f)
        print(rouge_score/len(tokens_data), file=f)
        print(f'em: {normalized_em/len(tokens_data)}', file=f)


if __name__ == '__main__':
    # Example usage for different models

    # Mistral
    # main(model_type='mistral',
    #      model_path='/mnt/data/models/Mistral-7B-Instruct-v0.3',
    #      model_name='Mistral-7B-Instruct-v0.3',
    #      data_name='musique-200.jsonl')

    # PanGu
    # main(model_type='pangu',
    #      model_path='/mnt/data/models/openPangu-Embedded-1B-V1.1',
    #      model_name='openPangu-Embedded-1B-V1.1',
    #      data_name='musique-200.jsonl')

    # Qwen
    # main(model_type='qwen',
    #      model_path='/mnt/data/models/Qwen2.5-7B-Instruct',
    #      model_name='Qwen2.5-7B-Instruct',
    #      cache_path='/mnt/data3/processCache/',
    #      data_name='musique-200.jsonl')

    # Llama
    # main(model_type='llama',
    #      model_path='/mnt/data/model/Llama-3.1-8B-Instruct',
    #      model_name='Llama-3.1-8B-Instruct',
    #      data_name='musique-200.jsonl')

    # Example: Run experiments
    for data_name in ['triviaqa-270-100-10-doc.jsonl', 'hotpotqa-260-100-10-doc.jsonl']:
        for topk in [10]:
            for rate in [1]:
                for method in ['cacheBlend']:
                    main(model_type='pangu',
                         model_path='/mnt/data/models/openPangu-Embedded-1B-V1.1/',
                         model_name='openPangu-Embedded-1B-V1.1',
                         rate=rate, preprocess=False, revert_rope=True,
                         cache_path='/mnt/data3/processCache/',
                         reprocess_method=method, data_name=data_name, topk=topk)
