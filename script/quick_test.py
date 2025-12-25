import sys                                                                                                                  
sys.path.insert(0, '/home/shm/document/FusionRAG')  
from ktransformers.unified_process_cache import main
#   CUDA_VISIBLE_DEVICES=0  /home/wjh/anaconda3/envs/FusionRAG-old/bin/python /home/shm/document/FusionRAG/script/quick_test.py  

if __name__ == '__main__':
    main(
        model_type='mistral',
        model_path='/mnt/data/models/Mistral-7B-Instruct-v0.3',
        model_name='Mistral-7B-Instruct-v0.3',
        data_name='musique-200.jsonl',
        rate=0.15,
        preprocess=True,
        topk=10,
        revert_rope=True,
        reprocess_method='FusionRAG',
        # cache_path='/home/shm/document/FusionRAG/cache'
    )