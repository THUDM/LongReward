import os, json, jsonlines
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import traceback
import torch.multiprocessing as mp
from collections import defaultdict
import time


model_path = "THUDM/LongReward-glm4-9b-DPO" # THUDM/LongReward-llama3.1-8b-DPO
max_input_length = 64000
gpus = [0,1,2,3,4,5,6,7]
gpu_per_model = 1

save_name = model_path.split('/')[-1]
save_dir = 'preds'
os.makedirs(f'{save_dir}/tmp', exist_ok=True)
fout_path = f'{save_dir}/tmp/{save_name}.jsonl'
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

def truncate_from_middle(prompt, max_input_length=None, tokenizer=None):
    if max_input_length is None:
        return prompt
    else:
        assert tokenizer is not None
        tokenized_prompt = tokenizer.encode(prompt, add_special_tokens=False)
        if len(tokenized_prompt) > max_input_length:
            half = int(max_input_length/2)
            prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        return prompt

ipts = json.load(open("longbench.json"))
parallel_num = len(gpus) // gpu_per_model
if os.path.exists(fout_path):
    with jsonlines.open(fout_path, 'r') as f:
        opts = [x for x in f]
else:
    opts = []
s = set(x['idx'] for x in opts)
need_list = [x for x in ipts if x['idx'] not in s]
print(f'Model: {model_path} | GPU per model: {gpu_per_model} | parallel num: {parallel_num}')
print(f'Already predict: {len(opts)} | Remain to predict: {len(need_list)}')

if len(need_list) > 0:
    def get_pred(rank, data):
        os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(x) for x in gpus[rank*gpu_per_model:(rank+1)*gpu_per_model])
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
        for js in tqdm(data):
            try:
                context, query = js['context'], js['query']
                prompt = context + '\n\n' + query
                prompt = truncate_from_middle(prompt, max_input_length, tokenizer)
                response, _ = model.chat(tokenizer, prompt, max_new_tokens=1024)
                res = {
                    'idx': js['idx'],
                    'dataset': js['dataset'],
                    'query': js['query'],
                    'answer': js['answer'],
                    'prediction': response,
                }
                with open(fout_path, "a", encoding="utf-8") as fout:
                    fout.write(json.dumps(res, ensure_ascii=False)+'\n')
                    fout.flush()
            except KeyboardInterrupt as e:
                raise e
            except:
                print(js['idx'])
                print(query)
                traceback.print_exc()
                print('-'*200)
        del model

    need_list_subsets = [need_list[i::parallel_num] for i in range(parallel_num)]
    processes = []
    for rank in range(parallel_num):
        p = mp.Process(target=get_pred, args=(rank, need_list_subsets[rank]))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

with jsonlines.open(fout_path, 'r') as f:
    opts = sorted([x for x in f], key=lambda x:x['idx'])
json.dump(opts, open(f'{save_dir}/{save_name}.json', 'w'), indent=2, ensure_ascii=False)