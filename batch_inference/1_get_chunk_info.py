import sys
import random
import json, jsonlines
from multiprocessing import Pool
import multiprocessing
from tqdm import tqdm
import traceback
import os
cur_dir = os.path.dirname(__file__)
sys.path.append(f'{cur_dir}/../')
from long_reward.auto_scorer import get_chunk_info

random.seed(666)
API_LIMIT_PER_MINUTE = 100
TIMEOUT = 600

dataset = "example"
ipts = [x for x in tqdm(jsonlines.open(f"../data/{dataset}/sft.jsonl"))]
fout_path = f"../data/{dataset}/chunk_info.jsonl"
os.makedirs(f"../data/{dataset}", exist_ok=True)
if os.path.exists(fout_path):
    with jsonlines.open(fout_path, 'r') as f:
        s = set(x['idx'] for x in tqdm(f))
else:
    s = set()
need_list = [x for x in tqdm(ipts) if x['idx'] not in s]#[:1]

def process(js):
    try:
        res =  get_chunk_info(js['context'], js['query'], chunk_size=4096)
        with open(fout_path, "a") as fout:
            fout.write(json.dumps({"idx": js['idx'], "query": js['query'], "info_list": res}, ensure_ascii=False) + '\n')
            fout.flush()
        return 1
    except KeyboardInterrupt as e:
        raise e
    except:
        traceback.print_exc()
        print("Error:", js['idx'])
        print('-'*100)
        return 0
    
with Pool(API_LIMIT_PER_MINUTE) as p:
    rst = list(tqdm(p.imap(process, need_list), total=len(need_list)))
    