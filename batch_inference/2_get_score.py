import os, sys
import json, jsonlines
import multiprocessing
from tqdm import tqdm
import traceback
cur_dir = os.path.dirname(__file__)
sys.path.append(f'{cur_dir}/../')
from long_reward.auto_scorer import get_all_score


class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, val):
        pass

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
import multiprocessing.pool
class NoDaemonProcessPool(multiprocessing.pool.Pool):
    def Process(self, *args, **kwds):
        proc = super(NoDaemonProcessPool, self).Process(*args, **kwds)
        proc.__class__ = NoDaemonProcess
        return proc
    
API_LIMIT_PER_MINUTE = 200

dataset = "example"
ipt_path = f"../data/{dataset}/sampled_responses.jsonl"
fout_path = f".,/data/{dataset}/scored_responses.jsonl"
os.makedirs(f'.,/data/{dataset}', exist_ok=True)
if os.path.exists(fout_path):
    with jsonlines.open(fout_path, 'r') as f:
        s = set(x['idx'] for x in tqdm(f))
else:
    s = set()
need_list = []
for i, js in tqdm(enumerate(jsonlines.open(ipt_path, 'r'))):
    if js['idx'] not in s:
        need_list.append(js)

idx2context = {x['idx']: x['context'] for x in tqdm(jsonlines.open(f"../data/{dataset}/sft.jsonl"))}
idx2chunk_info = {x['idx']: x['info_list'] for x in tqdm(jsonlines.open(f"../data/{dataset}/chunk_info.jsonl"))}

def process(js):
    idx, query, answer = js['idx'], js['query'], js['answer']
    c_idx = int(idx.split('-')[-2])
    try:
        context, info_list = idx2context[c_idx], idx2chunk_info[c_idx]
        res = get_all_score(context, query, answer, idx, info_list)
        with open(fout_path, "a") as fout:
            fout.write(json.dumps(res, ensure_ascii=False) + '\n')
            fout.flush()
        return 1
    except KeyboardInterrupt as e:
        raise e
    except:
        traceback.print_exc()
        print("Error:", idx)
        print('-'*100)
        return 0

with NoDaemonProcessPool(API_LIMIT_PER_MINUTE) as p:
    rst = list(tqdm(p.imap(process, need_list), total=len(need_list)))
    