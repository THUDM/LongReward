import os, jsonlines
from collections import defaultdict
from tqdm import tqdm

dataset = "example"
idx2context = {x['idx']: x['context'] for x in tqdm(jsonlines.open(f"../data/{dataset}/sft.jsonl"))}
ipt_path = f"../data/{dataset}/scored_responses.jsonl"
os.makedirs(f"../data/{dataset}", exist_ok=True)
fout = jsonlines.open(f"../data/{dataset}/preference_pairs.jsonl", "w")

q2ans = defaultdict(list)
for js in tqdm(jsonlines.open(ipt_path)):
    idx = int(js.get('idx', js.get('id')).split('-')[-2])
    q2ans[idx].append(js)
    
data = []
for idx in tqdm(q2ans):
    query = q2ans[idx][0]['query']
    context = idx2context.get(idx, None)
    if len(idx2context):
        assert context is not None
    results = q2ans[idx]
    if len(results) < 5:
        continue
    results = sorted(results, key=lambda x:x['scores']['total'], reverse=True)
    win, lose = results[0], results[-1]
    res = {
        'idx': idx,
        'context': context,
        'query': query,
        'win_response': win['answer'],
        'lose_response': lose['answer']
    }
    fout.write(res)
