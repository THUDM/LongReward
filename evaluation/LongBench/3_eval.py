import os, json, jsonlines
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool
import traceback
import re
import sys
sys.path.append('../../')
from utils.llm_api import query_llm

pred_paths = [
    './preds/LongReward-glm4-9b-DPO.json',
#    './preds/LongReward-llama3.1-8b-DPO.json',
]

GPT_MODEL = 'gpt-4o-2024-05-13'
system_prompt = "You're a good assistant at evaluating the quality of texts."

def gpt_score_qa(prediction, ground_truth, **kwargs):
    question = kwargs["query"]
    prompt_template = """You are asked to evaluate the quality of the AI assistant's answer to user questions as an impartial judge, and your evaluation should take into account factors including correctness (high priority), and comprehensiveness (whether the assistant's answer covers all points).\nRead the AI assistant's answer and compare against the reference answer, and give an overall integer rating in 1, 2, 3 (1 = wrong or irrelevant, 2 = partially correct, 3 = correct and comprehensive) based on the above principles, strictly in the following format:"[[rating]]", e.g. "[[2]]".\n\n[Question]\n$Q$\n[Reference answer]\n$A$\n[Assistant's answer]\n$P$\nRating:"""
    prompt = prompt_template.replace("$Q$", question).replace("$A$", ground_truth).replace("$P$", prediction)
    # print(prompt)
    trys = 0
    score = None
    while (score is None) and (trys < 5):
        msg = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
        response = query_llm(msg, GPT_MODEL, temperature=0, return_usage=True)
        if isinstance(response, str) and 'Trigger' in response:
            prompt_tokens, completion_tokens = 0, 0
            score = None
            break
        try:
            response, gpt_usage = response
            prompt_tokens, completion_tokens = gpt_usage["prompt_tokens"], gpt_usage["completion_tokens"]
            score = re.findall(r"\[\[([^\]]+)\]\]", response)[-1]
            matches = re.findall(r"\d+\.\d+|\d+", score)
            score = matches[0]
        except:
            trys += 1
            response = None
            prompt_tokens, completion_tokens = 0, 0
            score = None
    if score is None:
        return 0.5
    kwargs["gpt_usage"]["responses"].append(response)
    kwargs["gpt_usage"]["prompt_tokens"] += prompt_tokens
    kwargs["gpt_usage"]["completion_tokens"] += completion_tokens
    return (int(score)-1.)/2

def gpt_score_summ(prediction, ground_truth, **kwargs):
    prompt_template = """You are asked to evaluate the quality of the AI assistant's generated summary as an impartial judge, and your evaluation should take into account factors including correctness (high priority), comprehensiveness (whether the assistant's summary covers all points), and coherence.\nRead the AI assistant's summary and compare against the reference summary, and give an overall integer rating in on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the evaluation criteria, strictly in the following format:"[[rating]]", e.g. "[[3]]".\n\n[Reference summary]\n$A$\n[Assistant's summary]\n$P$\nRating:"""
    prompt = prompt_template.replace("$A$", ground_truth).replace("$P$", prediction)
    # print(prompt)
    trys = 0
    score = None
    while (score is None) and (trys < 5):
        msg = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
        response = query_llm(msg, GPT_MODEL, temperature=0, return_usage=True)
        if isinstance(response, str) and 'Trigger' in response:
            prompt_tokens, completion_tokens = 0, 0
            score = None
            break
        try:
            response, gpt_usage = response
            prompt_tokens, completion_tokens = gpt_usage["prompt_tokens"], gpt_usage["completion_tokens"]
            score = re.findall(r"\[\[([^\]]+)\]\]", response)[-1]
            matches = re.findall(r"\d+\.\d+|\d+", score)
            score = matches[0]
        except:
            trys += 1
            response = None
            prompt_tokens, completion_tokens = 0, 0
            score = None
    if score is None:
        return 0.5
    kwargs["gpt_usage"]["responses"].append(response)
    kwargs["gpt_usage"]["prompt_tokens"] += prompt_tokens
    kwargs["gpt_usage"]["completion_tokens"] += completion_tokens
    return (int(score)-1.)/4

def process(item):
    try:
        js, fout_path = item
        dataset = js['dataset']
        query, ground_truths, prediction = js['query'], js['answer'], js['prediction']
        gpt_usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'responses': []}
        score = 0
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, query=query, gpt_usage=gpt_usage))
        js['score'] = score
        js['gpt_usage'] = gpt_usage
        with open(fout_path, "a") as fout:
            fout.write(json.dumps(js, ensure_ascii=False)+'\n')
            fout.flush()
        return js
    except:
        raise
        print(js['dataset'], js['idx'])
        print(js['query'])
        traceback.print_exc()
        print('-'*200)
        return None

dataset2metric = {
    "narrativeqa": gpt_score_qa,
    "qasper": gpt_score_qa,
    "multifieldqa_en": gpt_score_qa,
    "multifieldqa_zh": gpt_score_qa,
    "hotpotqa": gpt_score_qa,
    "2wikimqa": gpt_score_qa,
    "musique": gpt_score_qa,
    "dureader": gpt_score_qa,
    "gov_report": gpt_score_summ,
    "qmsum": gpt_score_summ,
    "multi_news": gpt_score_summ,
    "vcsum": gpt_score_summ,
}
class2dataset = {
    "single_doc_qa": ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh"],
    "multi_doc_qa": ["hotpotqa", "2wikimqa", "musique", "dureader"],
    "summarization": ["gov_report", "qmsum", "multi_news", "vcsum"],
}

if __name__ == "__main__":
    os.makedirs(f"./scores/tmp", exist_ok=True)
    for path in pred_paths:
        print(path)
        assert path.endswith('.json')
        save_name = path.split('/')[-1][:-5]
        fout_path = f"./scores/tmp/{save_name}.jsonl"
        ipts = [x for x in json.load(open(path)) if x['dataset'] in dataset2metric] 
        for trie in range(2):
            if os.path.exists(fout_path):
                with jsonlines.open(fout_path, 'r') as f:
                    opts = [x for x in f]
            else:
                opts = []
            s = set(x['idx'] for x in opts)
            need_list = [(x, fout_path) for x in ipts if x['idx'] not in s]#[:1]
            # if len(need_list) > 0:
            #     continue
            with Pool(4) as p:
                rst = list(tqdm(p.imap(process, need_list), total=len(need_list)))
            opts = opts + [x for x in rst if x is not None]
            opts = sorted(opts, key=lambda x:x['idx'])
            
            result = {'scores': {}}
            for dataset in dataset2metric:
                parts = [x for x in opts if dataset in x['dataset']]
                score = np.mean([x['score'] for x in parts]) if len(parts) > 0 else 0
                finish = sum([1 for x in opts if dataset in x['dataset']]) == sum([1 for x in ipts if dataset in x['dataset']])
                result['scores'][dataset] = {
                    'gpt_score': score,
                    'finish': finish,
                }
            gpt_usage = {
                'prompt_tokens': sum(x['gpt_usage']['prompt_tokens'] for x in opts),
                'completion_tokens': sum(x['gpt_usage']['completion_tokens'] for x in opts),
                'gpt_model': GPT_MODEL
            }
            finish = len(opts) == len(ipts)
            result['class_scores'] = {}
            for c in class2dataset:
                result['class_scores'][c] = np.mean([x['gpt_score'] for k, x in result['scores'].items() if k in class2dataset[c]])
            result.update({
                "avg_gpt_score": np.mean([x['gpt_score'] for k, x in result['scores'].items()]),
                "finish": finish,
                "gpt_usage": gpt_usage,
            })
            opts = [result] + opts
            if finish:
                break
        print(json.dumps(result, indent=2))
        json.dump(opts, open(f"./scores/{save_name}.json", "w"), indent=2, ensure_ascii=False)
        
    