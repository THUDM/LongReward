import sys
import re
import random
import os
from tqdm import tqdm
import json, jsonlines
import traceback
from multiprocessing import Pool

cur_dir = os.path.dirname(__file__)
sys.path.append(f'{cur_dir}/../')
from utils.retrieve import batch_search, text_split
from utils.llm_api import query_llm
import time

MODEL = "glm-4-0520"

def get_lang(input_text):
    total_len = len(input_text)
    chinese_len = 0
    english_len = 0

    for word in input_text:
        for char in word:
            if '\u4e00' <= char <= '\u9fff':
                chinese_len += 1
            elif '\u0041' <= char <= '\u007a':
                english_len += 1

    if english_len > 4*chinese_len:
        return "en"
    else:
        return "zh"

def find_last_number(s):
    numbers = re.findall(r'\[\[(\d+\.\d+|\d+)\]\]', s)
    if numbers:
        return float(numbers[-1])
    else:
        return None

# ----------------------------------------------
# helpfulness
# ----------------------------------------------
with open(f'{cur_dir}/prompts/helpfulness_few_shot.txt', "r") as fp:
    helpfulness_prompt_template = fp.read()
def score_helpfulness(query, answer):
    prompt = helpfulness_prompt_template.replace("<<question>>", query).replace("<<answer>>", answer)
    # print(prompt)
    messages = [{'role': 'user', 'content': prompt}]
    for itr in range(5):
        output = query_llm(messages, MODEL, max_new_tokens=512, temperature=0.01 if itr==0 else 0.95)
        # print(output)
        if output:
            score = find_last_number(output)
            if score is not None:
                return {
                    "analysis": output,
                    "score": score
                }
        time.sleep(1)
    print({'Unexcepted score_helpfulness output': output})
    raise NotImplementedError

# ----------------------------------------------
# logicality
# ----------------------------------------------
with open(f'{cur_dir}/prompts/logicality_few_shot.txt', "r") as fp:
    logicality_prompt_template = fp.read()
def score_logicality(query, answer):
    prompt = logicality_prompt_template.replace("<<question>>", query).replace("<<answer>>", answer)
    messages = [{'role': 'user', 'content': prompt}]
    for itr in range(5):
        output = query_llm(messages, MODEL, max_new_tokens=512, temperature=0.01 if itr==0 else 0.95)
        score = find_last_number(output)
        if output:
            if score is not None:
                return {
                    "analysis": output,
                    "score": score
                }
        time.sleep(1)
    print({'Unexcepted score_logicality output': output})
    raise NotImplementedError
    
# ----------------------------------------------
# faithfulness
# ----------------------------------------------
with open(f'{cur_dir}/prompts/find_fact_few_shot.txt', "r") as fp:
    find_fact_prompt_template = fp.read() 
def find_facts(query, answer):
    prompt = find_fact_prompt_template.replace("<<question>>", query).replace("<<answer>>", answer)
    messages = [{'role': 'user', 'content': prompt}]
    for itr in range(5):
        output = query_llm(messages, MODEL, max_new_tokens=2048, temperature=0.01 if itr==0 else 0.95)
        try:
            facts = [x.strip() for x in re.findall(r'<statement>(.*?)</statement>', output, re.DOTALL) if len(x.strip()) > 0]
            if facts:
                return facts
        except:
            time.sleep(1)
            continue
    print({"Unexpected find_facts output": output})
    raise NotImplementedError

def faithfulness_level_to_score(s):
    str = re.findall(r'\[\[([ /a-zA-Z]+)\]\]', s)
    if str:
        str = str[-1]
        if "fully".lower() in str.lower():
            return 1
        elif "partially".lower() in str.lower():
            return 0.5
        else:
            return 0
    else:
        return None

with open(f'{cur_dir}/prompts/faithfulness_few_shot.txt', "r") as fp:
    faithfulness_prompt_template = fp.read() 
        
def score_faithfulness(context, query, answer):
    facts = find_facts(query, answer)
    all_c_chunks = text_split(context, chunk_size=128)
    output = batch_search(queries=facts, contexts=all_c_chunks, k=5)
    fact_evidences_list = []
    for js in output:
        fact_evidences_list.append((js['query'], sorted(js['retrieve_results'], key=lambda x:x['start_idx'])))
    total = 0
    fact2score = {}
    for i, (fact, evidences) in enumerate(fact_evidences_list):
        prompt = faithfulness_prompt_template.replace('<<statement>>', fact).replace('<<context>>', '\n\n'.join(['[文段{}]\n{}'.format(j+1, x['content']) for j, x in enumerate(evidences)]))
        messages = [{'role': 'user', 'content': prompt}]
        for itr in range(5):
            output = query_llm(messages, MODEL, max_new_tokens=512, temperature=0.01 if itr==0 else 0.95)
            if output:
                score = faithfulness_level_to_score(output)
                if score is not None:
                    fact_evidences_list[i] = {
                        "fact": fact,
                        "analysis": output,
                        "score": score,
                        "evidences": evidences, 
                    }
                    fact2score[fact] = score
                    total += score
                    break
            time.sleep(1)
        else:
            print({"Unexpected score_faithfulness output": output})
            raise NotImplementedError
    return {
        "score": total / len(facts) * 10,
        "facts": fact2score, 
        "details": fact_evidences_list,
    }

# ----------------------------------------------
# completeness
# ----------------------------------------------
extract_info_template = {}
with open(f'{cur_dir}/prompts/extract_info_en.txt', "r") as fp:
    extract_info_template['en'] = fp.read()
with open(f'{cur_dir}/prompts/extract_info_zh.txt', "r") as fp:
    extract_info_template['zh'] = fp.read()

def extract_info(chunk, query):
    lang = get_lang(chunk)
    prompt = extract_info_template[lang].replace('<<context>>', chunk).replace('<<question>>', query)
    messages = [{'role': 'user', 'content': prompt}]
    for itr in range(5):
        output = query_llm(messages, MODEL, max_new_tokens=1024, temperature=0.01 if itr==0 else 0.95)
        if output:
            if 'No relevant information' in output:
                return "No relevant information"
            if len(output) > 0:
                return output
    print({"Unexpected extract_info output": output})
    raise NotImplementedError

def get_chunk_info(context, query, chunk_size=4096):
    chunks = text_split(context, chunk_size)
    info_list = {}
    for chunk in chunks:
        info = extract_info(chunk['content'], query)
        st, ed = round(chunk['start_idx']/chunk['total_token']*100), round(chunk['end_idx']/chunk['total_token']*100)
        info_list[f'{st}% - {ed}%'] = info
    return info_list

with open(f'{cur_dir}/prompts/completeness_few_shot.txt', "r") as fp:
    completeness_prompt_template = fp.read() 
    
def score_completeness(context, query, answer, chunk_size=4096, info_list=None):
    if info_list is None:
        info_list = get_chunk_info(context, query, chunk_size=chunk_size)
    context = '\n\n'.join([f'[文档{x}部分的有关信息]\n{y}' for x, y in info_list.items()])
    prompt = completeness_prompt_template.replace("<<question>>", query).replace('<<context>>', context).replace("<<answer>>", answer)
    messages = [{'role': 'user', 'content': prompt}]
    for itr in range(5):
        output = query_llm(messages, MODEL, max_new_tokens=512, temperature=0.01 if itr==0 else 0.95)
        if output:
            score = find_last_number(output)
            if score is not None:
                return {
                "analysis": output, 
                "score": score, 
                "info_list": info_list
                }
            time.sleep(1)
    print({"Unexpected score_completeness output": output, 'info': info_list})
    raise NotImplementedError
            
def get_all_score(context, query, answer, idx=None, info_list=None):
    d1 = score_helpfulness(query, answer)
    d2 = score_logicality(query, answer)
    d3 = score_faithfulness(context, query, answer)
    d4 = score_completeness(context, query, answer, info_list=info_list)
    res = {
        "idx": idx,
        "query": query,
        "answer": answer,
        "scores": {
            "helpfulness": d1['score'],
            "logicality": d2['score'],
            "faithfulness": d3['score'],
            "completeness": d4['score'],
            'total': d1['score'] + d2['score'] + d3['score'] + d4['score']
        },
        "details": {
            "helpfulness": d1,
            "logicality": d2,
            "completeness": d4,
            "faithfulness": d3,
        }
    }
    return res

if __name__ == "__main__":
    for js in jsonlines.open("../data/example/sft.jsonl"):
        query, answer, context = js['query'], js['answer'], js['context']
        break
    res = get_all_score(context, query, answer)
    json.dump(res, open(f"{cur_dir}/test.json", "w"), indent=2, ensure_ascii=False)
    
    
