"""
Simply run `eval.py` to get the evaluation results. The evaluation results will be stored in `./scores`.

python eval.py

"""
import sys
sys.path.append('../../')
import os, json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
import random
import codecs
from copy import deepcopy
from tqdm import tqdm
import traceback
import re
from utils.llm_api import query_llm

model_path = "THUDM/LongReward-glm4-9b-DPO" # THUDM/LongReward-llama3.1-8b-DPO
max_input_length = 64000

os.makedirs("predictions", exist_ok=True)
os.makedirs("scores", exist_ok=True)
test_cases = json.load(open("test_cases.json", "r"))
few_shots = json.load(open('few_shots.json', 'r', encoding='utf-8'))
template = open("prompt_fs.txt", encoding="utf-8").read()
system_prompt = "You're a good assistant at evaluating the quality of texts."
GPT_MODEL = 'gpt-4o-2024-05-13'

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def get_predictions(path, max_length):
    save_name = path.split('/')[-1]
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
    with open(f"predictions/{save_name}.txt", "w", encoding='utf-8') as f:
        for case in tqdm(test_cases):
            seed_everything(42)
            history = []
            prompt = case["prompt"]
            print("\n\n\n")
            print(prompt.split('\n\n')[-1])
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
            print(case["idx"], len(tokenized_prompt))
            if len(tokenized_prompt) > max_length:
                half = int(max_length/2)
                prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
            response, _ = model.chat(tokenizer, prompt, max_new_tokens=1024, temperature=1.0)
            print(response)
            line = response.strip().replace('\n', ' ') + '\n'
            f.write(line)
            f.flush()
            # break
    
def get_score(path):
    save_name = path.split('/')[-1]
    predictions = []
    with open(f"predictions/{save_name}.txt", "r", encoding='utf-8') as f:
        for line in f:
            predictions.append(line.strip())
    assert len(predictions) == len(test_cases)
    result = []
    lines = []
    scores = []
    total_tokens = 0
    for case, prediction in tqdm(zip(deepcopy(test_cases), predictions)):
        question, answer = case["query"], case["answer"]
        few_shot_answers = [x['answer'] for x in few_shots[case['idx'] - 1]]
        few_shot_scores = [x['score'] for x in few_shots[case['idx'] - 1]]
        few_shot_ans_scores = []
        for k in range(len(few_shot_answers)):
            few_shot_ans_scores.append(few_shot_answers[k])
            few_shot_ans_scores.append(few_shot_scores[k])
        prompt = template.format(question, answer, *few_shot_ans_scores, prediction)
        score = "none"
        trys = 0
        while (score == "none") and (trys < 5):
            msg = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
            response = query_llm(msg, GPT_MODEL, temperature=0, return_usage=True)
            try:
                response, gpt_usage = response
                num_tokens = gpt_usage["total_tokens"]
                score = re.findall(r"\[\[([^\]]+)\]\]", response)[-1]
                matches = re.findall(r"\d+\.\d+|\d+", score)
                score = matches[0]
            except:
                trys += 1
                num_tokens = 0
                score = "none"
        total_tokens += num_tokens
        scores.append(score)
        lines.append(prediction + '\t' + score + '\n')
        case.update({
            "prediction": prediction,
            "gpt_analysis": response,
            "score": score,
            "used_tokens": num_tokens
        })
        case.pop("prompt")
        result.append(case)
    try:
        scores = [float(score) for score in scores]
        total_score = sum(scores)
    except Exception as e:
        traceback.print_exc()
        total_score = "none"
    
    result.append({
        "total_score": total_score/5,
        "total_tokens": total_tokens,
    })
    
    with codecs.open(f"scores/{save_name}.json", 'w', encoding='utf-8') as fout:
        json.dump(result, fout, indent=2, ensure_ascii=False)

get_predictions(model_path, max_input_length)
get_score(model_path)