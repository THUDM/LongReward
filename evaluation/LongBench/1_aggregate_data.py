import json, jsonlines

datasets = ['narrativeqa', 'qasper', 'multifieldqa_en', 'multifieldqa_zh', 'hotpotqa', 'musique', 'dureader', '2wikimqa', 'gov_report', 'qmsum', 'multi_news', 'vcsum']
dataset2prompt = json.load(open("./dataset2prompt.json", "r"))
data = []
for dataset in datasets:
    for i, js in enumerate(jsonlines.open(f"./datasets/{dataset}.jsonl")):
        prompt_format = dataset2prompt[dataset]
        prompt = prompt_format.format(**js)
        context, query = prompt.rsplit('\n\n', 1)
        query = query.strip()
        assert len(query) != 0
        if js["input"] != "":
            assert query == js['input'].strip(), query
        data.append({
            'idx': len(data),
            'dataset': js['dataset'],
            'context': context,
            'query': query,
            'answer': js['answers'],
        })

print(len(data))
json.dump(data, open("./longbench.json", 'w'), indent=2, ensure_ascii=False)
        